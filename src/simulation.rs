//! Conditional simulation via sequential Gaussian simulation (SGS).
//!
//! Given a set of observed stations and a variogram model, these functions draw one realization
//! of the underlying Gaussian random field at each requested target location such that:
//!
//! - The realization honors every conditioning observation exactly (up to kriging solver
//!   tolerance).
//! - Targets are visited in a user-chosen order (default is input order). At each target the
//!   value is sampled from `N(μ̂, σ²̂)` where `μ̂` and `σ²̂` come from kriging the target against
//!   *all* already-observed-or-simulated values (conditioning + previously simulated). The
//!   sampled value is then appended to the conditioning set for subsequent targets — this is
//!   the defining "sequential" step.
//!
//! SGS reproduces the covariance structure of the variogram (asymptotically, and exactly in
//! expectation) while honoring the data. It is appropriate when users need uncertainty via
//! multiple realizations rather than a single best-guess surface.
//!
//! ## Harness
//!
//! Convert a fitted kriging model with its `into_conditioner` method, then pass the resulting
//! [`KrigingConditioner`] to [`sequential_gaussian_simulate`] or
//! [`sequential_binomial_simulate`].
//!
//! All paths accept a shared [`SimulationOptions`] (seed + optional target visit order).
//!
//! ## RNG
//!
//! The simulator uses a seedable xoshiro-style PRNG so realizations are reproducible without
//! an external `rand` runtime dependency. For production work that needs rigorous random
//! number quality, callers can post-process or wrap this module's scalar outputs.

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::conditioner::{KrigingConditioner, LogitScale};
use crate::utils::logistic;

/// Options controlling conditional simulation.
#[derive(Debug, Clone)]
pub struct SimulationOptions {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Optional permutation of target indices specifying the order in which targets are
    /// visited. Must be a permutation of `0..targets.len()`. If `None`, targets are visited
    /// in input order.
    pub target_order: Option<Vec<usize>>,
}

impl SimulationOptions {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            target_order: None,
        }
    }
}

/// Result of a binomial conditional simulation. Contains samples on both the logit and
/// prevalence scales, in the **original** target input order.
///
/// Simulation happens on the logit scale (where the Gaussian assumption is natural);
/// `prevalence_samples[i] = logistic(logit_samples[i])`.
#[derive(Debug, Clone)]
pub struct BinomialSimulationResult {
    /// Simulated logit values (unbounded).
    pub logit_samples: Vec<Real>,
    /// Simulated prevalence values in (0, 1).
    pub prevalence_samples: Vec<Real>,
}

/// Result of a multi-realization binomial conditional simulation.
///
/// Each field is a flat row-major buffer of length `n_realizations * n_targets`. Row `k`
/// (entries `[k * n_targets .. (k + 1) * n_targets]`) holds realization `k` in the
/// **original** target input order. By construction
/// `prevalence_samples[i] = logistic(logit_samples[i])` element-wise.
#[derive(Debug, Clone)]
pub struct BinomialSimulationManyResult {
    /// Number of independent realizations stacked into the buffers.
    pub n_realizations: usize,
    /// Number of target locations per realization.
    pub n_targets: usize,
    /// Simulated logit values, row-major `[n_realizations * n_targets]`.
    pub logit_samples: Vec<Real>,
    /// Simulated prevalence values, row-major `[n_realizations * n_targets]`.
    pub prevalence_samples: Vec<Real>,
}

/// Alias for [`BinomialSimulationResult`] in SGS harness docs.
pub type BinomialSgsOutput = BinomialSimulationResult;

/// Kriging variance below this threshold means the target is already conditioned.
const SGS_CONDITIONED_VARIANCE_EPS: Real = 1e-10;

/// One continuous SGS realization in original target input order.
pub fn sequential_gaussian_simulate<S>(
    mut conditioner: KrigingConditioner<S>,
    targets: &[S],
    options: SimulationOptions,
) -> Result<Vec<Real>, KrigingError>
where
    S: Copy,
{
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;
    let mut rng = Rng::new(options.seed);
    let mut out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let target = targets[target_idx];
        let conditional = conditioner.predict(target)?;
        let sigma = conditional.variance.max(0.0).sqrt();
        let sampled = conditional.mean + sigma * rng.next_standard_normal();
        out[target_idx] = sampled;
        if conditional.variance > SGS_CONDITIONED_VARIANCE_EPS {
            conditioner.append_condition(target, sampled)?;
        }
    }

    Ok(out)
}

/// One binomial SGS realization (logit + prevalence) in original target input order.
pub fn sequential_binomial_simulate<S>(
    mut conditioner: KrigingConditioner<S, LogitScale>,
    targets: &[S],
    options: SimulationOptions,
) -> Result<BinomialSimulationResult, KrigingError>
where
    S: Copy,
{
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;
    let mut rng = Rng::new(options.seed);
    let mut logit_out = vec![0.0 as Real; n_targets];
    let mut prevalence_out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let target = targets[target_idx];
        let conditional = conditioner.predict(target)?;
        let sigma = conditional.variance.max(0.0).sqrt();
        let logit_sample = conditional.mean + sigma * rng.next_standard_normal();
        logit_out[target_idx] = logit_sample;
        prevalence_out[target_idx] = logistic(logit_sample);
        if conditional.variance > SGS_CONDITIONED_VARIANCE_EPS {
            conditioner.append_condition(target, logit_sample)?;
        }
    }

    Ok(BinomialSimulationResult {
        logit_samples: logit_out,
        prevalence_samples: prevalence_out,
    })
}

/// Multi-realization continuous SGS. Row `k` matches [`sequential_gaussian_simulate`] with
/// `seed = base_seed + k`.
pub fn sequential_gaussian_simulate_many<S>(
    template: KrigingConditioner<S>,
    targets: &[S],
    n_realizations: usize,
    base_seed: u64,
    target_order: Option<Vec<usize>>,
) -> Result<Vec<Real>, KrigingError>
where
    S: Copy,
{
    validate_n_realizations(n_realizations)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, target_order)?;
    let mut out = vec![0.0 as Real; n_realizations * n_targets];

    for k in 0..n_realizations {
        let mut rng = Rng::new(base_seed.wrapping_add(k as u64));
        let mut conditioner = template.clone();
        let row_offset = k * n_targets;
        for &target_idx in &order {
            let target = targets[target_idx];
            let conditional = conditioner.predict(target)?;
            let sigma = conditional.variance.max(0.0).sqrt();
            let sampled = conditional.mean + sigma * rng.next_standard_normal();
            out[row_offset + target_idx] = sampled;
            if conditional.variance > SGS_CONDITIONED_VARIANCE_EPS {
                conditioner.append_condition(target, sampled)?;
            }
        }
    }

    Ok(out)
}

/// Multi-realization binomial SGS. Row `k` matches [`sequential_binomial_simulate`] with
/// `seed = base_seed + k`.
pub fn sequential_binomial_simulate_many<S>(
    template: KrigingConditioner<S, LogitScale>,
    targets: &[S],
    n_realizations: usize,
    base_seed: u64,
    target_order: Option<Vec<usize>>,
) -> Result<BinomialSimulationManyResult, KrigingError>
where
    S: Copy,
{
    validate_n_realizations(n_realizations)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, target_order)?;
    let mut logit_out = vec![0.0 as Real; n_realizations * n_targets];
    let mut prevalence_out = vec![0.0 as Real; n_realizations * n_targets];

    for k in 0..n_realizations {
        let mut rng = Rng::new(base_seed.wrapping_add(k as u64));
        let mut conditioner = template.clone();
        let row_offset = k * n_targets;
        for &target_idx in &order {
            let target = targets[target_idx];
            let conditional = conditioner.predict(target)?;
            let sigma = conditional.variance.max(0.0).sqrt();
            let logit_sample = conditional.mean + sigma * rng.next_standard_normal();
            logit_out[row_offset + target_idx] = logit_sample;
            prevalence_out[row_offset + target_idx] = logistic(logit_sample);
            if conditional.variance > SGS_CONDITIONED_VARIANCE_EPS {
                conditioner.append_condition(target, logit_sample)?;
            }
        }
    }

    Ok(BinomialSimulationManyResult {
        n_realizations,
        n_targets,
        logit_samples: logit_out,
        prevalence_samples: prevalence_out,
    })
}

#[derive(Debug, Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut splitmix_state = seed;
        let mut next = || {
            splitmix_state = splitmix_state.wrapping_add(0x9E3779B97F4A7C15);
            let mut value = splitmix_state;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
            value ^ (value >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let temporary = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_unit(&mut self) -> Real {
        let value = (self.next_u64() >> 11) as Real;
        let scale = (1u64 << 53) as Real;
        (value + 0.5) / scale
    }

    fn next_standard_normal(&mut self) -> Real {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * (std::f64::consts::PI as Real) * u2;
        radius * theta.cos()
    }
}

fn resolve_target_order(
    n_targets: usize,
    target_order: Option<Vec<usize>>,
) -> Result<Vec<usize>, KrigingError> {
    match target_order {
        None => Ok((0..n_targets).collect()),
        Some(order) => {
            if order.len() != n_targets {
                return Err(KrigingError::InvalidInput(format!(
                    "target_order length ({}) must equal number of targets ({n_targets})",
                    order.len()
                )));
            }
            let mut seen = vec![false; n_targets];
            for &index in &order {
                if index >= n_targets {
                    return Err(KrigingError::InvalidInput(format!(
                        "target_order contains out-of-range index {index} (n_targets={n_targets})"
                    )));
                }
                if seen[index] {
                    return Err(KrigingError::InvalidInput(format!(
                        "target_order contains duplicate index {index}"
                    )));
                }
                seen[index] = true;
            }
            Ok(order)
        }
    }
}

fn validate_n_realizations(n_realizations: usize) -> Result<(), KrigingError> {
    if n_realizations == 0 {
        return Err(KrigingError::InvalidInput(
            "n_realizations must be >= 1".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::error::KrigingError;
    use crate::geo_dataset::GeoDataset;
    use crate::kriging::binomial::{
        BinomialKrigingModel, BinomialObservation, BinomialPrior, HeteroskedasticBinomialConfig,
    };
    use crate::kriging::conditioner::{KrigingConditioner, LogitScale};
    use crate::kriging::ordinary::OrdinaryKrigingModel;
    use crate::kriging::simple::SimpleKrigingModel;
    use crate::kriging::universal::UniversalKrigingModel;
    use crate::kriging::universal::UniversalTrend;
    use crate::projected::{
        Anisotropy2D, BinomialProjectedKrigingModel, ProjectedBinomialObservation, ProjectedCoord,
        ProjectedDataset, ProjectedKrigingModel,
    };
    use crate::spacetime::{
        GeoMetric, SpaceTimeBinomialKrigingModel, SpaceTimeBinomialObservation, SpaceTimeCoord,
        SpaceTimeDataset, SpaceTimeOrdinaryKrigingModel, SpaceTimeSimpleKrigingModel,
        SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend, SpaceTimeVariogram,
    };
    use crate::utils::logistic;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn setup() -> (Vec<GeoCoord>, Vec<Real>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        (coords, values, variogram)
    }

    fn ordinary_conditioner(
        coords: &[GeoCoord],
        values: &[Real],
        variogram: VariogramModel,
    ) -> Result<KrigingConditioner<GeoCoord>, KrigingError> {
        OrdinaryKrigingModel::new(
            GeoDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
        )?
        .into_conditioner()
    }

    fn simple_conditioner(
        coords: &[GeoCoord],
        values: &[Real],
        variogram: VariogramModel,
        mean: Real,
    ) -> Result<KrigingConditioner<GeoCoord>, KrigingError> {
        SimpleKrigingModel::new(
            GeoDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
            mean,
        )?
        .into_conditioner()
    }

    fn universal_conditioner(
        coords: &[GeoCoord],
        values: &[Real],
        variogram: VariogramModel,
        trend: UniversalTrend,
    ) -> Result<KrigingConditioner<GeoCoord>, KrigingError> {
        UniversalKrigingModel::new(
            GeoDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
            trend,
        )?
        .into_conditioner()
    }

    fn projected_conditioner(
        coords: &[ProjectedCoord],
        values: &[Real],
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
    ) -> Result<KrigingConditioner<ProjectedCoord>, KrigingError> {
        ProjectedKrigingModel::new(
            ProjectedDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
            anisotropy,
        )?
        .into_conditioner()
    }

    fn binomial_conditioner(
        coords: &[GeoCoord],
        successes: &[u32],
        trials: &[u32],
        variogram: VariogramModel,
        prior: BinomialPrior,
    ) -> Result<KrigingConditioner<GeoCoord, LogitScale>, KrigingError> {
        if coords.len() != successes.len() || coords.len() != trials.len() {
            return Err(KrigingError::DimensionMismatch(
                "conditioning arrays must have equal length".to_string(),
            ));
        }
        let observations = coords
            .iter()
            .copied()
            .zip(successes.iter().copied())
            .zip(trials.iter().copied())
            .filter(|(_, trials)| *trials > 0)
            .map(|((coord, successes), trials)| BinomialObservation::new(coord, successes, trials))
            .collect::<Result<Vec<_>, _>>()?;
        BinomialKrigingModel::new_with_prior(observations, variogram, prior)?
            .into_model()
            .into_conditioner()
    }

    fn projected_binomial_conditioner(
        coords: &[ProjectedCoord],
        successes: &[u32],
        trials: &[u32],
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
        prior: BinomialPrior,
    ) -> Result<KrigingConditioner<ProjectedCoord, LogitScale>, KrigingError> {
        if coords.len() != successes.len() || coords.len() != trials.len() {
            return Err(KrigingError::DimensionMismatch(
                "conditioning arrays must have equal length".to_string(),
            ));
        }
        let observations = coords
            .iter()
            .copied()
            .zip(successes.iter().copied())
            .zip(trials.iter().copied())
            .filter(|(_, trials)| *trials > 0)
            .map(|((coord, successes), trials)| {
                ProjectedBinomialObservation::new(coord, successes, trials)
            })
            .collect::<Result<Vec<_>, _>>()?;
        BinomialProjectedKrigingModel::new_with_prior(
            observations,
            variogram,
            anisotropy,
            prior,
            HeteroskedasticBinomialConfig::default(),
        )?
        .into_model()
        .into_conditioner()
    }

    fn spacetime_ordinary_conditioner<M>(
        metric: M,
        coords: &[SpaceTimeCoord<M::Coord>],
        values: &[Real],
        variogram: SpaceTimeVariogram,
    ) -> Result<KrigingConditioner<SpaceTimeCoord<M::Coord>>, KrigingError>
    where
        M: crate::spacetime::SpatialMetric + 'static,
        M::Coord: 'static,
        M::Prepared: 'static,
    {
        SpaceTimeOrdinaryKrigingModel::new(
            metric,
            SpaceTimeDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
        )?
        .into_conditioner()
    }

    fn spacetime_simple_conditioner<M>(
        metric: M,
        coords: &[SpaceTimeCoord<M::Coord>],
        values: &[Real],
        variogram: SpaceTimeVariogram,
        mean: Real,
    ) -> Result<KrigingConditioner<SpaceTimeCoord<M::Coord>>, KrigingError>
    where
        M: crate::spacetime::SpatialMetric + 'static,
        M::Coord: 'static,
        M::Prepared: 'static,
    {
        SpaceTimeSimpleKrigingModel::new(
            metric,
            SpaceTimeDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
            mean,
        )?
        .into_conditioner()
    }

    fn spacetime_universal_conditioner<M>(
        metric: M,
        coords: &[SpaceTimeCoord<M::Coord>],
        values: &[Real],
        variogram: SpaceTimeVariogram,
        trend: SpaceTimeUniversalTrend,
    ) -> Result<KrigingConditioner<SpaceTimeCoord<M::Coord>>, KrigingError>
    where
        M: crate::spacetime::SpatialBasis + 'static,
        M::Coord: 'static,
        M::Prepared: 'static,
    {
        SpaceTimeUniversalKrigingModel::new(
            metric,
            SpaceTimeDataset::new(coords.to_vec(), values.to_vec())?,
            variogram,
            trend,
        )?
        .into_conditioner()
    }

    fn spacetime_binomial_conditioner<M>(
        metric: M,
        coords: &[SpaceTimeCoord<M::Coord>],
        successes: &[u32],
        trials: &[u32],
        variogram: SpaceTimeVariogram,
        prior: BinomialPrior,
    ) -> Result<KrigingConditioner<SpaceTimeCoord<M::Coord>, LogitScale>, KrigingError>
    where
        M: crate::spacetime::SpatialMetric + 'static,
        M::Coord: 'static,
        M::Prepared: 'static,
    {
        if coords.len() != successes.len() || coords.len() != trials.len() {
            return Err(KrigingError::DimensionMismatch(
                "conditioning arrays must have equal length".to_string(),
            ));
        }
        let observations = coords
            .iter()
            .copied()
            .zip(successes.iter().copied())
            .zip(trials.iter().copied())
            .filter(|(_, trials)| *trials > 0)
            .map(|((coord, successes), trials)| {
                SpaceTimeBinomialObservation::new(coord, successes, trials)
            })
            .collect::<Result<Vec<_>, _>>()?;
        SpaceTimeBinomialKrigingModel::new_with_prior(
            metric,
            observations,
            variogram,
            prior,
            HeteroskedasticBinomialConfig::default(),
        )?
        .into_model()
        .into_conditioner()
    }

    // ---- Ordinary ----------------------------------------------------------

    #[test]
    fn same_seed_gives_identical_realization() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        assert_eq!(a, b, "same seed should yield identical realization");
    }

    #[test]
    fn different_seeds_give_different_realizations() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(1),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(2),
        )
        .unwrap();
        assert!(a != b, "different seeds should differ at least somewhere");
    }

    #[test]
    fn realization_honors_conditioning_when_target_coincides() {
        let (c, v, vg) = setup();
        let targets = vec![c[2]];
        let out = sequential_gaussian_simulate(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(123),
        )
        .unwrap();
        assert!(
            (out[0] - v[2]).abs() < 0.2,
            "sampled value {} should be close to observed {}",
            out[0],
            v[2]
        );
    }

    #[test]
    fn custom_target_order_still_returns_results_in_input_order() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
            GeoCoord::try_new(0.75, 0.25).unwrap(),
        ];
        let mut opts = SimulationOptions::new(7);
        opts.target_order = Some(vec![2, 0, 1]);
        let out =
            sequential_gaussian_simulate(ordinary_conditioner(&c, &v, vg).unwrap(), &targets, opts)
                .unwrap();
        assert_eq!(out.len(), targets.len());
        for v in out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn invalid_permutation_is_rejected() {
        let (c, v, vg) = setup();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap(); 3];
        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 0, 1]);
        assert!(
            sequential_gaussian_simulate(ordinary_conditioner(&c, &v, vg).unwrap(), &targets, opts)
                .is_err()
        );

        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 1, 5]);
        assert!(
            sequential_gaussian_simulate(ordinary_conditioner(&c, &v, vg).unwrap(), &targets, opts)
                .is_err()
        );

        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 1]);
        assert!(
            sequential_gaussian_simulate(ordinary_conditioner(&c, &v, vg).unwrap(), &targets, opts)
                .is_err()
        );
    }

    // ---- Simple ------------------------------------------------------------

    #[test]
    fn simple_simulation_is_deterministic_for_same_seed() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.3, 0.7).unwrap(),
        ];
        let mean = 2.5;
        let a = sequential_gaussian_simulate(
            simple_conditioner(&c, &v, vg, mean).unwrap(),
            &targets,
            SimulationOptions::new(9),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            simple_conditioner(&c, &v, vg, mean).unwrap(),
            &targets,
            SimulationOptions::new(9),
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), targets.len());
        for x in &a {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn simple_simulation_honors_conditioning_at_coincident_target() {
        let (c, v, vg) = setup();
        let targets = vec![c[1]];
        let out = sequential_gaussian_simulate(
            simple_conditioner(&c, &v, vg, 2.5).unwrap(),
            &targets,
            SimulationOptions::new(33),
        )
        .unwrap();
        assert!(
            (out[0] - v[1]).abs() < 0.2,
            "sampled value {} should be close to observed {}",
            out[0],
            v[1]
        );
    }

    // ---- Universal ---------------------------------------------------------

    #[test]
    fn universal_simulation_linear_trend_is_finite_and_deterministic() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.4, 0.6).unwrap(),
            GeoCoord::try_new(0.6, 0.4).unwrap(),
        ];
        let trend = UniversalTrend::Linear;
        let a = sequential_gaussian_simulate(
            universal_conditioner(&c, &v, vg, trend).unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            universal_conditioner(&c, &v, vg, trend).unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        assert_eq!(a, b);
        for x in &a {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn universal_simulation_rejects_too_few_conditioning_points() {
        // Linear trend needs n >= 3+1 = 4. Three conditioning points should fail up front.
        let c = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let v = vec![1.0, 2.0, 3.0];
        let vg = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let result = universal_conditioner(&c, &v, vg, UniversalTrend::Linear)
            .and_then(|sim| sequential_gaussian_simulate(sim, &targets, SimulationOptions::new(0)));
        match result {
            Err(KrigingError::InsufficientData(n)) => assert_eq!(n, 4),
            other => panic!("expected InsufficientData(4), got {other:?}"),
        }
    }

    // ---- Projected ---------------------------------------------------------

    #[test]
    fn projected_simulation_is_deterministic_for_same_seed() {
        let coords = vec![
            ProjectedCoord::new(0.0, 0.0),
            ProjectedCoord::new(0.0, 1.0),
            ProjectedCoord::new(1.0, 0.0),
            ProjectedCoord::new(1.0, 1.0),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let vg = VariogramModel::new(0.1, 5.0, 2.0, VariogramType::Exponential).unwrap();
        let targets = vec![
            ProjectedCoord::new(0.5, 0.5),
            ProjectedCoord::new(0.25, 0.75),
        ];
        let aniso = Anisotropy2D::isotropic();
        let a = sequential_gaussian_simulate(
            projected_conditioner(&coords, &values, vg, aniso).unwrap(),
            &targets,
            SimulationOptions::new(5),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            projected_conditioner(&coords, &values, vg, aniso).unwrap(),
            &targets,
            SimulationOptions::new(5),
        )
        .unwrap();
        assert_eq!(a, b);
        for x in &a {
            assert!(x.is_finite());
        }
    }

    // ---- Binomial projected -----------------------------------------------

    fn binomial_projected_setup() -> (Vec<ProjectedCoord>, Vec<u32>, Vec<u32>, VariogramModel) {
        let coords = vec![
            ProjectedCoord::new(0.0, 0.0),
            ProjectedCoord::new(0.0, 1.0),
            ProjectedCoord::new(1.0, 0.0),
            ProjectedCoord::new(1.0, 1.0),
        ];
        let successes = vec![3, 7, 4, 9];
        let trials = vec![10, 12, 9, 15];
        let vg = VariogramModel::new(0.1, 1.0, 1.5, VariogramType::Exponential).unwrap();
        (coords, successes, trials, vg)
    }

    #[test]
    fn binomial_projected_simulation_is_deterministic_for_same_seed() {
        let (coords, successes, trials, vg) = binomial_projected_setup();
        let targets = vec![ProjectedCoord::new(0.5, 0.5)];
        let a = sequential_binomial_simulate(
            projected_binomial_conditioner(
                &coords,
                &successes,
                &trials,
                vg,
                Anisotropy2D::isotropic(),
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        let b = sequential_binomial_simulate(
            projected_binomial_conditioner(
                &coords,
                &successes,
                &trials,
                vg,
                Anisotropy2D::isotropic(),
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        assert_eq!(a.logit_samples, b.logit_samples);
        assert_eq!(a.prevalence_samples, b.prevalence_samples);
        for p in &a.prevalence_samples {
            assert!(*p > 0.0 && *p < 1.0);
        }
    }

    #[test]
    fn binomial_projected_many_matches_seeded_singles() {
        let (coords, successes, trials, vg) = binomial_projected_setup();
        let targets = vec![
            ProjectedCoord::new(0.25, 0.25),
            ProjectedCoord::new(0.75, 0.5),
        ];
        let n_real = 3usize;
        let base = 100u64;
        let many = sequential_binomial_simulate_many(
            projected_binomial_conditioner(
                &coords,
                &successes,
                &trials,
                vg,
                Anisotropy2D::isotropic(),
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            n_real,
            base,
            None,
        )
        .unwrap();
        assert_eq!(many.n_realizations, n_real);
        assert_eq!(many.n_targets, targets.len());
        for k in 0..n_real {
            let one = sequential_binomial_simulate(
                projected_binomial_conditioner(
                    &coords,
                    &successes,
                    &trials,
                    vg,
                    Anisotropy2D::isotropic(),
                    BinomialPrior::default(),
                )
                .unwrap(),
                &targets,
                SimulationOptions::new(base + k as u64),
            )
            .unwrap();
            for j in 0..targets.len() {
                let off = k * targets.len() + j;
                assert!((many.logit_samples[off] - one.logit_samples[j]).abs() < 1e-9);
                assert!((many.prevalence_samples[off] - one.prevalence_samples[j]).abs() < 1e-9);
            }
        }
    }

    // ---- Binomial ----------------------------------------------------------

    fn binomial_setup() -> (Vec<GeoCoord>, Vec<u32>, Vec<u32>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let successes = vec![3, 7, 4, 9];
        let trials = vec![10, 12, 9, 15];
        let vg = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        (coords, successes, trials, vg)
    }

    #[test]
    fn binomial_simulation_reports_both_scales_consistently() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let prior = BinomialPrior::default();
        let result = sequential_binomial_simulate(
            binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(17),
        )
        .unwrap();
        assert_eq!(result.logit_samples.len(), targets.len());
        assert_eq!(result.prevalence_samples.len(), targets.len());
        for (logit, prev) in result
            .logit_samples
            .iter()
            .zip(result.prevalence_samples.iter())
        {
            assert!(logit.is_finite(), "logit must be finite: got {logit}");
            assert!(*prev > 0.0 && *prev < 1.0, "prevalence must be in (0,1)");
            assert!(
                (*prev - logistic(*logit)).abs() < 1e-12,
                "prevalence must equal logistic(logit): logit={logit}, prev={prev}"
            );
        }
    }

    #[test]
    fn binomial_simulation_is_deterministic_for_same_seed() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let prior = BinomialPrior::default();
        let a = sequential_binomial_simulate(
            binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(17),
        )
        .unwrap();
        let b = sequential_binomial_simulate(
            binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(17),
        )
        .unwrap();
        assert_eq!(a.logit_samples, b.logit_samples);
        assert_eq!(a.prevalence_samples, b.prevalence_samples);
    }

    #[test]
    fn binomial_simulation_drops_zero_trial_stations_from_conditioning() {
        // Injecting a trials==0 station between real observations must not poison the pool
        // and must not change the realization when compared against an equivalent
        // zero-station-free call.
        let (mut c, mut s, mut t, vg) = binomial_setup();
        c.insert(2, GeoCoord::try_new(0.5, 0.5).unwrap());
        s.insert(2, 0);
        t.insert(2, 0);
        let targets = vec![GeoCoord::try_new(0.75, 0.25).unwrap()];
        let prior = BinomialPrior::default();
        let with_zero = sequential_binomial_simulate(
            binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(4),
        )
        .unwrap();

        let (c2, s2, t2, _) = binomial_setup();
        let without = sequential_binomial_simulate(
            binomial_conditioner(&c2, &s2, &t2, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(4),
        )
        .unwrap();
        assert_eq!(with_zero.logit_samples, without.logit_samples);
        assert_eq!(with_zero.prevalence_samples, without.prevalence_samples);
    }

    #[test]
    fn binomial_simulation_rejects_all_zero_trials() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let successes = vec![0, 0];
        let trials = vec![0, 0];
        let vg = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let result =
            binomial_conditioner(&coords, &successes, &trials, vg, BinomialPrior::default())
                .and_then(|sim| {
                    sequential_binomial_simulate(sim, &targets, SimulationOptions::new(0))
                });
        match result {
            Err(KrigingError::InsufficientData(n)) => assert_eq!(n, 2),
            other => panic!("expected InsufficientData(2), got {other:?}"),
        }
    }

    #[test]
    fn binomial_simulation_respects_custom_prior() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let default_prior = BinomialPrior::default();
        let custom_prior = BinomialPrior::new(2.0, 5.0).unwrap();
        let a = sequential_binomial_simulate(
            binomial_conditioner(&c, &s, &t, vg, default_prior).unwrap(),
            &targets,
            SimulationOptions::new(21),
        )
        .unwrap();
        let b = sequential_binomial_simulate(
            binomial_conditioner(&c, &s, &t, vg, custom_prior).unwrap(),
            &targets,
            SimulationOptions::new(21),
        )
        .unwrap();
        assert!(
            (a.logit_samples[0] - b.logit_samples[0]).abs() > 1e-9,
            "different priors should produce different logit samples"
        );
    }

    // ----- Space–time SGS ---------------------------------------------------

    fn st_setup() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>, SpaceTimeVariogram) {
        let coords = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 1.5, 4.5];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        (coords, values, vg)
    }

    #[test]
    fn st_same_seed_gives_identical_realization() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn st_different_seeds_give_different_realizations() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(1),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(2),
        )
        .unwrap();
        assert!(a != b);
    }

    #[test]
    fn st_realization_honors_conditioning_when_target_coincides() {
        let (c, v, vg) = st_setup();
        let targets = vec![c[2]];
        let out = sequential_gaussian_simulate(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(7),
        )
        .unwrap();
        // Kriging at a known conditioning location returns the observed value with ~zero
        // variance, so the sample should equal the observation up to solver tolerance.
        assert!(
            (out[0] - v[2]).abs() < 1e-3,
            "expected ~{}, got {}",
            v[2],
            out[0]
        );
    }

    #[test]
    fn st_rejects_mismatched_arrays() {
        let (c, _, vg) = st_setup();
        let v_bad = vec![1.0, 2.0];
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap()];
        assert!(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v_bad, vg)
                .and_then(|sim| sequential_gaussian_simulate(
                    sim,
                    &targets,
                    SimulationOptions::new(1)
                ))
                .is_err()
        );
    }

    #[test]
    fn st_simple_sim_runs_with_known_mean() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let mean = v.iter().copied().sum::<Real>() / v.len() as Real;
        let out = sequential_gaussian_simulate(
            spacetime_simple_conditioner(GeoMetric, &c, &v, vg, mean).unwrap(),
            &targets,
            SimulationOptions::new(3),
        )
        .unwrap();
        assert_eq!(out.len(), targets.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn st_universal_sim_runs_with_linear_in_time_trend() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let out = sequential_gaussian_simulate(
            spacetime_universal_conditioner(
                GeoMetric,
                &c,
                &v,
                vg,
                SpaceTimeUniversalTrend::LinearInTime,
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(5),
        )
        .unwrap();
        assert_eq!(out.len(), targets.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn st_universal_sim_rejects_insufficient_points_for_trend() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
        ];
        let v = vec![1.0, 2.0];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.0).unwrap()];
        // LinearInSpaceAndTime requires 4 basis terms → at least 5 points.
        let r = spacetime_universal_conditioner(
            GeoMetric,
            &c,
            &v,
            vg,
            SpaceTimeUniversalTrend::LinearInSpaceAndTime,
        )
        .and_then(|sim| sequential_gaussian_simulate(sim, &targets, SimulationOptions::new(1)));
        assert!(matches!(r, Err(KrigingError::InsufficientData(_))));
    }

    #[test]
    fn st_binomial_sim_reports_both_scales_and_prevalence_in_range() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 1.0).unwrap(),
        ];
        let successes = vec![3u32, 6, 8, 15, 4, 16];
        let trials = vec![20u32, 20, 20, 20, 20, 20];
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let out = sequential_binomial_simulate(
            spacetime_binomial_conditioner(
                GeoMetric,
                &c,
                &successes,
                &trials,
                vg,
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        assert_eq!(out.logit_samples.len(), targets.len());
        assert_eq!(out.prevalence_samples.len(), targets.len());
        for (l, p) in out.logit_samples.iter().zip(out.prevalence_samples.iter()) {
            assert!(l.is_finite());
            assert!(p.is_finite());
            assert!(*p > 0.0 && *p < 1.0);
            // p must equal logistic(l) by construction.
            assert!((logistic(*l) - *p).abs() < 1e-6);
        }
    }

    #[test]
    fn st_binomial_sim_drops_zero_trials_from_pool() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
        ];
        // Only station 0 has trials == 0; the remaining 3 form the pool.
        let successes = vec![0u32, 6, 8, 15];
        let trials = vec![0u32, 20, 20, 20];
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap()];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let out = sequential_binomial_simulate(
            spacetime_binomial_conditioner(
                GeoMetric,
                &c,
                &successes,
                &trials,
                vg,
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(2),
        )
        .unwrap();
        assert_eq!(out.logit_samples.len(), 1);
        assert!(out.logit_samples[0].is_finite());
    }

    #[test]
    fn st_binomial_sim_rejects_too_few_observable_stations() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
        ];
        let successes = vec![0u32, 6];
        let trials = vec![0u32, 20];
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap()];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let r = spacetime_binomial_conditioner(
            GeoMetric,
            &c,
            &successes,
            &trials,
            vg,
            BinomialPrior::default(),
        )
        .and_then(|sim| sequential_binomial_simulate(sim, &targets, SimulationOptions::new(1)));
        assert!(matches!(r, Err(KrigingError::InsufficientData(_))));
    }

    // ---- Multi-realization variants ----------------------------------------

    /// Tight contract: the k-th row of `_many` MUST equal a fresh single-realization call
    /// using `seed = base_seed + k`. This is the primary correctness guarantee for the
    /// shared-input path.
    #[test]
    fn many_ordinary_matches_per_realization_single_call() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
            GeoCoord::try_new(0.75, 0.25).unwrap(),
        ];
        let n_real = 4usize;
        let base_seed = 100u64;
        let many = sequential_gaussian_simulate_many(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .expect("many ordinary call");
        assert_eq!(many.len(), n_real * targets.len());
        for k in 0..n_real {
            let one = sequential_gaussian_simulate(
                ordinary_conditioner(&c, &v, vg).unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let row = &many[k * targets.len()..(k + 1) * targets.len()];
            assert_eq!(
                row,
                one.as_slice(),
                "row {k} must match per-seed single call"
            );
        }
    }

    #[test]
    fn many_ordinary_rejects_zero_realizations() {
        let (c, v, vg) = setup();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let r = sequential_gaussian_simulate_many(
            ordinary_conditioner(&c, &v, vg).unwrap(),
            &targets,
            0,
            0,
            None,
        );
        assert!(matches!(r, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn many_binomial_matches_per_realization_single_call() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let prior = BinomialPrior::default();
        let n_real = 3usize;
        let base_seed = 555u64;
        let many = sequential_binomial_simulate_many(
            binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .expect("many binomial call");
        assert_eq!(many.n_realizations, n_real);
        assert_eq!(many.n_targets, targets.len());
        assert_eq!(many.logit_samples.len(), n_real * targets.len());
        assert_eq!(many.prevalence_samples.len(), n_real * targets.len());
        for k in 0..n_real {
            let one = sequential_binomial_simulate(
                binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let lo = k * targets.len();
            let hi = lo + targets.len();
            assert_eq!(
                &many.logit_samples[lo..hi],
                one.logit_samples.as_slice(),
                "logit row {k}"
            );
            assert_eq!(
                &many.prevalence_samples[lo..hi],
                one.prevalence_samples.as_slice(),
                "prevalence row {k}"
            );
        }
    }

    #[test]
    fn many_binomial_drops_zero_trial_stations() {
        let (mut c, mut s, mut t, vg) = binomial_setup();
        c.insert(2, GeoCoord::try_new(0.5, 0.5).unwrap());
        s.insert(2, 0);
        t.insert(2, 0);
        let targets = vec![GeoCoord::try_new(0.75, 0.25).unwrap()];
        let prior = BinomialPrior::default();
        let with_zero = sequential_binomial_simulate_many(
            binomial_conditioner(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            2,
            9,
            None,
        )
        .unwrap();
        let (c2, s2, t2, _) = binomial_setup();
        let without = sequential_binomial_simulate_many(
            binomial_conditioner(&c2, &s2, &t2, vg, prior).unwrap(),
            &targets,
            2,
            9,
            None,
        )
        .unwrap();
        assert_eq!(with_zero.logit_samples, without.logit_samples);
        assert_eq!(with_zero.prevalence_samples, without.prevalence_samples);
    }

    #[test]
    fn many_spacetime_matches_per_realization_single_call() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let n_real = 3usize;
        let base_seed = 12345u64;
        let many = sequential_gaussian_simulate_many(
            spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .expect("many st call");
        assert_eq!(many.len(), n_real * targets.len());
        for k in 0..n_real {
            let one = sequential_gaussian_simulate(
                spacetime_ordinary_conditioner(GeoMetric, &c, &v, vg).unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let row = &many[k * targets.len()..(k + 1) * targets.len()];
            assert_eq!(row, one.as_slice(), "row {k}");
        }
    }

    #[test]
    fn many_spacetime_binomial_matches_per_realization_single_call() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 1.0).unwrap(),
        ];
        let successes = vec![3u32, 6, 8, 15, 4, 16];
        let trials = vec![20u32, 20, 20, 20, 20, 20];
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let prior = BinomialPrior::default();
        let n_real = 3usize;
        let base_seed = 77u64;
        let many = sequential_binomial_simulate_many(
            spacetime_binomial_conditioner(GeoMetric, &c, &successes, &trials, vg, prior).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .unwrap();
        assert_eq!(many.n_realizations, n_real);
        assert_eq!(many.n_targets, targets.len());
        for k in 0..n_real {
            let one = sequential_binomial_simulate(
                spacetime_binomial_conditioner(GeoMetric, &c, &successes, &trials, vg, prior)
                    .unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let lo = k * targets.len();
            let hi = lo + targets.len();
            assert_eq!(&many.logit_samples[lo..hi], one.logit_samples.as_slice());
            assert_eq!(
                &many.prevalence_samples[lo..hi],
                one.prevalence_samples.as_slice()
            );
        }
    }
}
