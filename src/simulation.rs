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
//! ## Variants
//!
//! One function per kriging flavour, mirroring the cross-validation API:
//!
//! - [`conditional_simulate`] — ordinary kriging (no trend).
//! - [`conditional_simulate_simple`] — known mean.
//! - [`conditional_simulate_universal`] — polynomial drift.
//! - [`conditional_simulate_projected`] — planar coordinates with 2D anisotropy.
//! - [`conditional_simulate_binomial`] — count data on the logit scale. Returns samples on
//!   **both** the logit and prevalence scales.
//!
//! All variants accept a shared [`SimulationOptions`] (seed + optional target visit order).
//!
//! ## RNG
//!
//! The simulator uses a seedable xoshiro-style PRNG so realizations are reproducible without
//! an external `rand` runtime dependency. For production work that needs rigorous random
//! number quality, callers can post-process or wrap this module's scalar outputs.

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::binomial::{BinomialKrigingModel, BinomialObservation, BinomialPrior};
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::kriging::simple::SimpleKrigingModel;
use crate::kriging::universal::{UniversalKrigingModel, UniversalTrend};
use crate::projected::{Anisotropy2D, ProjectedCoord, ProjectedDataset, ProjectedKrigingModel};
use crate::utils::logistic;
use crate::variogram::models::VariogramModel;

/// A tiny splitmix64-seeded xoshiro256** PRNG. Deterministic given the same seed.
///
/// This is *not* a cryptographically secure RNG; it exists so simulation is reproducible
/// without pulling `rand` into the runtime dependency graph. It has well-known good
/// statistical properties for Monte-Carlo-style use.
#[derive(Debug, Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform `(0, 1)` — strictly positive so `ln` is safe.
    fn next_unit(&mut self) -> Real {
        let u = (self.next_u64() >> 11) as Real;
        let scale = (1u64 << 53) as Real;
        (u + 0.5) / scale
    }

    /// Standard normal sample via Box-Muller.
    fn next_standard_normal(&mut self) -> Real {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * (std::f64::consts::PI as Real) * u2;
        r * theta.cos()
    }
}

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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Validate and normalize the target visit order.
///
/// Returns a `Vec<usize>` permutation of `0..n_targets`, either echoing input order when the
/// caller left `target_order == None`, or validating the supplied permutation.
fn resolve_target_order(
    n_targets: usize,
    target_order: Option<Vec<usize>>,
) -> Result<Vec<usize>, KrigingError> {
    match target_order {
        None => Ok((0..n_targets).collect()),
        Some(p) => {
            if p.len() != n_targets {
                return Err(KrigingError::InvalidInput(format!(
                    "target_order length ({}) must equal number of targets ({})",
                    p.len(),
                    n_targets
                )));
            }
            let mut seen = vec![false; n_targets];
            for &idx in &p {
                if idx >= n_targets {
                    return Err(KrigingError::InvalidInput(format!(
                        "target_order contains out-of-range index {idx} (n_targets={n_targets})"
                    )));
                }
                if seen[idx] {
                    return Err(KrigingError::InvalidInput(format!(
                        "target_order contains duplicate index {idx}"
                    )));
                }
                seen[idx] = true;
            }
            Ok(p)
        }
    }
}

/// Validate parallel arrays and minimum count for a continuous SGS run.
fn validate_continuous_inputs<C>(coords: &[C], values: &[Real]) -> Result<(), KrigingError> {
    if coords.len() != values.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "conditioning_coords ({}) and conditioning_values ({}) must have equal length",
            coords.len(),
            values.len()
        )));
    }
    if coords.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ordinary SGS
// ---------------------------------------------------------------------------

/// Sequential Gaussian simulation using ordinary kriging.
///
/// Returns one sampled value per target in the original input order (regardless of the
/// visit order specified by [`SimulationOptions::target_order`]).
///
/// Errors:
/// - Mismatched conditioning array lengths → [`KrigingError::DimensionMismatch`].
/// - Fewer than 2 conditioning stations → [`KrigingError::InsufficientData`].
/// - `target_order` is not a valid permutation → [`KrigingError::InvalidInput`].
/// - Any underlying kriging solve failure propagates unchanged.
pub fn conditional_simulate(
    conditioning_coords: &[GeoCoord],
    conditioning_values: &[Real],
    targets: &[GeoCoord],
    variogram: VariogramModel,
    options: SimulationOptions,
) -> Result<Vec<Real>, KrigingError> {
    validate_continuous_inputs(conditioning_coords, conditioning_values)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;

    let mut rng = Rng::new(options.seed);
    let mut all_coords = conditioning_coords.to_vec();
    let mut all_values = conditioning_values.to_vec();
    let mut out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let dataset = GeoDataset::new(all_coords.clone(), all_values.clone())?;
        let model = OrdinaryKrigingModel::new(dataset, variogram)?;
        let pred = model.predict(targets[target_idx])?;
        let sigma = pred.variance.max(0.0).sqrt();
        let sampled = pred.value + sigma * rng.next_standard_normal();
        out[target_idx] = sampled;
        all_coords.push(targets[target_idx]);
        all_values.push(sampled);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Simple SGS (known mean)
// ---------------------------------------------------------------------------

/// Sequential Gaussian simulation using simple kriging (known `mean`).
///
/// Each step fits [`SimpleKrigingModel`] against the growing conditioning set and draws
/// from `N(μ̂, σ²̂)`. Sampled values are appended to the conditioning set for subsequent
/// targets.
///
/// Returns one sampled value per target in the original input order.
pub fn conditional_simulate_simple(
    conditioning_coords: &[GeoCoord],
    conditioning_values: &[Real],
    targets: &[GeoCoord],
    variogram: VariogramModel,
    mean: Real,
    options: SimulationOptions,
) -> Result<Vec<Real>, KrigingError> {
    validate_continuous_inputs(conditioning_coords, conditioning_values)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;

    let mut rng = Rng::new(options.seed);
    let mut all_coords = conditioning_coords.to_vec();
    let mut all_values = conditioning_values.to_vec();
    let mut out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let dataset = GeoDataset::new(all_coords.clone(), all_values.clone())?;
        let model = SimpleKrigingModel::new(dataset, variogram, mean)?;
        let pred = model.predict(targets[target_idx])?;
        let sigma = pred.variance.max(0.0).sqrt();
        let sampled = pred.value + sigma * rng.next_standard_normal();
        out[target_idx] = sampled;
        all_coords.push(targets[target_idx]);
        all_values.push(sampled);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Universal SGS (polynomial trend)
// ---------------------------------------------------------------------------

/// Sequential Gaussian simulation using universal kriging with polynomial `trend`.
///
/// Requires at least `trend.n_basis() + 1` conditioning stations (otherwise the normal
/// equations are rank-deficient). Returns one sampled value per target in the original input
/// order.
pub fn conditional_simulate_universal(
    conditioning_coords: &[GeoCoord],
    conditioning_values: &[Real],
    targets: &[GeoCoord],
    variogram: VariogramModel,
    trend: UniversalTrend,
    options: SimulationOptions,
) -> Result<Vec<Real>, KrigingError> {
    if conditioning_coords.len() != conditioning_values.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "conditioning_coords ({}) and conditioning_values ({}) must have equal length",
            conditioning_coords.len(),
            conditioning_values.len()
        )));
    }
    let min_required = trend.n_basis() + 1;
    if conditioning_coords.len() < min_required {
        return Err(KrigingError::InsufficientData(min_required));
    }
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;

    let mut rng = Rng::new(options.seed);
    let mut all_coords = conditioning_coords.to_vec();
    let mut all_values = conditioning_values.to_vec();
    let mut out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let dataset = GeoDataset::new(all_coords.clone(), all_values.clone())?;
        let model = UniversalKrigingModel::new(dataset, variogram, trend)?;
        let pred = model.predict(targets[target_idx])?;
        let sigma = pred.variance.max(0.0).sqrt();
        let sampled = pred.value + sigma * rng.next_standard_normal();
        out[target_idx] = sampled;
        all_coords.push(targets[target_idx]);
        all_values.push(sampled);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Projected SGS (planar + anisotropy)
// ---------------------------------------------------------------------------

/// Sequential Gaussian simulation on projected (planar) coordinates with 2D anisotropy.
///
/// Use [`Anisotropy2D::isotropic`] to disable directional scaling. Returns one sampled value
/// per target in the original input order.
pub fn conditional_simulate_projected(
    conditioning_coords: &[ProjectedCoord],
    conditioning_values: &[Real],
    targets: &[ProjectedCoord],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    options: SimulationOptions,
) -> Result<Vec<Real>, KrigingError> {
    validate_continuous_inputs(conditioning_coords, conditioning_values)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;

    let mut rng = Rng::new(options.seed);
    let mut all_coords = conditioning_coords.to_vec();
    let mut all_values = conditioning_values.to_vec();
    let mut out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let dataset = ProjectedDataset::new(all_coords.clone(), all_values.clone())?;
        let model = ProjectedKrigingModel::new(dataset, variogram, anisotropy)?;
        let pred = model.predict(targets[target_idx])?;
        let sigma = pred.variance.max(0.0).sqrt();
        let sampled = pred.value + sigma * rng.next_standard_normal();
        out[target_idx] = sampled;
        all_coords.push(targets[target_idx]);
        all_values.push(sampled);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Binomial SGS (dual-scale)
// ---------------------------------------------------------------------------

/// Sequential Gaussian simulation for binomial (count) data.
///
/// Simulation happens on the **logit** scale — where the Gaussian assumption is natural — and
/// the result carries both the raw logit samples and the back-transformed prevalence samples.
///
/// Steps:
/// 1. Drop stations with `trials == 0` (they carry no information).
/// 2. Convert remaining stations to smoothed logits using [`BinomialPrior`].
/// 3. At each target, fit a binomial kriging model against the growing pool of logits,
///    predict, draw `N(logit̂, σ²̂_logit)`, convert via `logistic` to prevalence, and append
///    the logit sample to the pool for subsequent targets.
///
/// Both `logit_samples` and `prevalence_samples` are returned in the **original** target
/// input order.
///
/// Errors:
/// - Mismatched `successes` / `trials` lengths → [`KrigingError::DimensionMismatch`].
/// - Fewer than 2 valid (`trials > 0`) stations → [`KrigingError::InsufficientData`].
/// - `successes[i] > trials[i]` → [`KrigingError::InvalidBinomialData`].
/// - `target_order` is not a valid permutation → [`KrigingError::InvalidInput`].
pub fn conditional_simulate_binomial(
    conditioning_coords: &[GeoCoord],
    successes: &[u32],
    trials: &[u32],
    targets: &[GeoCoord],
    variogram: VariogramModel,
    prior: BinomialPrior,
    options: SimulationOptions,
) -> Result<BinomialSimulationResult, KrigingError> {
    if conditioning_coords.len() != successes.len() || successes.len() != trials.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "conditioning arrays must have equal length (coords={}, successes={}, trials={})",
            conditioning_coords.len(),
            successes.len(),
            trials.len()
        )));
    }

    // Build initial logit pool from stations with trials > 0.
    let mut pool_coords: Vec<GeoCoord> = Vec::with_capacity(conditioning_coords.len());
    let mut pool_logits: Vec<Real> = Vec::with_capacity(conditioning_coords.len());
    for i in 0..conditioning_coords.len() {
        if trials[i] == 0 {
            continue;
        }
        let obs = BinomialObservation::new(conditioning_coords[i], successes[i], trials[i])?;
        pool_coords.push(obs.coord());
        pool_logits.push(obs.smoothed_logit_with_prior(prior));
    }
    if pool_coords.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }

    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;

    let mut rng = Rng::new(options.seed);
    let mut logit_out = vec![0.0 as Real; n_targets];
    let mut prev_out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let model = BinomialKrigingModel::from_precomputed_logits(
            pool_coords.clone(),
            pool_logits.clone(),
            variogram,
        )?;
        let pred = model.predict(targets[target_idx])?;
        let sigma = pred.variance.max(0.0).sqrt();
        let logit_sample = pred.logit_value + sigma * rng.next_standard_normal();
        let prevalence_sample = logistic(logit_sample);
        logit_out[target_idx] = logit_sample;
        prev_out[target_idx] = prevalence_sample;
        pool_coords.push(targets[target_idx]);
        pool_logits.push(logit_sample);
    }

    Ok(BinomialSimulationResult {
        logit_samples: logit_out,
        prevalence_samples: prev_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

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

    // ---- Ordinary ----------------------------------------------------------

    #[test]
    fn same_seed_gives_identical_realization() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let a = conditional_simulate(&c, &v, &targets, vg, SimulationOptions::new(42)).unwrap();
        let b = conditional_simulate(&c, &v, &targets, vg, SimulationOptions::new(42)).unwrap();
        assert_eq!(a, b, "same seed should yield identical realization");
    }

    #[test]
    fn different_seeds_give_different_realizations() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let a = conditional_simulate(&c, &v, &targets, vg, SimulationOptions::new(1)).unwrap();
        let b = conditional_simulate(&c, &v, &targets, vg, SimulationOptions::new(2)).unwrap();
        assert!(a != b, "different seeds should differ at least somewhere");
    }

    #[test]
    fn realization_honors_conditioning_when_target_coincides() {
        let (c, v, vg) = setup();
        let targets = vec![c[2]];
        let out = conditional_simulate(&c, &v, &targets, vg, SimulationOptions::new(123)).unwrap();
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
        let out = conditional_simulate(&c, &v, &targets, vg, opts).unwrap();
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
        assert!(conditional_simulate(&c, &v, &targets, vg, opts).is_err());

        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 1, 5]);
        assert!(conditional_simulate(&c, &v, &targets, vg, opts).is_err());

        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 1]);
        assert!(conditional_simulate(&c, &v, &targets, vg, opts).is_err());
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
        let a = conditional_simulate_simple(&c, &v, &targets, vg, mean, SimulationOptions::new(9))
            .unwrap();
        let b = conditional_simulate_simple(&c, &v, &targets, vg, mean, SimulationOptions::new(9))
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
        let out =
            conditional_simulate_simple(&c, &v, &targets, vg, 2.5, SimulationOptions::new(33))
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
        let a =
            conditional_simulate_universal(&c, &v, &targets, vg, trend, SimulationOptions::new(11))
                .unwrap();
        let b =
            conditional_simulate_universal(&c, &v, &targets, vg, trend, SimulationOptions::new(11))
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
        let result = conditional_simulate_universal(
            &c,
            &v,
            &targets,
            vg,
            UniversalTrend::Linear,
            SimulationOptions::new(0),
        );
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
        let a = conditional_simulate_projected(
            &coords,
            &values,
            &targets,
            vg,
            aniso,
            SimulationOptions::new(5),
        )
        .unwrap();
        let b = conditional_simulate_projected(
            &coords,
            &values,
            &targets,
            vg,
            aniso,
            SimulationOptions::new(5),
        )
        .unwrap();
        assert_eq!(a, b);
        for x in &a {
            assert!(x.is_finite());
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
        let result = conditional_simulate_binomial(
            &c,
            &s,
            &t,
            &targets,
            vg,
            prior,
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
        let a = conditional_simulate_binomial(
            &c,
            &s,
            &t,
            &targets,
            vg,
            prior,
            SimulationOptions::new(17),
        )
        .unwrap();
        let b = conditional_simulate_binomial(
            &c,
            &s,
            &t,
            &targets,
            vg,
            prior,
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
        let with_zero = conditional_simulate_binomial(
            &c,
            &s,
            &t,
            &targets,
            vg,
            prior,
            SimulationOptions::new(4),
        )
        .unwrap();

        let (c2, s2, t2, _) = binomial_setup();
        let without = conditional_simulate_binomial(
            &c2,
            &s2,
            &t2,
            &targets,
            vg,
            prior,
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
        let result = conditional_simulate_binomial(
            &coords,
            &successes,
            &trials,
            &targets,
            vg,
            BinomialPrior::default(),
            SimulationOptions::new(0),
        );
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
        let a = conditional_simulate_binomial(
            &c,
            &s,
            &t,
            &targets,
            vg,
            default_prior,
            SimulationOptions::new(21),
        )
        .unwrap();
        let b = conditional_simulate_binomial(
            &c,
            &s,
            &t,
            &targets,
            vg,
            custom_prior,
            SimulationOptions::new(21),
        )
        .unwrap();
        assert!(
            (a.logit_samples[0] - b.logit_samples[0]).abs() > 1e-9,
            "different priors should produce different logit samples"
        );
    }
}
