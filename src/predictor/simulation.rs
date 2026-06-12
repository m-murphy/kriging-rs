//! Generic sequential Gaussian simulation (SGS) harness for kriging simulators.
//!
//! Model-specific conditioning logic lives in [`KrigingSimulator`] and [`BinomialKrigingSimulator`]
//! implementors; [`sequential_gaussian_simulate`] and [`sequential_binomial_simulate`] drive the
//! target visit order and RNG.

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::kriging::binomial::{
    BinomialObservation, BinomialPrior, HeteroskedasticBinomialConfig,
    binomial_prediction_from_ordinary, build_calibrated_logit_ordinary,
    logit_observation_variance_empirical_bayes,
};
use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::kriging::simple_engine::SimpleKrigingEngine;
use crate::kriging::universal::UniversalTrend;
use crate::kriging::universal_engine::UniversalKrigingEngine;
use crate::projected::{Anisotropy2D, ProjectedBinomialObservation, ProjectedCoord};
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::kriging::binomial::SpaceTimeBinomialObservation;
use crate::spacetime::kriging::engine::SpaceTimeOrdinaryKrigingEngine;
use crate::spacetime::kriging::simple_engine::SpaceTimeSimpleKrigingEngine;
use crate::spacetime::kriging::universal::SpaceTimeUniversalTrend;
use crate::spacetime::kriging::universal_engine::SpaceTimeUniversalKrigingEngine;
use crate::spacetime::metric::{GeoMetric, ProjectedMetric, SpatialBasis, SpatialMetric};
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::utils::logistic;
use crate::variogram::models::VariogramModel;

pub use crate::simulation::{
    BinomialSimulationManyResult, BinomialSimulationResult, SimulationOptions,
};

/// Alias for [`BinomialSimulationResult`] in SGS harness docs.
pub type BinomialSgsOutput = BinomialSimulationResult;

type GeoBinomialPool = (Vec<GeoCoord>, Vec<Real>, Vec<Real>);
type ProjectedBinomialPool = (Vec<ProjectedCoord>, Vec<Real>, Vec<Real>);
type StBinomialPool<C> = (Vec<SpaceTimeCoord<C>>, Vec<Real>, Vec<Real>);

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A kriging model that supports sequential Gaussian simulation on a continuous scale.
pub trait KrigingSimulator {
    type Target;
    type Output;

    /// Kriging mean and variance at `target`.
    fn predict_at(&self, target: Self::Target) -> Result<(Real, Real), KrigingError>;

    /// Append a simulated value to the conditioning set for subsequent targets.
    fn append_sample(self, target: Self::Target, sample: Real) -> Result<Self, KrigingError>
    where
        Self: Sized;
}

/// A kriging model that supports sequential Gaussian simulation on the logit scale.
pub trait BinomialKrigingSimulator {
    type Target;

    /// Kriging mean and variance on the logit scale at `target`.
    fn predict_logit_at(&self, target: Self::Target) -> Result<(Real, Real), KrigingError>;

    /// Append a simulated logit sample (zero observation noise at simulated sites).
    fn append_logit_sample(
        self,
        target: Self::Target,
        logit_sample: Real,
    ) -> Result<Self, KrigingError>
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// Harnesses
// ---------------------------------------------------------------------------

/// Kriging variance below this threshold means the target is already in the conditioning set.
const SGS_CONDITIONED_VARIANCE_EPS: Real = 1e-10;

/// One continuous SGS realization in original target input order.
pub fn sequential_gaussian_simulate<S>(
    mut sim: S,
    targets: &[S::Target],
    options: SimulationOptions,
) -> Result<Vec<Real>, KrigingError>
where
    S: KrigingSimulator,
    S::Target: Copy,
{
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;
    let mut rng = Rng::new(options.seed);
    let mut out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let target = targets[target_idx];
        let (mu, variance) = sim.predict_at(target)?;
        let sigma = variance.max(0.0).sqrt();
        let sampled = mu + sigma * rng.next_standard_normal();
        out[target_idx] = sampled;
        if variance > SGS_CONDITIONED_VARIANCE_EPS {
            sim = sim.append_sample(target, sampled)?;
        }
    }

    Ok(out)
}

/// One binomial SGS realization (logit + prevalence) in original target input order.
pub fn sequential_binomial_simulate<B>(
    mut sim: B,
    targets: &[B::Target],
    options: SimulationOptions,
) -> Result<BinomialSimulationResult, KrigingError>
where
    B: BinomialKrigingSimulator,
    B::Target: Copy,
{
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, options.target_order)?;
    let mut rng = Rng::new(options.seed);
    let mut logit_out = vec![0.0 as Real; n_targets];
    let mut prev_out = vec![0.0 as Real; n_targets];

    for &target_idx in &order {
        let target = targets[target_idx];
        let (logit, logit_variance) = sim.predict_logit_at(target)?;
        let sigma = logit_variance.max(0.0).sqrt();
        let logit_sample = logit + sigma * rng.next_standard_normal();
        let prevalence_sample = logistic(logit_sample);
        logit_out[target_idx] = logit_sample;
        prev_out[target_idx] = prevalence_sample;
        if logit_variance > SGS_CONDITIONED_VARIANCE_EPS {
            sim = sim.append_logit_sample(target, logit_sample)?;
        }
    }

    Ok(BinomialSimulationResult {
        logit_samples: logit_out,
        prevalence_samples: prev_out,
    })
}

/// Multi-realization continuous SGS. Row `k` matches [`sequential_gaussian_simulate`] with
/// `seed = base_seed + k`.
pub fn sequential_gaussian_simulate_many<S>(
    template: S,
    targets: &[S::Target],
    n_realizations: usize,
    base_seed: u64,
    target_order: Option<Vec<usize>>,
) -> Result<Vec<Real>, KrigingError>
where
    S: KrigingSimulator + Clone,
    S::Target: Copy,
{
    validate_n_realizations(n_realizations)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, target_order)?;
    let mut out = vec![0.0 as Real; n_realizations * n_targets];

    for k in 0..n_realizations {
        let seed = base_seed.wrapping_add(k as u64);
        let mut rng = Rng::new(seed);
        let mut sim = template.clone();
        let row_off = k * n_targets;
        for &target_idx in &order {
            let target = targets[target_idx];
            let (mu, variance) = sim.predict_at(target)?;
            let sigma = variance.max(0.0).sqrt();
            let sampled = mu + sigma * rng.next_standard_normal();
            out[row_off + target_idx] = sampled;
            if variance > SGS_CONDITIONED_VARIANCE_EPS {
                sim = sim.append_sample(target, sampled)?;
            }
        }
    }

    Ok(out)
}

/// Multi-realization binomial SGS. Row `k` matches [`sequential_binomial_simulate`] with
/// `seed = base_seed + k`.
pub fn sequential_binomial_simulate_many<B>(
    template: B,
    targets: &[B::Target],
    n_realizations: usize,
    base_seed: u64,
    target_order: Option<Vec<usize>>,
) -> Result<BinomialSimulationManyResult, KrigingError>
where
    B: BinomialKrigingSimulator + Clone,
    B::Target: Copy,
{
    validate_n_realizations(n_realizations)?;
    let n_targets = targets.len();
    let order = resolve_target_order(n_targets, target_order)?;
    let mut logit_out = vec![0.0 as Real; n_realizations * n_targets];
    let mut prev_out = vec![0.0 as Real; n_realizations * n_targets];

    for k in 0..n_realizations {
        let seed = base_seed.wrapping_add(k as u64);
        let mut rng = Rng::new(seed);
        let mut sim = template.clone();
        let row_off = k * n_targets;
        for &target_idx in &order {
            let target = targets[target_idx];
            let (logit, logit_variance) = sim.predict_logit_at(target)?;
            let sigma = logit_variance.max(0.0).sqrt();
            let logit_sample = logit + sigma * rng.next_standard_normal();
            let prevalence_sample = logistic(logit_sample);
            logit_out[row_off + target_idx] = logit_sample;
            prev_out[row_off + target_idx] = prevalence_sample;
            if logit_variance > SGS_CONDITIONED_VARIANCE_EPS {
                sim = sim.append_logit_sample(target, logit_sample)?;
            }
        }
    }

    Ok(BinomialSimulationManyResult {
        n_realizations,
        n_targets,
        logit_samples: logit_out,
        prevalence_samples: prev_out,
    })
}

// ---------------------------------------------------------------------------
// Shared helpers (mirrors private helpers in `crate::simulation`)
// ---------------------------------------------------------------------------

/// A tiny splitmix64-seeded xoshiro256** PRNG. Deterministic given the same seed.
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

    fn next_unit(&mut self) -> Real {
        let u = (self.next_u64() >> 11) as Real;
        let scale = (1u64 << 53) as Real;
        (u + 0.5) / scale
    }

    fn next_standard_normal(&mut self) -> Real {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * (std::f64::consts::PI as Real) * u2;
        r * theta.cos()
    }
}

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

fn validate_universal_inputs(
    n_coords: usize,
    n_values: usize,
    trend: UniversalTrend,
) -> Result<(), KrigingError> {
    if n_coords != n_values {
        return Err(KrigingError::DimensionMismatch(format!(
            "conditioning_coords ({n_coords}) and conditioning_values ({n_values}) must have equal length"
        )));
    }
    let min_required = trend.n_basis() + 1;
    if n_coords < min_required {
        return Err(KrigingError::InsufficientData(min_required));
    }
    Ok(())
}

fn validate_st_universal_inputs(
    n_coords: usize,
    n_values: usize,
    trend: SpaceTimeUniversalTrend,
) -> Result<(), KrigingError> {
    if n_coords != n_values {
        return Err(KrigingError::DimensionMismatch(format!(
            "conditioning_coords ({n_coords}) and conditioning_values ({n_values}) must have equal length"
        )));
    }
    let min_required = trend.n_basis() + 1;
    if n_coords < min_required {
        return Err(KrigingError::InsufficientData(min_required));
    }
    Ok(())
}

fn validate_n_realizations(n: usize) -> Result<(), KrigingError> {
    if n == 0 {
        return Err(KrigingError::InvalidInput(
            "n_realizations must be >= 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_binomial_lengths(
    n_coords: usize,
    n_successes: usize,
    n_trials: usize,
) -> Result<(), KrigingError> {
    if n_coords != n_successes || n_coords != n_trials {
        return Err(KrigingError::DimensionMismatch(format!(
            "conditioning arrays must have equal length (coords={n_coords}, successes={n_successes}, trials={n_trials})"
        )));
    }
    Ok(())
}

fn single_prediction(
    preds: Result<Vec<crate::kriging::ordinary::Prediction>, KrigingError>,
) -> Result<crate::kriging::ordinary::Prediction, KrigingError> {
    preds?
        .into_iter()
        .next()
        .ok_or_else(|| KrigingError::InvalidInput("empty prediction batch".to_string()))
}

// ---------------------------------------------------------------------------
// Ordinary geo (engine + condition)
// ---------------------------------------------------------------------------

/// Ordinary kriging SGS backend (geographic coordinates, incremental engine).
#[derive(Debug, Clone)]
pub struct OrdinaryGeoSimulator {
    engine: OrdinaryKrigingEngine<GeoMetric>,
}

impl OrdinaryGeoSimulator {
    pub fn new(
        conditioning_coords: &[GeoCoord],
        conditioning_values: &[Real],
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        validate_continuous_inputs(conditioning_coords, conditioning_values)?;
        let engine = OrdinaryKrigingEngine::fit(
            GeoMetric,
            conditioning_coords.to_vec(),
            conditioning_values.to_vec(),
            variogram,
        )?;
        Ok(Self { engine })
    }
}

impl KrigingSimulator for OrdinaryGeoSimulator {
    type Target = GeoCoord;
    type Output = Vec<Real>;

    fn predict_at(&self, target: GeoCoord) -> Result<(Real, Real), KrigingError> {
        let pred = single_prediction(self.engine.predict(&[target]))?;
        Ok((pred.value, pred.variance))
    }

    fn append_sample(self, target: GeoCoord, sample: Real) -> Result<Self, KrigingError> {
        Ok(Self {
            engine: self.engine.condition(target, sample, 0.0)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Simple geo (engine + condition)
// ---------------------------------------------------------------------------

/// Simple kriging SGS backend (geographic coordinates, incremental engine).
#[derive(Debug, Clone)]
pub struct SimpleGeoSimulator {
    engine: SimpleKrigingEngine<GeoMetric>,
}

impl SimpleGeoSimulator {
    pub fn new(
        conditioning_coords: &[GeoCoord],
        conditioning_values: &[Real],
        variogram: VariogramModel,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        validate_continuous_inputs(conditioning_coords, conditioning_values)?;
        let engine = SimpleKrigingEngine::fit(
            GeoMetric,
            conditioning_coords.to_vec(),
            conditioning_values.to_vec(),
            variogram,
            mean,
        )?;
        Ok(Self { engine })
    }
}

impl KrigingSimulator for SimpleGeoSimulator {
    type Target = GeoCoord;
    type Output = Vec<Real>;

    fn predict_at(&self, target: GeoCoord) -> Result<(Real, Real), KrigingError> {
        let pred = single_prediction(self.engine.predict(&[target]))?;
        Ok((pred.value, pred.variance))
    }

    fn append_sample(self, target: GeoCoord, sample: Real) -> Result<Self, KrigingError> {
        Ok(Self {
            engine: self.engine.condition(target, sample, 0.0)?,
        })
    }
}

/// Universal kriging SGS backend (polynomial drift; constant trend uses ordinary engine).
#[derive(Debug, Clone)]
enum UniversalGeoSimulatorInner {
    Constant(OrdinaryKrigingEngine<GeoMetric>),
    Drift(UniversalKrigingEngine),
}

/// Universal kriging SGS backend (polynomial drift).
#[derive(Debug, Clone)]
pub struct UniversalGeoSimulator {
    inner: UniversalGeoSimulatorInner,
}

impl UniversalGeoSimulator {
    pub fn new(
        conditioning_coords: &[GeoCoord],
        conditioning_values: &[Real],
        variogram: VariogramModel,
        trend: UniversalTrend,
    ) -> Result<Self, KrigingError> {
        if trend == UniversalTrend::Constant {
            validate_continuous_inputs(conditioning_coords, conditioning_values)?;
            let engine = OrdinaryKrigingEngine::fit(
                GeoMetric,
                conditioning_coords.to_vec(),
                conditioning_values.to_vec(),
                variogram,
            )?;
            Ok(Self {
                inner: UniversalGeoSimulatorInner::Constant(engine),
            })
        } else {
            validate_universal_inputs(conditioning_coords.len(), conditioning_values.len(), trend)?;
            let engine = UniversalKrigingEngine::fit(
                conditioning_coords.to_vec(),
                conditioning_values.to_vec(),
                variogram,
                trend,
            )?;
            Ok(Self {
                inner: UniversalGeoSimulatorInner::Drift(engine),
            })
        }
    }
}

impl KrigingSimulator for UniversalGeoSimulator {
    type Target = GeoCoord;
    type Output = Vec<Real>;

    fn predict_at(&self, target: GeoCoord) -> Result<(Real, Real), KrigingError> {
        match &self.inner {
            UniversalGeoSimulatorInner::Constant(engine) => {
                let pred = single_prediction(engine.predict(&[target]))?;
                Ok((pred.value, pred.variance))
            }
            UniversalGeoSimulatorInner::Drift(engine) => {
                let pred = single_prediction(engine.predict(&[target]))?;
                Ok((pred.value, pred.variance))
            }
        }
    }

    fn append_sample(self, target: GeoCoord, sample: Real) -> Result<Self, KrigingError> {
        match self.inner {
            UniversalGeoSimulatorInner::Constant(engine) => Ok(Self {
                inner: UniversalGeoSimulatorInner::Constant(engine.condition(target, sample, 0.0)?),
            }),
            UniversalGeoSimulatorInner::Drift(engine) => Ok(Self {
                inner: UniversalGeoSimulatorInner::Drift(engine.condition(target, sample, 0.0)?),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Simple / universal refit loop (projected geo only below)
// ---------------------------------------------------------------------------

/// Projected ordinary kriging SGS backend (planar coordinates, incremental engine).
#[derive(Debug, Clone)]
pub struct ProjectedOrdinarySimulator {
    engine: OrdinaryKrigingEngine<ProjectedMetric>,
}

impl ProjectedOrdinarySimulator {
    pub fn new(
        conditioning_coords: &[ProjectedCoord],
        conditioning_values: &[Real],
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
    ) -> Result<Self, KrigingError> {
        validate_continuous_inputs(conditioning_coords, conditioning_values)?;
        let metric = ProjectedMetric::with_anisotropy(anisotropy);
        let engine = OrdinaryKrigingEngine::fit(
            metric,
            conditioning_coords.to_vec(),
            conditioning_values.to_vec(),
            variogram,
        )?;
        Ok(Self { engine })
    }
}

impl KrigingSimulator for ProjectedOrdinarySimulator {
    type Target = ProjectedCoord;
    type Output = Vec<Real>;

    fn predict_at(&self, target: ProjectedCoord) -> Result<(Real, Real), KrigingError> {
        let pred = single_prediction(self.engine.predict(&[target]))?;
        Ok((pred.value, pred.variance))
    }

    fn append_sample(self, target: ProjectedCoord, sample: Real) -> Result<Self, KrigingError> {
        Ok(Self {
            engine: self.engine.condition(target, sample, 0.0)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Binomial geo / projected (engine + condition)
// ---------------------------------------------------------------------------

/// Binomial kriging SGS backend (geographic coordinates, logit-scale engine).
#[derive(Debug, Clone)]
pub struct BinomialGeoSimulator {
    engine: OrdinaryKrigingEngine<GeoMetric>,
}

impl BinomialGeoSimulator {
    pub fn new(
        conditioning_coords: &[GeoCoord],
        successes: &[u32],
        trials: &[u32],
        variogram: VariogramModel,
        prior: BinomialPrior,
    ) -> Result<Self, KrigingError> {
        validate_binomial_lengths(conditioning_coords.len(), successes.len(), trials.len())?;
        let (pool_coords, pool_logits, pool_obs_var) =
            build_geo_binomial_pool(conditioning_coords, successes, trials, prior)?;
        let config = HeteroskedasticBinomialConfig::default();
        let engine = build_calibrated_logit_ordinary(
            pool_obs_var,
            &config,
            prior,
            &[],
            false,
            None,
            "binomial simulation engine build failed",
            |extra| {
                OrdinaryKrigingEngine::fit_with_extra_diagonal(
                    GeoMetric,
                    pool_coords.clone(),
                    pool_logits.clone(),
                    variogram,
                    &extra,
                )
            },
        )?
        .model;
        Ok(Self { engine })
    }
}

impl BinomialKrigingSimulator for BinomialGeoSimulator {
    type Target = GeoCoord;

    fn predict_logit_at(&self, target: GeoCoord) -> Result<(Real, Real), KrigingError> {
        let pred =
            binomial_prediction_from_ordinary(single_prediction(self.engine.predict(&[target]))?);
        Ok((pred.logit, pred.logit_variance))
    }

    fn append_logit_sample(
        self,
        target: GeoCoord,
        logit_sample: Real,
    ) -> Result<Self, KrigingError> {
        Ok(Self {
            engine: self.engine.condition(target, logit_sample, 0.0)?,
        })
    }
}

/// Binomial kriging SGS backend (projected coordinates, logit-scale engine).
#[derive(Debug, Clone)]
pub struct BinomialProjectedSimulator {
    engine: OrdinaryKrigingEngine<ProjectedMetric>,
}

impl BinomialProjectedSimulator {
    pub fn new(
        conditioning_coords: &[ProjectedCoord],
        successes: &[u32],
        trials: &[u32],
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
        prior: BinomialPrior,
    ) -> Result<Self, KrigingError> {
        validate_binomial_lengths(conditioning_coords.len(), successes.len(), trials.len())?;
        let (pool_coords, pool_logits, pool_obs_var) =
            build_projected_binomial_pool(conditioning_coords, successes, trials, prior)?;
        let config = HeteroskedasticBinomialConfig::default();
        let metric = ProjectedMetric::with_anisotropy(anisotropy);
        let engine = build_calibrated_logit_ordinary(
            pool_obs_var,
            &config,
            prior,
            &[],
            false,
            None,
            "binomial projected simulation engine build failed",
            |extra| {
                OrdinaryKrigingEngine::fit_with_extra_diagonal(
                    metric,
                    pool_coords.clone(),
                    pool_logits.clone(),
                    variogram,
                    &extra,
                )
            },
        )?
        .model;
        Ok(Self { engine })
    }
}

impl BinomialKrigingSimulator for BinomialProjectedSimulator {
    type Target = ProjectedCoord;

    fn predict_logit_at(&self, target: ProjectedCoord) -> Result<(Real, Real), KrigingError> {
        let pred =
            binomial_prediction_from_ordinary(single_prediction(self.engine.predict(&[target]))?);
        Ok((pred.logit, pred.logit_variance))
    }

    fn append_logit_sample(
        self,
        target: ProjectedCoord,
        logit_sample: Real,
    ) -> Result<Self, KrigingError> {
        Ok(Self {
            engine: self.engine.condition(target, logit_sample, 0.0)?,
        })
    }
}

fn build_geo_binomial_pool(
    conditioning_coords: &[GeoCoord],
    successes: &[u32],
    trials: &[u32],
    prior: BinomialPrior,
) -> Result<GeoBinomialPool, KrigingError> {
    let mut pool_coords = Vec::with_capacity(conditioning_coords.len());
    let mut pool_logits = Vec::with_capacity(conditioning_coords.len());
    let mut pool_obs_var = Vec::with_capacity(conditioning_coords.len());
    for i in 0..conditioning_coords.len() {
        if trials[i] == 0 {
            continue;
        }
        let obs = BinomialObservation::new(conditioning_coords[i], successes[i], trials[i])?;
        pool_coords.push(obs.coord());
        pool_logits.push(obs.smoothed_logit_with_prior(prior));
        pool_obs_var.push(logit_observation_variance_empirical_bayes(
            prior,
            successes[i],
            trials[i],
        ));
    }
    if pool_coords.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    Ok((pool_coords, pool_logits, pool_obs_var))
}

fn build_projected_binomial_pool(
    conditioning_coords: &[ProjectedCoord],
    successes: &[u32],
    trials: &[u32],
    prior: BinomialPrior,
) -> Result<ProjectedBinomialPool, KrigingError> {
    let mut pool_coords = Vec::with_capacity(conditioning_coords.len());
    let mut pool_logits = Vec::with_capacity(conditioning_coords.len());
    let mut pool_obs_var = Vec::with_capacity(conditioning_coords.len());
    for i in 0..conditioning_coords.len() {
        if trials[i] == 0 {
            continue;
        }
        let obs =
            ProjectedBinomialObservation::new(conditioning_coords[i], successes[i], trials[i])?;
        pool_coords.push(obs.coord());
        pool_logits.push(obs.smoothed_logit_with_prior(prior));
        pool_obs_var.push(logit_observation_variance_empirical_bayes(
            prior,
            successes[i],
            trials[i],
        ));
    }
    if pool_coords.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    Ok((pool_coords, pool_logits, pool_obs_var))
}

// ---------------------------------------------------------------------------
// Spacetime continuous (incremental ordinary/simple/universal-constant)
// ---------------------------------------------------------------------------

/// Space–time ordinary kriging SGS backend (incremental engine).
#[derive(Debug, Clone)]
pub struct SpacetimeOrdinarySimulator<M: SpatialMetric> {
    pub metric: M,
    engine: SpaceTimeOrdinaryKrigingEngine<M>,
}

impl<M: SpatialMetric> SpacetimeOrdinarySimulator<M> {
    pub fn new(
        metric: M,
        conditioning_coords: &[SpaceTimeCoord<M::Coord>],
        conditioning_values: &[Real],
        variogram: SpaceTimeVariogram,
    ) -> Result<Self, KrigingError> {
        validate_continuous_inputs(conditioning_coords, conditioning_values)?;
        let engine = SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
            metric,
            conditioning_coords.to_vec(),
            conditioning_values.to_vec(),
            variogram,
            &[],
        )?;
        Ok(Self { metric, engine })
    }
}

impl<M: SpatialMetric> KrigingSimulator for SpacetimeOrdinarySimulator<M> {
    type Target = SpaceTimeCoord<M::Coord>;
    type Output = Vec<Real>;

    fn predict_at(&self, target: Self::Target) -> Result<(Real, Real), KrigingError> {
        let pred = single_prediction(self.engine.predict(&[target]))?;
        Ok((pred.value, pred.variance))
    }

    fn append_sample(self, target: Self::Target, sample: Real) -> Result<Self, KrigingError> {
        Ok(Self {
            metric: self.metric,
            engine: self.engine.condition(target, sample, 0.0)?,
        })
    }
}

/// Space–time simple kriging SGS backend (incremental engine).
#[derive(Debug, Clone)]
pub struct SpacetimeSimpleSimulator<M: SpatialMetric> {
    pub metric: M,
    engine: SpaceTimeSimpleKrigingEngine<M>,
}

impl<M: SpatialMetric> SpacetimeSimpleSimulator<M> {
    pub fn new(
        metric: M,
        conditioning_coords: &[SpaceTimeCoord<M::Coord>],
        conditioning_values: &[Real],
        variogram: SpaceTimeVariogram,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        validate_continuous_inputs(conditioning_coords, conditioning_values)?;
        let engine = SpaceTimeSimpleKrigingEngine::fit(
            metric,
            conditioning_coords.to_vec(),
            conditioning_values.to_vec(),
            variogram,
            mean,
        )?;
        Ok(Self { metric, engine })
    }
}

impl<M: SpatialMetric> KrigingSimulator for SpacetimeSimpleSimulator<M> {
    type Target = SpaceTimeCoord<M::Coord>;
    type Output = Vec<Real>;

    fn predict_at(&self, target: Self::Target) -> Result<(Real, Real), KrigingError> {
        let pred = single_prediction(self.engine.predict(&[target]))?;
        Ok((pred.value, pred.variance))
    }

    fn append_sample(self, target: Self::Target, sample: Real) -> Result<Self, KrigingError> {
        Ok(Self {
            metric: self.metric,
            engine: self.engine.condition(target, sample, 0.0)?,
        })
    }
}

#[derive(Debug, Clone)]
enum SpacetimeUniversalSimulatorInner<M: SpatialBasis> {
    Constant(SpaceTimeOrdinaryKrigingEngine<M>),
    Drift(SpaceTimeUniversalKrigingEngine<M>),
}

/// Space–time universal kriging SGS backend (polynomial drift; constant trend uses ordinary engine).
#[derive(Debug, Clone)]
pub struct SpacetimeUniversalSimulator<M: SpatialBasis> {
    pub metric: M,
    inner: SpacetimeUniversalSimulatorInner<M>,
}

impl<M: SpatialBasis> SpacetimeUniversalSimulator<M> {
    pub fn new(
        metric: M,
        conditioning_coords: &[SpaceTimeCoord<M::Coord>],
        conditioning_values: &[Real],
        variogram: SpaceTimeVariogram,
        trend: SpaceTimeUniversalTrend,
    ) -> Result<Self, KrigingError> {
        if trend == SpaceTimeUniversalTrend::Constant {
            validate_continuous_inputs(conditioning_coords, conditioning_values)?;
            let engine = SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
                metric,
                conditioning_coords.to_vec(),
                conditioning_values.to_vec(),
                variogram,
                &[],
            )?;
            Ok(Self {
                metric,
                inner: SpacetimeUniversalSimulatorInner::Constant(engine),
            })
        } else {
            validate_st_universal_inputs(
                conditioning_coords.len(),
                conditioning_values.len(),
                trend,
            )?;
            let engine = SpaceTimeUniversalKrigingEngine::fit(
                metric,
                conditioning_coords.to_vec(),
                conditioning_values.to_vec(),
                variogram,
                trend,
            )?;
            Ok(Self {
                metric,
                inner: SpacetimeUniversalSimulatorInner::Drift(engine),
            })
        }
    }
}

impl<M: SpatialBasis> KrigingSimulator for SpacetimeUniversalSimulator<M> {
    type Target = SpaceTimeCoord<M::Coord>;
    type Output = Vec<Real>;

    fn predict_at(&self, target: Self::Target) -> Result<(Real, Real), KrigingError> {
        match &self.inner {
            SpacetimeUniversalSimulatorInner::Constant(engine) => {
                let pred = single_prediction(engine.predict(&[target]))?;
                Ok((pred.value, pred.variance))
            }
            SpacetimeUniversalSimulatorInner::Drift(engine) => {
                let pred = single_prediction(engine.predict(&[target]))?;
                Ok((pred.value, pred.variance))
            }
        }
    }

    fn append_sample(self, target: Self::Target, sample: Real) -> Result<Self, KrigingError> {
        match self.inner {
            SpacetimeUniversalSimulatorInner::Constant(engine) => Ok(Self {
                metric: self.metric,
                inner: SpacetimeUniversalSimulatorInner::Constant(
                    engine.condition(target, sample, 0.0)?,
                ),
            }),
            SpacetimeUniversalSimulatorInner::Drift(engine) => Ok(Self {
                metric: self.metric,
                inner: SpacetimeUniversalSimulatorInner::Drift(
                    engine.condition(target, sample, 0.0)?,
                ),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Spacetime binomial (engine + condition)
// ---------------------------------------------------------------------------

/// Space–time binomial kriging SGS backend (logit-scale engine).
#[derive(Debug, Clone)]
pub struct SpacetimeBinomialSimulator<M: SpatialMetric> {
    pub metric: M,
    engine: SpaceTimeOrdinaryKrigingEngine<M>,
}

impl<M: SpatialMetric> SpacetimeBinomialSimulator<M> {
    pub fn new(
        metric: M,
        conditioning_coords: &[SpaceTimeCoord<M::Coord>],
        successes: &[u32],
        trials: &[u32],
        variogram: SpaceTimeVariogram,
        prior: BinomialPrior,
    ) -> Result<Self, KrigingError> {
        validate_binomial_lengths(conditioning_coords.len(), successes.len(), trials.len())?;
        let (pool_coords, pool_logits, pool_obs_var) =
            build_st_binomial_pool(conditioning_coords, successes, trials, prior)?;
        let config = HeteroskedasticBinomialConfig::default();
        let engine = build_calibrated_logit_ordinary(
            pool_obs_var,
            &config,
            prior,
            &[],
            false,
            None,
            "space-time binomial simulation engine build failed",
            |extra| {
                SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
                    metric,
                    pool_coords.clone(),
                    pool_logits.clone(),
                    variogram,
                    &extra,
                )
            },
        )?
        .model;
        Ok(Self { metric, engine })
    }
}

impl<M: SpatialMetric> BinomialKrigingSimulator for SpacetimeBinomialSimulator<M> {
    type Target = SpaceTimeCoord<M::Coord>;

    fn predict_logit_at(&self, target: Self::Target) -> Result<(Real, Real), KrigingError> {
        let pred =
            binomial_prediction_from_ordinary(single_prediction(self.engine.predict(&[target]))?);
        Ok((pred.logit, pred.logit_variance))
    }

    fn append_logit_sample(
        self,
        target: Self::Target,
        logit_sample: Real,
    ) -> Result<Self, KrigingError> {
        Ok(Self {
            metric: self.metric,
            engine: self.engine.condition(target, logit_sample, 0.0)?,
        })
    }
}

fn build_st_binomial_pool<C: Copy>(
    conditioning_coords: &[SpaceTimeCoord<C>],
    successes: &[u32],
    trials: &[u32],
    prior: BinomialPrior,
) -> Result<StBinomialPool<C>, KrigingError> {
    let mut pool_coords = Vec::with_capacity(conditioning_coords.len());
    let mut pool_logits = Vec::with_capacity(conditioning_coords.len());
    let mut pool_obs_var = Vec::with_capacity(conditioning_coords.len());
    for i in 0..conditioning_coords.len() {
        if trials[i] == 0 {
            continue;
        }
        let obs =
            SpaceTimeBinomialObservation::new(conditioning_coords[i], successes[i], trials[i])?;
        pool_coords.push(obs.coord());
        pool_logits.push(obs.smoothed_logit_with_prior(prior));
        pool_obs_var.push(logit_observation_variance_empirical_bayes(
            prior,
            successes[i],
            trials[i],
        ));
    }
    if pool_coords.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    Ok((pool_coords, pool_logits, pool_obs_var))
}
