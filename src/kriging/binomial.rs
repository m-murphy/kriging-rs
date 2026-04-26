use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::utils::{Probability, logistic, logit};
use crate::variogram::models::VariogramModel;
use serde::Serialize;
use std::ops::Deref;

// Module overview (contract)
// ---------------------------------------------------------------------------
// **Prevalence mapping (default):** Empirical-Bayes Beta prior on each observed
// proportion, map to a logit working value, and apply **ordinary kriging with per-site
// logit observation variance** on the covariance diagonal (calibrated to binomial sampling
// on the link scale). The spatial signal is still modeled as a second-order stationary
// **Gaussian** random field on the logit link. The reported prevalence is the logistic of the
// predicted logit; the prevalence–scale variance uses a first-order **delta** approximation.
// This is not full binomial (hierarchical) maximum likelihood geostatistics, but is
// deterministic, defensible for mixed trial counts, and well-suited to interactive / WASM use.
//
// **Pre-computed logit path:** when logits are supplied without per-trial data, the library
// cannot form binomial observation variances; see [`BinomialKrigingModel::from_precomputed_logits`].

/// Contract version for binomial kriging calibration (`BinomialBuildNotes::calibration_version`).
pub const BINOMIAL_CALIBRATION_VERSION: u32 = 1;

/// A single binomial observation at a location: number of successes and trials.
///
/// Use with [`BinomialKrigingModel`] to build a prevalence surface from count data.
#[derive(Debug, Clone, Copy)]
pub struct BinomialObservation {
    coord: GeoCoord,
    successes: u32,
    trials: u32,
}

impl BinomialObservation {
    /// Creates an observation with validated `trials > 0` and `successes <= trials`.
    pub fn new(coord: GeoCoord, successes: u32, trials: u32) -> Result<Self, KrigingError> {
        if trials == 0 {
            return Err(KrigingError::InvalidBinomialData(
                "trials must be greater than 0".to_string(),
            ));
        }
        if successes > trials {
            return Err(KrigingError::InvalidBinomialData(format!(
                "successes ({}) cannot exceed trials ({})",
                successes, trials
            )));
        }
        Ok(Self {
            coord,
            successes,
            trials,
        })
    }

    #[inline]
    pub fn coord(self) -> GeoCoord {
        self.coord
    }

    #[inline]
    pub fn successes(self) -> u32 {
        self.successes
    }

    #[inline]
    pub fn trials(self) -> u32 {
        self.trials
    }

    pub fn smoothed_probability(&self) -> Real {
        self.smoothed_probability_with_prior(BinomialPrior::default())
    }

    pub fn smoothed_probability_with_prior(&self, prior: BinomialPrior) -> Real {
        let s = self.successes as Real;
        let n = self.trials as Real;
        (s + prior.alpha) / (n + prior.alpha + prior.beta)
    }

    pub fn smoothed_logit(&self) -> Real {
        self.smoothed_logit_with_prior(BinomialPrior::default())
    }

    pub fn smoothed_logit_with_prior(&self, prior: BinomialPrior) -> Real {
        let p = self.smoothed_probability_with_prior(prior);
        logit(Probability::from_known_in_range(p))
    }
}

/// Beta(alpha, beta) prior for smoothing binomial observations.
///
/// Used by [`BinomialObservation::smoothed_probability_with_prior`],
/// [`BinomialKrigingModel::new_with_prior`], and observation variance on the logit link.
/// Default is **Beta(1, 1)** (uniform on probability).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BinomialPrior {
    alpha: Real,
    beta: Real,
}

impl Default for BinomialPrior {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl BinomialPrior {
    /// Creates a prior with validated `alpha > 0` and `beta > 0`.
    pub fn new(alpha: Real, beta: Real) -> Result<Self, KrigingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(KrigingError::InvalidBinomialData(
                "prior alpha must be finite and positive".to_string(),
            ));
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(KrigingError::InvalidBinomialData(
                "prior beta must be finite and positive".to_string(),
            ));
        }
        Ok(Self { alpha, beta })
    }

    #[inline]
    pub fn alpha(self) -> Real {
        self.alpha
    }

    #[inline]
    pub fn beta(self) -> Real {
        self.beta
    }
}

/// Logit–scale “observation” variance for a binomial at one location: variance of a Beta
/// `Beta(s+α, f+β)` posterior on `p`, propagated through the delta method for `logit(p)`.
/// Use as extra diagonal noise for ordinary kriging in logit space.
pub fn logit_observation_variance_empirical_bayes(
    prior: BinomialPrior,
    successes: u32,
    trials: u32,
) -> Real {
    if trials == 0 {
        return 0.0;
    }
    let a = prior.alpha;
    let b = prior.beta;
    let s = successes as Real;
    let f = (trials - successes) as Real;
    // Posterior for p under Beta(α,β) prior: Beta(s+α, f+β)
    let ap = s + a;
    let bt = f + b;
    let n = ap + bt;
    // Var(p) for Beta(ap,bt)
    let var_p = (ap * bt) / (n * n * (n + 1.0));
    let p = ap / n;
    let d = p * (1.0 - p);
    if d <= 0.0 || !d.is_finite() {
        0.0
    } else {
        (var_p / (d * d)).max(0.0)
    }
}

/// Tuning and stability for heteroskedastic (calibrated) binomial kriging.
#[derive(Debug, Clone, Copy)]
pub struct HeteroskedasticBinomialConfig {
    /// Floor for each per-site logit observation variance.
    pub min_logit_observation_variance: Real,
    /// If the first ordinary-kriging build fails, multiply all observation variances by
    /// `2^(attempt-1)` for up to this many total attempts. Default: 6 (up to 32×).
    pub max_build_attempts: u32,
}

impl Default for HeteroskedasticBinomialConfig {
    fn default() -> Self {
        Self {
            min_logit_observation_variance: 1e-12,
            max_build_attempts: 6,
        }
    }
}

/// What happened when building a calibrated binomial model (always return from [`BinomialFit`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BinomialBuildNotes {
    /// Contract version; bump when the statistical pipeline meaningfully changes.
    pub calibration_version: u32,
    /// Total multiplier on base logit observation variances: `1` = first try, `2` = after one inflation, etc.
    pub logit_inflation: Real,
    /// Number of factorization attempts (1-based).
    pub n_build_attempts: u32,
    /// Prior used for EB-smoothed logits and observation variances.
    pub prior: BinomialPrior,
    /// Original input indices with `trials == 0` (dropped; no information).
    pub zero_trial_dropped_indices: Vec<usize>,
    /// `true` if the model was built from caller-supplied logits only (no per-site var).
    pub from_precomputed_logits_only: bool,
}

/// Fitted model plus the same auditable build notes for geographic, projected, and
/// space–time binomial families.
#[derive(Debug, Clone)]
pub struct BinomialCalibratedResult<T> {
    /// Fitted model (geographic, projected, or space–time binomial as `T`).
    pub model: T,
    /// Build diagnostics: include in logs, WASM responses, and UIs.
    pub notes: BinomialBuildNotes,
}

impl<T> BinomialCalibratedResult<T> {
    /// Keep only the model (e.g. for internal prediction or legacy call sites).
    pub fn into_model(self) -> T {
        self.model
    }
}

impl<T> Deref for BinomialCalibratedResult<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

/// Geographic **bin**omial kriging: [`BinomialCalibratedResult`] specialized to
/// [`BinomialKrigingModel`].
pub type BinomialFit = BinomialCalibratedResult<BinomialKrigingModel>;

/// Result of a binomial kriging prediction: prevalence (0–1), logit value, logit-space variance,
/// and a delta-method approximation of the variance on the prevalence scale.
#[derive(Debug, Clone, Copy)]
pub struct BinomialPrediction {
    /// Estimated prevalence at the target location (probability in (0, 1)).
    pub prevalence: Real,
    /// Logit of the prevalence (unbounded).
    pub logit_value: Real,
    /// Kriging variance of the logit prediction. This is **not** the variance of `prevalence`;
    /// see [`prevalence_variance`](Self::prevalence_variance) for a delta-method approximation.
    pub variance: Real,
    /// Delta-method approximation of the variance of `prevalence`:
    /// `Var(p) ≈ [p(1 - p)]² · Var(logit)`. Use for approximate CIs on the probability scale.
    pub prevalence_variance: Real,
}

#[inline]
fn delta_prevalence_variance(prevalence: Real, logit_variance: Real) -> Real {
    let factor = prevalence * (1.0 - prevalence);
    factor * factor * logit_variance.max(0.0)
}

/// Fitted binomial kriging model for prevalence surface estimation.
///
/// Build from [`BinomialObservation`]s and a [`VariogramModel`] with
/// [`new`](Self::new) or [`new_with_prior`](Self::new_with_prior) (returning [`BinomialFit`]
/// with build notes), then use [`BinomialKrigingModel::predict`].
#[derive(Debug, Clone)]
pub struct BinomialKrigingModel {
    ordinary_model: OrdinaryKrigingModel,
}

/// Collect indices with `trials == 0` in parallel with `successes` / `trials` slices.
pub fn indices_of_zero_trials(trials: &[u32]) -> Vec<usize> {
    trials
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| if t == 0 { Some(i) } else { None })
        .collect()
}

/// Build valid [`BinomialObservation`]s from parallel slices, **dropping** rows with
/// `trials == 0` and recording their original indices.
pub fn build_binomial_observations_dropping_zero_trials(
    coords: Vec<GeoCoord>,
    successes: &[u32],
    trials: &[u32],
) -> Result<(Vec<BinomialObservation>, Vec<usize>), KrigingError> {
    if coords.len() != successes.len() || successes.len() != trials.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({}), successes ({}), trials ({}) must have equal length",
            coords.len(),
            successes.len(),
            trials.len()
        )));
    }
    let mut dropped: Vec<usize> = Vec::new();
    let mut out: Vec<BinomialObservation> = Vec::new();
    for i in 0..coords.len() {
        if trials[i] == 0 {
            dropped.push(i);
            continue;
        }
        if successes[i] > trials[i] {
            return Err(KrigingError::InvalidBinomialData(format!(
                "successes ({}) cannot exceed trials ({}) at index {}",
                successes[i], trials[i], i
            )));
        }
        out.push(BinomialObservation::new(
            coords[i],
            successes[i],
            trials[i],
        )?);
    }
    Ok((out, dropped))
}

impl BinomialKrigingModel {
    /// Build a calibrated (heteroskedastic) model with the default `Beta(1, 1)` prior and
    /// default stability config.
    pub fn new(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
    ) -> Result<BinomialFit, KrigingError> {
        Self::new_with_config(
            observations,
            variogram,
            BinomialPrior::default(),
            HeteroskedasticBinomialConfig::default(),
            &[],
        )
    }

    /// As [`new`](Self::new), with an explicit Beta prior.
    pub fn new_with_prior(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
        prior: BinomialPrior,
    ) -> Result<BinomialFit, KrigingError> {
        Self::new_with_config(
            observations,
            variogram,
            prior,
            HeteroskedasticBinomialConfig::default(),
            &[],
        )
    }

    /// Calibrated build with full control over stability; `zero_trial_drops` is appended to
    /// the returned [`BinomialBuildNotes::zero_trial_dropped_indices`].
    pub fn new_with_config(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
        prior: BinomialPrior,
        config: HeteroskedasticBinomialConfig,
        extra_zero_trial_drops: &[usize],
    ) -> Result<BinomialFit, KrigingError> {
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let coords: Vec<GeoCoord> = observations.iter().map(|o| o.coord()).collect();
        let logits: Vec<Real> = observations
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect();
        let base: Vec<Real> = observations
            .iter()
            .map(|o| logit_observation_variance_empirical_bayes(prior, o.successes(), o.trials()))
            .map(|v| v.max(config.min_logit_observation_variance))
            .collect();

        let n_tries = config.max_build_attempts.max(1);
        let mut last_err: Option<KrigingError> = None;
        let mut inflation = 1.0 as Real;
        for attempt in 0..n_tries {
            let extra: Vec<Real> = base
                .iter()
                .map(|&v| (v * inflation).max(config.min_logit_observation_variance))
                .collect();
            let dataset = GeoDataset::new(coords.clone(), logits.clone())?;
            match OrdinaryKrigingModel::new_with_extra_diagonal(dataset, variogram, extra) {
                Ok(ordinary_model) => {
                    let mut z = extra_zero_trial_drops.to_vec();
                    z.sort_unstable();
                    return Ok(BinomialCalibratedResult {
                        model: Self { ordinary_model },
                        notes: BinomialBuildNotes {
                            calibration_version: BINOMIAL_CALIBRATION_VERSION,
                            logit_inflation: inflation,
                            n_build_attempts: attempt + 1,
                            prior,
                            zero_trial_dropped_indices: z,
                            from_precomputed_logits_only: false,
                        },
                    });
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
            inflation *= 2.0 as Real;
        }
        Err(last_err.unwrap_or_else(|| {
            KrigingError::MatrixError("binomial kriging build failed".to_string())
        }))
    }

    /// Build a binomial kriging model from pre-computed logit values.
    ///
    /// No per-trial information is available: the covariance uses **no** per-site
    /// observation noise beyond nugget, jitter, and the variogram (see notes).
    pub fn from_precomputed_logits(
        coords: Vec<GeoCoord>,
        logits: Vec<Real>,
        variogram: VariogramModel,
    ) -> Result<BinomialFit, KrigingError> {
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "logits must all be finite (no NaN/inf)".to_string(),
            ));
        }
        let dataset = GeoDataset::new(coords, logits)?;
        let ordinary_model = OrdinaryKrigingModel::new(dataset, variogram)?;
        Ok(BinomialCalibratedResult {
            model: Self { ordinary_model },
            notes: BinomialBuildNotes {
                calibration_version: BINOMIAL_CALIBRATION_VERSION,
                logit_inflation: 1.0,
                n_build_attempts: 1,
                prior: BinomialPrior::default(),
                zero_trial_dropped_indices: Vec::new(),
                from_precomputed_logits_only: true,
            },
        })
    }

    /// Pre-computed logit field with a **per-station** logit observation variance (diagonal) at
    /// every index (e.g. a simulation pool: data sites use [`logit_observation_variance_empirical_bayes`], latent draws use `0`).
    /// Reuses the same factorization / inflation policy as [`Self::new_with_config`].
    pub fn from_precomputed_logits_with_logit_observation_variances(
        coords: Vec<GeoCoord>,
        logits: Vec<Real>,
        variogram: VariogramModel,
        base_logit_observation_variance: Vec<Real>,
        config: HeteroskedasticBinomialConfig,
        prior_for_notes: BinomialPrior,
    ) -> Result<BinomialFit, KrigingError> {
        if logits.len() != coords.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "coords ({}) and logits ({}) must have equal length",
                coords.len(),
                logits.len()
            )));
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "logits must all be finite (no NaN/inf)".to_string(),
            ));
        }
        if base_logit_observation_variance.len() != coords.len() {
            return Err(KrigingError::InvalidInput(
                "base logit observation variance must match coords length".to_string(),
            ));
        }
        for &v in &base_logit_observation_variance {
            if !v.is_finite() || v < 0.0 {
                return Err(KrigingError::InvalidInput(
                    "logit observation variances must be finite and non-negative".to_string(),
                ));
            }
        }
        let n_tries = config.max_build_attempts.max(1);
        let mut last_err: Option<KrigingError> = None;
        let mut inflation = 1.0 as Real;
        for attempt in 0..n_tries {
            let extra: Vec<Real> = base_logit_observation_variance
                .iter()
                .map(|&v| (v * inflation).max(config.min_logit_observation_variance))
                .collect();
            let dataset = GeoDataset::new(coords.clone(), logits.clone())?;
            match OrdinaryKrigingModel::new_with_extra_diagonal(dataset, variogram, extra) {
                Ok(ordinary_model) => {
                    return Ok(BinomialCalibratedResult {
                        model: Self { ordinary_model },
                        notes: BinomialBuildNotes {
                            calibration_version: BINOMIAL_CALIBRATION_VERSION,
                            logit_inflation: inflation,
                            n_build_attempts: attempt + 1,
                            prior: prior_for_notes,
                            zero_trial_dropped_indices: Vec::new(),
                            from_precomputed_logits_only: false,
                        },
                    });
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
            inflation *= 2.0 as Real;
        }
        Err(last_err.unwrap_or_else(|| {
            KrigingError::MatrixError(
                "from_precomputed with observation variances: build failed".to_string(),
            )
        }))
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<BinomialPrediction, KrigingError> {
        let pred = self.ordinary_model.predict(coord)?;
        Ok(to_binomial_prediction(pred))
    }

    pub fn predict_batch(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch(coords)?;
        Ok(ordinary.into_iter().map(to_binomial_prediction).collect())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu(coords).await?;
        Ok(ordinary.into_iter().map(to_binomial_prediction).collect())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu_or_cpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu_or_cpu(coords).await?;
        Ok(ordinary.into_iter().map(to_binomial_prediction).collect())
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu_blocking(coords)?;
        Ok(ordinary.into_iter().map(to_binomial_prediction).collect())
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_or_cpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self
            .ordinary_model
            .predict_batch_gpu_or_cpu_blocking(coords)?;
        Ok(ordinary.into_iter().map(to_binomial_prediction).collect())
    }
}

fn to_binomial_prediction(pred: crate::kriging::ordinary::Prediction) -> BinomialPrediction {
    let prevalence = logistic(pred.value);
    BinomialPrediction {
        prevalence,
        logit_value: pred.value,
        variance: pred.variance,
        prevalence_variance: delta_prevalence_variance(prevalence, pred.variance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_prior_is_beta_1_1() {
        let p = BinomialPrior::default();
        assert!((p.alpha() - 1.0).abs() < 1e-5 && (p.beta() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn handles_zero_and_all_successes_with_smoothing() {
        let o1 = BinomialObservation::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0, 10).unwrap();
        let o2 = BinomialObservation::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 10, 10).unwrap();
        let p1 = o1.smoothed_probability();
        let p2 = o2.smoothed_probability();
        assert!(p1 > 0.0 && p1 < 1.0);
        assert!(p2 > 0.0 && p2 < 1.0);
    }

    #[test]
    fn calibrated_uses_extra_diagonal_on_covariance() {
        let p = BinomialPrior::default();
        let v = crate::variogram::models::VariogramModel::new(
            0.04,
            2.0,
            0.15,
            crate::variogram::models::VariogramType::Exponential,
        )
        .unwrap();
        let mut obs: Vec<BinomialObservation> = (0i32..18)
            .map(|i| {
                let lat = 40.0 as Real + (i as Real) * 0.12;
                let lon = -80.0 as Real + (i as Real) * 0.02;
                let t = 5u32 + ((i as u32) % 2) * 200;
                let s = t / 3;
                BinomialObservation::new(GeoCoord::try_new(lat, lon).unwrap(), s, t).expect("o")
            })
            .collect();
        let last = obs.len() - 1;
        let c = obs[last].coord();
        obs[last] = BinomialObservation::new(c, 0, 8).expect("o");
        let coords: Vec<GeoCoord> = obs.iter().map(|o| o.coord()).collect();
        let logits: Vec<Real> = obs.iter().map(|o| o.smoothed_logit_with_prior(p)).collect();
        let fit = super::BinomialKrigingModel::new_with_prior(obs, v, p).expect("fit");
        let fit2 = super::BinomialKrigingModel::from_precomputed_logits(coords, logits, v)
            .expect("logits only");
        let t = GeoCoord::try_new(40.1, -79.9).unwrap();
        let a = fit.model.predict(t).unwrap();
        let b = fit2.model.predict(t).unwrap();
        let d = (a.logit_value - b.logit_value).abs();
        assert!(
            d > 0.01,
            "expected precomputed (no obs var) to differ, got d={d}"
        );
    }

    #[test]
    fn from_precomputed_notes_flag() {
        let c = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let f = super::BinomialKrigingModel::from_precomputed_logits(c, vec![0.0, 0.0], {
            crate::variogram::models::VariogramModel::new(
                0.05,
                2.0,
                100.0,
                crate::variogram::models::VariogramType::Exponential,
            )
            .unwrap()
        })
        .expect("f");
        assert!(f.notes.from_precomputed_logits_only);
    }
}
