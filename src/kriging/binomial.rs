use crate::Real;
use crate::cv::BinomialCvSummary;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::predictor::cv::{BinomialGeoPredictor, leave_one_out_cv};
use crate::utils::{Probability, logistic, logit};
use crate::variogram::fitting::{FitResult, fit_variogram};
use crate::variogram::models::{VariogramModel, VariogramType};
use crate::variogram::{VariogramConfig, compute_empirical_variogram_binomial_calibrated};
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
pub const BINOMIAL_CALIBRATION_VERSION: u32 = 2;

/// Geometry-free binomial count pair `(successes, trials)` retained on count-based fits for
/// instance cross-validation and [`BinomialFit::diagnostics`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinomialCounts {
    successes: Vec<u32>,
    trials: Vec<u32>,
}

impl BinomialCounts {
    /// Parallel `(successes, trials)` slices with equal length (one row per training station).
    pub fn from_slices(successes: &[u32], trials: &[u32]) -> Result<Self, KrigingError> {
        if successes.len() != trials.len() {
            return Err(KrigingError::DimensionMismatch(
                "successes and trials must have equal length".to_string(),
            ));
        }
        Ok(Self {
            successes: successes.to_vec(),
            trials: trials.to_vec(),
        })
    }

    /// Count tensors extracted from geographic [`BinomialObservation`]s.
    pub fn from_geo_observations(observations: &[BinomialObservation]) -> Self {
        Self {
            successes: observations.iter().map(|o| o.successes()).collect(),
            trials: observations.iter().map(|o| o.trials()).collect(),
        }
    }

    #[inline]
    pub fn successes(&self) -> &[u32] {
        &self.successes
    }

    #[inline]
    pub fn trials(&self) -> &[u32] {
        &self.trials
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.successes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.successes.is_empty()
    }
}

/// [`BinomialBuildNotes::calibration_version`] when [`HeteroskedasticBinomialConfig::one_step_laplace_observation_variance`]
/// is enabled: per-site observation variance uses Fisher information at a **one-step Newton**
/// update of the logit from the EB-smoothed value (binomial score step).
pub const BINOMIAL_CALIBRATION_VERSION_ONE_STEP_LAPLACE_OBS_VAR: u32 = 3;

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

/// Logit-scale observation variance from a **Laplace / Fisher** approximation at the
/// empirical-Bayes smoothed proportion (inverse Fisher information on the logit link).
pub fn logit_observation_variance_laplace_binomial(
    prior: BinomialPrior,
    successes: u32,
    trials: u32,
) -> Real {
    if trials == 0 {
        return 0.0;
    }
    let s = successes as Real;
    let n = trials as Real;
    let p = (s + prior.alpha) / (n + prior.alpha + prior.beta);
    let denom = n * p * (1.0 - p);
    if denom <= 0.0 || !denom.is_finite() {
        0.0
    } else {
        (1.0 / denom).max(0.0)
    }
}

/// One Newton step on the binomial log-likelihood in **logit** space starting from the
/// empirical-Bayes smoothed proportion, then **Fisher** variance `1 / (n p(1-p))` at the
/// updated logit. Falls back to [`logit_observation_variance_laplace_binomial`] if the step
/// is numerically degenerate.
pub fn logit_observation_variance_one_step_laplace_binomial(
    prior: BinomialPrior,
    successes: u32,
    trials: u32,
) -> Real {
    if trials == 0 {
        return 0.0;
    }
    let s = successes as Real;
    let n = trials as Real;
    let a = prior.alpha;
    let b = prior.beta;
    let p_eb = (s + a) / (n + a + b);
    let lam0 = logit(Probability::from_known_in_range(
        crate::utils::clamp_probability(p_eb),
    ));
    let p0 = logistic(lam0);
    let score = s - n * p0;
    let info0 = n * p0 * (1.0 - p0);
    if info0 <= 0.0 || !info0.is_finite() {
        return logit_observation_variance_laplace_binomial(prior, successes, trials);
    }
    let lam1 = lam0 + score / info0;
    let p1 = logistic(lam1);
    let info1 = n * p1 * (1.0 - p1);
    if info1 <= 0.0 || !info1.is_finite() {
        return logit_observation_variance_laplace_binomial(prior, successes, trials);
    }
    (1.0 / info1).max(0.0)
}

/// Logit–scale “observation” variance for a binomial at one location: variance of a Beta
/// `Beta(s+α, f+β)` posterior on `p`, propagated through the delta method for `logit(p)`.
///
/// Used by simulation and diagnostics; the default calibrated binomial build uses
/// [`logit_observation_variance_laplace_binomial`].
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
    /// When `true`, each training-site logit observation variance uses
    /// [`logit_observation_variance_one_step_laplace_binomial`] instead of the default
    /// Laplace/Fisher value at the EB proportion. Sets [`BinomialBuildNotes::calibration_version`]
    /// to [`BINOMIAL_CALIBRATION_VERSION_ONE_STEP_LAPLACE_OBS_VAR`] on successful builds from counts.
    pub one_step_laplace_observation_variance: bool,
}

impl Default for HeteroskedasticBinomialConfig {
    fn default() -> Self {
        Self {
            min_logit_observation_variance: 1e-12,
            max_build_attempts: 6,
            one_step_laplace_observation_variance: false,
        }
    }
}

impl HeteroskedasticBinomialConfig {
    #[inline]
    pub(crate) fn raw_logit_observation_variance_binomial_site(
        &self,
        prior: BinomialPrior,
        successes: u32,
        trials: u32,
    ) -> Real {
        if self.one_step_laplace_observation_variance {
            logit_observation_variance_one_step_laplace_binomial(prior, successes, trials)
        } else {
            logit_observation_variance_laplace_binomial(prior, successes, trials)
        }
    }

    /// Value stored in [`BinomialBuildNotes::calibration_version`] for count-based calibrated builds.
    #[inline]
    pub fn calibration_version_for_notes(&self) -> u32 {
        if self.one_step_laplace_observation_variance {
            BINOMIAL_CALIBRATION_VERSION_ONE_STEP_LAPLACE_OBS_VAR
        } else {
            BINOMIAL_CALIBRATION_VERSION
        }
    }
}

/// Coarse stability preset for heteroskedastic binomial builds (maps to
/// [`HeteroskedasticBinomialConfig`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinomialStability {
    #[default]
    Default,
    Strict,
    Permissive,
}

impl BinomialStability {
    /// Maps to concrete inflation / floor settings.
    pub fn to_heteroskedastic_config(self) -> HeteroskedasticBinomialConfig {
        match self {
            BinomialStability::Default => HeteroskedasticBinomialConfig::default(),
            BinomialStability::Strict => HeteroskedasticBinomialConfig {
                max_build_attempts: 1,
                ..HeteroskedasticBinomialConfig::default()
            },
            BinomialStability::Permissive => HeteroskedasticBinomialConfig {
                max_build_attempts: 10,
                min_logit_observation_variance: 1e-10,
                ..HeteroskedasticBinomialConfig::default()
            },
        }
    }
}

#[cfg(test)]
mod stability_preset_tests {
    use super::{BinomialStability, HeteroskedasticBinomialConfig};

    #[test]
    fn stability_presets_match_expected_hetero_fields() {
        let def = BinomialStability::Default.to_heteroskedastic_config();
        let d0 = HeteroskedasticBinomialConfig::default();
        assert_eq!(def.max_build_attempts, d0.max_build_attempts);
        assert_eq!(
            def.min_logit_observation_variance,
            d0.min_logit_observation_variance
        );
        assert!(!def.one_step_laplace_observation_variance);
        let strict = BinomialStability::Strict.to_heteroskedastic_config();
        assert_eq!(strict.max_build_attempts, 1);
        assert_eq!(
            strict.min_logit_observation_variance,
            HeteroskedasticBinomialConfig::default().min_logit_observation_variance
        );
        assert!(!strict.one_step_laplace_observation_variance);
        let perm = BinomialStability::Permissive.to_heteroskedastic_config();
        assert_eq!(perm.max_build_attempts, 10);
        assert!(perm.min_logit_observation_variance > 1e-12);
        assert!(!perm.one_step_laplace_observation_variance);
    }
}

/// Heuristic Beta prior from pooled count data (for `"auto"` prior in bindings).
pub fn estimate_binomial_prior_from_counts(
    successes: &[u32],
    trials: &[u32],
) -> Result<BinomialPrior, KrigingError> {
    let mut s_tot: u64 = 0;
    let mut t_tot: u64 = 0;
    for (&s, &t) in successes.iter().zip(trials.iter()) {
        if t == 0 {
            continue;
        }
        if s > t {
            return Err(KrigingError::InvalidBinomialData(
                "estimate_binomial_prior_from_counts: successes exceed trials".to_string(),
            ));
        }
        s_tot += s as u64;
        t_tot += t as u64;
    }
    if t_tot == 0 {
        return Ok(BinomialPrior::default());
    }
    let p = (s_tot as Real) / (t_tot as Real);
    if p <= 0.0 || p >= 1.0 || !p.is_finite() {
        return Ok(BinomialPrior::default());
    }
    let eff = (t_tot as Real).sqrt().max(2.0);
    let alpha = (p * eff).max(0.5) + 0.5;
    let beta = ((1.0 - p) * eff).max(0.5) + 0.5;
    BinomialPrior::new(alpha, beta)
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
    /// Human-readable build warnings (e.g. logit variance inflation, prior auto-fallback).
    pub warnings: Vec<String>,
    /// Optional numeric diagnostics (reserved; usually `None` at build time).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_number: Option<Real>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_dof: Option<Real>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_msdr: Option<Real>,
}

/// Fitted model plus the same auditable build notes for geographic, projected, and
/// space–time binomial families.
#[derive(Debug, Clone)]
pub struct BinomialCalibratedResult<T> {
    /// Fitted model (geographic, projected, or space–time binomial as `T`).
    pub model: T,
    /// Build diagnostics: include in logs, WASM responses, and UIs.
    pub notes: BinomialBuildNotes,
    /// Count tensors when built from `(successes, trials)`; absent for precomputed-logit builds.
    pub(crate) training_counts: Option<BinomialCounts>,
}

impl<T> BinomialCalibratedResult<T> {
    /// Keep only the model (e.g. for internal prediction or legacy call sites).
    pub fn into_model(self) -> T {
        self.model
    }

    /// Count tensors retained at build time for LOO diagnostics and instance CV.
    pub fn training_counts(&self) -> Option<&BinomialCounts> {
        self.training_counts.as_ref()
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

/// Auditable snapshot of a geographic binomial fit: variogram, build notes, and optional
/// leave-one-out logit MSDR (see [`BinomialFit::diagnostics`]).
#[derive(Debug, Clone, PartialEq)]
pub struct BinomialDiagnostics {
    pub variogram: VariogramModel,
    pub build_notes: BinomialBuildNotes,
    /// Logit-scale MSDR from [`leave_one_out_binomial`] when count tensors were supplied.
    /// Each fold refits heteroskedastic inflation; values may differ slightly from the
    /// single held factorization in [`BinomialKrigingModel`].
    pub logit_loo_msdr: Option<Real>,
}

/// Logit-scale MSDR from leave-one-out cross-validation on a binomial [`KrigingPredictor`].
pub fn binomial_logit_loo_msdr<P>(predictor: &P) -> Result<Real, KrigingError>
where
    P: crate::predictor::cv::KrigingPredictor<Residual = crate::cv::BinomialCvResidual>,
{
    let residuals = leave_one_out_cv(predictor)?;
    Ok(BinomialCvSummary::from_residuals(&residuals).logit.msdr)
}

impl BinomialCalibratedResult<BinomialKrigingModel> {
    /// Bundle the fitted variogram, [`BinomialBuildNotes`], and optional LOO logit MSDR.
    ///
    /// When [`BinomialCalibratedResult::training_counts`] is present (count-based build),
    /// LOO MSDR is computed from the model’s training coordinates and retained counts using
    /// the notes’ prior and variogram (see [`BinomialDiagnostics::logit_loo_msdr`]).
    pub fn diagnostics(&self) -> Result<BinomialDiagnostics, KrigingError> {
        let variogram = self.model.variogram();
        let build_notes = self.notes.clone();
        let logit_loo_msdr = if let Some(counts) = self.training_counts.as_ref() {
            let coords = self.model.coords();
            if coords.len() != counts.len() {
                return Err(KrigingError::DimensionMismatch(format!(
                    "training coords length {} does not match counts length {}",
                    coords.len(),
                    counts.len()
                )));
            }
            Some(binomial_logit_loo_msdr(&BinomialGeoPredictor {
                coords,
                successes: counts.successes(),
                trials: counts.trials(),
                variogram,
                prior: build_notes.prior,
            })?)
        } else {
            None
        };
        Ok(BinomialDiagnostics {
            variogram,
            build_notes,
            logit_loo_msdr,
        })
    }
}

/// Result of a binomial kriging prediction on the logit link and probability scale.
#[derive(Debug, Clone, Copy)]
pub struct BinomialPrediction {
    /// Median prevalence (logistic of the predicted logit).
    pub prevalence_median: Real,
    /// Mean prevalence under a Gaussian predictive on the logit (Gauss–Hermite).
    pub prevalence_mean: Real,
    /// Predicted logit.
    pub logit: Real,
    /// Kriging variance of the logit prediction.
    pub logit_variance: Real,
    /// Delta-method variance of `prevalence_median` from `logit_variance`.
    pub prevalence_variance: Real,
}

#[inline]
pub(crate) fn finish_binomial_notes(mut notes: BinomialBuildNotes) -> BinomialBuildNotes {
    if notes.logit_inflation > 1.0 + 1e-6 {
        notes
            .warnings
            .push("logit_observation_variance_inflation".to_string());
    }
    notes
}

/// Shared inflation-retry loop for calibrated logit ordinary kriging across geometry backends.
///
/// Callers supply a closure that builds the inner ordinary model (geographic, projected, or
/// space–time) from the inflated per-site extra diagonal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_calibrated_logit_ordinary<O, F>(
    base_logit_observation_variance: Vec<Real>,
    config: &HeteroskedasticBinomialConfig,
    prior_for_notes: BinomialPrior,
    extra_zero_trial_drops: &[usize],
    from_precomputed_logits_only: bool,
    training_counts: Option<BinomialCounts>,
    build_failure_message: &str,
    build_with_inflated_extra: F,
) -> Result<BinomialCalibratedResult<O>, KrigingError>
where
    F: Fn(Vec<Real>) -> Result<O, KrigingError>,
{
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
        match build_with_inflated_extra(extra) {
            Ok(model) => {
                let mut z = extra_zero_trial_drops.to_vec();
                z.sort_unstable();
                return Ok(BinomialCalibratedResult {
                    model,
                    notes: finish_binomial_notes(BinomialBuildNotes {
                        calibration_version: config.calibration_version_for_notes(),
                        logit_inflation: inflation,
                        n_build_attempts: attempt + 1,
                        prior: prior_for_notes,
                        zero_trial_dropped_indices: z,
                        from_precomputed_logits_only,
                        warnings: Vec::new(),
                        condition_number: None,
                        effective_dof: None,
                        last_msdr: None,
                    }),
                    training_counts,
                });
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
        inflation *= 2.0 as Real;
    }
    Err(last_err.unwrap_or_else(|| KrigingError::MatrixError(build_failure_message.to_string())))
}

/// Delta-method variance of prevalence from logit mean and variance on the link scale.
pub fn delta_prevalence_variance(prevalence: Real, logit_variance: Real) -> Real {
    let factor = prevalence * (1.0 - prevalence);
    factor * factor * logit_variance.max(0.0)
}

/// E[logistic(Z)] for Z ~ N(μ, σ²) with σ = `sigma_logit`, via 5-point Gauss–Hermite quadrature.
fn gauss_hermite_mean_prevalence(mu: Real, sigma_logit: Real) -> Real {
    if !(mu.is_finite() && sigma_logit.is_finite()) {
        return logistic(mu);
    }
    if sigma_logit <= 0.0 {
        return logistic(mu);
    }
    // Physicist Gauss–Hermite nodes/weights for weight exp(-x²); standard normal expectation.
    const X: [f64; 5] = [
        -2.0201828704560856,
        -0.9585724646138185,
        0.0,
        0.9585724646138185,
        2.0201828704560856,
    ];
    const W: [f64; 5] = [
        0.0199532420585783,
        0.3936193231522412,
        0.9453087204829418,
        0.3936193231522412,
        0.0199532420585783,
    ];
    let inv_sqrt_pi = (1.0 / std::f64::consts::PI.sqrt()) as Real;
    let sqrt2 = (2.0 as Real).sqrt();
    let mut acc = 0.0 as Real;
    for i in 0..5 {
        let z = mu + sigma_logit * sqrt2 * (X[i] as Real);
        acc += (W[i] as Real) * logistic(z);
    }
    acc * inv_sqrt_pi
}

pub(crate) fn binomial_prediction_from_ordinary(
    pred: crate::kriging::ordinary::Prediction,
) -> BinomialPrediction {
    let logit = pred.value;
    let logit_variance = pred.variance.max(0.0);
    let prevalence_median = logistic(logit);
    let prevalence_mean = gauss_hermite_mean_prevalence(logit, logit_variance.sqrt());
    BinomialPrediction {
        prevalence_median,
        prevalence_mean,
        logit,
        logit_variance,
        prevalence_variance: delta_prevalence_variance(prevalence_median, logit_variance),
    }
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
    /// Variogram used by this model (same instance as at build time).
    pub fn variogram(&self) -> VariogramModel {
        self.ordinary_model.variogram()
    }

    /// Number of training stations.
    pub fn len(&self) -> usize {
        self.ordinary_model.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

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
        let training_counts = Some(BinomialCounts::from_geo_observations(&observations));
        let coords: Vec<GeoCoord> = observations.iter().map(|o| o.coord()).collect();
        let logits: Vec<Real> = observations
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect();
        let base: Vec<Real> = observations
            .iter()
            .map(|o| {
                config
                    .raw_logit_observation_variance_binomial_site(prior, o.successes(), o.trials())
                    .max(config.min_logit_observation_variance)
            })
            .collect();

        build_calibrated_logit_ordinary(
            base,
            &config,
            prior,
            extra_zero_trial_drops,
            false,
            training_counts,
            "binomial kriging build failed",
            |extra| {
                let dataset = GeoDataset::new(coords.clone(), logits.clone())?;
                OrdinaryKrigingModel::new_with_extra_diagonal(dataset, variogram, extra)
                    .map(|ordinary_model| Self { ordinary_model })
            },
        )
    }

    /// Training coordinates (same order as [`Self::len`] stations).
    pub fn coords(&self) -> &[GeoCoord] {
        self.ordinary_model.coords()
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
        let n = logits.len();
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "logits must all be finite (no NaN/inf)".to_string(),
            ));
        }
        if n != coords.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "coords ({}) and logits ({}) must have equal length",
                coords.len(),
                logits.len()
            )));
        }
        let zeros = vec![0.0 as Real; n];
        let mut fit = Self::from_precomputed_logits_with_logit_observation_variances(
            coords,
            logits,
            variogram,
            zeros,
            HeteroskedasticBinomialConfig::default(),
            BinomialPrior::default(),
        )?;
        fit.notes.from_precomputed_logits_only = true;
        Ok(fit)
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
        build_calibrated_logit_ordinary(
            base_logit_observation_variance,
            &config,
            prior_for_notes,
            &[],
            false,
            None,
            "from_precomputed with observation variances: build failed",
            |extra| {
                let dataset = GeoDataset::new(coords.clone(), logits.clone())?;
                OrdinaryKrigingModel::new_with_extra_diagonal(dataset, variogram, extra)
                    .map(|ordinary_model| Self { ordinary_model })
            },
        )
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<BinomialPrediction, KrigingError> {
        let pred = self.ordinary_model.predict(coord)?;
        Ok(binomial_prediction_from_ordinary(pred))
    }

    pub fn predict_batch(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch(coords)?;
        Ok(ordinary
            .into_iter()
            .map(binomial_prediction_from_ordinary)
            .collect())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu(coords).await?;
        Ok(ordinary
            .into_iter()
            .map(binomial_prediction_from_ordinary)
            .collect())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu_or_cpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu_or_cpu(coords).await?;
        Ok(ordinary
            .into_iter()
            .map(binomial_prediction_from_ordinary)
            .collect())
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu_blocking(coords)?;
        Ok(ordinary
            .into_iter()
            .map(binomial_prediction_from_ordinary)
            .collect())
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_or_cpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self
            .ordinary_model
            .predict_batch_gpu_or_cpu_blocking(coords)?;
        Ok(ordinary
            .into_iter()
            .map(binomial_prediction_from_ordinary)
            .collect())
    }
}

/// Fit a parametric variogram to binomial count data using a **noise-calibrated** empirical
/// variogram on EB-smoothed logits ([`compute_empirical_variogram_binomial_calibrated`]).
pub fn fit_binomial_variogram(
    coords: Vec<GeoCoord>,
    successes: &[u32],
    trials: &[u32],
    prior: BinomialPrior,
    variogram_config: &VariogramConfig,
    model_type: VariogramType,
    rel_weight_eps: Real,
) -> Result<FitResult, KrigingError> {
    let (obs, _) = build_binomial_observations_dropping_zero_trials(coords, successes, trials)?;
    if obs.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    let c2: Vec<GeoCoord> = obs.iter().map(|o| o.coord()).collect();
    let logits: Vec<Real> = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect();
    let per_site: Vec<Real> = obs
        .iter()
        .map(|o| logit_observation_variance_laplace_binomial(prior, o.successes(), o.trials()))
        .collect();
    let ds = GeoDataset::new(c2, logits)?;
    let empirical = compute_empirical_variogram_binomial_calibrated(
        &ds,
        &per_site,
        variogram_config,
        rel_weight_eps,
    )?;
    fit_variogram(&empirical, model_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_bundles_variogram_and_notes() {
        let p = BinomialPrior::default();
        let v = crate::variogram::models::VariogramModel::new(
            0.04,
            2.0,
            0.15,
            crate::variogram::models::VariogramType::Exponential,
        )
        .unwrap();
        let obs: Vec<BinomialObservation> = (0i32..10)
            .map(|i| {
                let lat = 40.0 as Real + (i as Real) * 0.05;
                let lon = -80.0 as Real + (i as Real) * 0.01;
                let t = 20u32;
                let s = 5u32 + (i as u32) % 8;
                BinomialObservation::new(GeoCoord::try_new(lat, lon).unwrap(), s, t).expect("o")
            })
            .collect();
        let fit = BinomialKrigingModel::new_with_prior(obs.clone(), v, p).expect("fit");
        let logits_fit = BinomialKrigingModel::from_precomputed_logits(
            obs.iter().map(|o| o.coord()).collect(),
            obs.iter().map(|o| o.smoothed_logit_with_prior(p)).collect(),
            v,
        )
        .expect("logits fit");
        let d0 = logits_fit.diagnostics().expect("d0");
        assert!(d0.logit_loo_msdr.is_none());
        let d = fit.diagnostics().expect("d");
        assert_eq!(d.variogram, v);
        assert_eq!(d.build_notes.prior, p);
        assert!(d.logit_loo_msdr.is_some());
        assert!(d.logit_loo_msdr.unwrap().is_finite());
    }

    #[test]
    fn default_prior_is_beta_1_1() {
        let p = BinomialPrior::default();
        assert!((p.alpha() - 1.0).abs() < 1e-5 && (p.beta() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn one_step_laplace_obs_var_differs_from_laplace_for_interior_counts() {
        let prior = BinomialPrior::default();
        let s = 40u32;
        let n = 100u32;
        let lap = logit_observation_variance_laplace_binomial(prior, s, n);
        let one = logit_observation_variance_one_step_laplace_binomial(prior, s, n);
        assert!(lap > 0.0 && one > 0.0);
        assert!(
            (one - lap).abs() > 1e-12,
            "expected one-step to differ from Laplace-at-EB-p, lap={lap} one={one}"
        );
    }

    #[test]
    fn new_with_config_one_step_sets_calibration_version_3_in_notes() {
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
        let config = HeteroskedasticBinomialConfig {
            one_step_laplace_observation_variance: true,
            ..Default::default()
        };
        let fit = BinomialKrigingModel::new_with_config(obs, v, p, config, &[]).expect("fit");
        assert_eq!(
            fit.notes.calibration_version,
            BINOMIAL_CALIBRATION_VERSION_ONE_STEP_LAPLACE_OBS_VAR
        );
    }

    #[test]
    fn estimate_binomial_prior_from_counts_returns_valid_beta() {
        let s = vec![3u32, 7, 5];
        let t = vec![10u32, 10, 10];
        let p = estimate_binomial_prior_from_counts(&s, &t).unwrap();
        assert!(p.alpha() > 0.0 && p.beta() > 0.0);
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
        let d = (a.logit - b.logit).abs();
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

    #[test]
    fn fit_binomial_variogram_nugget_not_above_classical_on_same_logits() {
        use crate::geo_dataset::GeoDataset;
        use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal, VariogramConfig};
        use crate::variogram::fitting::fit_variogram;
        use crate::variogram::models::VariogramType;
        use std::num::NonZeroUsize;

        let prior = BinomialPrior::default();
        let coords: Vec<GeoCoord> = (0..30)
            .map(|i| {
                GeoCoord::try_new(35.0 + (i as Real) * 0.1, -100.0 + (i as Real) * 0.08).unwrap()
            })
            .collect();
        let succ: Vec<u32> = (0..30).map(|i| 3u32 + (i % 5) as u32).collect();
        let trials: Vec<u32> = vec![15; 30];
        let nz = NonZeroUsize::new(15).unwrap();
        let config = VariogramConfig {
            max_distance: Some(PositiveReal::try_new(200.0 as Real).unwrap()),
            n_bins: nz,
            estimator: EmpiricalEstimator::Classical,
        };
        let (obs, _) =
            build_binomial_observations_dropping_zero_trials(coords.clone(), &succ, &trials)
                .unwrap();
        let c2: Vec<GeoCoord> = obs.iter().map(|o| o.coord()).collect();
        let logits: Vec<Real> = obs
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect();
        let ds = GeoDataset::new(c2, logits).unwrap();
        let emp_classical =
            crate::compute_empirical_variogram(&ds, &config).expect("empirical classical");
        let fit_classical = fit_variogram(&emp_classical, VariogramType::Exponential).unwrap();
        let fit_binomial = fit_binomial_variogram(
            coords,
            &succ,
            &trials,
            prior,
            &config,
            VariogramType::Exponential,
            1e-10 as Real,
        )
        .unwrap();
        let (n_c, _, _) = fit_classical.model.params();
        let (n_b, _, _) = fit_binomial.model.params();
        assert!(
            n_b <= n_c + 1e-3 * n_c.max(1e-6 as Real),
            "expected calibrated-fit nugget <= classical logit-fit nugget + tol; n_b={n_b} n_c={n_c}"
        );
    }
}
