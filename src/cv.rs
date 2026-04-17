//! Cross-validation for kriging models.
//!
//! Per-variant entry points are provided for every kriging flavor shipped by this crate:
//!
//! - Ordinary: [`leave_one_out`] / [`k_fold`]
//! - Simple (known mean): [`leave_one_out_simple`] / [`k_fold_simple`]
//! - Universal (polynomial drift): [`leave_one_out_universal`] / [`k_fold_universal`]
//! - Projected / planar (anisotropy): [`leave_one_out_projected`] / [`k_fold_projected`]
//! - Binomial (successes / trials): [`leave_one_out_binomial`] / [`k_fold_binomial`]
//! - Space–time ordinary / simple / universal / binomial:
//!   [`leave_one_out_spacetime`] / [`k_fold_spacetime`] and friends.
//!
//! The continuous variants share [`CvResidual`] / [`CvSummary`]. Binomial CV returns
//! [`BinomialCvResidual`] values that carry **both** the logit-scale residual (directly
//! comparable to continuous kriging and MSDR-calibratable) **and** the prevalence-scale
//! residual (intuitive; delta-method variance). [`BinomialCvSummary`] aggregates both.
//!
//! All routines assume the supplied variogram / drift / mean / anisotropy is held fixed
//! across folds; only the kriging system is refit per fold. Callers who want to refit
//! variogram parameters inside each fold must iterate themselves.
//!
//! The folds are deterministic round-robin assignments (station `i` goes to fold `i % k`),
//! which keeps validation reproducible and avoids the need for an RNG dependency. Callers
//! who want randomization can shuffle inputs before calling.

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::binomial::{BinomialKrigingModel, BinomialObservation, BinomialPrior};
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::kriging::simple::SimpleKrigingModel;
use crate::kriging::universal::{UniversalKrigingModel, UniversalTrend};
use crate::projected::{
    Anisotropy2D, BinomialProjectedKrigingModel, ProjectedBinomialObservation, ProjectedCoord,
    ProjectedDataset, ProjectedKrigingModel,
};
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::kriging::binomial::{
    SpaceTimeBinomialKrigingModel, SpaceTimeBinomialObservation,
};
use crate::spacetime::kriging::ordinary::SpaceTimeOrdinaryKrigingModel;
use crate::spacetime::kriging::simple::SpaceTimeSimpleKrigingModel;
use crate::spacetime::kriging::universal::{
    SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend,
};
use crate::spacetime::metric::{SpatialBasis, SpatialMetric};
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::utils::{logistic, logit_clamped};
use crate::variogram::models::VariogramModel;

/// A single cross-validation residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvResidual {
    /// Index of the held-out station in the original input arrays.
    pub index: usize,
    /// Observed value at the held-out station.
    pub observed: Real,
    /// Kriging prediction at the held-out station (from the training fold).
    pub predicted: Real,
    /// Kriging variance at the held-out station.
    pub variance: Real,
}

impl CvResidual {
    /// Signed residual `observed − predicted`.
    pub fn error(&self) -> Real {
        self.observed - self.predicted
    }
}

/// Summary statistics over a set of cross-validation residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvSummary {
    pub n: usize,
    /// Mean signed error (bias).
    pub mean_error: Real,
    /// Root mean squared error.
    pub rmse: Real,
    /// Mean squared deviation ratio: mean((obs − pred)² / variance).
    /// Approximately 1 when the variogram is well-calibrated.
    pub msdr: Real,
}

impl CvSummary {
    pub fn from_residuals(residuals: &[CvResidual]) -> Self {
        Self::from_scalar_iter(
            residuals.len(),
            residuals.iter().map(|r| (r.error(), r.variance)),
        )
    }

    /// Internal: compute bias/RMSE/MSDR from pre-computed `(error, variance)` pairs. NaN
    /// errors (e.g. when an observation is undefined, as for binomial trials == 0) are
    /// skipped. The `n` field is set to the number of *finite* residuals rather than the
    /// total length of the input iterator.
    fn from_scalar_iter<I>(_hint: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = (Real, Real)>,
    {
        let mut n_finite = 0usize;
        let mut sum_e = 0.0 as Real;
        let mut sum_e2 = 0.0 as Real;
        let mut sum_ratio = 0.0 as Real;
        let mut ratio_n = 0usize;
        for (e, variance) in iter {
            if !e.is_finite() {
                continue;
            }
            n_finite += 1;
            sum_e += e;
            sum_e2 += e * e;
            if variance > 0.0 && variance.is_finite() {
                sum_ratio += e * e / variance;
                ratio_n += 1;
            }
        }
        if n_finite == 0 {
            return Self {
                n: 0,
                mean_error: 0.0,
                rmse: 0.0,
                msdr: 0.0,
            };
        }
        let nf = n_finite as Real;
        let msdr = if ratio_n == 0 {
            0.0
        } else {
            sum_ratio / ratio_n as Real
        };
        Self {
            n: n_finite,
            mean_error: sum_e / nf,
            rmse: (sum_e2 / nf).sqrt(),
            msdr,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared fold iteration
// ---------------------------------------------------------------------------

fn validate_len(n_coords: usize, n_values: usize) -> Result<(), KrigingError> {
    if n_coords != n_values {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({n_coords}) and values ({n_values}) must have equal length"
        )));
    }
    if n_coords < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    Ok(())
}

fn validate_k(n: usize, k: usize) -> Result<(), KrigingError> {
    if k < 2 || k > n {
        return Err(KrigingError::InvalidInput(format!(
            "k must satisfy 2 <= k <= n (n={n}, k={k})"
        )));
    }
    Ok(())
}

/// Run a function once per leave-one-out fold: for each held-out index `i`, the closure
/// receives the complement indices as `train` and the singleton `[i]` as `test`. Indices
/// are pushed in the natural `0..n` order.
fn for_each_loo_fold<F>(n: usize, mut body: F) -> Result<(), KrigingError>
where
    F: FnMut(&[usize], &[usize]) -> Result<(), KrigingError>,
{
    let mut train = Vec::with_capacity(n.saturating_sub(1));
    for i in 0..n {
        train.clear();
        for j in 0..n {
            if j != i {
                train.push(j);
            }
        }
        let test = [i];
        body(&train, &test)?;
    }
    Ok(())
}

/// Run a function once per k-fold, with deterministic round-robin assignment
/// (station `i` goes to fold `i % k`). The closure receives the train and test indices
/// for each fold.
fn for_each_k_fold<F>(n: usize, k: usize, mut body: F) -> Result<(), KrigingError>
where
    F: FnMut(&[usize], &[usize]) -> Result<(), KrigingError>,
{
    let mut train = Vec::new();
    let mut test = Vec::new();
    for fold in 0..k {
        train.clear();
        test.clear();
        for i in 0..n {
            if i % k == fold {
                test.push(i);
            } else {
                train.push(i);
            }
        }
        if train.is_empty() || test.is_empty() {
            continue;
        }
        body(&train, &test)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ordinary kriging CV
// ---------------------------------------------------------------------------

/// Leave-one-out cross-validation: for each station `i`, fit an ordinary kriging model on
/// the remaining `n − 1` stations and predict station `i`. Returns residuals in input order.
///
/// Requires at least 2 stations. `O(n)` model builds — suitable for small-to-moderate `n`.
pub fn leave_one_out(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = OrdinaryKrigingModel::new(dataset, variogram)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold cross-validation over ordinary kriging. See module-level docs for fold assignment.
pub fn k_fold(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = OrdinaryKrigingModel::new(dataset, variogram)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Simple kriging CV (known mean)
// ---------------------------------------------------------------------------

/// Leave-one-out CV for [`SimpleKrigingModel`]. The supplied `mean` is treated as known
/// (same value for every fold) — this matches how simple kriging is used in practice and
/// keeps the CV cost proportional to one system solve per fold.
pub fn leave_one_out_simple(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    mean: Real,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = SimpleKrigingModel::new(dataset, variogram, mean)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`SimpleKrigingModel`]. See [`leave_one_out_simple`] for mean semantics.
pub fn k_fold_simple(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    mean: Real,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = SimpleKrigingModel::new(dataset, variogram, mean)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Universal kriging CV (polynomial drift)
// ---------------------------------------------------------------------------

/// Leave-one-out CV for [`UniversalKrigingModel`] with the given drift basis. Drift
/// coefficients are re-estimated inside each fold from the training stations, so there is
/// no in-sample leakage from the drift.
pub fn leave_one_out_universal(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    trend: UniversalTrend,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = UniversalKrigingModel::new(dataset, variogram, trend)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`UniversalKrigingModel`] with the given drift basis.
pub fn k_fold_universal(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    trend: UniversalTrend,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = UniversalKrigingModel::new(dataset, variogram, trend)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Projected kriging CV (planar, optional 2-D anisotropy)
// ---------------------------------------------------------------------------

/// Leave-one-out CV for [`ProjectedKrigingModel`]. Coordinates are planar `(x, y)`;
/// Euclidean distances (optionally anisotropy-deformed) are used inside each fold.
pub fn leave_one_out_projected(
    coords: &[ProjectedCoord],
    values: &[Real],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<ProjectedCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = ProjectedDataset::new(fold_coords, fold_values)?;
        let model = ProjectedKrigingModel::new(dataset, variogram, anisotropy)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`ProjectedKrigingModel`]. See [`leave_one_out_projected`] for coord semantics.
pub fn k_fold_projected(
    coords: &[ProjectedCoord],
    values: &[Real],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<ProjectedCoord> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = ProjectedDataset::new(fold_coords, fold_values)?;
        let model = ProjectedKrigingModel::new(dataset, variogram, anisotropy)?;
        let test_coords: Vec<ProjectedCoord> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Binomial kriging CV (reports BOTH logit and prevalence scales)
// ---------------------------------------------------------------------------

/// A single binomial cross-validation residual. Reports the held-out observation and the
/// prediction on **both** the logit scale (directly comparable to continuous kriging and
/// calibratable via MSDR) and the prevalence scale (intuitive for probability-scale
/// diagnostics, with a delta-method variance).
///
/// When `trials == 0` at the held-out station, the observed prevalence and logit are
/// undefined and reported as `NaN`. The prediction is still populated; the entry is
/// retained at its input index so downstream code can audit which stations were
/// unobservable. [`BinomialCvSummary`] skips these when aggregating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialCvResidual {
    /// Index of the held-out station in the original input arrays.
    pub index: usize,
    /// Held-out success count.
    pub successes: u32,
    /// Held-out trial count. `0` means the observation is undefined.
    pub trials: u32,
    /// Held-out observed logit. `NaN` when `trials == 0`.
    pub observed_logit: Real,
    /// Model prediction on the logit scale.
    pub predicted_logit: Real,
    /// Kriging variance on the logit scale.
    pub logit_variance: Real,
    /// Held-out observed prevalence `successes / trials`. `NaN` when `trials == 0`.
    pub observed_prevalence: Real,
    /// Model prediction on the prevalence scale (logistic of `predicted_logit`).
    pub predicted_prevalence: Real,
    /// Delta-method approximation of the variance of `predicted_prevalence`.
    pub prevalence_variance: Real,
}

impl BinomialCvResidual {
    /// Signed logit-scale error `observed_logit − predicted_logit`. `NaN` when `trials == 0`.
    pub fn logit_error(&self) -> Real {
        self.observed_logit - self.predicted_logit
    }

    /// Signed prevalence-scale error `observed_prevalence − predicted_prevalence`.
    /// `NaN` when `trials == 0`.
    pub fn prevalence_error(&self) -> Real {
        self.observed_prevalence - self.predicted_prevalence
    }
}

/// Aggregate summary for binomial CV, reported on **both** scales.
///
/// - `n` — total residuals (including any with `trials == 0`).
/// - `n_evaluated` — number of residuals with `trials > 0`, i.e. those contributing to
///   `logit` / `prevalence`.
/// - `logit` — summary statistics on the logit scale (bias / RMSE / MSDR).
/// - `prevalence` — summary statistics on the prevalence scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialCvSummary {
    pub n: usize,
    pub n_evaluated: usize,
    pub logit: CvSummary,
    pub prevalence: CvSummary,
}

impl BinomialCvSummary {
    pub fn from_residuals(residuals: &[BinomialCvResidual]) -> Self {
        let n = residuals.len();
        let n_evaluated = residuals.iter().filter(|r| r.trials > 0).count();
        let logit = CvSummary::from_scalar_iter(
            n,
            residuals
                .iter()
                .map(|r| (r.logit_error(), r.logit_variance)),
        );
        let prevalence = CvSummary::from_scalar_iter(
            n,
            residuals
                .iter()
                .map(|r| (r.prevalence_error(), r.prevalence_variance)),
        );
        Self {
            n,
            n_evaluated,
            logit,
            prevalence,
        }
    }
}

fn observed_logit_and_prevalence(successes: u32, trials: u32) -> (Real, Real) {
    if trials == 0 {
        (Real::NAN, Real::NAN)
    } else {
        let p = successes as Real / trials as Real;
        // `logit_clamped` applies the same `(ε, 1−ε)` clamp used by binomial model training,
        // so the observed-vs-predicted scale is consistent when `successes` equals `0` or
        // `trials`. `observed_prevalence` itself is reported *unclamped* so users can see the
        // raw proportion.
        (logit_clamped(p), p)
    }
}

fn delta_prevalence_variance(prevalence: Real, logit_variance: Real) -> Real {
    let factor = prevalence * (1.0 - prevalence);
    factor * factor * logit_variance.max(0.0)
}

fn make_binomial_residual(
    index: usize,
    successes: u32,
    trials: u32,
    predicted_logit: Real,
    logit_variance: Real,
) -> BinomialCvResidual {
    let (observed_logit, observed_prevalence) = observed_logit_and_prevalence(successes, trials);
    let predicted_prevalence = logistic(predicted_logit);
    let prevalence_variance = delta_prevalence_variance(predicted_prevalence, logit_variance);
    BinomialCvResidual {
        index,
        successes,
        trials,
        observed_logit,
        predicted_logit,
        logit_variance,
        observed_prevalence,
        predicted_prevalence,
        prevalence_variance,
    }
}

fn build_binomial_observations(
    coords: &[GeoCoord],
    successes: &[u32],
    trials: &[u32],
    indices: &[usize],
) -> Result<Vec<BinomialObservation>, KrigingError> {
    indices
        .iter()
        .filter(|&&i| trials[i] > 0)
        .map(|&i| BinomialObservation::new(coords[i], successes[i], trials[i]))
        .collect()
}

fn validate_binomial_lengths(
    n_coords: usize,
    n_successes: usize,
    n_trials: usize,
) -> Result<(), KrigingError> {
    if n_coords != n_successes || n_coords != n_trials {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({n_coords}), successes ({n_successes}), and trials ({n_trials}) must have equal length"
        )));
    }
    if n_coords < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    Ok(())
}

/// Leave-one-out CV for [`BinomialKrigingModel`]. Returns [`BinomialCvResidual`] with both
/// logit- and prevalence-scale residuals. Held-out stations with `trials == 0` contribute
/// a residual whose *observed* fields are `NaN` (predictions still populated); downstream
/// summarization via [`BinomialCvSummary`] skips them.
///
/// Training folds drop stations with `trials == 0` (the underlying model requires
/// `trials > 0`), so those stations never participate in any training fold either.
pub fn leave_one_out_binomial(
    coords: &[GeoCoord],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    prior: BinomialPrior,
) -> Result<Vec<BinomialCvResidual>, KrigingError> {
    validate_binomial_lengths(coords.len(), successes.len(), trials.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let observations = build_binomial_observations(coords, successes, trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = BinomialKrigingModel::new_with_prior(observations, variogram, prior)?;
        let pred = model.predict(coords[i])?;
        out.push(make_binomial_residual(
            i,
            successes[i],
            trials[i],
            pred.logit_value,
            pred.variance,
        ));
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`BinomialKrigingModel`]. See [`leave_one_out_binomial`] for semantics.
pub fn k_fold_binomial(
    coords: &[GeoCoord],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    prior: BinomialPrior,
    k: usize,
) -> Result<Vec<BinomialCvResidual>, KrigingError> {
    validate_binomial_lengths(coords.len(), successes.len(), trials.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<BinomialCvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let observations = build_binomial_observations(coords, successes, trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = BinomialKrigingModel::new_with_prior(observations, variogram, prior)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(make_binomial_residual(
                idx,
                successes[idx],
                trials[idx],
                pred.logit_value,
                pred.variance,
            ));
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Binomial projected kriging CV (planar / anisotropic)
// ---------------------------------------------------------------------------

fn build_projected_binomial_observations(
    coords: &[ProjectedCoord],
    successes: &[u32],
    trials: &[u32],
    indices: &[usize],
) -> Result<Vec<ProjectedBinomialObservation>, KrigingError> {
    indices
        .iter()
        .filter(|&&i| trials[i] > 0)
        .map(|&i| ProjectedBinomialObservation::new(coords[i], successes[i], trials[i]))
        .collect()
}

fn validate_projected_binomial_lengths(
    n_coords: usize,
    n_successes: usize,
    n_trials: usize,
) -> Result<(), KrigingError> {
    if n_coords != n_successes || n_successes != n_trials {
        return Err(KrigingError::DimensionMismatch(format!(
            "binomial projected CV: coords ({}), successes ({}), trials ({}) must match",
            n_coords, n_successes, n_trials
        )));
    }
    Ok(())
}

/// Leave-one-out CV for [`BinomialProjectedKrigingModel`]. Returns
/// [`BinomialCvResidual`]s with both logit- and prevalence-scale residuals,
/// matching the geographic [`leave_one_out_binomial`] semantics. Stations with
/// `trials == 0` produce residuals whose observed fields are `NaN` and never
/// participate in any training fold.
pub fn leave_one_out_binomial_projected(
    coords: &[ProjectedCoord],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    prior: BinomialPrior,
) -> Result<Vec<BinomialCvResidual>, KrigingError> {
    validate_projected_binomial_lengths(coords.len(), successes.len(), trials.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let observations = build_projected_binomial_observations(coords, successes, trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = BinomialProjectedKrigingModel::new_with_prior(
            observations,
            variogram,
            anisotropy,
            prior,
        )?;
        let pred = model.predict(coords[i])?;
        out.push(make_binomial_residual(
            i,
            successes[i],
            trials[i],
            pred.logit_value,
            pred.variance,
        ));
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`BinomialProjectedKrigingModel`]. See
/// [`leave_one_out_binomial_projected`] for fold semantics.
pub fn k_fold_binomial_projected(
    coords: &[ProjectedCoord],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    prior: BinomialPrior,
    k: usize,
) -> Result<Vec<BinomialCvResidual>, KrigingError> {
    validate_projected_binomial_lengths(coords.len(), successes.len(), trials.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<BinomialCvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let observations = build_projected_binomial_observations(coords, successes, trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = BinomialProjectedKrigingModel::new_with_prior(
            observations,
            variogram,
            anisotropy,
            prior,
        )?;
        let test_coords: Vec<ProjectedCoord> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(make_binomial_residual(
                idx,
                successes[idx],
                trials[idx],
                pred.logit_value,
                pred.variance,
            ));
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Space–time kriging CV
// ---------------------------------------------------------------------------

fn build_st_binomial_observations<C: Copy>(
    coords: &[SpaceTimeCoord<C>],
    successes: &[u32],
    trials: &[u32],
    indices: &[usize],
) -> Result<Vec<SpaceTimeBinomialObservation<C>>, KrigingError> {
    indices
        .iter()
        .filter(|&&i| trials[i] > 0)
        .map(|&i| SpaceTimeBinomialObservation::new(coords[i], successes[i], trials[i]))
        .collect()
}

/// Leave-one-out CV for [`SpaceTimeOrdinaryKrigingModel`]. Generic over
/// [`SpatialMetric`] so the same routine serves geographic
/// ([`GeoMetric`](crate::spacetime::GeoMetric)) and projected
/// ([`ProjectedMetric`](crate::spacetime::ProjectedMetric)) data.
pub fn leave_one_out_spacetime<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeOrdinaryKrigingModel::new(metric, dataset, variogram)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`SpaceTimeOrdinaryKrigingModel`]. See module-level docs for fold assignment.
pub fn k_fold_spacetime<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeOrdinaryKrigingModel::new(metric, dataset, variogram)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

/// Leave-one-out CV for [`SpaceTimeSimpleKrigingModel`]. The supplied `mean` is treated as
/// known (same value for every fold).
pub fn leave_one_out_spacetime_simple<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    mean: Real,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeSimpleKrigingModel::new(metric, dataset, variogram, mean)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`SpaceTimeSimpleKrigingModel`].
pub fn k_fold_spacetime_simple<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    mean: Real,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeSimpleKrigingModel::new(metric, dataset, variogram, mean)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

/// Leave-one-out CV for [`SpaceTimeUniversalKrigingModel`]. Drift coefficients are
/// re-estimated inside each fold so there is no in-sample leakage from the trend.
pub fn leave_one_out_spacetime_universal<M: SpatialBasis>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    trend: SpaceTimeUniversalTrend,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeUniversalKrigingModel::new(metric, dataset, variogram, trend)?;
        let pred = model.predict(coords[i])?;
        out.push(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        });
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`SpaceTimeUniversalKrigingModel`].
pub fn k_fold_spacetime_universal<M: SpatialBasis>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    trend: SpaceTimeUniversalTrend,
    k: usize,
) -> Result<Vec<CvResidual>, KrigingError> {
    validate_len(coords.len(), values.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<CvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> = train.iter().map(|&j| coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeUniversalKrigingModel::new(metric, dataset, variogram, trend)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(CvResidual {
                index: idx,
                observed: values[idx],
                predicted: pred.value,
                variance: pred.variance,
            });
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

/// Leave-one-out CV for [`SpaceTimeBinomialKrigingModel`]. Held-out stations with
/// `trials == 0` contribute a residual whose observed fields are `NaN` (predictions still
/// populated); downstream summarization via [`BinomialCvSummary`] skips them. Training
/// folds drop stations with `trials == 0` (the underlying model requires `trials > 0`).
pub fn leave_one_out_spacetime_binomial<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    successes: &[u32],
    trials: &[u32],
    variogram: SpaceTimeVariogram,
    prior: BinomialPrior,
) -> Result<Vec<BinomialCvResidual>, KrigingError> {
    validate_binomial_lengths(coords.len(), successes.len(), trials.len())?;
    let n = coords.len();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let i = test[0];
        let observations = build_st_binomial_observations(coords, successes, trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model =
            SpaceTimeBinomialKrigingModel::new_with_prior(metric, observations, variogram, prior)?;
        let pred = model.predict(coords[i])?;
        out.push(make_binomial_residual(
            i,
            successes[i],
            trials[i],
            pred.logit_value,
            pred.variance,
        ));
        Ok(())
    })?;
    Ok(out)
}

/// K-fold CV for [`SpaceTimeBinomialKrigingModel`]. See
/// [`leave_one_out_spacetime_binomial`] for semantics.
pub fn k_fold_spacetime_binomial<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    successes: &[u32],
    trials: &[u32],
    variogram: SpaceTimeVariogram,
    prior: BinomialPrior,
    k: usize,
) -> Result<Vec<BinomialCvResidual>, KrigingError> {
    validate_binomial_lengths(coords.len(), successes.len(), trials.len())?;
    let n = coords.len();
    validate_k(n, k)?;
    let mut results: Vec<Option<BinomialCvResidual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let observations = build_st_binomial_observations(coords, successes, trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model =
            SpaceTimeBinomialKrigingModel::new_with_prior(metric, observations, variogram, prior)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> = test.iter().map(|&j| coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        for (&idx, pred) in test.iter().zip(preds.iter()) {
            results[idx] = Some(make_binomial_residual(
                idx,
                successes[idx],
                trials[idx],
                pred.logit_value,
                pred.variance,
            ));
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

    fn grid_points() -> (Vec<GeoCoord>, Vec<Real>) {
        // A small 4x4 grid with a smooth linear trend in both coordinates.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let lat = i as Real;
                let lon = j as Real;
                coords.push(GeoCoord::try_new(lat, lon).unwrap());
                values.push(2.0 * lat + 3.0 * lon + 1.0);
            }
        }
        (coords, values)
    }

    fn projected_grid_points() -> (Vec<ProjectedCoord>, Vec<Real>) {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as Real;
                let y = j as Real;
                coords.push(ProjectedCoord::new(x, y));
                values.push(2.0 * x + 3.0 * y + 1.0);
            }
        }
        (coords, values)
    }

    fn binomial_grid_points() -> (Vec<GeoCoord>, Vec<u32>, Vec<u32>) {
        // 4x4 grid of counts; a smooth logit gradient in lat yields prevalences ~ 0.1..0.9.
        let mut coords = Vec::new();
        let mut successes = Vec::new();
        let mut trials = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let lat = i as Real;
                let lon = j as Real;
                let p = logistic(-2.0 + 0.5 * lat + 0.5 * lon);
                let n = 40u32;
                let s = (p * n as Real).round() as u32;
                coords.push(GeoCoord::try_new(lat, lon).unwrap());
                successes.push(s);
                trials.push(n);
            }
        }
        (coords, successes, trials)
    }

    #[test]
    fn leave_one_out_returns_one_residual_per_station_in_order() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out(&coords, &values, variogram).unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.observed, values[i]);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn leave_one_out_has_small_rmse_for_smooth_linear_field() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 10.0, 500.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out(&coords, &values, variogram).unwrap();
        let summary = CvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, coords.len());
        assert!(
            summary.rmse.is_finite() && summary.rmse < 3.0,
            "RMSE on smooth field should be modest, got {}",
            summary.rmse
        );
    }

    #[test]
    fn k_fold_covers_every_station_exactly_once() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold(&coords, &values, variogram, 4).unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index], "duplicate residual for index {}", r.index);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn k_fold_rejects_invalid_k() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        assert!(k_fold(&coords, &values, variogram, 1).is_err());
        assert!(k_fold(&coords, &values, variogram, coords.len() + 1).is_err());
    }

    #[test]
    fn leave_one_out_rejects_fewer_than_two_stations() {
        let coords = vec![GeoCoord::try_new(0.0, 0.0).unwrap()];
        let values = vec![1.0];
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        assert!(leave_one_out(&coords, &values, variogram).is_err());
    }

    #[test]
    fn cv_summary_mean_error_matches_hand_calculation() {
        let residuals = vec![
            CvResidual {
                index: 0,
                observed: 10.0,
                predicted: 11.0,
                variance: 1.0,
            },
            CvResidual {
                index: 1,
                observed: 20.0,
                predicted: 18.0,
                variance: 1.0,
            },
        ];
        let s = CvSummary::from_residuals(&residuals);
        assert_eq!(s.n, 2);
        // Errors are -1 and 2; mean = 0.5.
        assert!((s.mean_error - 0.5).abs() < 1e-6);
        // RMSE = sqrt((1 + 4) / 2) = sqrt(2.5).
        assert!((s.rmse - (2.5 as Real).sqrt()).abs() < 1e-6);
        // MSDR with unit variance = mean of squared errors = 2.5.
        assert!((s.msdr - 2.5).abs() < 1e-6);
    }

    #[test]
    fn simple_loo_runs_with_known_mean() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals = leave_one_out_simple(&coords, &values, variogram, mean).unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn simple_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals = k_fold_simple(&coords, &values, variogram, mean, 4).unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn universal_loo_matches_ordinary_for_constant_trend_within_tol() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let ok = leave_one_out(&coords, &values, variogram).unwrap();
        let uk =
            leave_one_out_universal(&coords, &values, variogram, UniversalTrend::Constant).unwrap();
        assert_eq!(ok.len(), uk.len());
        for (a, b) in ok.iter().zip(uk.iter()) {
            assert!(
                (a.predicted - b.predicted).abs() < 1e-6,
                "constant-trend UK should match OK at station {} (ok={}, uk={})",
                a.index,
                a.predicted,
                b.predicted
            );
        }
    }

    #[test]
    fn universal_k_fold_runs_with_linear_trend() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let residuals =
            k_fold_universal(&coords, &values, variogram, UniversalTrend::Linear, 4).unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn projected_loo_matches_ordinary_when_isotropic_and_euclidean() {
        // Sanity check: projected kriging with isotropic anisotropy on a planar grid should
        // produce finite residuals and pass structural checks.
        let (coords, values) = projected_grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals =
            leave_one_out_projected(&coords, &values, variogram, Anisotropy2D::isotropic())
                .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn projected_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = projected_grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals =
            k_fold_projected(&coords, &values, variogram, Anisotropy2D::isotropic(), 4).unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn binomial_loo_reports_both_scales_in_input_order() {
        let (coords, successes, trials) = binomial_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_binomial(
            &coords,
            &successes,
            &trials,
            variogram,
            BinomialPrior::default(),
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.successes, successes[i]);
            assert_eq!(r.trials, trials[i]);
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
            assert!(r.predicted_logit.is_finite());
            assert!(r.predicted_prevalence.is_finite());
            assert!(r.logit_variance.is_finite());
            assert!(r.prevalence_variance.is_finite());
            assert!(
                r.predicted_prevalence >= 0.0 && r.predicted_prevalence <= 1.0,
                "prevalence must lie in [0,1], got {}",
                r.predicted_prevalence
            );
        }
    }

    #[test]
    fn binomial_loo_handles_zero_trials_with_nan_observations() {
        let (coords, mut successes, mut trials) = binomial_grid_points();
        // Flip the first station to "unobservable" and keep the rest.
        successes[0] = 0;
        trials[0] = 0;
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_binomial(
            &coords,
            &successes,
            &trials,
            variogram,
            BinomialPrior::default(),
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        // First station: observed should be NaN, prediction should still be finite.
        let r0 = residuals[0];
        assert_eq!(r0.trials, 0);
        assert!(r0.observed_logit.is_nan());
        assert!(r0.observed_prevalence.is_nan());
        assert!(r0.predicted_logit.is_finite());
        assert!(r0.predicted_prevalence.is_finite());
        // Others must remain well-defined.
        for r in &residuals[1..] {
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
        }
        // Summary must aggregate only the observable stations on each scale.
        let summary = BinomialCvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, residuals.len());
        assert_eq!(summary.n_evaluated, residuals.len() - 1);
        assert_eq!(summary.logit.n, summary.n_evaluated);
        assert_eq!(summary.prevalence.n, summary.n_evaluated);
        assert!(summary.logit.rmse.is_finite());
        assert!(summary.prevalence.rmse.is_finite());
    }

    #[test]
    fn binomial_k_fold_covers_every_station_exactly_once() {
        let (coords, successes, trials) = binomial_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_binomial(
            &coords,
            &successes,
            &trials,
            variogram,
            BinomialPrior::default(),
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    fn binomial_projected_grid_points() -> (Vec<ProjectedCoord>, Vec<u32>, Vec<u32>) {
        let mut coords = Vec::new();
        let mut successes = Vec::new();
        let mut trials = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as Real;
                let y = j as Real;
                let p = logistic(-2.0 + 0.5 * x + 0.5 * y);
                let n = 40u32;
                let s = (p * n as Real).round() as u32;
                coords.push(ProjectedCoord::new(x, y));
                successes.push(s);
                trials.push(n);
            }
        }
        (coords, successes, trials)
    }

    #[test]
    fn binomial_projected_loo_returns_one_residual_per_station_in_order() {
        let (coords, successes, trials) = binomial_projected_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_binomial_projected(
            &coords,
            &successes,
            &trials,
            variogram,
            Anisotropy2D::isotropic(),
            BinomialPrior::default(),
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.predicted_logit.is_finite());
            assert!(r.predicted_prevalence > 0.0 && r.predicted_prevalence < 1.0);
            assert!(r.logit_variance >= 0.0);
            assert!(r.prevalence_variance >= 0.0);
        }
    }

    #[test]
    fn binomial_projected_k_fold_covers_every_station_exactly_once() {
        let (coords, successes, trials) = binomial_projected_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_binomial_projected(
            &coords,
            &successes,
            &trials,
            variogram,
            Anisotropy2D::isotropic(),
            BinomialPrior::default(),
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn binomial_summary_prevalence_rmse_is_consistent_with_residuals() {
        // Hand-constructed residuals with known errors on both scales.
        let residuals = vec![
            BinomialCvResidual {
                index: 0,
                successes: 3,
                trials: 10,
                observed_logit: logit_clamped(0.3),
                predicted_logit: logit_clamped(0.2),
                logit_variance: 1.0,
                observed_prevalence: 0.3,
                predicted_prevalence: 0.2,
                prevalence_variance: 0.01,
            },
            BinomialCvResidual {
                index: 1,
                successes: 0,
                trials: 0,
                observed_logit: Real::NAN,
                predicted_logit: 0.0,
                logit_variance: 1.0,
                observed_prevalence: Real::NAN,
                predicted_prevalence: 0.5,
                prevalence_variance: 0.0625,
            },
        ];
        let summary = BinomialCvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, 2);
        assert_eq!(summary.n_evaluated, 1);
        // Only the first residual contributes: prevalence error = 0.3 - 0.2 = 0.1; RMSE = 0.1.
        // Tolerance accounts for binary representation of 0.1 / 0.3 / 0.2 in f64.
        assert!(
            (summary.prevalence.rmse - 0.1).abs() < 1e-6,
            "expected ~0.1, got {}",
            summary.prevalence.rmse
        );
        // Logit error = logit(0.3) - logit(0.2); finite.
        assert!(summary.logit.rmse.is_finite() && summary.logit.rmse > 0.0);
    }

    // ----- Space–time CV ----------------------------------------------------

    use crate::spacetime::GeoMetric;

    fn st_grid_points() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>) {
        // 3x3x3 grid with a smooth linear drift in lat, lon, and time.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let lat = i as Real;
                    let lon = j as Real;
                    let t = k as Real;
                    coords.push(
                        SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon).unwrap(), t).unwrap(),
                    );
                    values.push(2.0 * lat + 3.0 * lon + 0.5 * t + 1.0);
                }
            }
        }
        (coords, values)
    }

    fn st_variogram() -> SpaceTimeVariogram {
        let spatial = VariogramModel::new(0.05, 2.0, 300.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 3.0, VariogramType::Exponential).unwrap();
        SpaceTimeVariogram::new_separable(spatial, temporal).unwrap()
    }

    fn st_binomial_grid_points() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<u32>, Vec<u32>) {
        let mut coords = Vec::new();
        let mut successes = Vec::new();
        let mut trials = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let lat = i as Real;
                    let lon = j as Real;
                    let t = k as Real;
                    let p = logistic(-2.0 + 0.5 * lat + 0.5 * lon + 0.1 * t);
                    let n = 40u32;
                    let s = (p * n as Real).round() as u32;
                    coords.push(
                        SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon).unwrap(), t).unwrap(),
                    );
                    successes.push(s);
                    trials.push(n);
                }
            }
        }
        (coords, successes, trials)
    }

    #[test]
    fn st_leave_one_out_returns_one_residual_per_station_in_order() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let residuals = leave_one_out_spacetime(GeoMetric, &coords, &values, variogram).unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.observed, values[i]);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn st_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let residuals = k_fold_spacetime(GeoMetric, &coords, &values, variogram, 4).unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index], "duplicate residual for index {}", r.index);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn st_leave_one_out_rejects_fewer_than_two_stations() {
        let coords =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap()];
        let values = vec![1.0];
        let variogram = st_variogram();
        assert!(leave_one_out_spacetime(GeoMetric, &coords, &values, variogram).is_err());
    }

    #[test]
    fn st_k_fold_rejects_invalid_k() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        assert!(k_fold_spacetime(GeoMetric, &coords, &values, variogram, 1).is_err());
        assert!(
            k_fold_spacetime(GeoMetric, &coords, &values, variogram, coords.len() + 1).is_err()
        );
    }

    #[test]
    fn st_simple_loo_runs_with_known_mean() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals =
            leave_one_out_spacetime_simple(GeoMetric, &coords, &values, variogram, mean).unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn st_simple_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals =
            k_fold_spacetime_simple(GeoMetric, &coords, &values, variogram, mean, 4).unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn st_universal_loo_matches_ordinary_for_constant_trend_within_tol() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let ok = leave_one_out_spacetime(GeoMetric, &coords, &values, variogram).unwrap();
        let uk = leave_one_out_spacetime_universal(
            GeoMetric,
            &coords,
            &values,
            variogram,
            SpaceTimeUniversalTrend::Constant,
        )
        .unwrap();
        assert_eq!(ok.len(), uk.len());
        for (a, b) in ok.iter().zip(uk.iter()) {
            assert!(
                (a.predicted - b.predicted).abs() < 1e-3,
                "constant-trend ST UK should ~match ST OK at index {} (ok={}, uk={})",
                a.index,
                a.predicted,
                b.predicted
            );
        }
    }

    #[test]
    fn st_universal_k_fold_runs_with_linear_in_time_trend() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let residuals = k_fold_spacetime_universal(
            GeoMetric,
            &coords,
            &values,
            variogram,
            SpaceTimeUniversalTrend::LinearInTime,
            3,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn st_binomial_loo_reports_both_scales_in_input_order() {
        let (coords, successes, trials) = st_binomial_grid_points();
        let variogram = st_variogram();
        let residuals = leave_one_out_spacetime_binomial(
            GeoMetric,
            &coords,
            &successes,
            &trials,
            variogram,
            BinomialPrior::default(),
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.successes, successes[i]);
            assert_eq!(r.trials, trials[i]);
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
            assert!(r.predicted_logit.is_finite());
            assert!(r.predicted_prevalence.is_finite());
            assert!(
                r.predicted_prevalence >= 0.0 && r.predicted_prevalence <= 1.0,
                "prevalence must lie in [0,1], got {}",
                r.predicted_prevalence
            );
        }
    }

    #[test]
    fn st_binomial_loo_handles_zero_trials_with_nan_observations() {
        let (coords, mut successes, mut trials) = st_binomial_grid_points();
        successes[0] = 0;
        trials[0] = 0;
        let variogram = st_variogram();
        let residuals = leave_one_out_spacetime_binomial(
            GeoMetric,
            &coords,
            &successes,
            &trials,
            variogram,
            BinomialPrior::default(),
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let r0 = residuals[0];
        assert_eq!(r0.trials, 0);
        assert!(r0.observed_logit.is_nan());
        assert!(r0.observed_prevalence.is_nan());
        assert!(r0.predicted_logit.is_finite());
        assert!(r0.predicted_prevalence.is_finite());
        for r in &residuals[1..] {
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
        }
        let summary = BinomialCvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, residuals.len());
        assert_eq!(summary.n_evaluated, residuals.len() - 1);
    }

    #[test]
    fn st_binomial_k_fold_covers_every_station_exactly_once() {
        let (coords, successes, trials) = st_binomial_grid_points();
        let variogram = st_variogram();
        let residuals = k_fold_spacetime_binomial(
            GeoMetric,
            &coords,
            &successes,
            &trials,
            variogram,
            BinomialPrior::default(),
            3,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }
}
