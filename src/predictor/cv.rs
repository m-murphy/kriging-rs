//! Generic cross-validation harness for kriging predictors.
//!
//! Model-specific fold logic lives in [`KrigingPredictor`] implementors; [`leave_one_out_cv`] and
//! [`k_fold_cv`] iterate folds deterministically (round-robin for k-fold) and aggregate residuals.

use crate::Real;
use crate::cv::{BinomialCvResidual, CvResidual};
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::binomial::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, HeteroskedasticBinomialConfig,
    delta_prevalence_variance,
};
use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::kriging::pairwise::{SpaceTimePairwiseCovariance, SpatialPairwiseCovariance};
use crate::kriging::simple::SimpleKrigingModel;
use crate::kriging::simple_engine::SimpleKrigingEngine;
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
use crate::spacetime::kriging::engine::SpaceTimeOrdinaryKrigingEngine;
use crate::spacetime::kriging::ordinary::SpaceTimeOrdinaryKrigingModel;
use crate::spacetime::kriging::simple::SpaceTimeSimpleKrigingModel;
use crate::spacetime::kriging::universal::{
    SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend,
};
use crate::spacetime::metric::{GeoMetric, ProjectedMetric};
use crate::spacetime::metric::{SpatialBasis, SpatialMetric};
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::utils::{logistic, logit_clamped};
use crate::variogram::models::VariogramModel;

/// A kriging model that can be refit on a training fold and evaluated on held-out indices.
pub trait KrigingPredictor {
    type Residual: Clone;

    fn n(&self) -> usize;

    fn validate(&self) -> Result<(), KrigingError>;

    /// Fit on `train` and predict stations in `test`. Returns one residual per test index, in
    /// `test` order.
    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<Self::Residual>, KrigingError>;

    /// Leave-one-out CV in input order. Engines override with O(n³) Cholesky-downdate paths.
    fn leave_one_out(&self) -> Result<Vec<Self::Residual>, KrigingError> {
        leave_one_out_via_folds(self)
    }
}

/// Leave-one-out cross-validation: for each station `i`, fit on the complement and predict `i`.
/// Residuals are returned in input order (`0..n`).
pub fn leave_one_out_cv<P: KrigingPredictor + ?Sized>(
    p: &P,
) -> Result<Vec<P::Residual>, KrigingError> {
    p.validate()?;
    p.leave_one_out()
}

fn leave_one_out_via_folds<P: KrigingPredictor + ?Sized>(
    p: &P,
) -> Result<Vec<P::Residual>, KrigingError> {
    let n = p.n();
    let mut out = Vec::with_capacity(n);
    for_each_loo_fold(n, |train, test| {
        let fold_residuals = p.predict_fold(train, test)?;
        out.extend(fold_residuals);
        Ok(())
    })?;
    Ok(out)
}

/// K-fold cross-validation with deterministic round-robin assignment (station `i` → fold `i % k`).
pub fn k_fold_cv<P: KrigingPredictor>(p: &P, k: usize) -> Result<Vec<P::Residual>, KrigingError> {
    p.validate()?;
    let n = p.n();
    validate_k(n, k)?;
    let mut results: Vec<Option<P::Residual>> = vec![None; n];
    for_each_k_fold(n, k, |train, test| {
        let fold_residuals = p.predict_fold(train, test)?;
        for (idx, residual) in test.iter().zip(fold_residuals) {
            results[*idx] = Some(residual);
        }
        Ok(())
    })?;
    Ok(results.into_iter().flatten().collect())
}

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
// Binomial CV helpers (mirrors `crate::cv` private helpers)
// ---------------------------------------------------------------------------

fn observed_logit_and_prevalence(successes: u32, trials: u32) -> (Real, Real) {
    if trials == 0 {
        (Real::NAN, Real::NAN)
    } else {
        let p = successes as Real / trials as Real;
        (logit_clamped(p), p)
    }
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

// ---------------------------------------------------------------------------
// Continuous geo / projected backends
// ---------------------------------------------------------------------------

/// Ordinary kriging CV backend (geographic coordinates).
pub struct OrdinaryGeoPredictor<'a> {
    pub coords: &'a [GeoCoord],
    pub values: &'a [Real],
    pub variogram: VariogramModel,
}

impl KrigingPredictor for OrdinaryGeoPredictor<'_> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn leave_one_out(&self) -> Result<Vec<CvResidual>, KrigingError> {
        self.validate()?;
        let engine = OrdinaryKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, self.variogram),
            self.coords.to_vec(),
            self.values.to_vec(),
        )?;
        let preds = engine.leave_one_out_predictions()?;
        Ok(preds
            .into_iter()
            .enumerate()
            .map(|(i, pred)| CvResidual {
                index: i,
                observed: self.values[i],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = OrdinaryKrigingModel::new(dataset, self.variogram)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

/// Simple kriging CV backend (geographic coordinates, known mean).
pub struct SimpleGeoPredictor<'a> {
    pub coords: &'a [GeoCoord],
    pub values: &'a [Real],
    pub variogram: VariogramModel,
    pub mean: Real,
}

impl KrigingPredictor for SimpleGeoPredictor<'_> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn leave_one_out(&self) -> Result<Vec<CvResidual>, KrigingError> {
        self.validate()?;
        let engine = SimpleKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, self.variogram),
            self.coords.to_vec(),
            self.values.to_vec(),
            self.mean,
        )?;
        let preds = engine.leave_one_out_predictions()?;
        Ok(preds
            .into_iter()
            .enumerate()
            .map(|(i, pred)| CvResidual {
                index: i,
                observed: self.values[i],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = SimpleKrigingModel::new(dataset, self.variogram, self.mean)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

/// Universal kriging CV backend (geographic coordinates, polynomial drift).
pub struct UniversalGeoPredictor<'a> {
    pub coords: &'a [GeoCoord],
    pub values: &'a [Real],
    pub variogram: VariogramModel,
    pub trend: UniversalTrend,
}

impl KrigingPredictor for UniversalGeoPredictor<'_> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn leave_one_out(&self) -> Result<Vec<CvResidual>, KrigingError> {
        if self.trend == UniversalTrend::Constant {
            return OrdinaryGeoPredictor {
                coords: self.coords,
                values: self.values,
                variogram: self.variogram,
            }
            .leave_one_out();
        }
        leave_one_out_via_folds(self)
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<GeoCoord> = train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = GeoDataset::new(fold_coords, fold_values)?;
        let model = UniversalKrigingModel::new(dataset, self.variogram, self.trend)?;
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

/// Projected ordinary kriging CV backend (planar coordinates, optional anisotropy).
pub struct ProjectedOrdinaryPredictor<'a> {
    pub coords: &'a [ProjectedCoord],
    pub values: &'a [Real],
    pub variogram: VariogramModel,
    pub anisotropy: Anisotropy2D,
}

impl KrigingPredictor for ProjectedOrdinaryPredictor<'_> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn leave_one_out(&self) -> Result<Vec<CvResidual>, KrigingError> {
        self.validate()?;
        let engine = OrdinaryKrigingEngine::fit(
            SpatialPairwiseCovariance::new(
                ProjectedMetric::with_anisotropy(self.anisotropy),
                self.variogram,
            ),
            self.coords.to_vec(),
            self.values.to_vec(),
        )?;
        let preds = engine.leave_one_out_predictions()?;
        Ok(preds
            .into_iter()
            .enumerate()
            .map(|(i, pred)| CvResidual {
                index: i,
                observed: self.values[i],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<ProjectedCoord> = train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = ProjectedDataset::new(fold_coords, fold_values)?;
        let model = ProjectedKrigingModel::new(dataset, self.variogram, self.anisotropy)?;
        let test_coords: Vec<ProjectedCoord> = test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Binomial geo / projected backends
// ---------------------------------------------------------------------------

/// Binomial kriging CV backend (geographic coordinates).
pub struct BinomialGeoPredictor<'a> {
    pub coords: &'a [GeoCoord],
    pub successes: &'a [u32],
    pub trials: &'a [u32],
    pub variogram: VariogramModel,
    pub prior: BinomialPrior,
}

impl KrigingPredictor for BinomialGeoPredictor<'_> {
    type Residual = BinomialCvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_binomial_lengths(self.coords.len(), self.successes.len(), self.trials.len())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<BinomialCvResidual>, KrigingError> {
        let observations =
            build_binomial_observations(self.coords, self.successes, self.trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = BinomialKrigingModel::new_with_prior(observations, self.variogram, self.prior)?
            .into_model();
        let test_coords: Vec<GeoCoord> = test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| {
                make_binomial_residual(
                    idx,
                    self.successes[idx],
                    self.trials[idx],
                    pred.logit,
                    pred.logit_variance,
                )
            })
            .collect())
    }
}

/// Binomial projected kriging CV backend (planar coordinates, anisotropy).
pub struct BinomialProjectedPredictor<'a> {
    pub coords: &'a [ProjectedCoord],
    pub successes: &'a [u32],
    pub trials: &'a [u32],
    pub variogram: VariogramModel,
    pub anisotropy: Anisotropy2D,
    pub prior: BinomialPrior,
}

impl KrigingPredictor for BinomialProjectedPredictor<'_> {
    type Residual = BinomialCvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_projected_binomial_lengths(
            self.coords.len(),
            self.successes.len(),
            self.trials.len(),
        )
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<BinomialCvResidual>, KrigingError> {
        let observations =
            build_projected_binomial_observations(self.coords, self.successes, self.trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = BinomialProjectedKrigingModel::new_with_prior(
            observations,
            self.variogram,
            self.anisotropy,
            self.prior,
            HeteroskedasticBinomialConfig::default(),
        )?
        .into_model();
        let test_coords: Vec<ProjectedCoord> = test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| {
                make_binomial_residual(
                    idx,
                    self.successes[idx],
                    self.trials[idx],
                    pred.logit,
                    pred.logit_variance,
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Spacetime backends
// ---------------------------------------------------------------------------

/// Space–time ordinary kriging CV backend.
pub struct SpacetimeOrdinaryPredictor<'a, M: SpatialMetric> {
    pub metric: M,
    pub coords: &'a [SpaceTimeCoord<M::Coord>],
    pub values: &'a [Real],
    pub variogram: SpaceTimeVariogram,
}

impl<M: SpatialMetric> KrigingPredictor for SpacetimeOrdinaryPredictor<'_, M> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn leave_one_out(&self) -> Result<Vec<CvResidual>, KrigingError> {
        self.validate()?;
        let engine = SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
            SpaceTimePairwiseCovariance::new(self.metric, self.variogram),
            self.coords.to_vec(),
            self.values.to_vec(),
            &[],
        )?;
        let preds = engine.leave_one_out_predictions()?;
        Ok(preds
            .into_iter()
            .enumerate()
            .map(|(i, pred)| CvResidual {
                index: i,
                observed: self.values[i],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> =
            train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model = SpaceTimeOrdinaryKrigingModel::new(self.metric, dataset, self.variogram)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> =
            test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

/// Space–time simple kriging CV backend (known mean).
pub struct SpacetimeSimplePredictor<'a, M: SpatialMetric> {
    pub metric: M,
    pub coords: &'a [SpaceTimeCoord<M::Coord>],
    pub values: &'a [Real],
    pub variogram: SpaceTimeVariogram,
    pub mean: Real,
}

impl<M: SpatialMetric> KrigingPredictor for SpacetimeSimplePredictor<'_, M> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> =
            train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model =
            SpaceTimeSimpleKrigingModel::new(self.metric, dataset, self.variogram, self.mean)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> =
            test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

/// Space–time universal kriging CV backend (polynomial drift).
pub struct SpacetimeUniversalPredictor<'a, M: SpatialBasis> {
    pub metric: M,
    pub coords: &'a [SpaceTimeCoord<M::Coord>],
    pub values: &'a [Real],
    pub variogram: SpaceTimeVariogram,
    pub trend: SpaceTimeUniversalTrend,
}

impl<M: SpatialBasis> KrigingPredictor for SpacetimeUniversalPredictor<'_, M> {
    type Residual = CvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_len(self.coords.len(), self.values.len())
    }

    fn leave_one_out(&self) -> Result<Vec<CvResidual>, KrigingError> {
        if self.trend == SpaceTimeUniversalTrend::Constant {
            return SpacetimeOrdinaryPredictor {
                metric: self.metric,
                coords: self.coords,
                values: self.values,
                variogram: self.variogram,
            }
            .leave_one_out();
        }
        leave_one_out_via_folds(self)
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<CvResidual>, KrigingError> {
        let fold_coords: Vec<SpaceTimeCoord<M::Coord>> =
            train.iter().map(|&j| self.coords[j]).collect();
        let fold_values: Vec<Real> = train.iter().map(|&j| self.values[j]).collect();
        let dataset = SpaceTimeDataset::new(fold_coords, fold_values)?;
        let model =
            SpaceTimeUniversalKrigingModel::new(self.metric, dataset, self.variogram, self.trend)?;
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> =
            test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| CvResidual {
                index: idx,
                observed: self.values[idx],
                predicted: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

/// Space–time binomial kriging CV backend.
pub struct SpacetimeBinomialPredictor<'a, M: SpatialMetric> {
    pub metric: M,
    pub coords: &'a [SpaceTimeCoord<M::Coord>],
    pub successes: &'a [u32],
    pub trials: &'a [u32],
    pub variogram: SpaceTimeVariogram,
    pub prior: BinomialPrior,
}

impl<M: SpatialMetric> KrigingPredictor for SpacetimeBinomialPredictor<'_, M> {
    type Residual = BinomialCvResidual;

    fn n(&self) -> usize {
        self.coords.len()
    }

    fn validate(&self) -> Result<(), KrigingError> {
        validate_binomial_lengths(self.coords.len(), self.successes.len(), self.trials.len())
    }

    fn predict_fold(
        &self,
        train: &[usize],
        test: &[usize],
    ) -> Result<Vec<BinomialCvResidual>, KrigingError> {
        let observations =
            build_st_binomial_observations(self.coords, self.successes, self.trials, train)?;
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let model = SpaceTimeBinomialKrigingModel::new_with_prior(
            self.metric,
            observations,
            self.variogram,
            self.prior,
            HeteroskedasticBinomialConfig::default(),
        )?
        .into_model();
        let test_coords: Vec<SpaceTimeCoord<M::Coord>> =
            test.iter().map(|&j| self.coords[j]).collect();
        let preds = model.predict_batch(&test_coords)?;
        Ok(test
            .iter()
            .zip(preds.iter())
            .map(|(&idx, pred)| {
                make_binomial_residual(
                    idx,
                    self.successes[idx],
                    self.trials[idx],
                    pred.logit,
                    pred.logit_variance,
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

    #[test]
    fn ordinary_geo_fast_loo_matches_fold_loo() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let p = OrdinaryGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
        };
        let fast = p.leave_one_out().unwrap();
        let slow = leave_one_out_via_folds(&p).unwrap();
        assert_eq!(fast.len(), slow.len());
        for (a, b) in fast.iter().zip(slow.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.predicted - b.predicted).abs() < 1e-4);
            assert!((a.variance - b.variance).abs() < 1e-4);
        }
    }

    #[test]
    fn ordinary_geo_loo_returns_finite_residuals() {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                coords.push(GeoCoord::try_new(i as Real, j as Real).unwrap());
                values.push(2.0 * i as Real + 3.0 * j as Real + 1.0);
            }
        }
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&OrdinaryGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }
}
