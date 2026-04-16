//! Binomial space–time kriging.
//!
//! Same trick as [`BinomialKrigingModel`](crate::BinomialKrigingModel): smooth each
//! observation's `(successes, trials)` into a logit, ordinary-krige the logit field, and
//! map predictions back to a prevalence in `(0, 1)` via the logistic function.

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::binomial::{BinomialPrediction, BinomialPrior};
use crate::spacetime::SpaceTimeCoord;
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::kriging::ordinary::SpaceTimeOrdinaryKrigingModel;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::utils::{Probability, logistic, logit};

/// A binomial observation at a space–time location: number of successes and trials.
#[derive(Debug, Clone, Copy)]
pub struct SpaceTimeBinomialObservation<C> {
    coord: SpaceTimeCoord<C>,
    successes: u32,
    trials: u32,
}

impl<C: Copy> SpaceTimeBinomialObservation<C> {
    pub fn new(
        coord: SpaceTimeCoord<C>,
        successes: u32,
        trials: u32,
    ) -> Result<Self, KrigingError> {
        if trials == 0 {
            return Err(KrigingError::InvalidBinomialData(
                "trials must be greater than 0".to_string(),
            ));
        }
        if successes > trials {
            return Err(KrigingError::InvalidBinomialData(format!(
                "successes ({successes}) cannot exceed trials ({trials})"
            )));
        }
        Ok(Self {
            coord,
            successes,
            trials,
        })
    }

    #[inline]
    pub fn coord(self) -> SpaceTimeCoord<C> {
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

    pub fn smoothed_probability_with_prior(&self, prior: BinomialPrior) -> Real {
        let s = self.successes as Real;
        let n = self.trials as Real;
        (s + prior.alpha()) / (n + prior.alpha() + prior.beta())
    }

    pub fn smoothed_logit_with_prior(&self, prior: BinomialPrior) -> Real {
        let p = self.smoothed_probability_with_prior(prior);
        logit(Probability::from_known_in_range(p))
    }
}

/// Fitted binomial space–time kriging model.
#[derive(Debug)]
pub struct SpaceTimeBinomialKrigingModel<M: SpatialMetric> {
    inner: SpaceTimeOrdinaryKrigingModel<M>,
}

impl<M: SpatialMetric> Clone for SpaceTimeBinomialKrigingModel<M>
where
    M::Coord: Clone,
    M::Prepared: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<M: SpatialMetric> SpaceTimeBinomialKrigingModel<M> {
    /// Build a model with the default Beta(0.5, 0.5) prior.
    pub fn new(
        metric: M,
        observations: Vec<SpaceTimeBinomialObservation<M::Coord>>,
        variogram: SpaceTimeVariogram,
    ) -> Result<Self, KrigingError> {
        Self::new_with_prior(metric, observations, variogram, BinomialPrior::default())
    }

    /// Build a model with an explicit Beta prior.
    pub fn new_with_prior(
        metric: M,
        observations: Vec<SpaceTimeBinomialObservation<M::Coord>>,
        variogram: SpaceTimeVariogram,
        prior: BinomialPrior,
    ) -> Result<Self, KrigingError> {
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let coords: Vec<SpaceTimeCoord<M::Coord>> =
            observations.iter().map(|o| o.coord()).collect();
        let logits: Vec<Real> = observations
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect();
        Self::from_precomputed_logits(metric, coords, logits, variogram)
    }

    /// Build a model from pre-computed logits (e.g. user-applied smoothing).
    pub fn from_precomputed_logits(
        metric: M,
        coords: Vec<SpaceTimeCoord<M::Coord>>,
        logits: Vec<Real>,
        variogram: SpaceTimeVariogram,
    ) -> Result<Self, KrigingError> {
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "logits must all be finite (no NaN/inf)".to_string(),
            ));
        }
        let dataset = SpaceTimeDataset::new(coords, logits)?;
        let inner = SpaceTimeOrdinaryKrigingModel::new(metric, dataset, variogram)?;
        Ok(Self { inner })
    }

    pub fn predict(
        &self,
        target: SpaceTimeCoord<M::Coord>,
    ) -> Result<BinomialPrediction, KrigingError> {
        let pred = self.inner.predict(target)?;
        Ok(to_binomial(pred))
    }

    pub fn predict_batch(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let inner = self.inner.predict_batch(targets)?;
        Ok(inner.into_iter().map(to_binomial).collect())
    }
}

fn to_binomial(p: crate::kriging::ordinary::Prediction) -> BinomialPrediction {
    let prevalence = logistic(p.value);
    let factor = prevalence * (1.0 - prevalence);
    BinomialPrediction {
        prevalence,
        logit_value: p.value,
        variance: p.variance,
        prevalence_variance: factor * factor * p.variance.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn variogram() -> SpaceTimeVariogram {
        SpaceTimeVariogram::new_separable(
            VariogramModel::new(0.05, 1.0, 200.0, VariogramType::Exponential).unwrap(),
            VariogramModel::new(0.05, 1.0, 5.0, VariogramType::Exponential).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn handles_zero_and_all_successes_with_smoothing() {
        let o1 = SpaceTimeBinomialObservation::new(
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            0,
            10,
        )
        .unwrap();
        let o2 = SpaceTimeBinomialObservation::new(
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            10,
            10,
        )
        .unwrap();
        let p1 = o1.smoothed_probability_with_prior(BinomialPrior::default());
        let p2 = o2.smoothed_probability_with_prior(BinomialPrior::default());
        assert!(p1 > 0.0 && p1 < 1.0);
        assert!(p2 > 0.0 && p2 < 1.0);
    }

    #[test]
    fn predictions_are_in_unit_interval() {
        let obs = vec![
            SpaceTimeBinomialObservation::new(
                SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
                3,
                10,
            )
            .unwrap(),
            SpaceTimeBinomialObservation::new(
                SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
                7,
                10,
            )
            .unwrap(),
            SpaceTimeBinomialObservation::new(
                SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
                5,
                10,
            )
            .unwrap(),
        ];
        let model = SpaceTimeBinomialKrigingModel::new(GeoMetric, obs, variogram()).unwrap();
        let pred = model
            .predict(SpaceTimeCoord::new(
                GeoCoord::try_new(0.5, 0.5).unwrap(),
                1.0,
            ))
            .unwrap();
        assert!(pred.prevalence > 0.0 && pred.prevalence < 1.0);
        assert!(pred.prevalence_variance >= 0.0);
        assert!(pred.variance >= 0.0);
    }

    #[test]
    fn rejects_non_finite_precomputed_logits() {
        let coords = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
        ];
        let logits = vec![0.0, Real::NAN];
        let res = SpaceTimeBinomialKrigingModel::from_precomputed_logits(
            GeoMetric,
            coords,
            logits,
            variogram(),
        );
        assert!(matches!(res, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn rejects_zero_trials() {
        let res = SpaceTimeBinomialObservation::new(
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            0,
            0,
        );
        assert!(matches!(res, Err(KrigingError::InvalidBinomialData(_))));
    }

    #[test]
    fn matches_ordinary_for_precomputed_logits() {
        let coords = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 1.0).unwrap(), 3.0),
        ];
        let logits = vec![-1.0, 0.0, 1.0, 0.5];
        let v = variogram();
        let bin = SpaceTimeBinomialKrigingModel::from_precomputed_logits(
            GeoMetric,
            coords.clone(),
            logits.clone(),
            v,
        )
        .unwrap();
        let ord = SpaceTimeOrdinaryKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords, logits).unwrap(),
            v,
        )
        .unwrap();
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let bp = bin.predict(target).unwrap();
        let op = ord.predict(target).unwrap();
        assert!((bp.logit_value - op.value).abs() < 1e-6);
        assert!((bp.variance - op.variance).abs() < 1e-6);
        assert!((bp.prevalence - logistic(op.value)).abs() < 1e-6);
    }
}
