//! Simple space–time kriging.
//!
//! Same role as [`SimpleKrigingModel`](crate::SimpleKrigingModel): interpolation with a
//! known, constant mean. The weights solve `C · w = c0` (no Lagrangian row), and the
//! predictor is `m + Σ_i w_i (z_i − m)`.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::ordinary::Prediction;
use crate::spacetime::coord::{SpaceTimeCoord, temporal_distance};
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::kriging::ordinary::spacetime_diagonal_jitter;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;

/// Fitted simple space–time kriging model.
#[derive(Debug)]
pub struct SpaceTimeSimpleKrigingModel<M: SpatialMetric> {
    metric: M,
    prepared_spatial: Vec<M::Prepared>,
    times: Vec<Real>,
    residuals: Vec<Real>,
    mean: Real,
    variogram: SpaceTimeVariogram,
    c_at_zero: Real,
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
}

impl<M: SpatialMetric> Clone for SpaceTimeSimpleKrigingModel<M>
where
    M::Prepared: Clone,
{
    fn clone(&self) -> Self {
        Self {
            metric: self.metric,
            prepared_spatial: self.prepared_spatial.clone(),
            times: self.times.clone(),
            residuals: self.residuals.clone(),
            mean: self.mean,
            variogram: self.variogram,
            c_at_zero: self.c_at_zero,
            system_lu: Arc::clone(&self.system_lu),
        }
    }
}

impl<M: SpatialMetric> SpaceTimeSimpleKrigingModel<M> {
    /// Build a simple ST kriging model with a known constant mean.
    pub fn new(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        if !mean.is_finite() {
            return Err(KrigingError::InvalidInput(
                "mean must be finite".to_string(),
            ));
        }
        let (coords, values) = dataset.into_parts();
        let prepared_spatial: Vec<M::Prepared> =
            coords.iter().map(|c| metric.prepare(c.spatial)).collect();
        let times: Vec<Real> = coords.iter().map(|c| c.time).collect();
        let residuals: Vec<Real> = values.iter().map(|v| *v - mean).collect();

        let system = build_system(&metric, &prepared_spatial, &times, variogram);
        let system_lu = Arc::new(system.lu());
        let probe = DVector::from_element(coords.len(), 1.0);
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize space-time simple kriging system".to_string(),
            ));
        }
        Ok(Self {
            metric,
            prepared_spatial,
            times,
            residuals,
            mean,
            variogram,
            c_at_zero: variogram.c_at_zero(),
            system_lu,
        })
    }

    /// Known mean used by the predictor.
    pub fn mean(&self) -> Real {
        self.mean
    }

    pub fn predict(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.times.len(), 0.0);
        self.predict_with_rhs(target, &mut rhs)
    }

    pub fn predict_batch(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.times.len();
            targets
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(n, 0.0),
                    |rhs, t| self.predict_with_rhs(*t, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(self.times.len(), 0.0);
            let mut out = Vec::with_capacity(targets.len());
            for &t in targets {
                out.push(self.predict_with_rhs(t, &mut rhs)?);
            }
            Ok(out)
        }
    }

    fn predict_with_rhs(
        &self,
        target: SpaceTimeCoord<M::Coord>,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        let n = self.times.len();
        let prepared_target = self.metric.prepare(target.spatial);
        for i in 0..n {
            let hs = self
                .metric
                .distance(self.prepared_spatial[i], prepared_target);
            let ht = temporal_distance(self.times[i], target.time);
            rhs[i] = self.variogram.covariance(hs, ht);
        }
        let w = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError(
                "could not solve space-time simple kriging system".to_string(),
            )
        })?;
        let mut residual_pred: Real = 0.0;
        let mut cov_dot: Real = 0.0;
        for i in 0..n {
            residual_pred += w[i] * self.residuals[i];
            cov_dot += w[i] * rhs[i];
        }
        let variance = (self.c_at_zero - cov_dot).max(0.0);
        Ok(Prediction {
            value: self.mean + residual_pred,
            variance,
        })
    }
}

fn build_system<M: SpatialMetric>(
    metric: &M,
    prepared: &[M::Prepared],
    times: &[Real],
    variogram: SpaceTimeVariogram,
) -> DMatrix<Real> {
    let n = prepared.len();
    let diag_eps = spacetime_diagonal_jitter(n, variogram);
    let mut m = DMatrix::from_element(n, n, 0.0);
    for i in 0..n {
        for j in i..n {
            let hs = metric.distance(prepared[i], prepared[j]);
            let ht = temporal_distance(times[i], times[j]);
            let mut cov = variogram.covariance(hs, ht);
            if i == j {
                cov += diag_eps;
            }
            m[(i, j)] = cov;
            m[(j, i)] = cov;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn spatial() -> VariogramModel {
        VariogramModel::new(0.01, 1.0, 300.0, VariogramType::Exponential).unwrap()
    }
    fn temporal() -> VariogramModel {
        VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap()
    }

    fn coords() -> Vec<SpaceTimeCoord<GeoCoord>> {
        vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 1.0).unwrap(), 3.0),
        ]
    }

    #[test]
    fn recovers_training_value_at_collocated_point() {
        let cs = coords();
        let values = vec![10.0, 20.0, 15.0, 25.0];
        let stv = SpaceTimeVariogram::new_separable(spatial(), temporal()).unwrap();
        let ds = SpaceTimeDataset::new(cs.clone(), values.clone()).unwrap();
        let model = SpaceTimeSimpleKrigingModel::new(GeoMetric, ds, stv, 17.5).unwrap();
        for (i, c) in cs.iter().enumerate() {
            let pred = model.predict(*c).unwrap();
            assert!(
                (pred.value - values[i]).abs() < 1e-2,
                "at i={i}: got {}, want {}",
                pred.value,
                values[i]
            );
            assert!(pred.variance >= 0.0);
        }
    }

    #[test]
    fn reverts_to_mean_far_from_any_training_point() {
        let cs = coords();
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let mean = 50.0;
        // Short ranges so the target is effectively uncorrelated with any observation.
        let spatial_short =
            VariogramModel::new(0.01, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let temporal_short =
            VariogramModel::new(0.01, 1.0, 0.5, VariogramType::Exponential).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_short, temporal_short).unwrap();
        let ds = SpaceTimeDataset::new(cs, values).unwrap();
        let model = SpaceTimeSimpleKrigingModel::new(GeoMetric, ds, stv, mean).unwrap();
        let pred = model
            .predict(SpaceTimeCoord::new(
                GeoCoord::try_new(50.0, 50.0).unwrap(),
                500.0,
            ))
            .unwrap();
        assert!(
            (pred.value - mean).abs() < 1e-2,
            "far-from-data prediction should revert to the mean, got {}",
            pred.value
        );
    }

    #[test]
    fn rejects_non_finite_mean() {
        let ds = SpaceTimeDataset::new(coords(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial(), temporal()).unwrap();
        assert!(SpaceTimeSimpleKrigingModel::new(GeoMetric, ds, stv, Real::NAN).is_err());
    }

    #[test]
    fn batch_matches_single_predictions() {
        let cs = coords();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let stv = SpaceTimeVariogram::new_separable(spatial(), temporal()).unwrap();
        let ds = SpaceTimeDataset::new(cs, values).unwrap();
        let model = SpaceTimeSimpleKrigingModel::new(GeoMetric, ds, stv, 2.5).unwrap();
        let targets = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.3, 0.7).unwrap(), 1.5),
            SpaceTimeCoord::new(GeoCoord::try_new(0.9, 0.1).unwrap(), 2.5),
        ];
        let batch = model.predict_batch(&targets).unwrap();
        for (i, t) in targets.iter().enumerate() {
            let single = model.predict(*t).unwrap();
            assert!((batch[i].value - single.value).abs() < 1e-5);
        }
    }
}
