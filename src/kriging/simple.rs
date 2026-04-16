//! Simple kriging: interpolation with a known, constant mean.
//!
//! Unlike ordinary kriging (which treats the mean as unknown and adds a Lagrangian
//! constraint that weights sum to one), simple kriging assumes the global mean `m` is known.
//! The predictor is
//!
//! ```text
//!   Z*(x0) = m + Σ_i w_i [Z(x_i) − m]
//! ```
//!
//! where the weights solve the plain covariance system `C · w = c0` (no border row/col).
//! The kriging variance is `σ²_K(x0) = C(0) − wᵀ c0`.
//!
//! Use simple kriging when you have an independently estimated mean (e.g. from a calibration
//! dataset) and want slightly lower variance than ordinary kriging buys you.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::distance::{GeoCoord, PreparedGeoCoord, haversine_distance_prepared, prepare_geo_coord};
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::ordinary::{Prediction, kriging_diagonal_jitter};
use crate::variogram::models::VariogramModel;

/// Fitted simple kriging model.
#[derive(Debug)]
pub struct SimpleKrigingModel {
    coords: Vec<GeoCoord>,
    prepared_coords: Vec<PreparedGeoCoord>,
    residuals: Vec<Real>,
    mean: Real,
    variogram: VariogramModel,
    cov_at_zero: Real,
    system: DMatrix<Real>,
    /// Shared LU factorization; `Clone` just bumps the `Arc`.
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
}

impl Clone for SimpleKrigingModel {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.clone(),
            prepared_coords: self.prepared_coords.clone(),
            residuals: self.residuals.clone(),
            mean: self.mean,
            variogram: self.variogram,
            cov_at_zero: self.cov_at_zero,
            system: self.system.clone(),
            system_lu: Arc::clone(&self.system_lu),
        }
    }
}

impl SimpleKrigingModel {
    /// Build a simple kriging model using a known `mean`.
    pub fn new(
        dataset: GeoDataset,
        variogram: VariogramModel,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let prepared_coords = coords
            .iter()
            .copied()
            .map(prepare_geo_coord)
            .collect::<Vec<_>>();
        let residuals: Vec<Real> = values.iter().map(|v| *v - mean).collect();

        let system = build_simple_system(&prepared_coords, variogram);
        let system_lu = Arc::new(system.clone().lu());
        // Probe solvability up front.
        let probe = DVector::from_element(coords.len(), 1.0);
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize simple kriging system".to_string(),
            ));
        }
        Ok(Self {
            coords,
            prepared_coords,
            residuals,
            mean,
            variogram,
            cov_at_zero: variogram.covariance(0.0),
            system,
            system_lu,
        })
    }

    /// The known mean used by the model.
    pub fn mean(&self) -> Real {
        self.mean
    }

    /// Predict at a single target.
    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.coords.len(), 0.0);
        self.predict_with_rhs(coord, &mut rhs)
    }

    /// Batch predictions; parallel on native builds.
    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            coords
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(n, 0.0),
                    |rhs, c| self.predict_with_rhs(*c, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(self.coords.len(), 0.0);
            let mut out = Vec::with_capacity(coords.len());
            for &c in coords {
                out.push(self.predict_with_rhs(c, &mut rhs)?);
            }
            Ok(out)
        }
    }

    fn predict_with_rhs(
        &self,
        coord: GeoCoord,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        let n = self.coords.len();
        let prepared = prepare_geo_coord(coord);
        for i in 0..n {
            rhs[i] = self.variogram.covariance(haversine_distance_prepared(
                self.prepared_coords[i],
                prepared,
            ));
        }
        let w = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve simple kriging system".to_string())
        })?;
        let mut residual_pred: Real = 0.0;
        let mut cov_dot: Real = 0.0;
        for i in 0..n {
            residual_pred += w[i] * self.residuals[i];
            cov_dot += w[i] * rhs[i];
        }
        let variance = (self.cov_at_zero - cov_dot).max(0.0);
        Ok(Prediction {
            value: self.mean + residual_pred,
            variance,
        })
    }
}

fn build_simple_system(coords: &[PreparedGeoCoord], variogram: VariogramModel) -> DMatrix<Real> {
    let n = coords.len();
    let diag_eps = kriging_diagonal_jitter(n, variogram);
    let mut m = DMatrix::from_element(n, n, 0.0);
    for i in 0..n {
        for j in i..n {
            let mut cov = variogram.covariance(haversine_distance_prepared(coords[i], coords[j]));
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
    use crate::variogram::models::VariogramType;

    #[test]
    fn recovers_training_value_at_collocated_point() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values).unwrap();
        let model = SimpleKrigingModel::new(dataset, variogram, 15.0).expect("model");
        let pred = model.predict(coords[0]).expect("prediction");
        assert!((pred.value - 10.0).abs() < 1e-3);
        assert!(pred.variance >= 0.0);
    }

    #[test]
    fn reverts_to_mean_far_from_any_station() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 0.1).unwrap(),
            GeoCoord::try_new(0.1, 0.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0];
        let mean = 20.0;
        let variogram = VariogramModel::new(0.01, 1.0, 5.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let model = SimpleKrigingModel::new(dataset, variogram, mean).expect("model");
        // Target far from all stations (many range units away) has near-zero
        // covariance with the data, so weights ~ 0 and the prediction reverts to the mean.
        let pred = model
            .predict(GeoCoord::try_new(50.0, 50.0).unwrap())
            .expect("prediction");
        assert!((pred.value - mean).abs() < 1e-3, "got {}", pred.value);
    }

    #[test]
    fn batch_matches_single_predictions() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords, values.clone()).unwrap();
        let model = SimpleKrigingModel::new(dataset, variogram, 13.0).expect("model");
        let queries = vec![
            GeoCoord::try_new(0.2, 0.3).unwrap(),
            GeoCoord::try_new(0.7, 0.4).unwrap(),
        ];
        let batch = model.predict_batch(&queries).expect("batch");
        for (i, q) in queries.iter().enumerate() {
            let single = model.predict(*q).expect("single");
            assert!((batch[i].value - single.value).abs() < 1e-5);
            assert!((batch[i].variance - single.variance).abs() < 1e-5);
        }
    }
}
