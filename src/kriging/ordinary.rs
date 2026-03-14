use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::distance::{GeoCoord, PreparedGeoCoord, haversine_distance_prepared, prepare_geo_coord};
use crate::error::KrigingError;
use crate::variogram::models::VariogramModel;

#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    pub value: Real,
    pub variance: Real,
}

#[derive(Debug)]
pub struct OrdinaryKrigingModel {
    coords: Vec<GeoCoord>,
    prepared_coords: Vec<PreparedGeoCoord>,
    values: Vec<Real>,
    variogram: VariogramModel,
    cov_at_zero: Real,
    system: DMatrix<Real>,
    system_lu: LU<Real, Dyn, Dyn>,
}

impl Clone for OrdinaryKrigingModel {
    fn clone(&self) -> Self {
        let system = self.system.clone();
        let system_lu = system.clone().lu();
        Self {
            coords: self.coords.clone(),
            prepared_coords: self.prepared_coords.clone(),
            values: self.values.clone(),
            variogram: self.variogram,
            cov_at_zero: self.cov_at_zero,
            system,
            system_lu,
        }
    }
}

impl OrdinaryKrigingModel {
    pub fn new(
        coords: Vec<GeoCoord>,
        values: Vec<Real>,
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        if coords.len() != values.len() {
            return Err(KrigingError::DimensionMismatch(
                "coords and values length must match".to_string(),
            ));
        }
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        for coord in &coords {
            coord.validate()?;
        }
        let prepared_coords = coords
            .iter()
            .copied()
            .map(prepare_geo_coord)
            .collect::<Vec<_>>();

        let system = build_ordinary_system(&prepared_coords, variogram);
        let system_lu = system.clone().lu();
        // Validate solvability up front so prediction failures are not deferred.
        let mut probe_rhs = DVector::from_element(coords.len() + 1, 0.0);
        probe_rhs[coords.len()] = 1.0;
        if system_lu.solve(&probe_rhs).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize ordinary kriging system".to_string(),
            ));
        }
        Ok(Self {
            coords,
            prepared_coords,
            values,
            variogram,
            cov_at_zero: variogram.covariance(0.0),
            system,
            system_lu,
        })
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
        self.predict_with_rhs(coord, &mut rhs)
    }

    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            return coords
                .par_iter()
                .map(|coord| {
                    let mut rhs = DVector::from_element(n + 1, 0.0);
                    self.predict_with_rhs(*coord, &mut rhs)
                })
                .collect();
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
            let mut out = Vec::with_capacity(coords.len());
            for &coord in coords {
                out.push(self.predict_with_rhs(coord, &mut rhs)?);
            }
            Ok(out)
        }
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        for coord in coords {
            coord.validate()?;
        }
        match crate::gpu::build_rhs_covariances_gpu(&self.coords, coords, self.variogram).await {
            Ok(covariances) => self.predict_batch_with_covariances(coords, &covariances),
            Err(_) => self.predict_batch(coords),
        }
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        for coord in coords {
            coord.validate()?;
        }
        match crate::gpu::build_rhs_covariances_gpu_blocking(&self.coords, coords, self.variogram) {
            Ok(covariances) => self.predict_batch_with_covariances(coords, &covariances),
            Err(_) => self.predict_batch(coords),
        }
    }

    fn predict_with_rhs(
        &self,
        coord: GeoCoord,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        coord.validate()?;
        let n = self.coords.len();
        let prepared_coord = prepare_geo_coord(coord);
        for i in 0..n {
            rhs[i] = self.variogram.covariance(haversine_distance_prepared(
                self.prepared_coords[i],
                prepared_coord,
            ));
        }
        rhs[n] = 1.0;

        let sol = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve ordinary kriging system".to_string())
        })?;
        let mut value = 0.0;
        let mut cov_dot = 0.0;
        for i in 0..n {
            value += sol[i] * self.values[i];
            cov_dot += sol[i] * rhs[i];
        }
        let mu = sol[n];
        let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
        Ok(Prediction { value, variance })
    }

    #[cfg(feature = "gpu")]
    fn predict_batch_with_covariances(
        &self,
        coords: &[GeoCoord],
        covariances: &[Real],
    ) -> Result<Vec<Prediction>, KrigingError> {
        let n = self.coords.len();
        let expected = n.checked_mul(coords.len()).ok_or_else(|| {
            KrigingError::MatrixError("covariance dimensions overflowed".to_string())
        })?;
        if covariances.len() != expected {
            return Err(KrigingError::MatrixError(format!(
                "expected {} covariance entries, got {}",
                expected,
                covariances.len()
            )));
        }
        let mut rhs = DVector::from_element(n + 1, 0.0);
        let mut out = Vec::with_capacity(coords.len());
        for pred_idx in 0..coords.len() {
            for i in 0..n {
                rhs[i] = covariances[pred_idx * n + i];
            }
            rhs[n] = 1.0;
            let sol = self.system_lu.solve(&rhs).ok_or_else(|| {
                KrigingError::MatrixError("could not solve ordinary kriging system".to_string())
            })?;
            let mut value = 0.0;
            let mut cov_dot = 0.0;
            for i in 0..n {
                value += sol[i] * self.values[i];
                cov_dot += sol[i] * rhs[i];
            }
            let mu = sol[n];
            let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
            out.push(Prediction { value, variance });
        }
        Ok(out)
    }
}

fn build_ordinary_system(coords: &[PreparedGeoCoord], variogram: VariogramModel) -> DMatrix<Real> {
    let n = coords.len();
    let mut m = DMatrix::from_element(n + 1, n + 1, 0.0);
    for i in 0..n {
        for j in i..n {
            let mut cov = variogram.covariance(haversine_distance_prepared(coords[i], coords[j]));
            if i == j {
                cov += 1e-10;
            }
            m[(i, j)] = cov;
            m[(j, i)] = cov;
        }
        m[(i, n)] = 1.0;
        m[(n, i)] = 1.0;
    }
    m[(n, n)] = 0.0;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_close_to_training_value_for_collocated_point() {
        let coords = vec![
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord { lat: 0.0, lon: 1.0 },
            GeoCoord { lat: 1.0, lon: 0.0 },
        ];
        let values = vec![10.0, 20.0, 15.0];
        let variogram = VariogramModel::Exponential {
            nugget: 0.01,
            sill: 5.0,
            range: 300.0,
        };
        let model = OrdinaryKrigingModel::new(coords.clone(), values, variogram).expect("model");
        let pred = model.predict(coords[0]).expect("prediction");
        assert!((pred.value - 10.0).abs() < 1e-6);
        assert!(pred.variance >= 0.0);
    }

    #[test]
    fn batch_predictions_match_repeated_single_predictions() {
        let coords = vec![
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord { lat: 0.0, lon: 1.0 },
            GeoCoord { lat: 1.0, lon: 0.0 },
            GeoCoord { lat: 1.0, lon: 1.0 },
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::Gaussian {
            nugget: 0.01,
            sill: 10.0,
            range: 400.0,
        };
        let model = OrdinaryKrigingModel::new(coords, values, variogram).expect("model");
        let query_coords = vec![
            GeoCoord { lat: 0.2, lon: 0.3 },
            GeoCoord { lat: 0.7, lon: 0.4 },
            GeoCoord { lat: 0.5, lon: 0.8 },
        ];
        let batch = model.predict_batch(&query_coords).expect("batch");
        let singles = query_coords
            .iter()
            .map(|coord| model.predict(*coord).expect("single"))
            .collect::<Vec<_>>();
        assert_eq!(batch.len(), singles.len());
        for (b, s) in batch.iter().zip(singles.iter()) {
            assert!((b.value - s.value).abs() < 1e-4);
            assert!((b.variance - s.variance).abs() < 1e-4);
        }
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    #[test]
    fn gpu_batch_predictions_match_cpu_batch_predictions() {
        let coords = vec![
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord { lat: 0.0, lon: 1.0 },
            GeoCoord { lat: 1.0, lon: 0.0 },
            GeoCoord { lat: 1.0, lon: 1.0 },
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::Gaussian {
            nugget: 0.01,
            sill: 10.0,
            range: 400.0,
        };
        let model = OrdinaryKrigingModel::new(coords, values, variogram).expect("model");
        let query_coords = vec![
            GeoCoord { lat: 0.2, lon: 0.3 },
            GeoCoord { lat: 0.7, lon: 0.4 },
            GeoCoord { lat: 0.5, lon: 0.8 },
        ];
        let cpu = model.predict_batch(&query_coords).expect("cpu batch");
        let gpu = model
            .predict_batch_gpu_blocking(&query_coords)
            .expect("gpu batch");
        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert!((g.value - c.value).abs() < 1e-3);
            assert!((g.variance - c.variance).abs() < 1e-3);
        }
    }
}
