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

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::ordinary::Prediction;
use crate::kriging::simple_engine::SimpleKrigingEngine;
use crate::spacetime::metric::GeoMetric;
use crate::variogram::models::VariogramModel;

/// Fitted simple kriging model.
#[derive(Debug, Clone)]
pub struct SimpleKrigingModel {
    engine: SimpleKrigingEngine<GeoMetric>,
}

impl SimpleKrigingModel {
    /// Build a simple kriging model using a known `mean`.
    pub fn new(
        dataset: GeoDataset,
        variogram: VariogramModel,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let engine = SimpleKrigingEngine::fit(GeoMetric, coords, values, variogram, mean)?;
        Ok(Self { engine })
    }

    /// The known mean used by the model.
    pub fn mean(&self) -> Real {
        self.engine.mean()
    }

    /// Predict at a single target.
    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        self.engine
            .predict(&[coord])
            .map(|mut v| v.pop().expect("single prediction"))
    }

    /// Batch predictions; parallel on native builds.
    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        self.engine.predict(coords)
    }
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
