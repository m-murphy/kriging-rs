//! Simple space–time kriging.
//!
//! Same role as [`SimpleKrigingModel`](crate::SimpleKrigingModel): interpolation with a
//! known, constant mean. The weights solve `C · w = c0` (no Lagrangian row), and the
//! predictor is `m + Σ_i w_i (z_i − m)`.

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::conditioner::KrigingConditioner;
use crate::kriging::ordinary::Prediction;
use crate::kriging::pairwise::SpaceTimePairwiseCovariance;
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::kriging::simple_engine::SpaceTimeSimpleKrigingEngine;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;

/// Fitted simple space–time kriging model.
#[derive(Debug, Clone)]
pub struct SpaceTimeSimpleKrigingModel<M: SpatialMetric> {
    engine: SpaceTimeSimpleKrigingEngine<M>,
}

impl<M: SpatialMetric> SpaceTimeSimpleKrigingModel<M> {
    /// Build a simple ST kriging model with a known constant mean.
    pub fn new(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let engine = SpaceTimeSimpleKrigingEngine::fit(
            SpaceTimePairwiseCovariance::new(metric, variogram),
            coords,
            values,
            mean,
        )?;
        Ok(Self { engine })
    }

    /// Known mean used by the predictor.
    pub fn mean(&self) -> Real {
        self.engine.mean()
    }

    /// Space–time variogram used by the model.
    pub fn variogram(&self) -> SpaceTimeVariogram {
        self.engine.pairwise_covariance().variogram()
    }

    /// Consume this fitted model as live state for sequential Gaussian simulation.
    pub fn into_conditioner(
        self,
    ) -> Result<KrigingConditioner<SpaceTimeCoord<M::Coord>>, KrigingError>
    where
        M: 'static,
        M::Coord: 'static,
        M::Prepared: 'static,
    {
        Ok(KrigingConditioner::from_simple(self.engine))
    }

    pub fn predict(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        self.engine
            .predict(&[target])
            .map(|mut v| v.pop().expect("single prediction"))
    }

    pub fn predict_batch(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<Prediction>, KrigingError> {
        self.engine.predict(targets)
    }
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
