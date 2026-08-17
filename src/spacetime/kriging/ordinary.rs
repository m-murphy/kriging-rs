//! Ordinary space–time kriging.
//!
//! Build a model from a [`SpaceTimeDataset`] and a [`SpaceTimeVariogram`] and predict at
//! arbitrary [`SpaceTimeCoord`] targets. Uses the dual SPD engine (ADR-0001).

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::ordinary::Prediction;
use crate::kriging::pairwise::SpaceTimePairwiseCovariance;
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::kriging::engine::SpaceTimeOrdinaryKrigingEngine;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;

/// Fitted ordinary space–time kriging model.
///
/// Generic over a [`SpatialMetric`] so the same implementation serves geographic data
/// ([`GeoMetric`](crate::spacetime::GeoMetric)) and projected data
/// ([`ProjectedMetric`](crate::spacetime::ProjectedMetric)).
#[derive(Debug, Clone)]
pub struct SpaceTimeOrdinaryKrigingModel<M: SpatialMetric> {
    engine: SpaceTimeOrdinaryKrigingEngine<M>,
}

impl<M: SpatialMetric> SpaceTimeOrdinaryKrigingModel<M> {
    /// Build a model. The caller must supply a compatible [`SpatialMetric`] — in particular
    /// the variogram's spatial range must be expressed in the same units the metric returns.
    pub fn new(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
    ) -> Result<Self, KrigingError> {
        Self::new_with_extra_diagonal_internal(metric, dataset, variogram, &[])
    }

    /// Per-station observation noise on the covariance main diagonal.
    pub fn new_with_extra_diagonal(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
        extra: Vec<Real>,
    ) -> Result<Self, KrigingError> {
        let n = dataset.len();
        if !extra.is_empty() && extra.len() != n {
            return Err(KrigingError::InvalidInput(
                "extra observation diagonal must be empty (homoscedastic) or the same length as the dataset"
                    .to_string(),
            ));
        }
        for &v in &extra {
            if !v.is_finite() || v < 0.0 {
                return Err(KrigingError::InvalidInput(
                    "observation diagonal entries must be finite and non-negative".to_string(),
                ));
            }
        }
        Self::new_with_extra_diagonal_internal(metric, dataset, variogram, &extra)
    }

    fn new_with_extra_diagonal_internal(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
        extra: &[Real],
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let engine = SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
            SpaceTimePairwiseCovariance::new(metric, variogram),
            coords,
            values,
            extra,
        )?;
        Ok(Self { engine })
    }

    /// Metric used to measure spatial distances.
    pub fn metric(&self) -> M {
        self.engine.pairwise_covariance().metric()
    }

    /// Number of training points.
    pub fn len(&self) -> usize {
        self.engine.len()
    }

    /// Whether the model has no training points. Always `false` because construction
    /// enforces `len() >= 2`.
    pub fn is_empty(&self) -> bool {
        self.engine.len() == 0
    }

    /// Space–time variogram used by the model.
    pub fn variogram(&self) -> SpaceTimeVariogram {
        self.engine.pairwise_covariance().variogram()
    }

    /// Training space–time coordinates in station order.
    pub fn coords(&self) -> Vec<SpaceTimeCoord<M::Coord>>
    where
        M::Coord: Copy,
    {
        self.engine.coords().to_vec()
    }

    /// Single-target prediction.
    pub fn predict(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        self.engine
            .predict(&[target])
            .map(|mut v| v.pop().expect("single prediction"))
    }

    /// Batched predictions. Parallel on native builds; sequential on wasm32.
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

    fn spatial_var() -> VariogramModel {
        VariogramModel::new(0.01, 1.0, 300.0, VariogramType::Exponential).unwrap()
    }

    fn temporal_var() -> VariogramModel {
        VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap()
    }

    fn make_coords() -> Vec<SpaceTimeCoord<GeoCoord>> {
        vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 1.0).unwrap(), 3.0),
        ]
    }

    #[test]
    fn predicts_close_to_training_value_at_collocated_target() {
        let coords = make_coords();
        let values = vec![10.0, 20.0, 15.0, 25.0];
        let dataset = SpaceTimeDataset::new(coords.clone(), values.clone()).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();

        for (i, c) in coords.iter().enumerate() {
            let pred = model.predict(*c).expect("prediction");
            assert!(
                (pred.value - values[i]).abs() < 1e-2,
                "at training point {i}: got {}, expected {}",
                pred.value,
                values[i]
            );
            assert!(pred.variance >= 0.0);
        }
    }

    #[test]
    fn predict_batch_matches_single_predictions() {
        let coords = make_coords();
        let values = vec![10.0, 20.0, 15.0, 25.0];
        let dataset = SpaceTimeDataset::new(coords.clone(), values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let targets = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5),
            SpaceTimeCoord::new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5),
        ];
        let batch = model.predict_batch(&targets).unwrap();
        for (t, b) in targets.iter().zip(batch.iter()) {
            let single = model.predict(*t).unwrap();
            assert!((single.value - b.value).abs() < 1e-6);
            assert!((single.variance - b.variance).abs() < 1e-6);
        }
    }

    #[test]
    fn clone_produces_equivalent_model() {
        let coords = make_coords();
        let values = vec![10.0, 20.0, 15.0, 25.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let cloned = model.clone();
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let a = model.predict(target).unwrap();
        let b = cloned.predict(target).unwrap();
        assert!((a.value - b.value).abs() < 1e-6);
        assert!((a.variance - b.variance).abs() < 1e-6);
    }

    #[test]
    fn rejects_unfactorizable_single_point_dataset() {
        let coords = vec![SpaceTimeCoord::new(
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            0.0,
        )];
        let values = vec![1.0];
        let r = SpaceTimeDataset::new(coords, values);
        assert!(matches!(r, Err(KrigingError::InsufficientData(2))));
    }

    #[test]
    fn prediction_variance_is_non_negative() {
        let coords = make_coords();
        let values = vec![10.0, 20.0, 15.0, 25.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let pred = model.predict(target).unwrap();
        assert!(pred.variance >= 0.0);
    }

    #[test]
    fn variance_increases_when_far_in_both_space_and_time() {
        let coords = make_coords();
        let values = vec![10.0, 20.0, 15.0, 25.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let near = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let far = SpaceTimeCoord::new(GeoCoord::try_new(5.0, 5.0).unwrap(), 10.0);
        let near_var = model.predict(near).unwrap().variance;
        let far_var = model.predict(far).unwrap().variance;
        assert!(far_var > near_var);
    }

    #[test]
    fn symmetric_under_time_reversal_around_midpoint() {
        // Training set is time-symmetric around t = 2: every training point's reflection
        // through t = 2 is also in the set with the same value. Predictions at a target
        // and its time-mirror must therefore agree.
        let coord_a = GeoCoord::try_new(0.0, 0.0).unwrap();
        let coord_b = GeoCoord::try_new(1.0, 1.0).unwrap();
        let coords = vec![
            SpaceTimeCoord::new(coord_a, 0.0),
            SpaceTimeCoord::new(coord_a, 4.0),
            SpaceTimeCoord::new(coord_b, 1.0),
            SpaceTimeCoord::new(coord_b, 3.0),
        ];
        let values = vec![5.0, 5.0, 7.0, 7.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();

        let target = GeoCoord::try_new(0.5, 0.5).unwrap();
        let a = model.predict(SpaceTimeCoord::new(target, 1.0)).unwrap();
        let b = model.predict(SpaceTimeCoord::new(target, 3.0)).unwrap();
        assert!(
            (a.value - b.value).abs() < 1e-3,
            "{} vs {}",
            a.value,
            b.value
        );
        assert!((a.variance - b.variance).abs() < 1e-3);
    }

    #[test]
    fn weights_sum_to_one_implicitly_via_constant_field() {
        let coords = make_coords();
        let values = vec![5.0; 4];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let pred = model.predict(target).unwrap();
        assert!((pred.value - 5.0).abs() < 1e-2);
    }

    #[test]
    fn works_with_projected_metric() {
        use crate::projected::ProjectedCoord;
        use crate::spacetime::metric::ProjectedMetric;

        let coords = vec![
            SpaceTimeCoord::new(ProjectedCoord::new(0.0, 0.0), 0.0),
            SpaceTimeCoord::new(ProjectedCoord::new(10.0, 0.0), 1.0),
            SpaceTimeCoord::new(ProjectedCoord::new(0.0, 10.0), 2.0),
        ];
        let values = vec![1.0, 2.0, 3.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model =
            SpaceTimeOrdinaryKrigingModel::new(ProjectedMetric::isotropic(), dataset, stv).unwrap();
        let target = SpaceTimeCoord::new(ProjectedCoord::new(5.0, 5.0), 1.0);
        let pred = model.predict(target).unwrap();
        assert!(pred.value.is_finite());
        assert!(pred.variance >= 0.0);
    }
}
