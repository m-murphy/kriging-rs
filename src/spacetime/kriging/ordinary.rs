//! Ordinary space–time kriging.
//!
//! Build a model from a [`SpaceTimeDataset`] and a [`SpaceTimeVariogram`] and predict at
//! arbitrary [`SpaceTimeCoord`] targets. The underlying system has the same shape as 2-D
//! ordinary kriging: an `(n + 1) × (n + 1)` covariance matrix with a Lagrangian row/column
//! enforcing unit-sum weights.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::ordinary::Prediction;
use crate::spacetime::coord::{SpaceTimeCoord, temporal_distance};
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::variogram::models::VariogramType;

/// Fitted ordinary space–time kriging model.
///
/// Generic over a [`SpatialMetric`] so the same implementation serves geographic data
/// ([`GeoMetric`](crate::spacetime::GeoMetric)) and projected data
/// ([`ProjectedMetric`](crate::spacetime::ProjectedMetric)).
#[derive(Debug)]
pub struct SpaceTimeOrdinaryKrigingModel<M: SpatialMetric> {
    metric: M,
    coords: Vec<SpaceTimeCoord<M::Coord>>,
    prepared_spatial: Vec<M::Prepared>,
    times: Vec<Real>,
    values: Vec<Real>,
    variogram: SpaceTimeVariogram,
    c_at_zero: Real,
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
}

impl<M: SpatialMetric> Clone for SpaceTimeOrdinaryKrigingModel<M>
where
    M::Coord: Clone,
    M::Prepared: Clone,
{
    fn clone(&self) -> Self {
        Self {
            metric: self.metric,
            coords: self.coords.clone(),
            prepared_spatial: self.prepared_spatial.clone(),
            times: self.times.clone(),
            values: self.values.clone(),
            variogram: self.variogram,
            c_at_zero: self.c_at_zero,
            system_lu: Arc::clone(&self.system_lu),
        }
    }
}

impl<M: SpatialMetric> SpaceTimeOrdinaryKrigingModel<M> {
    /// Build a model. The caller must supply a compatible [`SpatialMetric`] — in particular
    /// the variogram's spatial range must be expressed in the same units the metric returns.
    pub fn new(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let prepared_spatial: Vec<M::Prepared> =
            coords.iter().map(|c| metric.prepare(c.spatial)).collect();
        let times: Vec<Real> = coords.iter().map(|c| c.time).collect();

        let system = build_ordinary_system(&metric, &prepared_spatial, &times, variogram);
        let system_lu = Arc::new(system.lu());
        let n = coords.len();
        let mut probe = DVector::from_element(n + 1, 0.0);
        probe[n] = 1.0;
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize space-time ordinary kriging system".to_string(),
            ));
        }
        Ok(Self {
            metric,
            coords,
            prepared_spatial,
            times,
            values,
            variogram,
            c_at_zero: variogram.c_at_zero(),
            system_lu,
        })
    }

    /// Metric used to measure spatial distances.
    pub fn metric(&self) -> M {
        self.metric
    }

    /// Number of training points.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether the model has no training points. Always `false` because construction
    /// enforces `len() >= 2`.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Space–time variogram used by the model.
    pub fn variogram(&self) -> SpaceTimeVariogram {
        self.variogram
    }

    /// Single-target prediction.
    pub fn predict(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
        self.predict_with_rhs(target, &mut rhs)
    }

    /// Batched predictions. Parallel on native builds; sequential on wasm32.
    pub fn predict_batch(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            targets
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(n + 1, 0.0),
                    |rhs, t| self.predict_with_rhs(*t, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
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
        let n = self.coords.len();
        let prepared_target = self.metric.prepare(target.spatial);
        for i in 0..n {
            let hs = self
                .metric
                .distance(self.prepared_spatial[i], prepared_target);
            let ht = temporal_distance(self.times[i], target.time);
            rhs[i] = self.variogram.covariance(hs, ht);
        }
        rhs[n] = 1.0;

        let sol = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError(
                "could not solve space-time ordinary kriging system".to_string(),
            )
        })?;
        let mut value: Real = 0.0;
        let mut cov_dot: Real = 0.0;
        for i in 0..n {
            value += sol[i] * self.values[i];
            cov_dot += sol[i] * rhs[i];
        }
        let mu = sol[n];
        let variance = (self.c_at_zero - cov_dot - mu).max(0.0);
        Ok(Prediction { value, variance })
    }
}

/// Diagonal regularization added to the covariance block of a space–time kriging matrix to
/// keep it solvable under `f32`. Picks the stronger of the spatial and temporal marginal
/// jitters (Gaussian and Cubic marginals are the ill-conditioning cases) and scales by the
/// ST `C(0, 0)` so the absolute magnitude matches the matrix entries.
pub(crate) fn spacetime_diagonal_jitter(n_stations: usize, variogram: SpaceTimeVariogram) -> Real {
    let c0 = variogram.c_at_zero();
    let scale = (n_stations as Real).sqrt().max(1.0);
    let worst_frac = variogram
        .marginal_variogram_types()
        .iter()
        .map(|vt| match vt {
            VariogramType::Gaussian => 1e-5 as Real,
            VariogramType::Cubic => 1e-4 as Real,
            _ => 1e-8 as Real,
        })
        .fold(1e-8 as Real, Real::max);
    let floor = (1e-10 * c0).max(Real::MIN_POSITIVE);
    (worst_frac * c0 * scale).max(floor)
}

fn build_ordinary_system<M: SpatialMetric>(
    metric: &M,
    prepared_spatial: &[M::Prepared],
    times: &[Real],
    variogram: SpaceTimeVariogram,
) -> DMatrix<Real> {
    let n = prepared_spatial.len();
    let diag_eps = spacetime_diagonal_jitter(n, variogram);
    let mut m = DMatrix::from_element(n + 1, n + 1, 0.0);
    for i in 0..n {
        for j in i..n {
            let hs = metric.distance(prepared_spatial[i], prepared_spatial[j]);
            let ht = temporal_distance(times[i], times[j]);
            let mut cov = variogram.covariance(hs, ht);
            if i == j {
                cov += diag_eps;
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
    use crate::distance::GeoCoord;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::VariogramModel;

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
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();

        let targets = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.3, 0.4).unwrap(), 0.5),
            SpaceTimeCoord::new(GeoCoord::try_new(0.7, 0.8).unwrap(), 2.5),
            SpaceTimeCoord::new(GeoCoord::try_new(0.1, 0.9).unwrap(), 1.2),
        ];
        let batch = model.predict_batch(&targets).expect("batch");
        for (i, t) in targets.iter().enumerate() {
            let single = model.predict(*t).expect("single");
            assert!((batch[i].value - single.value).abs() < 1e-5);
            assert!((batch[i].variance - single.variance).abs() < 1e-5);
        }
    }

    #[test]
    fn prediction_variance_is_non_negative() {
        let coords = make_coords();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();

        for lat in [-0.5, 0.2, 0.9, 2.0] {
            for t in [-1.0, 0.5, 4.0, 100.0] {
                let pred = model
                    .predict(SpaceTimeCoord::new(GeoCoord::try_new(lat, 0.3).unwrap(), t))
                    .expect("predict");
                assert!(pred.variance >= 0.0, "variance at (lat={lat}, t={t})");
                assert!(pred.variance.is_finite());
                assert!(pred.value.is_finite());
            }
        }
    }

    #[test]
    fn variance_increases_when_far_in_both_space_and_time() {
        let coords = make_coords();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();

        let near = model
            .predict(SpaceTimeCoord::new(
                GeoCoord::try_new(0.5, 0.5).unwrap(),
                1.5,
            ))
            .unwrap();
        let far = model
            .predict(SpaceTimeCoord::new(
                GeoCoord::try_new(0.5, 0.5).unwrap(),
                1000.0,
            ))
            .unwrap();
        assert!(
            far.variance > near.variance,
            "far-in-time variance {} should exceed near variance {}",
            far.variance,
            near.variance
        );
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
    fn clone_produces_equivalent_model() {
        let coords = make_coords();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let cloned = model.clone();

        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let original = model.predict(target).unwrap();
        let duplicate = cloned.predict(target).unwrap();
        assert!((original.value - duplicate.value).abs() < 1e-6);
        assert!((original.variance - duplicate.variance).abs() < 1e-6);
    }

    #[test]
    fn weights_sum_to_one_implicitly_via_constant_field() {
        // Ordinary kriging of a constant field must return the constant at every target.
        let coords = make_coords();
        let values = vec![42.0; coords.len()];
        let dataset = SpaceTimeDataset::new(coords, values).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial_var(), temporal_var()).unwrap();
        let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, stv).unwrap();
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.2, 0.7).unwrap(), 1.2);
        let pred = model.predict(target).expect("prediction");
        assert!((pred.value - 42.0).abs() < 1e-3, "got {}", pred.value);
    }

    #[test]
    fn rejects_unfactorizable_single_point_dataset() {
        let coords = vec![SpaceTimeCoord::new(
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            0.0,
        )];
        let values = vec![1.0];
        assert!(SpaceTimeDataset::new(coords, values).is_err());
    }

    #[test]
    fn works_with_projected_metric() {
        use crate::projected::ProjectedCoord;
        use crate::spacetime::metric::ProjectedMetric;

        let coords = vec![
            SpaceTimeCoord::new(ProjectedCoord::new(0.0, 0.0), 0.0),
            SpaceTimeCoord::new(ProjectedCoord::new(0.0, 1.0), 1.0),
            SpaceTimeCoord::new(ProjectedCoord::new(1.0, 0.0), 2.0),
            SpaceTimeCoord::new(ProjectedCoord::new(1.0, 1.0), 3.0),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let dataset = SpaceTimeDataset::new(coords.clone(), values).unwrap();
        let spatial = VariogramModel::new(0.01, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.01, 2.0, 3.0, VariogramType::Exponential).unwrap();
        let stv = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let model =
            SpaceTimeOrdinaryKrigingModel::new(ProjectedMetric::isotropic(), dataset, stv).unwrap();
        for c in &coords {
            let pred = model.predict(*c).unwrap();
            assert!(pred.value.is_finite() && pred.variance.is_finite());
        }
    }
}
