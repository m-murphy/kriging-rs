//! Universal space–time kriging.
//!
//! Generalizes ordinary ST kriging by allowing a deterministic polynomial trend in space,
//! time, or both. Non-constant trends use
//! [`SpaceTimeUniversalKrigingEngine`](crate::spacetime::kriging::universal_engine::SpaceTimeUniversalKrigingEngine);
//! constant trend delegates to [`SpaceTimeOrdinaryKrigingEngine`](crate::spacetime::kriging::engine::SpaceTimeOrdinaryKrigingEngine).

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::ordinary::Prediction;
use crate::kriging::pairwise::SpaceTimePairwiseCovariance;
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::kriging::engine::SpaceTimeOrdinaryKrigingEngine;
use crate::spacetime::kriging::universal_engine::{
    SpaceTimeTrendEval, SpaceTimeUniversalKrigingEngine,
};
use crate::spacetime::metric::SpatialBasis;
use crate::spacetime::variogram::SpaceTimeVariogram;

/// Polynomial trend bases for universal space–time kriging. Each variant lists the
/// terms it contributes to the design matrix `F` (one column per term).
///
/// Spatial components `s1, s2` come from
/// [`SpatialBasis::spatial_components`](crate::spacetime::metric::SpatialBasis::spatial_components):
/// `(lat, lon)` for [`GeoMetric`](crate::spacetime::GeoMetric), `(x, y)` for
/// [`ProjectedMetric`](crate::spacetime::ProjectedMetric).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceTimeUniversalTrend {
    /// Constant only: `[1]` (1 term). Equivalent to ordinary ST kriging.
    Constant,
    /// Constant + linear time: `[1, t]` (2 terms).
    LinearInTime,
    /// Constant + linear time + quadratic time: `[1, t, t²]` (3 terms).
    QuadraticInTime,
    /// Constant + linear space: `[1, s1, s2]` (3 terms).
    LinearInSpace,
    /// Constant + linear space + linear time: `[1, s1, s2, t]` (4 terms).
    LinearInSpaceAndTime,
    /// Constant + linear space + quadratic space + linear/quadratic time, with the
    /// space×time cross terms suppressed: `[1, s1, s2, s1², s1·s2, s2², t, t²]` (8 terms).
    QuadraticInSpaceAndTime,
}

impl SpaceTimeUniversalTrend {
    /// Number of basis functions (columns of `F`) added by this trend.
    pub fn n_basis(self) -> usize {
        match self {
            Self::Constant => 1,
            Self::LinearInTime => 2,
            Self::QuadraticInTime => 3,
            Self::LinearInSpace => 3,
            Self::LinearInSpaceAndTime => 4,
            Self::QuadraticInSpaceAndTime => 8,
        }
    }

    /// Evaluate the basis at `(s1, s2, t)` and write into `out` (length must match `n_basis`).
    pub fn eval(self, s1: Real, s2: Real, t: Real, out: &mut [Real]) {
        debug_assert_eq!(out.len(), self.n_basis());
        match self {
            Self::Constant => out[0] = 1.0,
            Self::LinearInTime => {
                out[0] = 1.0;
                out[1] = t;
            }
            Self::QuadraticInTime => {
                out[0] = 1.0;
                out[1] = t;
                out[2] = t * t;
            }
            Self::LinearInSpace => {
                out[0] = 1.0;
                out[1] = s1;
                out[2] = s2;
            }
            Self::LinearInSpaceAndTime => {
                out[0] = 1.0;
                out[1] = s1;
                out[2] = s2;
                out[3] = t;
            }
            Self::QuadraticInSpaceAndTime => {
                out[0] = 1.0;
                out[1] = s1;
                out[2] = s2;
                out[3] = s1 * s1;
                out[4] = s1 * s2;
                out[5] = s2 * s2;
                out[6] = t;
                out[7] = t * t;
            }
        }
    }
}

#[derive(Debug, Clone)]
enum SpaceTimeUniversalInner<M: SpatialBasis> {
    Constant(SpaceTimeOrdinaryKrigingEngine<M>),
    Drift(SpaceTimeUniversalKrigingEngine<M>),
}

/// Fitted universal space–time kriging model.
#[derive(Debug, Clone)]
pub struct SpaceTimeUniversalKrigingModel<M: SpatialBasis> {
    trend: SpaceTimeUniversalTrend,
    inner: SpaceTimeUniversalInner<M>,
}

impl<M: SpatialBasis> SpaceTimeUniversalKrigingModel<M> {
    pub fn new(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
        trend: SpaceTimeUniversalTrend,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let inner = if trend == SpaceTimeUniversalTrend::Constant {
            SpaceTimeUniversalInner::Constant(
                SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
                    SpaceTimePairwiseCovariance::new(metric, variogram),
                    coords,
                    values,
                    &[],
                )?,
            )
        } else {
            SpaceTimeUniversalInner::Drift(SpaceTimeUniversalKrigingEngine::fit(
                SpaceTimePairwiseCovariance::new(metric, variogram),
                coords,
                values,
                SpaceTimeTrendEval::new(metric, trend),
            )?)
        };
        Ok(Self { trend, inner })
    }

    pub fn trend(&self) -> SpaceTimeUniversalTrend {
        self.trend
    }

    pub fn variogram(&self) -> SpaceTimeVariogram {
        match &self.inner {
            SpaceTimeUniversalInner::Constant(engine) => engine.pairwise_covariance().variogram(),
            SpaceTimeUniversalInner::Drift(engine) => engine.pairwise_covariance().variogram(),
        }
    }

    pub fn predict(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        match &self.inner {
            SpaceTimeUniversalInner::Constant(engine) => engine
                .predict(&[target])
                .map(|mut v| v.pop().expect("single prediction")),
            SpaceTimeUniversalInner::Drift(engine) => engine
                .predict(&[target])
                .map(|mut v| v.pop().expect("single prediction")),
        }
    }

    pub fn predict_batch(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<Prediction>, KrigingError> {
        match &self.inner {
            SpaceTimeUniversalInner::Constant(engine) => engine.predict(targets),
            SpaceTimeUniversalInner::Drift(engine) => engine.predict(targets),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::SpaceTimeOrdinaryKrigingModel;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn variogram() -> SpaceTimeVariogram {
        SpaceTimeVariogram::new_separable(
            VariogramModel::new(0.05, 1.0, 300.0, VariogramType::Exponential).unwrap(),
            VariogramModel::new(0.05, 1.0, 5.0, VariogramType::Exponential).unwrap(),
        )
        .unwrap()
    }

    fn make_grid() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>) {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for t in 0..3 {
                    let lat = i as Real * 0.1;
                    let lon = j as Real * 0.1;
                    let tval = t as Real;
                    coords.push(SpaceTimeCoord::new(
                        GeoCoord::try_new(lat, lon).unwrap(),
                        tval,
                    ));
                    values.push(1.0 + 2.0 * lat + 0.5 * lon + 3.0 * tval);
                }
            }
        }
        (coords, values)
    }

    #[test]
    fn n_basis_matches_eval_length() {
        for trend in [
            SpaceTimeUniversalTrend::Constant,
            SpaceTimeUniversalTrend::LinearInTime,
            SpaceTimeUniversalTrend::QuadraticInTime,
            SpaceTimeUniversalTrend::LinearInSpace,
            SpaceTimeUniversalTrend::LinearInSpaceAndTime,
            SpaceTimeUniversalTrend::QuadraticInSpaceAndTime,
        ] {
            let mut buf = vec![0.0 as Real; trend.n_basis()];
            trend.eval(0.5, 0.7, 1.3, &mut buf);
            assert!(buf.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn constant_trend_matches_ordinary_kriging_closely() {
        let (coords, values) = make_grid();
        let v = variogram();
        let uk = SpaceTimeUniversalKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords.clone(), values.clone()).unwrap(),
            v,
            SpaceTimeUniversalTrend::Constant,
        )
        .unwrap();
        let ok = SpaceTimeOrdinaryKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords, values).unwrap(),
            v,
        )
        .unwrap();
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.15, 0.0).unwrap(), 1.5);
        let uk_p = uk.predict(target).unwrap();
        let ok_p = ok.predict(target).unwrap();
        assert!((uk_p.value - ok_p.value).abs() < 1e-3);
        assert!((uk_p.variance - ok_p.variance).abs() < 1e-3);
    }

    #[test]
    fn linear_in_time_recovers_pure_temporal_trend() {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..3 {
            for t in 0..4 {
                coords.push(SpaceTimeCoord::new(
                    GeoCoord::try_new(i as Real * 0.05, 0.0).unwrap(),
                    t as Real,
                ));
                values.push(1.0 + 3.0 * t as Real);
            }
        }
        let v = variogram();
        let model = SpaceTimeUniversalKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords, values).unwrap(),
            v,
            SpaceTimeUniversalTrend::LinearInTime,
        )
        .unwrap();
        let pred = model
            .predict(SpaceTimeCoord::new(
                GeoCoord::try_new(0.025, 0.0).unwrap(),
                10.0,
            ))
            .unwrap();
        let expected = 1.0 + 3.0 * 10.0;
        assert!(
            (pred.value - expected).abs() < 0.5,
            "got {}, expected {}",
            pred.value,
            expected
        );
    }

    #[test]
    fn linear_in_space_and_time_recovers_planar_drift() {
        let (coords, values) = make_grid();
        let v = variogram();
        let model = SpaceTimeUniversalKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords, values).unwrap(),
            v,
            SpaceTimeUniversalTrend::LinearInSpaceAndTime,
        )
        .unwrap();
        let lat = 0.15;
        let lon = 0.15;
        let t = 1.5;
        let pred = model
            .predict(SpaceTimeCoord::new(GeoCoord::try_new(lat, lon).unwrap(), t))
            .unwrap();
        let expected = 1.0 + 2.0 * lat + 0.5 * lon + 3.0 * t;
        assert!(
            (pred.value - expected).abs() < 0.5,
            "got {}, expected {}",
            pred.value,
            expected
        );
    }

    #[test]
    fn rejects_insufficient_data_for_quadratic_trend() {
        let coords = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.1).unwrap(), 0.5),
            SpaceTimeCoord::new(GeoCoord::try_new(0.1, 0.0).unwrap(), 1.0),
        ];
        let values = vec![1.0, 2.0, 3.0];
        let v = variogram();
        let err = SpaceTimeUniversalKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords, values).unwrap(),
            v,
            SpaceTimeUniversalTrend::QuadraticInSpaceAndTime,
        )
        .expect_err("should reject insufficient data");
        assert!(matches!(err, KrigingError::InsufficientData(_)));
    }

    #[test]
    fn batch_matches_single() {
        let (coords, values) = make_grid();
        let v = variogram();
        let model = SpaceTimeUniversalKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords, values).unwrap(),
            v,
            SpaceTimeUniversalTrend::LinearInSpaceAndTime,
        )
        .unwrap();
        let targets = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.05, 0.05).unwrap(), 0.5),
            SpaceTimeCoord::new(GeoCoord::try_new(0.15, 0.25).unwrap(), 1.5),
        ];
        let batch = model.predict_batch(&targets).unwrap();
        for (i, t) in targets.iter().enumerate() {
            let single = model.predict(*t).unwrap();
            assert!((batch[i].value - single.value).abs() < 1e-5);
        }
    }
}
