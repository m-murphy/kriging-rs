//! Universal kriging: interpolation with a deterministic polynomial trend.
//!
//! Universal kriging models the process as `Z(x) = Σ_l β_l f_l(x) + Y(x)` where `f_l` are
//! known basis functions of the coordinates (the "trend" or "drift") and `Y(x)` is a
//! zero-mean stationary residual with the given variogram. The unknown coefficients `β`
//! are handled as Lagrangian constraints solved via the dual SPD engine
//! ([`UniversalKrigingEngine`](crate::kriging::universal_engine::UniversalKrigingEngine))
//! or, for a constant trend, [`OrdinaryKrigingEngine`](crate::kriging::engine::OrdinaryKrigingEngine).
//!
//! Supported trends (see [`UniversalTrend`]):
//!
//! - [`UniversalTrend::Constant`] — `[1]`. Equivalent to ordinary kriging.
//! - [`UniversalTrend::Linear`] — `[1, lat, lon]`.
//! - [`UniversalTrend::Quadratic`] — `[1, lat, lon, lat², lat·lon, lon²]`.

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::kriging::ordinary::Prediction;
use crate::kriging::universal_engine::UniversalKrigingEngine;
use crate::spacetime::metric::GeoMetric;
use crate::variogram::models::VariogramModel;

/// Polynomial trend used by [`UniversalKrigingModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniversalTrend {
    /// Constant mean (1 basis function). Equivalent to ordinary kriging.
    Constant,
    /// Linear drift in (lat, lon). Basis = `[1, lat, lon]`.
    Linear,
    /// Quadratic drift in (lat, lon). Basis = `[1, lat, lon, lat², lat·lon, lon²]`.
    Quadratic,
}

impl UniversalTrend {
    /// Number of basis functions (columns in the `F` matrix).
    pub fn n_basis(self) -> usize {
        match self {
            UniversalTrend::Constant => 1,
            UniversalTrend::Linear => 3,
            UniversalTrend::Quadratic => 6,
        }
    }

    /// Evaluate basis functions at `coord`, writing into `out` (must have length `n_basis()`).
    pub(crate) fn eval_basis(self, coord: GeoCoord, out: &mut [Real]) {
        let lat = coord.lat();
        let lon = coord.lon();
        match self {
            UniversalTrend::Constant => {
                out[0] = 1.0;
            }
            UniversalTrend::Linear => {
                out[0] = 1.0;
                out[1] = lat;
                out[2] = lon;
            }
            UniversalTrend::Quadratic => {
                out[0] = 1.0;
                out[1] = lat;
                out[2] = lon;
                out[3] = lat * lat;
                out[4] = lat * lon;
                out[5] = lon * lon;
            }
        }
    }
}

#[derive(Debug, Clone)]
enum UniversalKrigingInner {
    Constant(OrdinaryKrigingEngine<GeoMetric>),
    Drift(UniversalKrigingEngine),
}

/// Fitted universal kriging model.
#[derive(Debug, Clone)]
pub struct UniversalKrigingModel {
    trend: UniversalTrend,
    inner: UniversalKrigingInner,
}

impl UniversalKrigingModel {
    pub fn new(
        dataset: GeoDataset,
        variogram: VariogramModel,
        trend: UniversalTrend,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let inner = if trend == UniversalTrend::Constant {
            UniversalKrigingInner::Constant(OrdinaryKrigingEngine::fit(
                GeoMetric, coords, values, variogram,
            )?)
        } else {
            UniversalKrigingInner::Drift(UniversalKrigingEngine::fit(
                coords, values, variogram, trend,
            )?)
        };
        Ok(Self { trend, inner })
    }

    pub fn trend(&self) -> UniversalTrend {
        self.trend
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        match &self.inner {
            UniversalKrigingInner::Constant(engine) => engine
                .predict(&[coord])
                .map(|mut v| v.pop().expect("single prediction")),
            UniversalKrigingInner::Drift(engine) => engine
                .predict(&[coord])
                .map(|mut v| v.pop().expect("single prediction")),
        }
    }

    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        match &self.inner {
            UniversalKrigingInner::Constant(engine) => engine.predict(coords),
            UniversalKrigingInner::Drift(engine) => engine.predict(coords),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

    #[test]
    fn constant_trend_matches_ordinary_kriging_closely() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let uk = UniversalKrigingModel::new(
            GeoDataset::new(coords.clone(), values.clone()).unwrap(),
            variogram,
            UniversalTrend::Constant,
        )
        .expect("uk");
        let ok =
            crate::OrdinaryKrigingModel::new(GeoDataset::new(coords, values).unwrap(), variogram)
                .expect("ok");

        let target = GeoCoord::try_new(0.5, 0.5).unwrap();
        let uk_pred = uk.predict(target).expect("uk predict");
        let ok_pred = ok.predict(target).expect("ok predict");
        assert!((uk_pred.value - ok_pred.value).abs() < 1e-3);
        assert!((uk_pred.variance - ok_pred.variance).abs() < 1e-3);
    }

    #[test]
    fn linear_trend_fits_planar_surface_exactly() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
            GeoCoord::try_new(2.0, 0.5).unwrap(),
        ];
        let values: Vec<Real> = coords
            .iter()
            .map(|c| 1.0 + 2.0 * c.lat() + 3.0 * c.lon())
            .collect();
        let variogram = VariogramModel::new(0.01, 1.0, 500.0, VariogramType::Exponential).unwrap();
        let model = UniversalKrigingModel::new(
            GeoDataset::new(coords, values).unwrap(),
            variogram,
            UniversalTrend::Linear,
        )
        .expect("uk");
        let target = GeoCoord::try_new(0.7, 0.3).unwrap();
        let expected = 1.0 + 2.0 * 0.7 + 3.0 * 0.3;
        let pred = model.predict(target).expect("prediction");
        assert!(
            (pred.value - expected).abs() < 0.1,
            "got {}, expected {}",
            pred.value,
            expected
        );
    }

    #[test]
    fn rejects_insufficient_data_for_quadratic_trend() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];
        let variogram = VariogramModel::new(0.01, 1.0, 100.0, VariogramType::Exponential).unwrap();
        let err = UniversalKrigingModel::new(
            GeoDataset::new(coords, values).unwrap(),
            variogram,
            UniversalTrend::Quadratic,
        )
        .expect_err("should fail");
        match err {
            KrigingError::InsufficientData(_) => {}
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }
}
