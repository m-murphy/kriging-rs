//! Universal kriging: interpolation with a deterministic polynomial trend.
//!
//! Universal kriging models the process as `Z(x) = Σ_l β_l f_l(x) + Y(x)` where `f_l` are
//! known basis functions of the coordinates (the "trend" or "drift") and `Y(x)` is a
//! zero-mean stationary residual with the given variogram. The unknown coefficients `β`
//! are handled as Lagrangian constraints inside the kriging system:
//!
//! ```text
//! | C   F | | w |   | c0 |
//! | Fᵀ  0 | | μ | = | f0 |
//! ```
//!
//! Supported trends (see [`UniversalTrend`]):
//!
//! - [`UniversalTrend::Constant`] — `[1]`. Equivalent to ordinary kriging.
//! - [`UniversalTrend::Linear`] — `[1, lat, lon]`.
//! - [`UniversalTrend::Quadratic`] — `[1, lat, lon, lat², lat·lon, lon²]`.
//!
//! Prediction returns a [`Prediction`] with the usual interpolated value and kriging
//! variance (adjusted for the trend constraints).

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
    fn eval(self, coord: GeoCoord, out: &mut [Real]) {
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

/// Fitted universal kriging model.
#[derive(Debug)]
pub struct UniversalKrigingModel {
    coords: Vec<GeoCoord>,
    prepared_coords: Vec<PreparedGeoCoord>,
    values: Vec<Real>,
    variogram: VariogramModel,
    trend: UniversalTrend,
    cov_at_zero: Real,
    system: DMatrix<Real>,
    /// Shared LU factorization; `Clone` just bumps the `Arc`.
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
}

impl Clone for UniversalKrigingModel {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.clone(),
            prepared_coords: self.prepared_coords.clone(),
            values: self.values.clone(),
            variogram: self.variogram,
            trend: self.trend,
            cov_at_zero: self.cov_at_zero,
            system: self.system.clone(),
            system_lu: Arc::clone(&self.system_lu),
        }
    }
}

impl UniversalKrigingModel {
    pub fn new(
        dataset: GeoDataset,
        variogram: VariogramModel,
        trend: UniversalTrend,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let n = coords.len();
        let p = trend.n_basis();
        if n < p + 1 {
            return Err(KrigingError::InsufficientData(p + 1));
        }
        let prepared_coords = coords
            .iter()
            .copied()
            .map(prepare_geo_coord)
            .collect::<Vec<_>>();

        let system = build_universal_system(&coords, &prepared_coords, variogram, trend);
        let system_lu = Arc::new(system.clone().lu());
        // Probe solvability with an arbitrary compatible RHS.
        let probe = DVector::from_element(n + p, 0.0);
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize universal kriging system".to_string(),
            ));
        }
        Ok(Self {
            coords,
            prepared_coords,
            values,
            variogram,
            trend,
            cov_at_zero: variogram.covariance(0.0),
            system,
            system_lu,
        })
    }

    pub fn trend(&self) -> UniversalTrend {
        self.trend
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        let n = self.coords.len();
        let p = self.trend.n_basis();
        let mut rhs = DVector::from_element(n + p, 0.0);
        self.predict_with_rhs(coord, &mut rhs)
    }

    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            let p = self.trend.n_basis();
            coords
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(n + p, 0.0),
                    |rhs, c| self.predict_with_rhs(*c, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let n = self.coords.len();
            let p = self.trend.n_basis();
            let mut rhs = DVector::from_element(n + p, 0.0);
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
        let p = self.trend.n_basis();
        let prepared = prepare_geo_coord(coord);
        for i in 0..n {
            rhs[i] = self.variogram.covariance(haversine_distance_prepared(
                self.prepared_coords[i],
                prepared,
            ));
        }
        let mut f0 = vec![0.0 as Real; p];
        self.trend.eval(coord, &mut f0);
        for l in 0..p {
            rhs[n + l] = f0[l];
        }

        let sol = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve universal kriging system".to_string())
        })?;
        let mut value: Real = 0.0;
        let mut cov_dot: Real = 0.0;
        for i in 0..n {
            value += sol[i] * self.values[i];
            cov_dot += sol[i] * rhs[i];
        }
        let mut mu_dot: Real = 0.0;
        for l in 0..p {
            mu_dot += sol[n + l] * f0[l];
        }
        let variance = (self.cov_at_zero - cov_dot - mu_dot).max(0.0);
        Ok(Prediction { value, variance })
    }
}

fn build_universal_system(
    coords: &[GeoCoord],
    prepared: &[PreparedGeoCoord],
    variogram: VariogramModel,
    trend: UniversalTrend,
) -> DMatrix<Real> {
    let n = coords.len();
    let p = trend.n_basis();
    let diag_eps = kriging_diagonal_jitter(n, variogram);

    let mut m = DMatrix::from_element(n + p, n + p, 0.0);
    // Covariance block.
    for i in 0..n {
        for j in i..n {
            let mut cov =
                variogram.covariance(haversine_distance_prepared(prepared[i], prepared[j]));
            if i == j {
                cov += diag_eps;
            }
            m[(i, j)] = cov;
            m[(j, i)] = cov;
        }
    }
    // Trend matrix F and its transpose.
    let mut fi = vec![0.0 as Real; p];
    for i in 0..n {
        trend.eval(coords[i], &mut fi);
        for l in 0..p {
            m[(i, n + l)] = fi[l];
            m[(n + l, i)] = fi[l];
        }
    }
    // Zero block in the bottom-right (already 0 from initialization).
    m
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
        // Construct data on a plane z = 1 + 2*lat + 3*lon. With a linear universal trend,
        // the predictor should recover this plane (up to numerical noise) at unsampled points.
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
        // Quadratic needs >= 7 stations.
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
