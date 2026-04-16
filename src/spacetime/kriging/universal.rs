//! Universal space–time kriging.
//!
//! Generalizes ordinary ST kriging by allowing a deterministic polynomial trend in space,
//! time, or both. Mirrors the 2-D [`UniversalKrigingModel`](crate::UniversalKrigingModel)
//! pattern: stack `[C  F; F^T  0]` and solve the augmented system.

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

/// Fitted universal space–time kriging model.
#[derive(Debug)]
pub struct SpaceTimeUniversalKrigingModel<M: SpatialBasis> {
    metric: M,
    coords: Vec<SpaceTimeCoord<M::Coord>>,
    prepared_spatial: Vec<M::Prepared>,
    times: Vec<Real>,
    values: Vec<Real>,
    variogram: SpaceTimeVariogram,
    trend: SpaceTimeUniversalTrend,
    c_at_zero: Real,
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
}

impl<M: SpatialBasis> Clone for SpaceTimeUniversalKrigingModel<M>
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
            trend: self.trend,
            c_at_zero: self.c_at_zero,
            system_lu: Arc::clone(&self.system_lu),
        }
    }
}

impl<M: SpatialBasis> SpaceTimeUniversalKrigingModel<M> {
    pub fn new(
        metric: M,
        dataset: SpaceTimeDataset<M::Coord>,
        variogram: SpaceTimeVariogram,
        trend: SpaceTimeUniversalTrend,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let n = coords.len();
        let p = trend.n_basis();
        if n < p + 1 {
            return Err(KrigingError::InsufficientData(p + 1));
        }
        let prepared_spatial: Vec<M::Prepared> =
            coords.iter().map(|c| metric.prepare(c.spatial)).collect();
        let times: Vec<Real> = coords.iter().map(|c| c.time).collect();

        let system = build_system(
            &metric,
            &coords,
            &prepared_spatial,
            &times,
            variogram,
            trend,
        );
        let system_lu = Arc::new(system.lu());
        let probe = DVector::from_element(n + p, 0.0);
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize space-time universal kriging system".to_string(),
            ));
        }
        Ok(Self {
            metric,
            coords,
            prepared_spatial,
            times,
            values,
            variogram,
            trend,
            c_at_zero: variogram.c_at_zero(),
            system_lu,
        })
    }

    pub fn trend(&self) -> SpaceTimeUniversalTrend {
        self.trend
    }

    pub fn predict(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        let n = self.coords.len();
        let p = self.trend.n_basis();
        let mut rhs = DVector::from_element(n + p, 0.0);
        self.predict_with_rhs(target, &mut rhs)
    }

    pub fn predict_batch(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            let p = self.trend.n_basis();
            targets
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(n + p, 0.0),
                    |rhs, t| self.predict_with_rhs(*t, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let n = self.coords.len();
            let p = self.trend.n_basis();
            let mut rhs = DVector::from_element(n + p, 0.0);
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
        let p = self.trend.n_basis();
        let prepared_target = self.metric.prepare(target.spatial);
        for i in 0..n {
            let hs = self
                .metric
                .distance(self.prepared_spatial[i], prepared_target);
            let ht = temporal_distance(self.times[i], target.time);
            rhs[i] = self.variogram.covariance(hs, ht);
        }
        let (s1, s2) = self.metric.spatial_components(target.spatial);
        let mut f0 = vec![0.0 as Real; p];
        self.trend.eval(s1, s2, target.time, &mut f0);
        for l in 0..p {
            rhs[n + l] = f0[l];
        }
        let sol = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError(
                "could not solve space-time universal kriging system".to_string(),
            )
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
        let variance = (self.c_at_zero - cov_dot - mu_dot).max(0.0);
        Ok(Prediction { value, variance })
    }
}

fn build_system<M: SpatialBasis>(
    metric: &M,
    coords: &[SpaceTimeCoord<M::Coord>],
    prepared: &[M::Prepared],
    times: &[Real],
    variogram: SpaceTimeVariogram,
    trend: SpaceTimeUniversalTrend,
) -> DMatrix<Real> {
    let n = prepared.len();
    let p = trend.n_basis();
    let diag_eps = spacetime_diagonal_jitter(n, variogram);
    let mut m = DMatrix::from_element(n + p, n + p, 0.0);
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
    let mut fi = vec![0.0 as Real; p];
    for i in 0..n {
        let (s1, s2) = metric.spatial_components(coords[i].spatial);
        trend.eval(s1, s2, times[i], &mut fi);
        for l in 0..p {
            m[(i, n + l)] = fi[l];
            m[(n + l, i)] = fi[l];
        }
    }
    m
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
        // Field z = 1 + 3*t with no spatial dependence; LinearInTime should fit exactly.
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
        // z = 1 + 2*lat + 0.5*lon + 3*t. With both spatial axes varying the design matrix
        // is full-rank and LinearInSpaceAndTime should recover the plane closely.
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
