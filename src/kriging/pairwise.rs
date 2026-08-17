//! Pairwise covariance seam for ordinary, simple, and universal kriging engines.
//!
//! Adapters hide spatial domain and the semivariogram model. The engine sees `C_ij`, `C(0)`,
//! and model jitter. Per-site observation variance is applied by the engine, not here.
//! See ADR-0003.

use crate::Real;
use crate::kriging::numerics::{kriging_diagonal_jitter, spacetime_diagonal_jitter};
use crate::spacetime::coord::{SpaceTimeCoord, temporal_distance};
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::variogram::models::VariogramModel;

/// Map from a pair of prepared sites to an entry of C.
///
/// `covariance` is the off-diagonal (no jitter). `diagonal` is `C(0) +` model jitter.
/// Collocated distinct stations use `covariance`, not `diagonal`.
pub trait PairwiseCovariance: Copy + Clone + std::fmt::Debug + Send + Sync {
    type Site: Copy + Send + Sync + std::fmt::Debug + PartialEq;
    type Prepared: Copy + Send + Sync + std::fmt::Debug;

    fn prepare(&self, site: Self::Site) -> Self::Prepared;

    fn covariance(&self, a: Self::Prepared, b: Self::Prepared) -> Real;

    fn cov_at_zero(&self) -> Real;

    fn jitter(&self) -> Real;

    fn diagonal(&self) -> Real {
        self.cov_at_zero() + self.jitter()
    }
}

/// 2-D pairwise covariance: spatial metric + semivariogram model.
#[derive(Debug, Clone, Copy)]
pub struct SpatialPairwiseCovariance<M: SpatialMetric> {
    metric: M,
    variogram: VariogramModel,
}

impl<M: SpatialMetric> SpatialPairwiseCovariance<M> {
    pub fn new(metric: M, variogram: VariogramModel) -> Self {
        Self { metric, variogram }
    }

    pub fn metric(self) -> M {
        self.metric
    }

    pub fn variogram(self) -> VariogramModel {
        self.variogram
    }
}

impl<M: SpatialMetric> PairwiseCovariance for SpatialPairwiseCovariance<M> {
    type Site = M::Coord;
    type Prepared = M::Prepared;

    fn prepare(&self, site: Self::Site) -> Self::Prepared {
        self.metric.prepare(site)
    }

    fn covariance(&self, a: Self::Prepared, b: Self::Prepared) -> Real {
        self.variogram.covariance(self.metric.distance(a, b))
    }

    fn cov_at_zero(&self) -> Real {
        self.variogram.covariance(0.0)
    }

    fn jitter(&self) -> Real {
        kriging_diagonal_jitter(self.variogram)
    }
}

/// Prepared space–time site: warped spatial coordinate plus scalar time.
#[derive(Debug, Clone, Copy)]
pub struct PreparedSpaceTime<P> {
    spatial: P,
    time: Real,
}

/// Spatio-temporal pairwise covariance: space-time metric + space-time variogram.
#[derive(Debug, Clone, Copy)]
pub struct SpaceTimePairwiseCovariance<M: SpatialMetric> {
    metric: M,
    variogram: SpaceTimeVariogram,
}

impl<M: SpatialMetric> SpaceTimePairwiseCovariance<M> {
    pub fn new(metric: M, variogram: SpaceTimeVariogram) -> Self {
        Self { metric, variogram }
    }

    pub fn metric(self) -> M {
        self.metric
    }

    pub fn variogram(self) -> SpaceTimeVariogram {
        self.variogram
    }
}

impl<M: SpatialMetric> PairwiseCovariance for SpaceTimePairwiseCovariance<M> {
    type Site = SpaceTimeCoord<M::Coord>;
    type Prepared = PreparedSpaceTime<M::Prepared>;

    fn prepare(&self, site: Self::Site) -> Self::Prepared {
        PreparedSpaceTime {
            spatial: self.metric.prepare(site.spatial),
            time: site.time,
        }
    }

    fn covariance(&self, a: Self::Prepared, b: Self::Prepared) -> Real {
        let hs = self.metric.distance(a.spatial, b.spatial);
        let ht = temporal_distance(a.time, b.time);
        self.variogram.covariance(hs, ht)
    }

    fn cov_at_zero(&self) -> Real {
        self.variogram.c_at_zero()
    }

    fn jitter(&self) -> Real {
        spacetime_diagonal_jitter(self.variogram)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::kriging::numerics::{kriging_diagonal_jitter, spacetime_diagonal_jitter};
    use crate::projected::ProjectedCoord;
    use crate::spacetime::metric::{GeoMetric, ProjectedMetric};
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn exponential() -> VariogramModel {
        VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn spatial_covariance_is_symmetric() {
        let cov = SpatialPairwiseCovariance::new(GeoMetric, exponential());
        let a = cov.prepare(GeoCoord::try_new(0.0, 0.0).unwrap());
        let b = cov.prepare(GeoCoord::try_new(0.3, 0.5).unwrap());
        assert_relative_eq!(cov.covariance(a, b), cov.covariance(b, a), epsilon = 1e-12);
    }

    #[test]
    fn spatial_diagonal_is_c0_plus_model_jitter() {
        let variogram = exponential();
        let cov = SpatialPairwiseCovariance::new(GeoMetric, variogram);
        assert_relative_eq!(
            cov.diagonal(),
            cov.cov_at_zero() + kriging_diagonal_jitter(variogram),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            cov.cov_at_zero(),
            variogram.covariance(0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn spatial_collocated_off_diagonal_has_no_jitter() {
        let cov = SpatialPairwiseCovariance::new(GeoMetric, exponential());
        let p = cov.prepare(GeoCoord::try_new(1.0, 2.0).unwrap());
        assert_relative_eq!(cov.covariance(p, p), cov.cov_at_zero(), epsilon = 1e-12);
        assert!(cov.diagonal() > cov.covariance(p, p));
    }

    #[test]
    fn projected_spatial_adapter_uses_euclidean_distance() {
        let cov = SpatialPairwiseCovariance::new(ProjectedMetric::isotropic(), exponential());
        let a = cov.prepare(ProjectedCoord::new(0.0, 0.0));
        let b = cov.prepare(ProjectedCoord::new(3.0, 4.0));
        assert_relative_eq!(
            cov.covariance(a, b),
            exponential().covariance(5.0),
            epsilon = 1e-12
        );
    }

    fn st_variogram() -> SpaceTimeVariogram {
        let spatial = VariogramModel::new(0.01, 1.0, 300.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
        SpaceTimeVariogram::new_separable(spatial, temporal).unwrap()
    }

    #[test]
    fn spacetime_covariance_is_symmetric() {
        let cov = SpaceTimePairwiseCovariance::new(GeoMetric, st_variogram());
        let a = cov.prepare(SpaceTimeCoord::new(
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            0.0,
        ));
        let b = cov.prepare(SpaceTimeCoord::new(
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            2.0,
        ));
        assert_relative_eq!(cov.covariance(a, b), cov.covariance(b, a), epsilon = 1e-12);
    }

    #[test]
    fn spacetime_diagonal_is_c0_plus_model_jitter() {
        let variogram = st_variogram();
        let cov = SpaceTimePairwiseCovariance::new(GeoMetric, variogram);
        assert_relative_eq!(
            cov.diagonal(),
            cov.cov_at_zero() + spacetime_diagonal_jitter(variogram),
            epsilon = 1e-12
        );
        assert_relative_eq!(cov.cov_at_zero(), variogram.c_at_zero(), epsilon = 1e-12);
    }

    #[test]
    fn spacetime_collocated_off_diagonal_has_no_jitter() {
        let cov = SpaceTimePairwiseCovariance::new(GeoMetric, st_variogram());
        let p = cov.prepare(SpaceTimeCoord::new(
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            1.0,
        ));
        assert_relative_eq!(cov.covariance(p, p), cov.cov_at_zero(), epsilon = 1e-12);
        assert_relative_eq!(
            cov.diagonal(),
            cov.covariance(p, p) + cov.jitter(),
            epsilon = 1e-12
        );
    }
}
