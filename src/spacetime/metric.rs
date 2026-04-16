//! Abstraction over spatial distance engines used by space–time kriging.
//!
//! The [`SpatialMetric`] trait pairs a coordinate type with a prepared form and a distance
//! function so that the same generic ST kriging implementation can work over both geographic
//! (Haversine) and projected (Euclidean, optionally anisotropic) spatial data.
//!
//! Two concrete metrics ship with the crate:
//!
//! - [`GeoMetric`] — Haversine distance on [`GeoCoord`] in kilometers.
//! - [`ProjectedMetric`] — Euclidean distance on [`ProjectedCoord`] with optional 2-D
//!   geometric anisotropy.

use crate::Real;
use crate::distance::{GeoCoord, PreparedGeoCoord, haversine_distance_prepared, prepare_geo_coord};
use crate::projected::{Anisotropy2D, ProjectedCoord};

/// Associates a coordinate type with a distance function and a cheap prepared form.
///
/// Implementors must be cheap to `Copy` and safe to share across threads — space–time
/// kriging stores the metric by value and reuses it inside rayon parallel iterators.
pub trait SpatialMetric: Copy + Send + Sync + std::fmt::Debug {
    /// The raw (user-facing) coordinate type.
    type Coord: Copy + Send + Sync + std::fmt::Debug + PartialEq;
    /// A precomputed form used for repeated distance evaluations.
    type Prepared: Copy + Send + Sync + std::fmt::Debug;

    /// Build the prepared form of a coordinate.
    fn prepare(&self, coord: Self::Coord) -> Self::Prepared;

    /// Distance between two prepared coordinates, in the same units as the variogram range.
    fn distance(&self, a: Self::Prepared, b: Self::Prepared) -> Real;
}

/// Geographic metric: Haversine distance on [`GeoCoord`]. Returned distances are in kilometers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GeoMetric;

impl SpatialMetric for GeoMetric {
    type Coord = GeoCoord;
    type Prepared = PreparedGeoCoord;

    #[inline]
    fn prepare(&self, coord: Self::Coord) -> Self::Prepared {
        prepare_geo_coord(coord)
    }

    #[inline]
    fn distance(&self, a: Self::Prepared, b: Self::Prepared) -> Real {
        haversine_distance_prepared(a, b)
    }
}

/// Projected (planar) metric with optional 2-D geometric anisotropy.
///
/// Use [`ProjectedMetric::isotropic`] for plain Euclidean distance, or pass an explicit
/// [`Anisotropy2D`] to model direction-dependent correlation ranges (see the projected
/// module for the rotation + scaling convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedMetric {
    pub anisotropy: Anisotropy2D,
}

impl ProjectedMetric {
    /// Isotropic metric: plain Euclidean distance.
    pub fn isotropic() -> Self {
        Self {
            anisotropy: Anisotropy2D::isotropic(),
        }
    }

    /// Metric with an explicit 2-D anisotropy.
    pub fn with_anisotropy(anisotropy: Anisotropy2D) -> Self {
        Self { anisotropy }
    }
}

impl Default for ProjectedMetric {
    fn default() -> Self {
        Self::isotropic()
    }
}

impl SpatialMetric for ProjectedMetric {
    type Coord = ProjectedCoord;
    type Prepared = ProjectedCoord;

    #[inline]
    fn prepare(&self, coord: Self::Coord) -> Self::Prepared {
        coord
    }

    #[inline]
    fn distance(&self, a: Self::Prepared, b: Self::Prepared) -> Real {
        self.anisotropy.distance(a, b)
    }
}

/// Extracts two scalar components from a spatial coordinate for use in polynomial trend
/// bases. For [`GeoMetric`] these are `(lat, lon)` in degrees; for [`ProjectedMetric`] they
/// are `(x, y)` in the coordinate's linear units.
pub trait SpatialBasis: SpatialMetric {
    fn spatial_components(&self, coord: Self::Coord) -> (Real, Real);
}

impl SpatialBasis for GeoMetric {
    #[inline]
    fn spatial_components(&self, coord: Self::Coord) -> (Real, Real) {
        (coord.lat(), coord.lon())
    }
}

impl SpatialBasis for ProjectedMetric {
    #[inline]
    fn spatial_components(&self, coord: Self::Coord) -> (Real, Real) {
        (coord.x, coord.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projected::ProjectedCoord;
    use approx::assert_relative_eq;

    #[test]
    fn geo_metric_matches_haversine() {
        let m = GeoMetric;
        let a = m.prepare(GeoCoord::try_new(0.0, 0.0).unwrap());
        let b = m.prepare(GeoCoord::try_new(0.0, 1.0).unwrap());
        assert_relative_eq!(m.distance(a, b), 111.1949, epsilon = 0.05);
    }

    #[test]
    fn projected_isotropic_matches_pythagoras() {
        let m = ProjectedMetric::isotropic();
        let a = m.prepare(ProjectedCoord::new(0.0, 0.0));
        let b = m.prepare(ProjectedCoord::new(3.0, 4.0));
        assert_relative_eq!(m.distance(a, b), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn projected_anisotropy_shrinks_along_minor_axis() {
        let aniso = Anisotropy2D::new(0.0, 0.5).unwrap();
        let m = ProjectedMetric::with_anisotropy(aniso);
        let along_major = m.distance(ProjectedCoord::new(0.0, 0.0), ProjectedCoord::new(1.0, 0.0));
        let along_minor = m.distance(ProjectedCoord::new(0.0, 0.0), ProjectedCoord::new(0.0, 1.0));
        assert_relative_eq!(along_major, 1.0, epsilon = 1e-6);
        assert_relative_eq!(along_minor, 2.0, epsilon = 1e-6);
    }
}
