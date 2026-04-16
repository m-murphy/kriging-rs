//! Space–time coordinate: pairs a spatial coordinate with a scalar time index.

use crate::Real;
use crate::error::KrigingError;

/// A location in space and time. `spatial` is any coordinate type recognized by a
/// [`SpatialMetric`](crate::spacetime::SpatialMetric); `time` is a scalar in whatever unit
/// the temporal variogram's range is expressed in (seconds, days, fractional years, etc.).
///
/// `time` must be finite. Construct with [`SpaceTimeCoord::try_new`] for validated input, or
/// [`SpaceTimeCoord::new`] if `time` is known to be finite already.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceTimeCoord<C> {
    pub spatial: C,
    pub time: Real,
}

impl<C> SpaceTimeCoord<C> {
    /// Pair a spatial coordinate with a finite time index without validation.
    ///
    /// Prefer [`try_new`](Self::try_new) at data-ingestion boundaries; use this constructor
    /// once you already know `time` is finite (e.g. loop indices, clipped timestamps).
    #[inline]
    pub const fn new(spatial: C, time: Real) -> Self {
        Self { spatial, time }
    }

    /// Pair a spatial coordinate with a time index; errors if `time` is `NaN` or `±∞`.
    pub fn try_new(spatial: C, time: Real) -> Result<Self, KrigingError> {
        if !time.is_finite() {
            return Err(KrigingError::InvalidInput(format!(
                "time must be finite, got {time}"
            )));
        }
        Ok(Self { spatial, time })
    }
}

/// Absolute temporal lag `|t_a - t_b|`.
#[inline]
pub(crate) fn temporal_distance(a: Real, b: Real) -> Real {
    (a - b).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;

    #[test]
    fn try_new_accepts_finite_time() {
        let coord = SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.5).unwrap();
        assert_eq!(coord.time, 1.5);
    }

    #[test]
    fn try_new_rejects_nan_and_infinity() {
        let coord = GeoCoord::try_new(0.0, 0.0).unwrap();
        assert!(SpaceTimeCoord::try_new(coord, Real::NAN).is_err());
        assert!(SpaceTimeCoord::try_new(coord, Real::INFINITY).is_err());
        assert!(SpaceTimeCoord::try_new(coord, Real::NEG_INFINITY).is_err());
    }

    #[test]
    fn temporal_distance_is_absolute() {
        assert_eq!(temporal_distance(3.0, 1.0), 2.0);
        assert_eq!(temporal_distance(1.0, 3.0), 2.0);
        assert_eq!(temporal_distance(-2.0, 5.0), 7.0);
    }
}
