//! Space–time coordinate/value dataset.

use crate::Real;
use crate::error::KrigingError;
use crate::spacetime::coord::SpaceTimeCoord;

/// Space–time coord–value pairs with matching length and at least two points, enforced at
/// construction. Mirrors [`GeoDataset`](crate::GeoDataset) but carries `(spatial, time)`
/// coordinates.
#[derive(Debug, Clone)]
pub struct SpaceTimeDataset<C> {
    coords: Vec<SpaceTimeCoord<C>>,
    values: Vec<Real>,
}

impl<C: Copy> SpaceTimeDataset<C> {
    /// Build a dataset.
    ///
    /// Errors:
    /// - [`KrigingError::DimensionMismatch`] when `coords.len() != values.len()`.
    /// - [`KrigingError::InsufficientData`] when there are fewer than two points.
    /// - [`KrigingError::InvalidInput`] when any value or time is not finite.
    pub fn new(coords: Vec<SpaceTimeCoord<C>>, values: Vec<Real>) -> Result<Self, KrigingError> {
        if coords.len() != values.len() {
            return Err(KrigingError::DimensionMismatch(
                "coords and values length must match".to_string(),
            ));
        }
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(format!(
                "value at index {bad} is not finite"
            )));
        }
        if let Some(bad) = coords.iter().position(|c| !c.time.is_finite()) {
            return Err(KrigingError::InvalidInput(format!(
                "time at index {bad} is not finite"
            )));
        }
        Ok(Self { coords, values })
    }

    #[inline]
    pub fn coords(&self) -> &[SpaceTimeCoord<C>] {
        &self.coords
    }

    #[inline]
    pub fn values(&self) -> &[Real] {
        &self.values
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Consume the dataset and return the component vectors.
    pub fn into_parts(self) -> (Vec<SpaceTimeCoord<C>>, Vec<Real>) {
        (self.coords, self.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;

    fn coord(lat: Real, lon: Real, t: Real) -> SpaceTimeCoord<GeoCoord> {
        SpaceTimeCoord::new(GeoCoord::try_new(lat, lon).unwrap(), t)
    }

    #[test]
    fn rejects_length_mismatch() {
        let coords = vec![coord(0.0, 0.0, 0.0), coord(0.0, 1.0, 1.0)];
        let values = vec![1.0];
        assert!(matches!(
            SpaceTimeDataset::new(coords, values),
            Err(KrigingError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn rejects_fewer_than_two_points() {
        let coords = vec![coord(0.0, 0.0, 0.0)];
        let values = vec![1.0];
        assert!(matches!(
            SpaceTimeDataset::new(coords, values),
            Err(KrigingError::InsufficientData(2))
        ));
    }

    #[test]
    fn rejects_non_finite_values() {
        let coords = vec![coord(0.0, 0.0, 0.0), coord(0.0, 1.0, 1.0)];
        let values = vec![1.0, Real::NAN];
        assert!(matches!(
            SpaceTimeDataset::new(coords, values),
            Err(KrigingError::InvalidInput(_))
        ));
    }

    #[test]
    fn accepts_valid_input() {
        let coords = vec![coord(0.0, 0.0, 0.0), coord(0.0, 1.0, 1.0)];
        let values = vec![1.0, 2.0];
        let ds = SpaceTimeDataset::new(coords, values).unwrap();
        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
    }
}
