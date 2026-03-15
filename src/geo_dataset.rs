//! Coordinate–value datasets for kriging (pairs of locations and observed values).

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;

/// Coord–value pairs with matching length and at least two points, enforced at construction.
#[derive(Debug, Clone)]
pub struct GeoDataset {
    coords: Vec<GeoCoord>,
    values: Vec<Real>,
}

impl GeoDataset {
    /// Builds a dataset. Fails if `coords.len() != values.len()` or if there are fewer than two points.
    pub fn new(coords: Vec<GeoCoord>, values: Vec<Real>) -> Result<Self, KrigingError> {
        if coords.len() != values.len() {
            return Err(KrigingError::DimensionMismatch(
                "coords and values length must match".to_string(),
            ));
        }
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        Ok(Self { coords, values })
    }

    #[inline]
    pub fn coords(&self) -> &[GeoCoord] {
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

    /// Consumes the dataset and returns coords and values.
    pub fn into_parts(self) -> (Vec<GeoCoord>, Vec<Real>) {
        (self.coords, self.values)
    }
}
