//! Coordinate–value datasets for kriging (pairs of locations and observed values).
//!
//! A [`GeoDataset`] pairs a list of [`GeoCoord`] with a list of observed values. The
//! constructor enforces the two invariants every downstream solver relies on:
//!
//! 1. `coords.len() == values.len()` (returns [`KrigingError::DimensionMismatch`] otherwise).
//! 2. At least two observations are present (returns [`KrigingError::InsufficientData`]
//!    otherwise — one point cannot define any covariance structure).
//!
//! Empty inputs (`coords.len() == 0`) are rejected under the same [`InsufficientData`]
//! branch. Upstream modules (`compute_empirical_variogram`, `fit_variogram`, kriging model
//! constructors) rely on this invariant and will similarly return a clear error rather than
//! panic if given an empty or single-point dataset.

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
    /// Builds a dataset.
    ///
    /// Errors:
    /// - [`KrigingError::DimensionMismatch`] when `coords.len() != values.len()`.
    /// - [`KrigingError::InsufficientData`] when there are fewer than two points, **including
    ///   the empty-input case**.
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
