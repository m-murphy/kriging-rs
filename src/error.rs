use crate::Real;
use thiserror::Error;

/// Errors returned by kriging, variogram, and dataset operations in this crate.
#[derive(Error, Debug)]
pub enum KrigingError {
    #[error("Insufficient data: need at least {0} observations")]
    InsufficientData(usize),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Invalid coordinate: lat={lat}, lon={lon}")]
    InvalidCoordinate { lat: Real, lon: Real },
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),
    #[error("Variogram fitting failed: {0}")]
    FittingError(String),
    #[error("Invalid binomial data: {0}")]
    InvalidBinomialData(String),
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),
}
