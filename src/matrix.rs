use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::error::KrigingError;

pub fn solve_linear_system(
    a: &DMatrix<Real>,
    b: &DVector<Real>,
) -> Result<DVector<Real>, KrigingError> {
    if a.nrows() != a.ncols() {
        return Err(KrigingError::MatrixError(
            "matrix must be square".to_string(),
        ));
    }
    if a.nrows() != b.nrows() {
        return Err(KrigingError::DimensionMismatch(
            "matrix/vector shape mismatch".to_string(),
        ));
    }

    if let Some(ch) = a.clone().cholesky() {
        return Ok(ch.solve(b));
    }
    if let Some(sol) = a.clone().lu().solve(b) {
        return Ok(sol);
    }

    Err(KrigingError::MatrixError(
        "could not solve linear system".to_string(),
    ))
}
