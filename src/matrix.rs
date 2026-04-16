//! Dense linear-system solver used by the kriging models.
//!
//! The kriging system has the Lagrangian border
//!
//! ```text
//! [ C   1 ]
//! [ 1ᵀ  0 ]
//! ```
//!
//! which is symmetric but **indefinite**, so Cholesky is not applicable. We therefore
//! rely on a partial-pivoted LU factorization.

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::error::KrigingError;

/// Solve `a · x = b` using partial-pivoted LU factorization.
///
/// Kriging systems with a Lagrangian border are symmetric but not positive definite,
/// so this solver deliberately skips Cholesky. Returns [`KrigingError::MatrixError`]
/// when the system is singular or non-square.
#[allow(dead_code)] // Kept available for downstream crates that pull this crate as a path
// dependency; the crate-private visibility prevents public-API drift but we still
// want the module itself to compile clean when linked.
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

    a.clone().lu().solve(b).ok_or_else(|| {
        KrigingError::MatrixError("could not solve linear system (singular matrix)".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_indefinite_kriging_style_border_matrix() {
        // Symmetric indefinite system of ordinary-kriging shape: PSD block C plus
        // Lagrangian border of ones with a zero in the bottom-right corner. Cholesky
        // fails on this pattern; the LU path must succeed.
        let a = DMatrix::from_row_slice(
            3,
            3,
            &[
                2.0, 0.5, 1.0, //
                0.5, 2.0, 1.0, //
                1.0, 1.0, 0.0, //
            ],
        );
        let x_expected = DVector::from_column_slice(&[0.3, 0.7, 0.25]);
        let b = &a * &x_expected;

        let x = solve_linear_system(&a, &b).expect("solve should succeed");
        for (got, want) in x.iter().zip(x_expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn rejects_singular_matrix() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let b = DVector::from_column_slice(&[3.0, 6.0]);
        let err = solve_linear_system(&a, &b).expect_err("singular system must fail");
        match err {
            KrigingError::MatrixError(_) => {}
            other => panic!("expected MatrixError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_square_matrix() {
        let a = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b = DVector::from_column_slice(&[1.0, 1.0]);
        let err = solve_linear_system(&a, &b).expect_err("non-square must fail");
        match err {
            KrigingError::MatrixError(_) => {}
            other => panic!("expected MatrixError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_shape_mismatch() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let err = solve_linear_system(&a, &b).expect_err("shape mismatch must fail");
        match err {
            KrigingError::DimensionMismatch(_) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }
}
