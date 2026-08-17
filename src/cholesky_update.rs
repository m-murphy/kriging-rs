//! Cholesky maintenance for **symmetric positive definite** (SPD) covariance blocks.
//!
//! Production ordinary / universal / binomial kriging solves on the SPD covariance block `C`
//! via Cholesky (ADR-0001). Bordered LU remains in [`crate::kriging::numerics`] for regression
//! tests only. SPD blocks satisfy `A = L Lᵀ` and support cheap updates:
//!
//! - **Rank-1 update** `A ← A + x xᵀ` with [`cholesky_rank1_update_lower_inplace`].
//! - **Bordered extension** when one observation is appended: `A` grows by one row/column;
//!   given existing `L`, the new factor is obtained with one forward solve and a scalar
//!   square root ([`cholesky_extend_spd_lower`]).
//! - **Row/column deletion** for LOO CV: [`cholesky_delete_index`] (Krause & Igel 2015 rank-one
//!   downdate, O(n²)).
//!
//! All routines assume `L` is **lower triangular** (`L[i,j] = 0` for `i < j`) with
//! `A = L · Lᵀ`. Garbage in the strict upper triangle is ignored.
//!
//! Used by [`crate::kriging::engine::OrdinaryKrigingEngine::condition`] for incremental SGS.

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::error::KrigingError;

/// Solve `L · x = b` where `L` is lower triangular (only entries `i ≥ j` are read).
pub(crate) fn forward_solve_lower(
    l: &DMatrix<Real>,
    b: &DVector<Real>,
) -> Result<DVector<Real>, KrigingError> {
    let n = l.nrows();
    if l.ncols() != n || b.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "forward_solve_lower: shape mismatch".to_string(),
        ));
    }
    let mut x = DVector::zeros(n);
    for i in 0..n {
        let diag = l[(i, i)];
        if diag.abs() < Real::EPSILON {
            return Err(KrigingError::MatrixError(
                "forward_solve_lower: near-zero diagonal".to_string(),
            ));
        }
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * x[k];
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Extend an SPD Cholesky factor when the matrix gains one row/column:
///
/// ```text
/// A_new = [ A      v ]
///         [ vᵀ     α ]
/// ```
///
/// with `A = L Lᵀ`. Returns `L_new` with `A_new = L_new L_newᵀ`.
#[allow(dead_code)] // engine::condition; simulation.rs next
pub(crate) fn cholesky_extend_spd_lower(
    l: &DMatrix<Real>,
    cross_cov: &DVector<Real>,
    new_diag: Real,
) -> Result<DMatrix<Real>, KrigingError> {
    let n = l.nrows();
    if l.ncols() != n {
        return Err(KrigingError::MatrixError(
            "cholesky_extend_spd_lower: L must be square".to_string(),
        ));
    }
    if cross_cov.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "cholesky_extend_spd_lower: cross_cov length must match L order".to_string(),
        ));
    }
    let w = forward_solve_lower(l, cross_cov)?;
    let schur = new_diag - w.dot(&w);
    if schur <= Real::EPSILON {
        return Err(KrigingError::MatrixError(
            "cholesky_extend_spd_lower: Schur complement not positive".to_string(),
        ));
    }
    let ell = schur.sqrt();
    let mut l_new = DMatrix::zeros(n + 1, n + 1);
    l_new.view_mut((0, 0), (n, n)).copy_from(l);
    for i in 0..n {
        l_new[(n, i)] = w[i];
    }
    l_new[(n, n)] = ell;
    Ok(l_new)
}

fn zero_strict_upper(l: &mut DMatrix<Real>) {
    let n = l.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            l[(i, j)] = 0.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_submat(
    src: &DMatrix<Real>,
    src_row: usize,
    src_col: usize,
    dst: &mut DMatrix<Real>,
    dst_row: usize,
    dst_col: usize,
    nrows: usize,
    ncols: usize,
) {
    for c in 0..ncols {
        for r in 0..nrows {
            dst[(dst_row + r, dst_col + c)] = src[(src_row + r, src_col + c)];
        }
    }
}

/// Rank-one Cholesky downdate after deleting row/column `del` from `C = L Lᵀ`.
///
/// Krause & Igel (2015), FOGA — O(n²). `w_scratch` must hold at least `n` elements.
fn chol_row_del_update_lower(
    l: &DMatrix<Real>,
    del: usize,
    w_scratch: &mut [Real],
) -> Result<DMatrix<Real>, KrigingError> {
    let n = l.nrows();
    if n != l.ncols() {
        return Err(KrigingError::MatrixError(
            "chol_row_del_update_lower: L must be square".to_string(),
        ));
    }
    if n < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    if del >= n {
        return Err(KrigingError::DimensionMismatch(
            "chol_row_del_update_lower: index out of range".to_string(),
        ));
    }
    if w_scratch.len() < n {
        return Err(KrigingError::DimensionMismatch(
            "chol_row_del_update_lower: scratch too short".to_string(),
        ));
    }

    let n1 = n - 1;
    let mut l1 = DMatrix::zeros(n1, n1);

    if del == n - 1 {
        copy_submat(l, 0, 0, &mut l1, 0, 0, n1, n1);
        zero_strict_upper(&mut l1);
        return Ok(l1);
    }

    if del == 0 {
        let nk = n1;
        let del_plus_one = 1;
        for i in 0..nk {
            w_scratch[i] = l[(i + 1, 0)];
        }
        let mut b = 1.0 as Real;

        for j in 0..nk {
            let l_jj = l[(del_plus_one + j, del_plus_one + j)];
            if l_jj.abs() <= Real::EPSILON {
                return Err(KrigingError::MatrixError(
                    "chol_row_del_update_lower: degenerate pivot".to_string(),
                ));
            }
            let w_j = w_scratch[j];
            let gamma = l_jj * l_jj * b + w_j * w_j;
            let new_diag = (l_jj * l_jj + w_j * w_j / b).sqrt();
            l1[(j, j)] = new_diag;

            if j < nk - 1 {
                for k in (j + 1)..nk {
                    let l_kj = l[(del_plus_one + k, del_plus_one + j)];
                    let ratio = l_kj / l_jj;
                    w_scratch[k] -= ratio * w_j;
                    let coeff = ratio + (w_j * w_scratch[k]) / gamma;
                    l1[(k, j)] = coeff * new_diag;
                }
                b += w_j * w_j / (l_jj * l_jj);
            }
            zero_strict_upper(&mut l1);
        }
        return Ok(l1);
    }

    let del_plus_one = del + 1;
    let nk = n - del_plus_one;

    copy_submat(l, 0, 0, &mut l1, 0, 0, del, del);
    copy_submat(l, del_plus_one, 0, &mut l1, del, 0, nk, del);

    for i in 0..nk {
        w_scratch[i] = l[(del_plus_one + i, del)];
    }
    let mut b = 1.0 as Real;

    for j in 0..nk {
        let l_jj = l[(del_plus_one + j, del_plus_one + j)];
        if l_jj.abs() <= Real::EPSILON {
            return Err(KrigingError::MatrixError(
                "chol_row_del_update_lower: degenerate pivot".to_string(),
            ));
        }
        let w_j = w_scratch[j];
        let gamma = l_jj * l_jj * b + w_j * w_j;
        let new_diag = (l_jj * l_jj + w_j * w_j / b).sqrt();
        l1[(del + j, del + j)] = new_diag;

        if j < nk - 1 {
            for k in (j + 1)..nk {
                let l_kj = l[(del_plus_one + k, del_plus_one + j)];
                let ratio = l_kj / l_jj;
                w_scratch[k] -= ratio * w_j;
                let coeff = ratio + (w_j * w_scratch[k]) / gamma;
                l1[(del + k, del + j)] = coeff * new_diag;
            }
            b += w_j * w_j / (l_jj * l_jj);
        }
    }
    zero_strict_upper(&mut l1);
    Ok(l1)
}

/// Delete row/column `del` from an SPD Cholesky factor (lower `L` with `C = L Lᵀ`). O(n²).
pub(crate) fn cholesky_delete_index(
    l: &DMatrix<Real>,
    del: usize,
) -> Result<DMatrix<Real>, KrigingError> {
    if l.nrows() != l.ncols() {
        return Err(KrigingError::MatrixError(
            "cholesky_delete_index: L must be square".to_string(),
        ));
    }
    let n = l.nrows();
    let mut w_scratch = vec![0.0 as Real; n];
    chol_row_del_update_lower(l, del, &mut w_scratch)
}

#[cfg(test)]
fn extract_principal_submatrix(m: &DMatrix<Real>, del: usize) -> DMatrix<Real> {
    let n = m.nrows();
    let mut out = DMatrix::zeros(n - 1, n - 1);
    let mut out_i = 0;
    for i in 0..n {
        if i == del {
            continue;
        }
        let mut out_j = 0;
        for j in 0..n {
            if j == del {
                continue;
            }
            out[(out_i, out_j)] = m[(i, j)];
            out_j += 1;
        }
        out_i += 1;
    }
    out
}

#[cfg(test)]
fn cholesky_delete_by_refactor(
    l: &DMatrix<Real>,
    del: usize,
) -> Result<DMatrix<Real>, KrigingError> {
    let c = l * l.transpose();
    let c_red = extract_principal_submatrix(&c, del);
    let chol = c_red
        .cholesky()
        .ok_or_else(|| KrigingError::MatrixError("refactor delete failed".to_string()))?;
    Ok(chol.l().clone())
}

/// Rank-1 Cholesky **update**: if `A = L Lᵀ` then after this call `A' ≈ L' L'ᵀ` for
/// `A' = A + x xᵀ`. On input, `x` is the update vector; it is overwritten with scratch values.
///
/// Reference: Golub & Van Loan, *Matrix Computations* (stabilized `hypot`-style step is
/// expressed here via `sqrt` on sums of squares).
#[allow(dead_code)]
pub(crate) fn cholesky_rank1_update_lower_inplace(l: &mut DMatrix<Real>, x: &mut DVector<Real>) {
    let n = l.nrows();
    debug_assert_eq!(l.ncols(), n);
    debug_assert_eq!(x.len(), n);
    for i in 0..n {
        let l_ii = l[(i, i)];
        let xi = x[i];
        let r = (l_ii * l_ii + xi * xi).sqrt();
        if r < Real::EPSILON {
            continue;
        }
        let c = r / l_ii;
        let s = xi / l_ii;
        l[(i, i)] = r;
        for j in i + 1..n {
            let l_ji = l[(j, i)];
            let x_j = x[j];
            l[(j, i)] = (l_ji + s * x_j) / c;
            x[j] = c * x_j - s * l_ji;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SymmetricEigen;

    fn spd_3x3() -> DMatrix<Real> {
        DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0])
    }

    fn reconstruct(l: &DMatrix<Real>) -> DMatrix<Real> {
        l * l.transpose()
    }

    #[test]
    fn extend_matches_full_cholesky() {
        let a = spd_3x3();
        let chol = a.clone().cholesky().expect("SPD");
        let l = chol.l();
        let v = DVector::from_column_slice(&[0.75, -0.5, 0.25]);
        let alpha = 2.5 as Real;
        let mut a_ext = DMatrix::zeros(4, 4);
        a_ext.view_mut((0, 0), (3, 3)).copy_from(&a);
        for i in 0..3 {
            a_ext[(i, 3)] = v[i];
            a_ext[(3, i)] = v[i];
        }
        a_ext[(3, 3)] = alpha;
        let l_ext = cholesky_extend_spd_lower(&l, &v, alpha).expect("extend");
        let got = reconstruct(&l_ext);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (got[(i, j)] - a_ext[(i, j)]).abs() < 1e-2,
                    "({i},{j}) got {} want {}",
                    got[(i, j)],
                    a_ext[(i, j)]
                );
            }
        }
    }

    #[test]
    fn delete_index_matches_recomputed_submatrix() {
        let a = spd_3x3();
        let l = a.clone().cholesky().unwrap().l();
        for del in 0..3 {
            let l_del = cholesky_delete_index(&l, del).expect("delete");
            let a_sub = extract_principal_submatrix(&a, del);
            let got = reconstruct(&l_del);
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (got[(i, j)] - a_sub[(i, j)]).abs() < 1e-4,
                        "del={del} ({i},{j}) got={} want={}",
                        got[(i, j)],
                        a_sub[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn delete_index_matches_refactor_path() {
        let a = spd_3x3();
        let l = a.cholesky().unwrap().l();
        for del in 0..3 {
            let fast = cholesky_delete_index(&l, del).expect("fast");
            let slow = cholesky_delete_by_refactor(&l, del).expect("refactor");
            let diff = (&fast - &slow).norm();
            assert!(diff < 1e-4, "del={del} factor diff={diff}");
            let cov_diff = (reconstruct(&fast) - reconstruct(&slow)).norm();
            assert!(cov_diff < 1e-4, "del={del} cov diff={cov_diff}");
        }
    }

    #[test]
    fn delete_index_random_spd_matches_refactor() {
        let n = 16_usize;
        let mut a = DMatrix::<Real>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = (1.0 + (i * 17 + j * 31) as Real * 0.001).sin() * 0.05;
                if i == j {
                    a[(i, j)] += n as Real;
                }
            }
        }
        a = &a * a.transpose() + DMatrix::identity(n, n);
        let l = a.clone().cholesky().unwrap().l();
        for del in 0..n {
            let fast = cholesky_delete_index(&l, del).expect("fast");
            let slow = cholesky_delete_by_refactor(&l, del).expect("refactor");
            let cov_diff = (reconstruct(&fast) - reconstruct(&slow)).norm();
            assert!(cov_diff < 1e-3, "n={n} del={del} cov diff={cov_diff}");
        }
    }

    #[test]
    fn extend_rejects_indefinite_schur() {
        let a = spd_3x3();
        let l = a.cholesky().unwrap().l();
        let v = DVector::from_column_slice(&[10.0 as Real, 10.0, 10.0]);
        let alpha = 1.0 as Real;
        assert!(cholesky_extend_spd_lower(&l, &v, alpha).is_err());
    }

    #[test]
    fn rank1_update_matches_naive() {
        let a = spd_3x3();
        let chol = a.clone().cholesky().expect("SPD");
        let l = chol.l();
        let x = DVector::from_column_slice(&[0.3 as Real, -0.2, 0.1]);
        let mut l_copy = l.clone();
        let mut x_copy = x.clone();
        cholesky_rank1_update_lower_inplace(&mut l_copy, &mut x_copy);
        let updated = &a + &x * x.transpose();
        let got = reconstruct(&l_copy);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (got[(i, j)] - updated[(i, j)]).abs() < 1e-2,
                    "({i},{j}) got {} want {}",
                    got[(i, j)],
                    updated[(i, j)]
                );
            }
        }
        // Original L still valid for A
        let _ = reconstruct(&l);
    }

    #[test]
    fn rank1_update_preserves_spd_random() {
        use nalgebra::DVector as NV;
        let n = 12_usize;
        let mut a = DMatrix::<Real>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = (1.0 + (i * 17 + j * 31) as Real * 0.001).sin() * 0.1;
                if i == j {
                    a[(i, j)] += n as Real;
                }
            }
        }
        a = &a * a.transpose() + DMatrix::identity(n, n);
        let l = a.clone().cholesky().unwrap().l();
        let x = NV::from_iterator(n, (0..n).map(|i| 0.05 * (i as Real + 1.0).sin()));
        let mut l2 = l.clone();
        let mut x2 = x.clone();
        cholesky_rank1_update_lower_inplace(&mut l2, &mut x2);
        let approx = reconstruct(&l2);
        let truth = &a + &x * x.transpose();
        let diff = (&approx - &truth).norm();
        assert!(diff < 1e-2, "diff={diff}");
        let se = SymmetricEigen::new(approx.clone());
        assert!(
            se.eigenvalues.iter().all(|&ev| ev > -1e-3),
            "min eig {:?}",
            se.eigenvalues
        );
    }
}
