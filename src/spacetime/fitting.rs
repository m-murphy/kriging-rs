//! Fit parametric space–time variograms to an empirical ST variogram.
//!
//! Two families are supported:
//!
//! - [`SpaceTimeVariogramType::Separable`] — independently fit the spatial marginal
//!   (smallest-`h_t` row) and the temporal marginal (smallest-`h_s` column) with the
//!   existing 1-D [`fit_variogram`] and combine them into
//!   [`SpaceTimeVariogram::Separable`].
//! - [`SpaceTimeVariogramType::ProductSum`] — fit the two marginals the same way, then
//!   solve a 3-variable non-negative least-squares problem for the `(k1, k2, k3)`
//!   coefficients in `C(h_s, h_t) = k1·C_s(h_s)·C_t(h_t) + k2·C_s(h_s) + k3·C_t(h_t)`.
//!
//! The marginal model types are supplied by the caller via [`SpaceTimeFitConfig`].

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::error::KrigingError;
use crate::spacetime::empirical::{
    EmpiricalSpaceTimeVariogram, spatial_marginal, temporal_marginal,
};
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::variogram::fitting::fit_variogram;
use crate::variogram::models::{VariogramModel, VariogramType};

/// Which space–time variogram family to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceTimeVariogramType {
    Separable,
    ProductSum,
}

/// Input to [`fit_spacetime_variogram`]: which ST family to fit and which 1-D variogram
/// types to use for the spatial and temporal marginals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceTimeFitConfig {
    pub family: SpaceTimeVariogramType,
    pub spatial_model: VariogramType,
    pub temporal_model: VariogramType,
}

/// Result of [`fit_spacetime_variogram`]: the fitted model and the residual sum-of-squares
/// across non-empty 2-D bins, weighted by `n_pairs`.
#[derive(Debug, Clone, Copy)]
pub struct SpaceTimeFitResult {
    pub model: SpaceTimeVariogram,
    pub residuals: Real,
}

/// Fit a space–time variogram sequentially: marginals first, then (for product-sum) the
/// three mixing coefficients via constrained linear least squares.
pub fn fit_spacetime_variogram(
    empirical: &EmpiricalSpaceTimeVariogram,
    config: SpaceTimeFitConfig,
) -> Result<SpaceTimeFitResult, KrigingError> {
    let spatial_marg = spatial_marginal(empirical);
    let temporal_marg = temporal_marginal(empirical);
    if spatial_marg.distances.is_empty() || temporal_marg.distances.is_empty() {
        return Err(KrigingError::FittingError(
            "empirical ST variogram lacks a populated spatial or temporal marginal".to_string(),
        ));
    }
    let spatial_fit = fit_variogram(&spatial_marg, config.spatial_model)?;
    let temporal_fit = fit_variogram(&temporal_marg, config.temporal_model)?;

    match config.family {
        SpaceTimeVariogramType::Separable => {
            let model = SpaceTimeVariogram::new_separable(spatial_fit.model, temporal_fit.model)?;
            let residuals = weighted_residuals(empirical, model);
            Ok(SpaceTimeFitResult { model, residuals })
        }
        SpaceTimeVariogramType::ProductSum => {
            let (k1, k2, k3) =
                fit_product_sum_coefficients(empirical, spatial_fit.model, temporal_fit.model)?;
            let model = SpaceTimeVariogram::new_product_sum(
                spatial_fit.model,
                temporal_fit.model,
                k1,
                k2,
                k3,
            )?;
            let residuals = weighted_residuals(empirical, model);
            Ok(SpaceTimeFitResult { model, residuals })
        }
    }
}

fn weighted_residuals(emp: &EmpiricalSpaceTimeVariogram, model: SpaceTimeVariogram) -> Real {
    let mut sum: Real = 0.0;
    for i in 0..emp.n_spatial_bins {
        for j in 0..emp.n_temporal_bins {
            let idx = emp.index(i, j);
            let n = emp.n_pairs[idx];
            if n == 0 {
                continue;
            }
            let diff = emp.semivariances[idx]
                - model.semivariance(emp.spatial_lags[idx], emp.temporal_lags[idx]);
            sum += (n as Real) * diff * diff;
        }
    }
    sum
}

/// Solve the weighted linear least-squares problem
///   `γ̂(h_s, h_t) = k1·f1 + k2·f2 + k3·f3`
/// with
///   `f1 = C_s(0)C_t(0) − C_s(h_s)C_t(h_t)`,
///   `f2 = C_s(0) − C_s(h_s)`,
///   `f3 = C_t(0) − C_t(h_t)`,
/// under `k_i ≥ 0` and weights `sqrt(n_pairs)`. Uses unconstrained LS followed by a simple
/// projected re-solve on the active set (sufficient for three unknowns).
fn fit_product_sum_coefficients(
    emp: &EmpiricalSpaceTimeVariogram,
    spatial: VariogramModel,
    temporal: VariogramModel,
) -> Result<(Real, Real, Real), KrigingError> {
    let cs0 = spatial.covariance(0.0);
    let ct0 = temporal.covariance(0.0);

    let mut rows = Vec::new();
    let mut targets = Vec::new();
    let mut weights = Vec::new();
    for i in 0..emp.n_spatial_bins {
        for j in 0..emp.n_temporal_bins {
            let idx = emp.index(i, j);
            let n = emp.n_pairs[idx];
            if n == 0 {
                continue;
            }
            let cs = spatial.covariance(emp.spatial_lags[idx]);
            let ct = temporal.covariance(emp.temporal_lags[idx]);
            let f1 = cs0 * ct0 - cs * ct;
            let f2 = cs0 - cs;
            let f3 = ct0 - ct;
            rows.push([f1, f2, f3]);
            targets.push(emp.semivariances[idx]);
            weights.push((n as Real).sqrt());
        }
    }
    if rows.len() < 3 {
        return Err(KrigingError::FittingError(
            "need at least 3 populated bins to fit product-sum coefficients".to_string(),
        ));
    }

    let k_all = solve_nnls_3(&rows, &targets, &weights);
    Ok((k_all[0], k_all[1], k_all[2]))
}

/// Weighted NNLS for exactly three unknowns via enumeration of the eight active sets
/// (each coefficient is either free or pinned to 0). For each active set we solve the
/// reduced unconstrained normal equations; the candidate with the smallest weighted SSE
/// that also satisfies non-negativity is returned. Robust and deterministic for the tiny
/// fixed dimensionality we need.
fn solve_nnls_3(rows: &[[Real; 3]], targets: &[Real], weights: &[Real]) -> [Real; 3] {
    let m = rows.len();
    let mut best: Option<([Real; 3], Real)> = None;
    for mask in 0u8..8 {
        let free: Vec<usize> = (0..3).filter(|k| (mask >> k) & 1 == 1).collect();
        if free.is_empty() {
            // All zero; evaluate for completeness (residual = Σ w² y²).
            let residuals = weighted_sse(rows, targets, weights, &[0.0, 0.0, 0.0]);
            match best {
                None => best = Some(([0.0, 0.0, 0.0], residuals)),
                Some((_, r)) if residuals < r => best = Some(([0.0, 0.0, 0.0], residuals)),
                _ => {}
            }
            continue;
        }
        let p = free.len();
        let mut a = DMatrix::<Real>::zeros(m, p);
        let mut b = DVector::<Real>::zeros(m);
        for (r, row) in rows.iter().enumerate() {
            let w = weights[r];
            for (c, &k) in free.iter().enumerate() {
                a[(r, c)] = w * row[k];
            }
            b[r] = w * targets[r];
        }
        let ata = a.transpose() * &a;
        let atb = a.transpose() * &b;
        let Some(sol) = ata.lu().solve(&atb) else {
            continue;
        };
        if sol.iter().any(|v| !v.is_finite() || *v < 0.0) {
            continue;
        }
        let mut full = [0.0 as Real; 3];
        for (c, &k) in free.iter().enumerate() {
            full[k] = sol[c];
        }
        let residuals = weighted_sse(rows, targets, weights, &full);
        match best {
            None => best = Some((full, residuals)),
            Some((_, r)) if residuals < r => best = Some((full, residuals)),
            _ => {}
        }
    }
    best.map(|(k, _)| k).unwrap_or([0.0, 0.0, 0.0])
}

fn weighted_sse(rows: &[[Real; 3]], targets: &[Real], weights: &[Real], k: &[Real; 3]) -> Real {
    rows.iter()
        .zip(targets.iter())
        .zip(weights.iter())
        .map(|((row, y), w)| {
            let fit = k[0] * row[0] + k[1] * row[1] + k[2] * row[2];
            let d = (*y - fit) * w;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::SpaceTimeCoord;
    use crate::spacetime::dataset::SpaceTimeDataset;
    use crate::spacetime::empirical::{
        SpaceTimeVariogramConfig, compute_empirical_spacetime_variogram,
    };
    use crate::spacetime::metric::GeoMetric;
    use std::num::NonZeroUsize;

    fn synthetic_dataset(
        true_model: SpaceTimeVariogram,
        n_spatial: usize,
        n_time: usize,
    ) -> SpaceTimeDataset<GeoCoord> {
        // Generate a dataset whose empirical variogram matches `true_model` in expectation by
        // drawing values from a zero-mean field with the right correlation via Cholesky.
        // For fitting tests we just evaluate the true model analytically: we construct a
        // dataset, compute its empirical variogram, then inject the analytical γ̂ from the
        // true model — but that would trivially pass. Instead, we use a *simple* dataset
        // and check that fit residuals are small (stable monotone surfaces like z = f(t)
        // are already a good sanity test).
        let _ = true_model;
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..n_spatial {
            for t in 0..n_time {
                coords.push(SpaceTimeCoord::new(
                    GeoCoord::try_new(i as Real * 0.1, 0.0).unwrap(),
                    t as Real,
                ));
                // Noiseless separable structure: z = sin(i) + 0.5*cos(t).
                values.push(((i as Real) * 0.7).sin() + 0.5 * ((t as Real) * 0.5).cos());
            }
        }
        SpaceTimeDataset::new(coords, values).unwrap()
    }

    #[test]
    fn separable_fit_returns_positive_marginals() {
        let spatial = VariogramModel::new(0.01, 1.0, 30.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.01, 0.5, 3.0, VariogramType::Exponential).unwrap();
        let truth = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let ds = synthetic_dataset(truth, 8, 6);
        let emp = compute_empirical_spacetime_variogram(
            &GeoMetric,
            &ds,
            &SpaceTimeVariogramConfig {
                n_spatial_bins: NonZeroUsize::new(5).unwrap(),
                n_temporal_bins: NonZeroUsize::new(5).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let fit = fit_spacetime_variogram(
            &emp,
            SpaceTimeFitConfig {
                family: SpaceTimeVariogramType::Separable,
                spatial_model: VariogramType::Exponential,
                temporal_model: VariogramType::Exponential,
            },
        )
        .unwrap();
        let (n_s, s_s, r_s) = fit.model.spatial().params();
        let (n_t, s_t, r_t) = fit.model.temporal().params();
        assert!(n_s >= 0.0 && s_s > n_s && r_s > 0.0);
        assert!(n_t >= 0.0 && s_t > n_t && r_t > 0.0);
        assert!(fit.residuals.is_finite());
    }

    #[test]
    fn product_sum_coefficients_are_non_negative() {
        let ds = synthetic_dataset(
            SpaceTimeVariogram::new_separable(
                VariogramModel::new(0.01, 1.0, 20.0, VariogramType::Exponential).unwrap(),
                VariogramModel::new(0.01, 1.0, 3.0, VariogramType::Exponential).unwrap(),
            )
            .unwrap(),
            10,
            8,
        );
        let emp = compute_empirical_spacetime_variogram(
            &GeoMetric,
            &ds,
            &SpaceTimeVariogramConfig {
                n_spatial_bins: NonZeroUsize::new(6).unwrap(),
                n_temporal_bins: NonZeroUsize::new(6).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let fit = fit_spacetime_variogram(
            &emp,
            SpaceTimeFitConfig {
                family: SpaceTimeVariogramType::ProductSum,
                spatial_model: VariogramType::Exponential,
                temporal_model: VariogramType::Exponential,
            },
        )
        .unwrap();
        match fit.model {
            SpaceTimeVariogram::ProductSum { k1, k2, k3, .. } => {
                assert!(k1 >= 0.0 && k2 >= 0.0 && k3 >= 0.0);
                assert!(k1 + k2 + k3 > 0.0);
            }
            _ => panic!("expected ProductSum"),
        }
        assert!(fit.residuals.is_finite());
    }

    #[test]
    fn product_sum_fit_is_not_worse_than_separable_fit() {
        // The product-sum family is strictly more flexible than the separable family
        // (separable ≡ (k1, k2, k3) = (1, 0, 0)), so weighted residuals should be ≤
        // the pure separable residuals on the same empirical variogram.
        let ds = synthetic_dataset(
            SpaceTimeVariogram::new_separable(
                VariogramModel::new(0.05, 1.0, 15.0, VariogramType::Exponential).unwrap(),
                VariogramModel::new(0.05, 1.0, 2.5, VariogramType::Exponential).unwrap(),
            )
            .unwrap(),
            10,
            10,
        );
        let emp = compute_empirical_spacetime_variogram(
            &GeoMetric,
            &ds,
            &SpaceTimeVariogramConfig {
                n_spatial_bins: NonZeroUsize::new(6).unwrap(),
                n_temporal_bins: NonZeroUsize::new(6).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let sep = fit_spacetime_variogram(
            &emp,
            SpaceTimeFitConfig {
                family: SpaceTimeVariogramType::Separable,
                spatial_model: VariogramType::Exponential,
                temporal_model: VariogramType::Exponential,
            },
        )
        .unwrap();
        let ps = fit_spacetime_variogram(
            &emp,
            SpaceTimeFitConfig {
                family: SpaceTimeVariogramType::ProductSum,
                spatial_model: VariogramType::Exponential,
                temporal_model: VariogramType::Exponential,
            },
        )
        .unwrap();
        assert!(
            ps.residuals <= sep.residuals * 1.0001,
            "product-sum residuals {} should not exceed separable residuals {}",
            ps.residuals,
            sep.residuals
        );
    }

    #[test]
    fn fitting_rejects_empty_marginal() {
        let emp = EmpiricalSpaceTimeVariogram {
            n_spatial_bins: 3,
            n_temporal_bins: 3,
            spatial_lags: vec![0.0; 9],
            temporal_lags: vec![0.0; 9],
            semivariances: vec![0.0; 9],
            n_pairs: vec![0; 9],
        };
        assert!(
            fit_spacetime_variogram(
                &emp,
                SpaceTimeFitConfig {
                    family: SpaceTimeVariogramType::Separable,
                    spatial_model: VariogramType::Exponential,
                    temporal_model: VariogramType::Exponential,
                },
            )
            .is_err()
        );
    }
}
