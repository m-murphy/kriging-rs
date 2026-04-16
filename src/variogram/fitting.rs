use crate::Real;
use crate::error::KrigingError;
use crate::variogram::empirical::EmpiricalVariogram;
use crate::variogram::models::{VariogramModel, VariogramType};

#[derive(Debug, Clone)]
pub struct FitResult {
    pub model: VariogramModel,
    pub residuals: Real,
}

fn model_from_params(
    nugget: Real,
    sill: Real,
    range: Real,
    model_type: VariogramType,
    shape: Option<Real>,
) -> VariogramModel {
    match shape {
        None => VariogramModel::new(nugget, sill, range, model_type)
            .expect("grid ensures nugget >= 0, sill > nugget, range > 0"),
        Some(s) => VariogramModel::new_with_shape(nugget, sill, range, model_type, s)
            .expect("grid ensures valid shape for Stable/Matérn"),
    }
}

/// Fits a parametric variogram by minimizing weighted sum of squared residuals over a 5×5×5 grid.
///
/// The empirical variogram must be non-empty and have matching-length distance/semivariance/n_pairs
/// arrays (e.g. from [`compute_empirical_variogram`](crate::compute_empirical_variogram)). Returns
/// [`KrigingError::FittingError`] if these preconditions are violated.
///
/// The grid spans plausible scales around data-derived guesses (sill, range, nugget). Accuracy is
/// limited by grid resolution: the best point may be 20–40% away from the continuous optimum in
/// sill/range. For typical empirical variograms (noisy, few bins) this is usually acceptable; for
/// noiseless synthetic data the grid can pick a different local minimum than the true parameters.
pub fn fit_variogram(
    empirical: &EmpiricalVariogram,
    model_type: VariogramType,
) -> Result<FitResult, KrigingError> {
    if empirical.semivariances.is_empty()
        || empirical.distances.is_empty()
        || empirical.semivariances.len() != empirical.distances.len()
        || empirical.semivariances.len() != empirical.n_pairs.len()
    {
        return Err(KrigingError::FittingError(
            "empirical variogram is empty or has mismatched arrays".to_string(),
        ));
    }
    let sill_guess = empirical
        .semivariances
        .iter()
        .copied()
        .fold(0.0 as Real, |a, b| a.max(b))
        .max(Real::EPSILON);
    let range_guess = empirical
        .distances
        .iter()
        .copied()
        .fold(0.0 as Real, |a, b| a.max(b))
        .max(Real::EPSILON);
    let nugget_guess = empirical.semivariances[0].min(sill_guess * 0.5).max(0.0);

    let shape_values: Option<&[Real]> = match model_type {
        VariogramType::Stable => Some(&[0.5, 1.0, 1.5, 2.0]),
        VariogramType::Matern => Some(&[0.5, 1.0, 2.0, 3.0]),
        // Power exponent must lie in (0, 2); sample plausible values (avoiding the endpoints).
        VariogramType::Power => Some(&[0.5, 1.0, 1.5, 1.9]),
        _ => None,
    };

    let mut best = None::<FitResult>;
    for nugget_frac in [0.0, 0.05, 0.1, 0.2, 0.3] {
        for sill_scale in [0.7, 0.9, 1.0, 1.1, 1.3] {
            for range_scale in [0.4, 0.7, 1.0, 1.4, 1.8] {
                let nugget = (nugget_guess * (1.0 + nugget_frac)).min(sill_guess * sill_scale);
                let sill = (sill_guess * sill_scale).max(nugget + 1e-9);
                let range = (range_guess * range_scale).max(1e-9);
                let shapes: Vec<Option<Real>> = match shape_values {
                    None => vec![None],
                    Some(slices) => slices.iter().copied().map(Some).collect(),
                };
                for shape in shapes {
                    let model = model_from_params(nugget, sill, range, model_type, shape);
                    let residuals = weighted_residuals(empirical, model);
                    let candidate = FitResult { model, residuals };
                    best = Some(match best {
                        None => candidate,
                        Some(ref curr) if residuals < curr.residuals => candidate,
                        Some(curr) => curr,
                    });
                }
            }
        }
    }
    let best = best.expect("grid has at least one iteration");
    // Refine the grid minimum with a few Nelder–Mead iterations over (nugget, sill, range)
    // (shape stays fixed at whatever the grid picked). This typically recovers the continuous
    // optimum from a nearby grid point while staying numerically cheap.
    Ok(refine_nelder_mead(empirical, model_type, best))
}

/// A light Nelder–Mead simplex over `(nugget, sill, range)` starting from an existing fit.
/// Shape (for Stable/Matérn/Power) is held fixed — the grid has already sampled it. Only valid
/// candidates (those satisfying the model's constructor preconditions) are evaluated.
fn refine_nelder_mead(
    empirical: &EmpiricalVariogram,
    model_type: VariogramType,
    start: FitResult,
) -> FitResult {
    let shape = start.model.shape();
    let (n0, s0, r0) = start.model.params();
    let build = |p: [Real; 3]| -> Option<VariogramModel> {
        let (nugget, sill, range) = (p[0], p[1], p[2]);
        if !(nugget.is_finite() && sill.is_finite() && range.is_finite()) {
            return None;
        }
        if nugget < 0.0 || range <= 0.0 {
            return None;
        }
        match model_type {
            VariogramType::Power => VariogramModel::new_power(nugget, sill, range).ok(),
            _ => match shape {
                Some(s) => VariogramModel::new_with_shape(nugget, sill, range, model_type, s).ok(),
                None => VariogramModel::new(nugget, sill, range, model_type).ok(),
            },
        }
    };
    let eval = |p: [Real; 3]| -> Real {
        match build(p) {
            Some(m) => weighted_residuals(empirical, m),
            None => Real::INFINITY,
        }
    };
    let step_n = (s0 * 0.05).max(1e-6);
    let step_s = (s0 * 0.1).max(1e-6);
    let step_r = (r0 * 0.1).max(1e-6);
    let mut simplex: [([Real; 3], Real); 4] = [
        ([n0, s0, r0], start.residuals),
        ([n0 + step_n, s0, r0], 0.0),
        ([n0, s0 + step_s, r0], 0.0),
        ([n0, s0, r0 + step_r], 0.0),
    ];
    for entry in simplex.iter_mut().skip(1) {
        entry.1 = eval(entry.0);
    }
    for _ in 0..64 {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best, worst) = (simplex[0], simplex[3]);
        if !worst.1.is_finite() && !best.1.is_finite() {
            break;
        }
        // Centroid of all but worst.
        let c = [
            (simplex[0].0[0] + simplex[1].0[0] + simplex[2].0[0]) / 3.0,
            (simplex[0].0[1] + simplex[1].0[1] + simplex[2].0[1]) / 3.0,
            (simplex[0].0[2] + simplex[1].0[2] + simplex[2].0[2]) / 3.0,
        ];
        let reflect = [
            c[0] + (c[0] - worst.0[0]),
            c[1] + (c[1] - worst.0[1]),
            c[2] + (c[2] - worst.0[2]),
        ];
        let r_val = eval(reflect);
        if r_val < simplex[2].1 && r_val >= best.1 {
            simplex[3] = (reflect, r_val);
            continue;
        }
        if r_val < best.1 {
            let expand = [
                c[0] + 2.0 * (c[0] - worst.0[0]),
                c[1] + 2.0 * (c[1] - worst.0[1]),
                c[2] + 2.0 * (c[2] - worst.0[2]),
            ];
            let e_val = eval(expand);
            simplex[3] = if e_val < r_val {
                (expand, e_val)
            } else {
                (reflect, r_val)
            };
            continue;
        }
        let contract = [
            c[0] + 0.5 * (worst.0[0] - c[0]),
            c[1] + 0.5 * (worst.0[1] - c[1]),
            c[2] + 0.5 * (worst.0[2] - c[2]),
        ];
        let k_val = eval(contract);
        if k_val < worst.1 {
            simplex[3] = (contract, k_val);
            continue;
        }
        // Shrink toward best.
        for slot in simplex.iter_mut().skip(1) {
            let p = [
                best.0[0] + 0.5 * (slot.0[0] - best.0[0]),
                best.0[1] + 0.5 * (slot.0[1] - best.0[1]),
                best.0[2] + 0.5 * (slot.0[2] - best.0[2]),
            ];
            *slot = (p, eval(p));
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (p, r) = simplex[0];
    match build(p) {
        Some(m) if r < start.residuals => FitResult {
            model: m,
            residuals: r,
        },
        _ => start,
    }
}

pub(crate) fn weighted_residuals(emp: &EmpiricalVariogram, model: VariogramModel) -> Real {
    emp.distances
        .iter()
        .zip(emp.semivariances.iter())
        .zip(emp.n_pairs.iter())
        .map(|((d, y), w)| {
            let diff = y - model.semivariance(*d);
            (*w as Real) * diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_variogram_rejects_empty_empirical() {
        let empirical = EmpiricalVariogram {
            distances: vec![],
            semivariances: vec![],
            n_pairs: vec![],
        };
        let result = fit_variogram(&empirical, VariogramType::Exponential);
        assert!(result.is_err(), "empty empirical must be rejected");
    }

    #[test]
    fn fit_variogram_returns_finite_solution() {
        let empirical = EmpiricalVariogram {
            distances: vec![10.0, 20.0, 30.0, 40.0],
            semivariances: vec![0.2, 0.4, 0.6, 0.75],
            n_pairs: vec![8, 9, 7, 6],
        };
        let fit = fit_variogram(&empirical, VariogramType::Exponential).expect("fit should work");
        assert!(fit.residuals.is_finite());
        let (_, sill, range) = fit.model.params();
        assert!(sill > 0.0);
        assert!(range > 0.0);
    }

    #[test]
    fn fit_synthetic_exponential_returns_valid_params() {
        let true_model = VariogramModel::new(0.1, 2.0, 25.0, VariogramType::Exponential).unwrap();
        let distances = vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];
        let semivariances: Vec<Real> = distances
            .iter()
            .map(|&d| true_model.semivariance(d))
            .collect();
        let n_pairs = vec![10, 12, 11, 9, 8, 7, 6, 5];
        let empirical = EmpiricalVariogram {
            distances,
            semivariances,
            n_pairs,
        };
        let fit = fit_variogram(&empirical, VariogramType::Exponential).expect("fit should work");
        assert!(fit.residuals.is_finite());
        let (nugget, sill, range) = fit.model.params();
        assert!(nugget >= 0.0, "nugget {} should be non-negative", nugget);
        assert!(
            sill > nugget,
            "sill {} should exceed nugget {}",
            sill,
            nugget
        );
        assert!(range > 0.0, "range {} should be positive", range);
    }

    #[test]
    fn fit_spherical_and_gaussian_return_finite() {
        let empirical = EmpiricalVariogram {
            distances: vec![10.0, 20.0, 30.0, 40.0],
            semivariances: vec![0.2, 0.4, 0.6, 0.75],
            n_pairs: vec![8, 9, 7, 6],
        };
        for vt in [VariogramType::Spherical, VariogramType::Gaussian] {
            let fit = fit_variogram(&empirical, vt).expect("fit should work");
            assert!(fit.residuals.is_finite());
            let (_, sill, range) = fit.model.params();
            assert!(sill > 0.0);
            assert!(range > 0.0);
        }
    }

    #[test]
    fn nelder_mead_refinement_does_not_worsen_grid_fit() {
        // True exponential with parameters between grid points, so the grid itself will
        // land at a suboptimal point and refinement should tighten the fit.
        let true_model = VariogramModel::new(0.07, 1.83, 27.3, VariogramType::Exponential).unwrap();
        let distances: Vec<Real> = (1..=20).map(|i| i as Real * 2.5).collect();
        let semivariances: Vec<Real> = distances
            .iter()
            .map(|&d| true_model.semivariance(d))
            .collect();
        let n_pairs = vec![10usize; distances.len()];
        let empirical = EmpiricalVariogram {
            distances: distances.clone(),
            semivariances: semivariances.clone(),
            n_pairs: n_pairs.clone(),
        };
        let refined = fit_variogram(&empirical, VariogramType::Exponential).unwrap();
        // Compare against the residuals of the best grid-only fit by rerunning in a local
        // wrapper with refinement disabled (we just recompute the grid search inline).
        let sill_guess = empirical
            .semivariances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        let range_guess = empirical
            .distances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        let nugget_guess = empirical.semivariances[0].min(sill_guess * 0.5).max(0.0);
        let mut grid_best = Real::INFINITY;
        for nf in [0.0, 0.05, 0.1, 0.2, 0.3] {
            for ss in [0.7, 0.9, 1.0, 1.1, 1.3] {
                for rs in [0.4, 0.7, 1.0, 1.4, 1.8] {
                    let nug = (nugget_guess * (1.0 + nf)).min(sill_guess * ss);
                    let sill = (sill_guess * ss).max(nug + 1e-9);
                    let range = (range_guess * rs).max(1e-9);
                    let m =
                        VariogramModel::new(nug, sill, range, VariogramType::Exponential).unwrap();
                    let r = weighted_residuals(&empirical, m);
                    if r < grid_best {
                        grid_best = r;
                    }
                }
            }
        }
        assert!(
            refined.residuals <= grid_best * 1.000001,
            "refined residuals {} should not exceed grid-best {}",
            refined.residuals,
            grid_best
        );
    }

    #[test]
    fn fit_cubic_stable_matern_return_finite() {
        let empirical = EmpiricalVariogram {
            distances: vec![10.0, 20.0, 30.0, 40.0],
            semivariances: vec![0.2, 0.4, 0.6, 0.75],
            n_pairs: vec![8, 9, 7, 6],
        };
        for vt in [
            VariogramType::Cubic,
            VariogramType::Stable,
            VariogramType::Matern,
        ] {
            let fit = fit_variogram(&empirical, vt).expect("fit should work");
            assert!(fit.residuals.is_finite());
            let (nugget, sill, range) = fit.model.params();
            assert!(nugget >= 0.0);
            assert!(sill > nugget);
            assert!(range > 0.0);
            if let Some(shape) = fit.model.shape() {
                assert!(shape.is_finite());
                assert!(shape > 0.0);
            }
        }
    }
}
