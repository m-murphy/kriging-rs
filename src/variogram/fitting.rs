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
/// The empirical variogram must be non-empty (e.g. from `compute_empirical_variogram`). The grid
/// spans plausible scales around data-derived guesses (sill, range, nugget). Accuracy is limited by
/// grid resolution: the best point may be 20–40% away from the continuous optimum in sill/range.
/// For typical empirical variograms (noisy, few bins) this is usually acceptable; for noiseless
/// synthetic data the grid can pick a different local minimum than the true parameters.
pub fn fit_variogram(
    empirical: &EmpiricalVariogram,
    model_type: VariogramType,
) -> Result<FitResult, KrigingError> {
    let sill_guess = empirical
        .semivariances
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b))
        .max(Real::EPSILON);
    let range_guess = empirical
        .distances
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b))
        .max(Real::EPSILON);
    let nugget_guess = empirical.semivariances[0].min(sill_guess * 0.5).max(0.0);

    let shape_values: Option<&[Real]> = match model_type {
        VariogramType::Stable => Some(&[0.5, 1.0, 1.5, 2.0]),
        VariogramType::Matern => Some(&[0.5, 1.0, 2.0, 3.0]),
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
    Ok(best.expect("grid has at least one iteration"))
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
