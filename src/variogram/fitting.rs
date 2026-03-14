use crate::Real;
use crate::error::KrigingError;
use crate::variogram::empirical::EmpiricalVariogram;
use crate::variogram::models::{VariogramModel, VariogramType};

#[derive(Debug, Clone)]
pub struct FitResult {
    pub model: VariogramModel,
    pub residuals: Real,
}

pub fn fit_variogram(
    empirical: &EmpiricalVariogram,
    model_type: VariogramType,
) -> Result<FitResult, KrigingError> {
    if empirical.distances.is_empty() || empirical.semivariances.is_empty() {
        return Err(KrigingError::FittingError(
            "empirical variogram cannot be empty".to_string(),
        ));
    }

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

    let mut best: Option<FitResult> = None;
    for nugget_frac in [0.0, 0.05, 0.1, 0.2, 0.3] {
        for sill_scale in [0.7, 0.9, 1.0, 1.1, 1.3] {
            for range_scale in [0.4, 0.7, 1.0, 1.4, 1.8] {
                let nugget = (nugget_guess * (1.0 + nugget_frac)).min(sill_guess * sill_scale);
                let sill = (sill_guess * sill_scale).max(nugget + 1e-9);
                let range = (range_guess * range_scale).max(1e-9);
                let model = match model_type {
                    VariogramType::Spherical => VariogramModel::Spherical {
                        nugget,
                        sill,
                        range,
                    },
                    VariogramType::Exponential => VariogramModel::Exponential {
                        nugget,
                        sill,
                        range,
                    },
                    VariogramType::Gaussian => VariogramModel::Gaussian {
                        nugget,
                        sill,
                        range,
                    },
                };
                let residuals = weighted_residuals(empirical, model);
                match &best {
                    Some(curr) if residuals >= curr.residuals => {}
                    _ => best = Some(FitResult { model, residuals }),
                }
            }
        }
    }

    best.ok_or_else(|| KrigingError::FittingError("could not fit variogram".to_string()))
}

fn weighted_residuals(emp: &EmpiricalVariogram, model: VariogramModel) -> Real {
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
}
