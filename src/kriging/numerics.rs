//! Shared numerical policy for kriging engines: diagonal jitter, neighborhood selection,
//! and bordered-LU reference solvers for regression tests (ADR-0001).

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::ordinary::{Neighborhood, Prediction};
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::variogram::models::{VariogramModel, VariogramType};

/// Diagonal regularization for an SPD covariance block: variogram-type fraction × `scale_at_zero`,
/// floored at `floor_at_zero`.
///
/// Size-independent so Cholesky extend/downdate (LOO, SGS) does not change ε when the block
/// grows or shrinks by one station.
pub fn covariance_diagonal_jitter(
    worst_type_fraction: Real,
    scale_at_zero: Real,
    floor_at_zero: Real,
) -> Real {
    let floor = floor_at_zero.max(Real::MIN_POSITIVE);
    (worst_type_fraction * scale_at_zero).max(floor)
}

fn variogram_type_jitter_fraction(vt: VariogramType) -> Real {
    match vt {
        VariogramType::Gaussian => 1e-5 as Real,
        VariogramType::Cubic => 1e-4 as Real,
        _ => 1e-8 as Real,
    }
}

/// Diagonal jitter for a scalar [`VariogramModel`] covariance block.
pub fn kriging_diagonal_jitter(variogram: VariogramModel) -> Real {
    let (nugget, sill, _) = variogram.params();
    let frac = variogram_type_jitter_fraction(variogram.variogram_type());
    covariance_diagonal_jitter(frac, sill, (0.01 * nugget).max(1e-10))
}

/// Diagonal jitter for a space–time covariance block: stronger of the spatial and temporal
/// marginal type fractions, scaled by `C(0, 0)`.
pub fn spacetime_diagonal_jitter(variogram: SpaceTimeVariogram) -> Real {
    let c0 = variogram.c_at_zero();
    let worst_frac = worst_variogram_type_jitter_fraction(&variogram.marginal_variogram_types());
    covariance_diagonal_jitter(worst_frac, c0, 1e-10 * c0)
}

/// Worst-case variogram-type fraction across a list (space–time marginals).
pub fn worst_variogram_type_jitter_fraction(types: &[VariogramType]) -> Real {
    types
        .iter()
        .map(|vt| variogram_type_jitter_fraction(*vt))
        .fold(1e-8 as Real, Real::max)
}

/// Select station indices for a search neighborhood using the spatial metric distance.
pub fn select_neighborhood_indices<M: SpatialMetric>(
    metric: &M,
    prepared: &[M::Prepared],
    prepared_target: M::Prepared,
    neighborhood: Neighborhood,
) -> Vec<usize> {
    let n_total = prepared.len();
    let mut indexed: Vec<(usize, Real)> = (0..n_total)
        .map(|i| (i, metric.distance(prepared[i], prepared_target)))
        .collect();

    if let Some(r) = neighborhood.max_radius {
        indexed.retain(|(_, d)| *d <= r);
    }
    if let Some(k) = neighborhood.max_neighbors
        && indexed.len() > k
    {
        indexed.select_nth_unstable_by(k, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
    }
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Dual-SPD ordinary kriging prediction from a local covariance factor and cross-covariance.
pub fn predict_dual_spd(
    chol_l: &DMatrix<Real>,
    beta: &DVector<Real>,
    one_t_beta: Real,
    c0: &DVector<Real>,
    values: &[Real],
    cov_at_zero: Real,
) -> Result<Prediction, KrigingError> {
    use crate::kriging::engine::solve_spd_lower;

    let n = chol_l.nrows();
    if c0.len() != n || beta.len() != n || values.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "predict_dual_spd: length mismatch".to_string(),
        ));
    }
    let gamma0 = solve_spd_lower(chol_l, c0)?;
    let mu = (beta.dot(c0) - 1.0) / one_t_beta;
    let w = gamma0 - mu * beta;

    let mut value = 0.0 as Real;
    let mut cov_dot = 0.0 as Real;
    for i in 0..n {
        value += w[i] * values[i];
        cov_dot += w[i] * c0[i];
    }
    Ok(Prediction {
        value,
        variance: (cov_at_zero - cov_dot - mu).max(0.0),
    })
}

/// Reference ordinary kriging via bordered LU `[C 1; 1ᵀ 0]` — regression / ADR-0001 contract tests.
#[cfg_attr(not(test), allow(dead_code))]
pub fn predict_bordered_lu(
    c: &DMatrix<Real>,
    c0: &DVector<Real>,
    values: &[Real],
    cov_at_zero: Real,
) -> Result<Prediction, KrigingError> {
    let n = c.nrows();
    if c.ncols() != n || c0.len() != n || values.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "predict_bordered_lu: shape mismatch".to_string(),
        ));
    }
    let mut a = DMatrix::from_element(n + 1, n + 1, 0.0);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = c[(i, j)];
        }
        a[(i, n)] = 1.0;
        a[(n, i)] = 1.0;
    }
    let mut rhs = DVector::from_element(n + 1, 0.0);
    for i in 0..n {
        rhs[i] = c0[i];
    }
    rhs[n] = 1.0;

    let sol = a
        .lu()
        .solve(&rhs)
        .ok_or_else(|| KrigingError::MatrixError("bordered LU kriging solve failed".to_string()))?;

    let mut value = 0.0 as Real;
    let mut cov_dot = 0.0 as Real;
    for i in 0..n {
        value += sol[i] * values[i];
        cov_dot += sol[i] * c0[i];
    }
    let mu = sol[n];
    Ok(Prediction {
        value,
        variance: (cov_at_zero - cov_dot - mu).max(0.0),
    })
}

/// Extend β = C⁻¹1 when C grows by one row/column with cross-covariance `v`, γᵥ = C⁻¹v, Schur `s`.
pub fn extend_constraint_beta(
    beta: &DVector<Real>,
    cross: &DVector<Real>,
    gamma_v: &DVector<Real>,
    schur: Real,
) -> Result<(DVector<Real>, Real), KrigingError> {
    if schur <= Real::EPSILON {
        return Err(KrigingError::MatrixError(
            "extend_constraint_beta: non-positive Schur complement".to_string(),
        ));
    }
    let n = beta.len();
    if cross.len() != n || gamma_v.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "extend_constraint_beta: length mismatch".to_string(),
        ));
    }
    let v_dot_beta = beta.dot(cross);
    let y = (1.0 - v_dot_beta) / schur;
    let mut beta_new = DVector::zeros(n + 1);
    for i in 0..n {
        beta_new[i] = beta[i] - gamma_v[i] * y;
    }
    beta_new[n] = y;
    let one_t_beta = beta_new.sum();
    Ok((beta_new, one_t_beta))
}

/// Extend one column of β = C⁻¹F when C grows (same block formula as the constraint vector).
pub fn extend_beta_column(
    gamma_col: &DVector<Real>,
    cross: &DVector<Real>,
    gamma_v: &DVector<Real>,
    schur: Real,
    f_new: Real,
) -> Result<DVector<Real>, KrigingError> {
    if schur <= Real::EPSILON {
        return Err(KrigingError::MatrixError(
            "extend_beta_column: non-positive Schur complement".to_string(),
        ));
    }
    let n = gamma_col.len();
    if cross.len() != n || gamma_v.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "extend_beta_column: length mismatch".to_string(),
        ));
    }
    let v_dot_gamma = cross.dot(gamma_col);
    let y = (f_new - v_dot_gamma) / schur;
    let mut col_new = DVector::zeros(n + 1);
    for i in 0..n {
        col_new[i] = gamma_col[i] - gamma_v[i] * y;
    }
    col_new[n] = y;
    Ok(col_new)
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::geo_dataset::GeoDataset;
    use crate::kriging::engine::{
        OrdinaryKrigingEngine, build_covariance, factor_spd, solve_spd_lower,
    };
    use crate::kriging::ordinary::{Neighborhood, OrdinaryKrigingModel};
    use crate::kriging::pairwise::{PairwiseCovariance, SpatialPairwiseCovariance};
    use crate::kriging::universal::UniversalTrend;
    use crate::kriging::universal_engine::UniversalKrigingEngine;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn rel_err(a: Real, b: Real) -> Real {
        let scale = a.abs().max(b.abs()).max(1e-12);
        (a - b).abs() / scale
    }

    fn assert_prediction_close(
        actual: &Prediction,
        expected: &Prediction,
        max_rel: Real,
        label: &str,
    ) {
        assert!(
            rel_err(actual.value, expected.value) <= max_rel,
            "{label} value: actual={} expected={} rel={}",
            actual.value,
            expected.value,
            rel_err(actual.value, expected.value)
        );
        assert!(
            rel_err(actual.variance, expected.variance) <= max_rel,
            "{label} variance: actual={} expected={} rel={}",
            actual.variance,
            expected.variance,
            rel_err(actual.variance, expected.variance)
        );
    }

    fn fixture() -> (Vec<GeoCoord>, Vec<Real>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        (coords, values, variogram)
    }

    #[test]
    fn dual_spd_matches_bordered_lu() {
        let (coords, values, variogram) = fixture();
        let target = GeoCoord::try_new(0.5, 0.5).unwrap();
        let cov = SpatialPairwiseCovariance::new(GeoMetric, variogram);
        let prepared: Vec<_> = coords.iter().map(|&c| cov.prepare(c)).collect();
        let c = build_covariance(&cov, &prepared, &[]).expect("cov");
        let chol_l = factor_spd(c.clone()).expect("factor");
        let beta =
            solve_spd_lower(&chol_l, &DVector::from_element(coords.len(), 1.0)).expect("beta");
        let one_t_beta = beta.sum();
        let prepared_target = cov.prepare(target);
        let mut c0 = DVector::zeros(coords.len());
        for i in 0..coords.len() {
            c0[i] = cov.covariance(prepared[i], prepared_target);
        }
        let cov_at_zero = cov.cov_at_zero();
        let dual =
            predict_dual_spd(&chol_l, &beta, one_t_beta, &c0, &values, cov_at_zero).expect("dual");
        let bordered = predict_bordered_lu(&c, &c0, &values, cov_at_zero).expect("bordered");
        assert_prediction_close(&dual, &bordered, 1e-6, "dual vs bordered");
    }

    #[test]
    fn neighborhood_dual_matches_full() {
        let (coords, values, variogram) = fixture();
        let dataset = GeoDataset::new(coords.clone(), values.clone()).unwrap();
        let full = OrdinaryKrigingModel::new(dataset.clone(), variogram).expect("full");
        let local = OrdinaryKrigingModel::new(dataset, variogram)
            .expect("local")
            .with_neighborhood(Some(Neighborhood::nearest(coords.len())));
        let target = GeoCoord::try_new(0.5, 0.5).unwrap();
        let full_pred = full.predict(target).expect("full");
        let local_pred = local.predict(target).expect("local");
        assert_prediction_close(&local_pred, &full_pred, 1e-6, "neighborhood vs full");
    }

    #[test]
    fn leave_one_out_matches_fold_refit() {
        let (coords, values, variogram) = fixture();
        let engine = OrdinaryKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            coords,
            values,
        )
        .expect("fit");
        let loo = engine.leave_one_out_predictions().expect("loo");
        let n = loo.len();
        for (i, loo_pred) in loo.iter().enumerate() {
            let mut train_idx = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j != i {
                    train_idx.push(j);
                }
            }
            let fold_pred = engine
                .predict_subset(&train_idx, engine.coords()[i])
                .expect("subset");
            assert_prediction_close(loo_pred, &fold_pred, 1e-4, &format!("loo[{i}]"));
        }
    }

    #[test]
    fn ordinary_condition_matches_refit() {
        let (coords, values, variogram) = fixture();
        let new_site = GeoCoord::try_new(0.2, 0.2).unwrap();
        let target = GeoCoord::try_new(0.4, 0.4).unwrap();

        let mut conditioned = OrdinaryKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            coords.clone(),
            values.clone(),
        )
        .expect("fit");
        conditioned
            .append_condition(new_site, 17.5, 0.25)
            .expect("condition");
        let cond_pred = conditioned
            .predict(&[target])
            .expect("predict")
            .pop()
            .unwrap();

        let mut all_coords = coords;
        let mut all_values = values;
        all_coords.push(new_site);
        all_values.push(17.5);
        let refit = OrdinaryKrigingEngine::fit_with_extra_diagonal(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            all_coords,
            all_values,
            &[0.0, 0.0, 0.0, 0.0, 0.25],
        )
        .expect("refit");
        let refit_pred = refit.predict(&[target]).expect("predict").pop().unwrap();
        assert_prediction_close(&cond_pred, &refit_pred, 1e-5, "condition vs refit");
    }

    #[test]
    fn universal_condition_matches_refit() {
        let (coords, values, variogram) = fixture();
        let trend = UniversalTrend::Constant;
        let new_site = GeoCoord::try_new(0.2, 0.2).unwrap();
        let target = GeoCoord::try_new(0.4, 0.4).unwrap();

        let mut conditioned = UniversalKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            coords.clone(),
            values.clone(),
            trend,
        )
        .expect("fit");
        conditioned
            .append_condition(new_site, 17.5, 0.25)
            .expect("condition");
        let cond_pred = conditioned.predict(&[target]).expect("predict")[0];

        let mut all_coords = coords;
        let mut all_values = values;
        all_coords.push(new_site);
        all_values.push(17.5);
        let refit = UniversalKrigingEngine::fit_with_extra_diagonal(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            all_coords,
            all_values,
            trend,
            &[0.0, 0.0, 0.0, 0.0, 0.25],
        )
        .expect("refit");
        let refit_pred = refit.predict(&[target]).expect("predict")[0];
        assert_prediction_close(&cond_pred, &refit_pred, 1e-5, "universal condition");
    }
}
