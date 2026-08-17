//! Dual-SPD universal kriging engine, generic over [`PairwiseCovariance`] and [`TrendBasis`].
//!
//! Single solver behind geographic and space-time universal kriging. See ADR-0003.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::{cholesky_extend_spd_lower, forward_solve_lower};
use crate::error::KrigingError;
use crate::kriging::engine::{build_covariance, factor_spd, solve_spd_lower};
use crate::kriging::numerics::extend_beta_column;
use crate::kriging::ordinary::Prediction;
use crate::kriging::pairwise::PairwiseCovariance;
use crate::kriging::universal::TrendBasis;

/// Fitted universal kriging engine: Cholesky on `C` plus `β = C⁻¹F` and Schur `Fᵀβ`.
#[derive(Debug, Clone)]
pub struct UniversalKrigingEngine<K: PairwiseCovariance, T: TrendBasis<Site = K::Site>> {
    cov: K,
    trend: T,
    sites: Vec<K::Site>,
    prepared: Vec<K::Prepared>,
    values: Vec<Real>,
    cov_at_zero: Real,
    design: DMatrix<Real>,
    chol_l: DMatrix<Real>,
    beta: DMatrix<Real>,
    schur_l: DMatrix<Real>,
}

impl<K: PairwiseCovariance, T: TrendBasis<Site = K::Site>> UniversalKrigingEngine<K, T> {
    pub fn fit(
        cov: K,
        sites: Vec<K::Site>,
        values: Vec<Real>,
        trend: T,
    ) -> Result<Self, KrigingError> {
        Self::fit_with_extra_diagonal(cov, sites, values, trend, &[])
    }

    pub fn fit_with_extra_diagonal(
        cov: K,
        sites: Vec<K::Site>,
        values: Vec<Real>,
        trend: T,
        extra_diagonal: &[Real],
    ) -> Result<Self, KrigingError> {
        let n = sites.len();
        let p = trend.n_basis();
        if n != values.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "coords ({n}) and values ({}) must have equal length",
                values.len()
            )));
        }
        if n < p + 1 {
            return Err(KrigingError::InsufficientData(p + 1));
        }
        if !extra_diagonal.is_empty() && extra_diagonal.len() != n {
            return Err(KrigingError::InvalidInput(
                "extra observation diagonal must be empty or match coords length".to_string(),
            ));
        }

        let prepared: Vec<K::Prepared> = sites.iter().map(|&s| cov.prepare(s)).collect();
        let design = build_design(&sites, trend, n, p);
        let c = build_covariance(&cov, &prepared, extra_diagonal)?;
        let chol_l = factor_spd(c)?;
        let beta = solve_beta_columns(&chol_l, &design, n, p)?;
        let schur = design.transpose() * &beta;
        let schur_l = factor_spd(schur)?;

        Ok(Self {
            cov,
            trend,
            sites,
            prepared,
            values,
            cov_at_zero: cov.cov_at_zero(),
            design,
            chol_l,
            beta,
            schur_l,
        })
    }

    pub fn pairwise_covariance(&self) -> K {
        self.cov
    }

    pub fn coords(&self) -> &[K::Site] {
        &self.sites
    }

    pub fn values(&self) -> &[Real] {
        &self.values
    }

    pub fn predict(&self, targets: &[K::Site]) -> Result<Vec<Prediction>, KrigingError> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            targets
                .par_iter()
                .map(|&target| self.predict_one(target))
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            targets.iter().map(|&t| self.predict_one(t)).collect()
        }
    }

    pub(crate) fn append_condition(
        &mut self,
        site: K::Site,
        value: Real,
        obs_var: Real,
    ) -> Result<(), KrigingError> {
        if !obs_var.is_finite() || obs_var < 0.0 {
            return Err(KrigingError::InvalidInput(
                "observation variance must be finite and non-negative".to_string(),
            ));
        }
        let n = self.prepared.len();
        let p = self.trend.n_basis();
        let prepared_site = self.cov.prepare(site);
        let mut cross = DVector::zeros(n);
        for i in 0..n {
            cross[i] = self.cov.covariance(self.prepared[i], prepared_site);
        }
        let new_diag = self.cov.diagonal() + obs_var;

        let gamma_v = solve_spd_lower(&self.chol_l, &cross)?;
        let w_fwd = forward_solve_lower(&self.chol_l, &cross)?;
        let schur = new_diag - w_fwd.dot(&w_fwd);

        let chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;

        let mut f_row = vec![0.0 as Real; p];
        self.trend.eval(site, &mut f_row);

        let mut new_beta = DMatrix::zeros(n + 1, p);
        for l in 0..p {
            let mut col = DVector::zeros(n);
            for i in 0..n {
                col[i] = self.beta[(i, l)];
            }
            let col_new = extend_beta_column(&col, &cross, &gamma_v, schur, f_row[l])?;
            for i in 0..=n {
                new_beta[(i, l)] = col_new[i];
            }
        }

        let mut new_design = DMatrix::zeros(n + 1, p);
        for i in 0..n {
            for l in 0..p {
                new_design[(i, l)] = self.design[(i, l)];
            }
        }
        for l in 0..p {
            new_design[(n, l)] = f_row[l];
        }

        let schur_mat = new_design.transpose() * &new_beta;
        let schur_l = factor_spd(schur_mat)?;

        self.chol_l = chol_l;
        self.beta = new_beta;
        self.design = new_design;
        self.schur_l = schur_l;
        self.sites.push(site);
        self.prepared.push(prepared_site);
        self.values.push(value);
        Ok(())
    }

    fn predict_one(&self, target: K::Site) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
        let prepared_target = self.cov.prepare(target);
        let mut c0 = DVector::zeros(n);
        for i in 0..n {
            c0[i] = self.cov.covariance(self.prepared[i], prepared_target);
        }
        let mut f0 = vec![0.0 as Real; self.trend.n_basis()];
        self.trend.eval(target, &mut f0);
        self.predict_from_cross_cov(&c0, &f0)
    }

    fn predict_from_cross_cov(
        &self,
        c0: &DVector<Real>,
        f0: &[Real],
    ) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
        let p = self.trend.n_basis();
        let gamma0 = solve_spd_lower(&self.chol_l, c0)?;

        let mut rhs = DVector::zeros(p);
        for l in 0..p {
            let mut col_dot = 0.0 as Real;
            for i in 0..n {
                col_dot += self.design[(i, l)] * gamma0[i];
            }
            rhs[l] = col_dot - f0[l];
        }
        let lambda = solve_spd_lower(&self.schur_l, &rhs)?;

        let mut value = 0.0 as Real;
        let mut cov_dot = 0.0 as Real;
        for i in 0..n {
            let mut wi = gamma0[i];
            for l in 0..p {
                wi -= self.beta[(i, l)] * lambda[l];
            }
            value += wi * self.values[i];
            cov_dot += wi * c0[i];
        }
        let mut mu_dot = 0.0 as Real;
        for l in 0..p {
            mu_dot += lambda[l] * f0[l];
        }
        Ok(Prediction {
            value,
            variance: (self.cov_at_zero - cov_dot - mu_dot).max(0.0),
        })
    }
}

fn build_design<T: TrendBasis>(sites: &[T::Site], trend: T, n: usize, p: usize) -> DMatrix<Real> {
    let mut design = DMatrix::zeros(n, p);
    let mut buf = vec![0.0 as Real; p];
    for i in 0..n {
        trend.eval(sites[i], &mut buf);
        for l in 0..p {
            design[(i, l)] = buf[l];
        }
    }
    design
}

fn solve_beta_columns(
    chol_l: &DMatrix<Real>,
    design: &DMatrix<Real>,
    n: usize,
    p: usize,
) -> Result<DMatrix<Real>, KrigingError> {
    let mut beta = DMatrix::zeros(n, p);
    for l in 0..p {
        let mut col = DVector::zeros(n);
        for i in 0..n {
            col[i] = design[(i, l)];
        }
        let gamma = solve_spd_lower(chol_l, &col)?;
        for i in 0..n {
            beta[(i, l)] = gamma[i];
        }
    }
    Ok(beta)
}

#[cfg(test)]
mod golden_tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::geo_dataset::GeoDataset;
    use crate::kriging::pairwise::SpatialPairwiseCovariance;
    use crate::kriging::universal::{UniversalKrigingModel, UniversalTrend};
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    #[test]
    fn geographic_linear_prediction_matches_universal_kriging_model() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 20.0, 12.0, 22.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let trend = UniversalTrend::Linear;
        let target = GeoCoord::try_new(0.25, 0.25).unwrap();

        let golden = UniversalKrigingModel::new(
            GeoDataset::new(coords.clone(), values.clone()).unwrap(),
            variogram,
            trend,
        )
        .expect("golden model");
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine = UniversalKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            coords,
            values,
            trend,
        )
        .expect("engine fit");
        let engine_pred = engine.predict(&[target]).expect("engine predict")[0];

        assert!((engine_pred.value - golden_pred.value).abs() < 1e-3);
        assert!((engine_pred.variance - golden_pred.variance).abs() < 1e-3);
    }
}
