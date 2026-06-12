//! Dual-SPD universal kriging engine (geographic polynomial drift).

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::cholesky_extend_spd_lower;
use crate::distance::{GeoCoord, PreparedGeoCoord};
use crate::error::KrigingError;
use crate::kriging::engine::{build_covariance, factor_spd, solve_spd_lower};
use crate::kriging::numerics::{extend_beta_column, kriging_diagonal_jitter};
use crate::kriging::ordinary::Prediction;
use crate::kriging::universal::UniversalTrend;
use crate::spacetime::metric::{GeoMetric, SpatialMetric};
use crate::variogram::models::VariogramModel;

/// Fitted universal kriging engine: Cholesky on `C` plus precomputed `β = C⁻¹F` and Schur `Fᵀβ`.
#[derive(Debug, Clone)]
pub struct UniversalKrigingEngine {
    coords: Vec<GeoCoord>,
    prepared: Vec<PreparedGeoCoord>,
    values: Vec<Real>,
    variogram: VariogramModel,
    trend: UniversalTrend,
    cov_at_zero: Real,
    observation_diagonal: Vec<Real>,
    design: DMatrix<Real>,
    chol_l: DMatrix<Real>,
    beta: DMatrix<Real>,
    schur_l: DMatrix<Real>,
}

impl UniversalKrigingEngine {
    pub fn fit(
        coords: Vec<GeoCoord>,
        values: Vec<Real>,
        variogram: VariogramModel,
        trend: UniversalTrend,
    ) -> Result<Self, KrigingError> {
        Self::fit_with_extra_diagonal(coords, values, variogram, trend, &[])
    }

    pub fn fit_with_extra_diagonal(
        coords: Vec<GeoCoord>,
        values: Vec<Real>,
        variogram: VariogramModel,
        trend: UniversalTrend,
        extra_diagonal: &[Real],
    ) -> Result<Self, KrigingError> {
        let n = coords.len();
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

        let prepared: Vec<_> = coords.iter().map(|&c| GeoMetric.prepare(c)).collect();
        let design = build_design(&coords, trend, n, p);
        let c = build_covariance(&GeoMetric, &prepared, variogram, extra_diagonal)?;
        let chol_l = factor_spd(c)?;
        let beta = solve_beta_columns(&chol_l, &design, n, p)?;
        let schur = design.transpose() * &beta;
        let schur_l = factor_spd(schur)?;

        Ok(Self {
            coords,
            prepared,
            values,
            variogram,
            trend,
            cov_at_zero: variogram.covariance(0.0),
            observation_diagonal: extra_diagonal.to_vec(),
            design,
            chol_l,
            beta,
            schur_l,
        })
    }

    pub fn coords(&self) -> &[GeoCoord] {
        &self.coords
    }

    pub fn values(&self) -> &[Real] {
        &self.values
    }

    pub fn variogram(&self) -> VariogramModel {
        self.variogram
    }

    pub fn predict(&self, targets: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
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

    pub fn condition(
        mut self,
        site: GeoCoord,
        value: Real,
        obs_var: Real,
    ) -> Result<Self, KrigingError> {
        if !obs_var.is_finite() || obs_var < 0.0 {
            return Err(KrigingError::InvalidInput(
                "observation variance must be finite and non-negative".to_string(),
            ));
        }
        let n = self.prepared.len();
        let p = self.trend.n_basis();
        let prepared_site = GeoMetric.prepare(site);
        let mut cross = DVector::zeros(n);
        for i in 0..n {
            cross[i] = self
                .variogram
                .covariance(GeoMetric.distance(self.prepared[i], prepared_site));
        }
        let diag_eps = kriging_diagonal_jitter(self.variogram);
        let new_diag = self.cov_at_zero + diag_eps + obs_var;

        let gamma_v = solve_spd_lower(&self.chol_l, &cross)?;
        let w_fwd = crate::cholesky_update::forward_solve_lower(&self.chol_l, &cross)?;
        let schur = new_diag - w_fwd.dot(&w_fwd);

        self.chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;
        self.coords.push(site);
        self.prepared.push(prepared_site);
        self.values.push(value);
        self.observation_diagonal.push(obs_var);

        let mut f_row = vec![0.0 as Real; p];
        self.trend.eval_basis(site, &mut f_row);

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
        self.beta = new_beta;

        let mut new_design = DMatrix::zeros(n + 1, p);
        for i in 0..n {
            for l in 0..p {
                new_design[(i, l)] = self.design[(i, l)];
            }
        }
        for l in 0..p {
            new_design[(n, l)] = f_row[l];
        }
        self.design = new_design;

        let schur_mat = self.design.transpose() * &self.beta;
        self.schur_l = factor_spd(schur_mat)?;
        Ok(self)
    }

    fn predict_one(&self, target: GeoCoord) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
        let prepared_target = GeoMetric.prepare(target);
        let mut c0 = DVector::zeros(n);
        for i in 0..n {
            c0[i] = self
                .variogram
                .covariance(GeoMetric.distance(self.prepared[i], prepared_target));
        }
        let mut f0 = vec![0.0 as Real; self.trend.n_basis()];
        self.trend.eval_basis(target, &mut f0);
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

fn build_design(coords: &[GeoCoord], trend: UniversalTrend, n: usize, p: usize) -> DMatrix<Real> {
    let mut design = DMatrix::zeros(n, p);
    let mut buf = vec![0.0 as Real; p];
    for i in 0..n {
        trend.eval_basis(coords[i], &mut buf);
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
