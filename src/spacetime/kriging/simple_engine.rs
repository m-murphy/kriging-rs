//! Cholesky simple space–time kriging engine for incremental SGS.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::cholesky_extend_spd_lower;
use crate::error::KrigingError;
use crate::kriging::engine::{factor_spd, solve_spd_lower};
use crate::kriging::ordinary::Prediction;
use crate::spacetime::coord::{SpaceTimeCoord, temporal_distance};
use crate::spacetime::kriging::engine::build_covariance;
use crate::spacetime::kriging::ordinary::spacetime_diagonal_jitter;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;

/// Fitted simple space–time kriging engine: SPD Cholesky on the covariance block.
#[derive(Debug, Clone)]
pub struct SpaceTimeSimpleKrigingEngine<M: SpatialMetric> {
    metric: M,
    prepared_spatial: Vec<M::Prepared>,
    times: Vec<Real>,
    residuals: Vec<Real>,
    mean: Real,
    variogram: SpaceTimeVariogram,
    cov_at_zero: Real,
    observation_diagonal: Vec<Real>,
    chol_l: DMatrix<Real>,
}

impl<M: SpatialMetric> SpaceTimeSimpleKrigingEngine<M> {
    pub fn fit(
        metric: M,
        coords: Vec<SpaceTimeCoord<M::Coord>>,
        values: Vec<Real>,
        variogram: SpaceTimeVariogram,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        Self::fit_with_extra_diagonal(metric, coords, values, variogram, mean, &[])
    }

    pub fn mean(&self) -> Real {
        self.mean
    }

    pub fn variogram(&self) -> SpaceTimeVariogram {
        self.variogram
    }

    pub fn fit_with_extra_diagonal(
        metric: M,
        coords: Vec<SpaceTimeCoord<M::Coord>>,
        values: Vec<Real>,
        variogram: SpaceTimeVariogram,
        mean: Real,
        extra_diagonal: &[Real],
    ) -> Result<Self, KrigingError> {
        if !mean.is_finite() {
            return Err(KrigingError::InvalidInput(
                "mean must be finite".to_string(),
            ));
        }
        let n = coords.len();
        if n != values.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "coords ({n}) and values ({}) must have equal length",
                values.len()
            )));
        }
        if n < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        if !extra_diagonal.is_empty() && extra_diagonal.len() != n {
            return Err(KrigingError::InvalidInput(
                "extra observation diagonal must be empty or match coords length".to_string(),
            ));
        }

        let prepared_spatial: Vec<M::Prepared> =
            coords.iter().map(|c| metric.prepare(c.spatial)).collect();
        let times: Vec<Real> = coords.iter().map(|c| c.time).collect();
        let residuals: Vec<Real> = values.iter().map(|v| *v - mean).collect();
        let c = build_covariance(
            &metric,
            &prepared_spatial,
            &times,
            variogram,
            extra_diagonal,
        )?;
        let chol_l = factor_spd(c)?;

        Ok(Self {
            metric,
            prepared_spatial,
            times,
            residuals,
            mean,
            variogram,
            cov_at_zero: variogram.c_at_zero(),
            observation_diagonal: extra_diagonal.to_vec(),
            chol_l,
        })
    }

    pub fn predict(
        &self,
        targets: &[SpaceTimeCoord<M::Coord>],
    ) -> Result<Vec<Prediction>, KrigingError> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            targets
                .par_iter()
                .map(|target| self.predict_one(*target))
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            targets.iter().map(|&t| self.predict_one(t)).collect()
        }
    }

    pub fn condition(
        mut self,
        site: SpaceTimeCoord<M::Coord>,
        value: Real,
        obs_var: Real,
    ) -> Result<Self, KrigingError> {
        if !obs_var.is_finite() || obs_var < 0.0 {
            return Err(KrigingError::InvalidInput(
                "observation variance must be finite and non-negative".to_string(),
            ));
        }
        let n = self.prepared_spatial.len();
        let prepared_site = self.metric.prepare(site.spatial);
        let time_site = site.time;
        let mut cross = DVector::zeros(n);
        for i in 0..n {
            let hs = self
                .metric
                .distance(self.prepared_spatial[i], prepared_site);
            let ht = temporal_distance(self.times[i], time_site);
            cross[i] = self.variogram.covariance(hs, ht);
        }
        let _n_new = n + 1;
        let diag_eps = spacetime_diagonal_jitter(self.variogram);
        let new_diag = self.cov_at_zero + diag_eps + obs_var;

        self.chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;
        self.prepared_spatial.push(prepared_site);
        self.times.push(time_site);
        self.residuals.push(value - self.mean);
        self.observation_diagonal.push(obs_var);
        Ok(self)
    }

    fn predict_one(&self, target: SpaceTimeCoord<M::Coord>) -> Result<Prediction, KrigingError> {
        let n = self.prepared_spatial.len();
        let prepared_target = self.metric.prepare(target.spatial);
        let mut c0 = DVector::zeros(n);
        for i in 0..n {
            let hs = self
                .metric
                .distance(self.prepared_spatial[i], prepared_target);
            let ht = temporal_distance(self.times[i], target.time);
            c0[i] = self.variogram.covariance(hs, ht);
        }
        let w = solve_spd_lower(&self.chol_l, &c0)?;
        let mut residual_pred = 0.0 as Real;
        let mut cov_dot = 0.0 as Real;
        for i in 0..n {
            residual_pred += w[i] * self.residuals[i];
            cov_dot += w[i] * c0[i];
        }
        Ok(Prediction {
            value: self.mean + residual_pred,
            variance: (self.cov_at_zero - cov_dot).max(0.0),
        })
    }
}
