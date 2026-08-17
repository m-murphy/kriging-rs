//! Cholesky simple kriging engine for incremental SGS, generic over [`PairwiseCovariance`].
//!
//! Single solver behind geographic, projected, and space-time simple kriging. See ADR-0003.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::{cholesky_delete_index, cholesky_extend_spd_lower};
use crate::error::KrigingError;
use crate::kriging::engine::{build_covariance, factor_spd, solve_spd_lower};
use crate::kriging::ordinary::Prediction;
use crate::kriging::pairwise::PairwiseCovariance;

/// Fitted simple kriging engine: SPD Cholesky factorization on the covariance block.
#[derive(Debug, Clone)]
pub struct SimpleKrigingEngine<K: PairwiseCovariance> {
    cov: K,
    sites: Vec<K::Site>,
    prepared: Vec<K::Prepared>,
    residuals: Vec<Real>,
    mean: Real,
    cov_at_zero: Real,
    chol_l: DMatrix<Real>,
}

impl<K: PairwiseCovariance> SimpleKrigingEngine<K> {
    pub fn fit(
        cov: K,
        sites: Vec<K::Site>,
        values: Vec<Real>,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        Self::fit_with_extra_diagonal(cov, sites, values, mean, &[])
    }

    /// Known mean used by the predictor.
    pub fn mean(&self) -> Real {
        self.mean
    }

    /// Pairwise covariance adapter used when fitting this engine.
    pub fn pairwise_covariance(&self) -> K {
        self.cov
    }

    /// Original observations (not mean-centered residuals).
    pub fn observed_values(&self) -> Vec<Real> {
        self.residuals.iter().map(|&r| r + self.mean).collect()
    }

    pub fn fit_with_extra_diagonal(
        cov: K,
        sites: Vec<K::Site>,
        values: Vec<Real>,
        mean: Real,
        extra_diagonal: &[Real],
    ) -> Result<Self, KrigingError> {
        if !mean.is_finite() {
            return Err(KrigingError::InvalidInput(
                "mean must be finite".to_string(),
            ));
        }
        let n = sites.len();
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

        let prepared: Vec<K::Prepared> = sites.iter().map(|&s| cov.prepare(s)).collect();
        let residuals: Vec<Real> = values.iter().map(|v| *v - mean).collect();
        let c = build_covariance(&cov, &prepared, extra_diagonal)?;
        let chol_l = factor_spd(c)?;

        Ok(Self {
            cov,
            sites,
            prepared,
            residuals,
            mean,
            cov_at_zero: cov.cov_at_zero(),
            chol_l,
        })
    }

    pub fn coords(&self) -> &[K::Site] {
        &self.sites
    }

    /// Leave-one-out predictions via Cholesky downdate (O(n³) total).
    pub fn leave_one_out_predictions(&self) -> Result<Vec<Prediction>, KrigingError> {
        let n = self.prepared.len();
        if n < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let predict_hold = |hold: usize| -> Result<Prediction, KrigingError> {
            let l_red = cholesky_delete_index(&self.chol_l, hold)?;
            let prepared_target = self.cov.prepare(self.sites[hold]);
            let mut c0 = DVector::zeros(n - 1);
            let mut out_i = 0;
            for i in 0..n {
                if i == hold {
                    continue;
                }
                c0[out_i] = self.cov.covariance(self.prepared[i], prepared_target);
                out_i += 1;
            }
            let w = solve_spd_lower(&l_red, &c0)?;
            let mut residual_pred = 0.0 as Real;
            let mut cov_dot = 0.0 as Real;
            out_i = 0;
            for i in 0..n {
                if i == hold {
                    continue;
                }
                residual_pred += w[out_i] * self.residuals[i];
                cov_dot += w[out_i] * c0[out_i];
                out_i += 1;
            }
            Ok(Prediction {
                value: self.mean + residual_pred,
                variance: (self.cov_at_zero - cov_dot).max(0.0),
            })
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            (0..n).into_par_iter().map(predict_hold).collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut out = Vec::with_capacity(n);
            for hold in 0..n {
                out.push(predict_hold(hold)?);
            }
            Ok(out)
        }
    }

    pub fn predict(&self, targets: &[K::Site]) -> Result<Vec<Prediction>, KrigingError> {
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
        let prepared_site = self.cov.prepare(site);
        let mut cross = DVector::zeros(n);
        for i in 0..n {
            cross[i] = self.cov.covariance(self.prepared[i], prepared_site);
        }
        let new_diag = self.cov.diagonal() + obs_var;

        self.chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;
        self.sites.push(site);
        self.prepared.push(prepared_site);
        self.residuals.push(value - self.mean);
        Ok(())
    }

    fn predict_one(&self, target: K::Site) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
        let prepared_target = self.cov.prepare(target);
        let mut c0 = DVector::zeros(n);
        for i in 0..n {
            c0[i] = self.cov.covariance(self.prepared[i], prepared_target);
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

#[cfg(test)]
mod golden_tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::geo_dataset::GeoDataset;
    use crate::kriging::pairwise::SpatialPairwiseCovariance;
    use crate::kriging::simple::SimpleKrigingModel;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn sample_fixture() -> (Vec<GeoCoord>, Vec<Real>, VariogramModel, Real) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let mean = 12.0;
        (coords, values, variogram, mean)
    }

    #[test]
    fn geographic_prediction_matches_simple_kriging_model() {
        let (coords, values, variogram, mean) = sample_fixture();
        let target = GeoCoord::try_new(0.1, 0.1).unwrap();

        let golden = SimpleKrigingModel::new(
            GeoDataset::new(coords.clone(), values.clone()).unwrap(),
            variogram,
            mean,
        )
        .expect("golden model");
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine = SimpleKrigingEngine::fit(
            SpatialPairwiseCovariance::new(GeoMetric, variogram),
            coords,
            values,
            mean,
        )
        .expect("engine fit");
        let engine_pred = engine.predict(&[target]).expect("engine predict")[0];

        assert!((engine_pred.value - golden_pred.value).abs() < 1e-3);
        assert!((engine_pred.variance - golden_pred.variance).abs() < 1e-3);
    }
}
