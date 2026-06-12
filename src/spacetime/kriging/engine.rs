//! Dual-SPD ordinary space–time kriging engine, generic over [`SpatialMetric`].

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::cholesky_extend_spd_lower;
use crate::error::KrigingError;
use crate::kriging::engine::{factor_spd, solve_spd_lower};
use crate::kriging::ordinary::Prediction;
use crate::spacetime::coord::{SpaceTimeCoord, temporal_distance};
use crate::spacetime::kriging::ordinary::spacetime_diagonal_jitter;
use crate::spacetime::metric::SpatialMetric;
use crate::spacetime::variogram::SpaceTimeVariogram;

/// Fitted ordinary space–time kriging engine: dual SPD factorization + constraint vector β = C⁻¹·1.
#[derive(Debug, Clone)]
pub struct SpaceTimeOrdinaryKrigingEngine<M: SpatialMetric> {
    metric: M,
    prepared_spatial: Vec<M::Prepared>,
    times: Vec<Real>,
    values: Vec<Real>,
    variogram: SpaceTimeVariogram,
    cov_at_zero: Real,
    observation_diagonal: Vec<Real>,
    chol_l: DMatrix<Real>,
    beta: DVector<Real>,
    one_t_beta: Real,
}

impl<M: SpatialMetric> SpaceTimeOrdinaryKrigingEngine<M> {
    pub fn fit_with_extra_diagonal(
        metric: M,
        coords: Vec<SpaceTimeCoord<M::Coord>>,
        values: Vec<Real>,
        variogram: SpaceTimeVariogram,
        extra_diagonal: &[Real],
    ) -> Result<Self, KrigingError> {
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
        for &v in extra_diagonal {
            if !v.is_finite() || v < 0.0 {
                return Err(KrigingError::InvalidInput(
                    "observation diagonal entries must be finite and non-negative".to_string(),
                ));
            }
        }

        let prepared_spatial: Vec<M::Prepared> =
            coords.iter().map(|c| metric.prepare(c.spatial)).collect();
        let times: Vec<Real> = coords.iter().map(|c| c.time).collect();
        let c = build_covariance(
            &metric,
            &prepared_spatial,
            &times,
            variogram,
            extra_diagonal,
        )?;
        let chol_l = factor_spd(&c)?;
        let ones = DVector::from_element(n, 1.0);
        let beta = solve_spd_lower(&chol_l, &ones)?;
        let one_t_beta = beta.sum();

        Ok(Self {
            metric,
            prepared_spatial,
            times,
            values,
            variogram,
            cov_at_zero: variogram.c_at_zero(),
            observation_diagonal: extra_diagonal.to_vec(),
            chol_l,
            beta,
            one_t_beta,
        })
    }

    pub fn len(&self) -> usize {
        self.prepared_spatial.len()
    }

    pub fn variogram(&self) -> SpaceTimeVariogram {
        self.variogram
    }

    pub(crate) fn metric(&self) -> M {
        self.metric
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
        let n_new = n + 1;
        let diag_eps = spacetime_diagonal_jitter(n_new, self.variogram);
        let new_diag = self.cov_at_zero + diag_eps + obs_var;

        self.chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;
        self.prepared_spatial.push(prepared_site);
        self.times.push(time_site);
        self.values.push(value);
        self.observation_diagonal.push(obs_var);

        let ones = DVector::from_element(n_new, 1.0);
        self.beta = solve_spd_lower(&self.chol_l, &ones)?;
        self.one_t_beta = self.beta.sum();
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
        self.predict_from_cross_cov_inner(&c0)
    }

    fn predict_from_cross_cov_inner(&self, c0: &DVector<Real>) -> Result<Prediction, KrigingError> {
        let n = self.prepared_spatial.len();
        let gamma0 = solve_spd_lower(&self.chol_l, c0)?;
        let mu = (self.beta.dot(c0) - 1.0) / self.one_t_beta;
        let w = gamma0 - mu * &self.beta;

        let mut value = 0.0 as Real;
        let mut cov_dot = 0.0 as Real;
        for i in 0..n {
            value += w[i] * self.values[i];
            cov_dot += w[i] * c0[i];
        }
        let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
        Ok(Prediction { value, variance })
    }
}

pub(crate) fn build_covariance<M: SpatialMetric>(
    metric: &M,
    prepared_spatial: &[M::Prepared],
    times: &[Real],
    variogram: SpaceTimeVariogram,
    obs_extra: &[Real],
) -> Result<DMatrix<Real>, KrigingError> {
    let n = prepared_spatial.len();
    let diag_eps = spacetime_diagonal_jitter(n, variogram);

    let fill_row = |i: usize| -> Vec<Real> {
        let mut row = Vec::with_capacity(n - i);
        for j in i..n {
            let hs = metric.distance(prepared_spatial[i], prepared_spatial[j]);
            let ht = temporal_distance(times[i], times[j]);
            let mut cov = variogram.covariance(hs, ht);
            if i == j {
                cov += diag_eps;
                if let Some(&d) = obs_extra.get(i) {
                    cov += d;
                }
            }
            row.push(cov);
        }
        row
    };

    #[cfg(not(target_arch = "wasm32"))]
    let rows: Vec<Vec<Real>> = (0..n).into_par_iter().map(fill_row).collect();
    #[cfg(target_arch = "wasm32")]
    let rows: Vec<Vec<Real>> = (0..n).map(fill_row).collect();

    let mut c = DMatrix::zeros(n, n);
    for (i, row) in rows.into_iter().enumerate() {
        for (off, cov) in row.into_iter().enumerate() {
            let j = i + off;
            c[(i, j)] = cov;
            c[(j, i)] = cov;
        }
    }
    Ok(c)
}

#[cfg(test)]
mod golden_tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::dataset::SpaceTimeDataset;
    use crate::spacetime::kriging::ordinary::SpaceTimeOrdinaryKrigingModel;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn fixture() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>, SpaceTimeVariogram) {
        let coords = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let spatial = VariogramModel::new(0.01, 1.0, 300.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let variogram = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        (coords, values, variogram)
    }

    fn assert_prediction_close(engine: &Prediction, golden: &Prediction, label: &str) {
        assert!(
            (engine.value - golden.value).abs() < 1e-3,
            "{label}: value engine={} golden={}",
            engine.value,
            golden.value
        );
        assert!(
            (engine.variance - golden.variance).abs() < 1e-3,
            "{label}: variance engine={} golden={}",
            engine.variance,
            golden.variance
        );
    }

    #[test]
    fn geographic_prediction_matches_spacetime_ordinary_model_at_training_site() {
        let (coords, values, variogram) = fixture();
        let target = coords[1];
        let dataset = SpaceTimeDataset::new(coords.clone(), values.clone()).unwrap();
        let golden = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, variogram)
            .unwrap()
            .predict(target)
            .unwrap();
        let engine = SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
            GeoMetric,
            coords,
            values,
            variogram,
            &[],
        )
        .unwrap();
        let pred = engine.predict(&[target]).unwrap().pop().unwrap();
        assert_prediction_close(&pred, &golden, "training site");
    }

    #[test]
    fn condition_then_predict_matches_refit_spacetime_ordinary_model() {
        let (coords, values, variogram) = fixture();
        let append = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let append_value = 17.5;
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5);

        let mut engine = SpaceTimeOrdinaryKrigingEngine::fit_with_extra_diagonal(
            GeoMetric,
            coords.clone(),
            values.clone(),
            variogram,
            &[],
        )
        .unwrap();
        engine = engine.condition(append, append_value, 0.0).unwrap();
        let engine_pred = engine.predict(&[target]).unwrap().pop().unwrap();

        let mut all_coords = coords;
        let mut all_values = values;
        all_coords.push(append);
        all_values.push(append_value);
        let dataset = SpaceTimeDataset::new(all_coords, all_values).unwrap();
        let golden = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, variogram)
            .unwrap()
            .predict(target)
            .unwrap();
        assert_prediction_close(&engine_pred, &golden, "after condition");
    }
}
