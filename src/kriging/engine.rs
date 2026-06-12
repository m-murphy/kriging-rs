//! Dual-SPD ordinary kriging engine, generic over [`SpatialMetric`].
//!
//! Single solver behind geographic, projected, and (via composition) binomial ordinary kriging.
//! See ADR-0001.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::{cholesky_extend_spd_lower, forward_solve_lower};
use crate::error::KrigingError;
use crate::kriging::ordinary::{Prediction, kriging_diagonal_jitter};
use crate::spacetime::metric::SpatialMetric;
use crate::variogram::models::VariogramModel;

/// Fitted ordinary kriging engine: dual SPD factorization + constraint vector β = C⁻¹·1.
#[derive(Debug, Clone)]
pub struct OrdinaryKrigingEngine<M: SpatialMetric> {
    metric: M,
    coords: Vec<M::Coord>,
    prepared: Vec<M::Prepared>,
    values: Vec<Real>,
    variogram: VariogramModel,
    cov_at_zero: Real,
    observation_diagonal: Vec<Real>,
    chol_l: DMatrix<Real>,
    beta: DVector<Real>,
    one_t_beta: Real,
}

impl<M: SpatialMetric> OrdinaryKrigingEngine<M> {
    /// Build from station coordinates, values, and a variogram (homoscedastic path).
    pub fn fit(
        metric: M,
        coords: Vec<M::Coord>,
        values: Vec<Real>,
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        Self::fit_with_extra_diagonal(metric, coords, values, variogram, &[])
    }

    /// Like [`fit`](Self::fit) but adds per-station observation variance on the covariance diagonal.
    pub fn fit_with_extra_diagonal(
        metric: M,
        coords: Vec<M::Coord>,
        values: Vec<Real>,
        variogram: VariogramModel,
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

        let prepared: Vec<M::Prepared> = coords.iter().map(|&c| metric.prepare(c)).collect();
        let c = build_covariance(&metric, &prepared, variogram, extra_diagonal)?;
        let chol_l = factor_spd(&c)?;
        let ones = DVector::from_element(n, 1.0);
        let beta = solve_spd_lower(&chol_l, &ones)?;
        let one_t_beta = beta.sum();

        Ok(Self {
            metric,
            coords,
            prepared,
            values,
            variogram,
            cov_at_zero: variogram.covariance(0.0),
            observation_diagonal: extra_diagonal.to_vec(),
            chol_l,
            beta,
            one_t_beta,
        })
    }

    /// Number of training stations.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Variogram used when fitting this engine.
    pub fn variogram(&self) -> VariogramModel {
        self.variogram
    }

    /// Training coordinates (same order as values).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn coords(&self) -> &[M::Coord] {
        &self.coords
    }

    /// Training values.
    pub fn values(&self) -> &[Real] {
        &self.values
    }

    /// Prepared coordinates for distance lookups.
    pub(crate) fn prepared(&self) -> &[M::Prepared] {
        &self.prepared
    }

    /// Per-station extra diagonal observation variance (empty if homoscedastic).
    pub(crate) fn observation_diagonal(&self) -> &[Real] {
        &self.observation_diagonal
    }

    pub(crate) fn metric(&self) -> M {
        self.metric
    }

    pub(crate) fn cov_at_zero(&self) -> Real {
        self.cov_at_zero
    }

    /// Predict at one or more targets (batch API; single target = one-element slice).
    pub fn predict(&self, targets: &[M::Coord]) -> Result<Vec<Prediction>, KrigingError> {
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

    /// Append a conditioning site (SGS / incremental path). Returns a new engine with extended
    /// factorization via [`cholesky_extend_spd_lower`].
    pub fn condition(
        mut self,
        site: M::Coord,
        value: Real,
        obs_var: Real,
    ) -> Result<Self, KrigingError> {
        if !obs_var.is_finite() || obs_var < 0.0 {
            return Err(KrigingError::InvalidInput(
                "observation variance must be finite and non-negative".to_string(),
            ));
        }
        let n = self.prepared.len();
        let prepared_site = self.metric.prepare(site);
        let mut cross = DVector::zeros(n);
        for i in 0..n {
            cross[i] = self
                .variogram
                .covariance(self.metric.distance(self.prepared[i], prepared_site));
        }
        let n_new = n + 1;
        let diag_eps = kriging_diagonal_jitter(n_new, self.variogram);
        let new_diag = self.cov_at_zero + diag_eps + obs_var;

        self.chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;
        self.coords.push(site);
        self.prepared.push(prepared_site);
        self.values.push(value);
        self.observation_diagonal.push(obs_var);

        let ones = DVector::from_element(n_new, 1.0);
        self.beta = solve_spd_lower(&self.chol_l, &ones)?;
        self.one_t_beta = self.beta.sum();
        Ok(self)
    }

    /// Predict from a precomputed cross-covariance vector `c0` (e.g. GPU RHS assembly).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn predict_from_cross_cov(&self, c0: DVector<Real>) -> Result<Prediction, KrigingError> {
        if c0.len() != self.prepared.len() {
            return Err(KrigingError::DimensionMismatch(
                "cross-covariance length must match number of stations".to_string(),
            ));
        }
        self.predict_from_cross_cov_inner(&c0)
    }

    fn predict_one(&self, target: M::Coord) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
        let prepared_target = self.metric.prepare(target);
        let mut c0 = DVector::zeros(n);
        for i in 0..n {
            c0[i] = self
                .variogram
                .covariance(self.metric.distance(self.prepared[i], prepared_target));
        }
        self.predict_from_cross_cov_inner(&c0)
    }

    fn predict_from_cross_cov_inner(&self, c0: &DVector<Real>) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
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
    prepared: &[M::Prepared],
    variogram: VariogramModel,
    obs_extra: &[Real],
) -> Result<DMatrix<Real>, KrigingError> {
    let n = prepared.len();
    let diag_eps = kriging_diagonal_jitter(n, variogram);

    let fill_row = |i: usize| -> Vec<Real> {
        let mut row = Vec::with_capacity(n - i);
        for j in i..n {
            let mut cov = variogram.covariance(metric.distance(prepared[i], prepared[j]));
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

pub(crate) fn factor_spd(c: &DMatrix<Real>) -> Result<DMatrix<Real>, KrigingError> {
    c.clone()
        .cholesky()
        .ok_or_else(|| {
            KrigingError::MatrixError("could not factorize ordinary kriging covariance".to_string())
        })
        .map(|chol| chol.l().clone())
}

pub(crate) fn solve_spd_lower(
    l: &DMatrix<Real>,
    b: &DVector<Real>,
) -> Result<DVector<Real>, KrigingError> {
    let y = forward_solve_lower(l, b)?;
    backward_solve_lower_transpose(l, &y)
}

fn backward_solve_lower_transpose(
    l: &DMatrix<Real>,
    y: &DVector<Real>,
) -> Result<DVector<Real>, KrigingError> {
    let n = l.nrows();
    if l.ncols() != n || y.len() != n {
        return Err(KrigingError::DimensionMismatch(
            "backward_solve_lower_transpose: shape mismatch".to_string(),
        ));
    }
    let mut x = DVector::zeros(n);
    for i in (0..n).rev() {
        let diag = l[(i, i)];
        if diag.abs() < Real::EPSILON {
            return Err(KrigingError::MatrixError(
                "backward_solve_lower_transpose: near-zero diagonal".to_string(),
            ));
        }
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * x[k];
        }
        x[i] = s / diag;
    }
    Ok(x)
}

#[cfg(test)]
mod golden_tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::geo_dataset::GeoDataset;
    use crate::kriging::ordinary::OrdinaryKrigingModel;
    use crate::spacetime::metric::GeoMetric;
    use crate::variogram::models::VariogramType;

    fn sample_fixture() -> (Vec<GeoCoord>, Vec<Real>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        (coords, values, variogram)
    }

    fn assert_prediction_close(engine: &Prediction, golden: &Prediction, label: &str) {
        assert!(
            (engine.value - golden.value).abs() < 1e-3,
            "{label} value: engine={} golden={}",
            engine.value,
            golden.value
        );
        assert!(
            (engine.variance - golden.variance).abs() < 1e-3,
            "{label} variance: engine={} golden={}",
            engine.variance,
            golden.variance
        );
    }

    fn golden_model(
        coords: &[GeoCoord],
        values: &[Real],
        variogram: VariogramModel,
        extra: &[Real],
    ) -> OrdinaryKrigingModel {
        let dataset = GeoDataset::new(coords.to_vec(), values.to_vec()).unwrap();
        if extra.is_empty() {
            OrdinaryKrigingModel::new(dataset, variogram).expect("golden model")
        } else {
            OrdinaryKrigingModel::new_with_extra_diagonal(dataset, variogram, extra.to_vec())
                .expect("golden model")
        }
    }

    #[test]
    fn geographic_prediction_matches_ordinary_kriging_model_at_training_site() {
        let (coords, values, variogram) = sample_fixture();
        let target = coords[0];

        let golden = golden_model(&coords, &values, variogram, &[]);
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine =
            OrdinaryKrigingEngine::fit(GeoMetric, coords, values, variogram).expect("engine fit");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_eq!(engine_preds.len(), 1);
        assert_prediction_close(&engine_preds[0], &golden_pred, "training site");
    }

    #[test]
    fn geographic_prediction_matches_ordinary_kriging_model_at_interior_point() {
        let (coords, values, variogram) = sample_fixture();
        let target = GeoCoord::try_new(0.3, 0.3).unwrap();

        let golden = golden_model(&coords, &values, variogram, &[]);
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine =
            OrdinaryKrigingEngine::fit(GeoMetric, coords, values, variogram).expect("engine fit");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_prediction_close(&engine_preds[0], &golden_pred, "interior");
    }

    #[test]
    fn geographic_batch_prediction_matches_ordinary_kriging_model() {
        let (coords, values, variogram) = sample_fixture();
        let targets = vec![
            coords[0],
            GeoCoord::try_new(0.3, 0.3).unwrap(),
            GeoCoord::try_new(0.5, 0.5).unwrap(),
        ];

        let golden = golden_model(&coords, &values, variogram, &[]);
        let golden_preds = golden.predict_batch(&targets).expect("golden batch");

        let engine =
            OrdinaryKrigingEngine::fit(GeoMetric, coords, values, variogram).expect("engine fit");
        let engine_preds = engine.predict(&targets).expect("engine batch");

        assert_eq!(engine_preds.len(), golden_preds.len());
        for (i, (e, g)) in engine_preds.iter().zip(golden_preds.iter()).enumerate() {
            assert_prediction_close(e, g, &format!("batch[{i}]"));
        }
    }

    #[test]
    fn extra_diagonal_matches_ordinary_kriging_model() {
        let (coords, values, variogram) = sample_fixture();
        let extra = vec![0.0, 0.0, 2.0];
        let target = GeoCoord::try_new(0.1, 0.1).unwrap();

        let golden = golden_model(&coords, &values, variogram, &extra);
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine = OrdinaryKrigingEngine::fit_with_extra_diagonal(
            GeoMetric, coords, values, variogram, &extra,
        )
        .expect("engine fit");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_prediction_close(&engine_preds[0], &golden_pred, "extra diagonal");
    }

    #[test]
    fn condition_extends_coords_and_values() {
        let (coords, values, variogram) = sample_fixture();
        let site = GeoCoord::try_new(0.5, 0.5).unwrap();
        let engine =
            OrdinaryKrigingEngine::fit(GeoMetric, coords, values, variogram).expect("engine fit");
        let extended = engine.condition(site, 42.0, 0.0).expect("condition");
        assert_eq!(extended.coords().len(), 4);
        assert_eq!(extended.coords()[3], site);
        assert_eq!(extended.values()[3], 42.0);
    }

    #[test]
    fn projected_prediction_matches_projected_kriging_model() {
        use crate::projected::{
            Anisotropy2D, ProjectedCoord, ProjectedDataset, ProjectedKrigingModel,
        };
        use crate::spacetime::metric::ProjectedMetric;

        let coords = vec![
            ProjectedCoord::new(0.0, 0.0),
            ProjectedCoord::new(0.0, 10.0),
            ProjectedCoord::new(10.0, 0.0),
        ];
        let values = vec![1.0, 3.0, 2.0];
        let variogram = VariogramModel::new(0.01, 4.0, 50.0, VariogramType::Exponential).unwrap();
        let anisotropy = Anisotropy2D::isotropic();
        let target = ProjectedCoord::new(2.0, 2.0);

        let golden = ProjectedKrigingModel::new(
            ProjectedDataset::new(coords.clone(), values.clone()).unwrap(),
            variogram,
            anisotropy,
        )
        .expect("golden projected");
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine = OrdinaryKrigingEngine::fit(
            ProjectedMetric::with_anisotropy(anisotropy),
            coords,
            values,
            variogram,
        )
        .expect("engine fit");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_prediction_close(&engine_preds[0], &golden_pred, "projected");
    }

    #[test]
    fn condition_then_predict_matches_refit_ordinary_kriging_model() {
        let (coords, values, variogram) = sample_fixture();
        let new_site = GeoCoord::try_new(0.2, 0.2).unwrap();
        let new_value = 17.5;
        let obs_var = 0.25;
        let target = GeoCoord::try_new(0.4, 0.4).unwrap();

        let mut extended_coords = coords.clone();
        extended_coords.push(new_site);
        let mut extended_values = values.clone();
        extended_values.push(new_value);
        let mut extended_extra = vec![0.0; coords.len()];
        extended_extra.push(obs_var);

        let golden = golden_model(
            &extended_coords,
            &extended_values,
            variogram,
            &extended_extra,
        );
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine = OrdinaryKrigingEngine::fit(GeoMetric, coords, values, variogram)
            .expect("engine fit")
            .condition(new_site, new_value, obs_var)
            .expect("condition");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_prediction_close(&engine_preds[0], &golden_pred, "after condition");
    }
}
