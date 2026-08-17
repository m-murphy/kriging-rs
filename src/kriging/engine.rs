//! Dual-SPD ordinary kriging engine, generic over [`PairwiseCovariance`].
//!
//! Single solver behind geographic, projected, and space-time ordinary kriging
//! (and, by composition, binomial). See ADR-0001 and ADR-0003.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use nalgebra::{DMatrix, DVector};

use crate::Real;
use crate::cholesky_update::{
    cholesky_delete_index, cholesky_extend_spd_lower, forward_solve_lower,
};
use crate::error::KrigingError;
use crate::kriging::numerics::{extend_constraint_beta, predict_dual_spd};
use crate::kriging::ordinary::Prediction;
use crate::kriging::pairwise::PairwiseCovariance;

/// Fitted ordinary kriging engine: dual SPD factorization + constraint vector β = C⁻¹·1.
#[derive(Debug, Clone)]
pub struct OrdinaryKrigingEngine<K: PairwiseCovariance> {
    cov: K,
    sites: Vec<K::Site>,
    prepared: Vec<K::Prepared>,
    values: Vec<Real>,
    cov_at_zero: Real,
    observation_diagonal: Vec<Real>,
    chol_l: DMatrix<Real>,
    beta: DVector<Real>,
    one_t_beta: Real,
}

impl<K: PairwiseCovariance> OrdinaryKrigingEngine<K> {
    /// Build from pairwise covariance, sites, and values (homoscedastic path).
    pub fn fit(cov: K, sites: Vec<K::Site>, values: Vec<Real>) -> Result<Self, KrigingError> {
        Self::fit_with_extra_diagonal(cov, sites, values, &[])
    }

    /// Like [`fit`](Self::fit) but adds per-station observation variance on the covariance diagonal.
    pub fn fit_with_extra_diagonal(
        cov: K,
        sites: Vec<K::Site>,
        values: Vec<Real>,
        extra_diagonal: &[Real],
    ) -> Result<Self, KrigingError> {
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
        for &v in extra_diagonal {
            if !v.is_finite() || v < 0.0 {
                return Err(KrigingError::InvalidInput(
                    "observation diagonal entries must be finite and non-negative".to_string(),
                ));
            }
        }

        let prepared: Vec<K::Prepared> = sites.iter().map(|&s| cov.prepare(s)).collect();
        let c = build_covariance(&cov, &prepared, extra_diagonal)?;
        let chol_l = factor_spd(c)?;
        let ones = DVector::from_element(n, 1.0);
        let beta = solve_spd_lower(&chol_l, &ones)?;
        let one_t_beta = beta.sum();

        Ok(Self {
            cov,
            sites,
            prepared,
            values,
            cov_at_zero: cov.cov_at_zero(),
            observation_diagonal: extra_diagonal.to_vec(),
            chol_l,
            beta,
            one_t_beta,
        })
    }

    /// Number of training stations.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Pairwise covariance adapter used when fitting this engine.
    pub fn pairwise_covariance(&self) -> K {
        self.cov
    }

    /// Training sites (same order as values).
    pub fn coords(&self) -> &[K::Site] {
        &self.sites
    }

    /// Training values.
    pub fn values(&self) -> &[Real] {
        &self.values
    }

    pub(crate) fn prepared(&self) -> &[K::Prepared] {
        &self.prepared
    }

    /// Predict at one or more targets (batch API; single target = one-element slice).
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

    /// Append a conditioning site (SGS / incremental path). Returns a new engine with extended
    /// factorization via [`cholesky_extend_spd_lower`].
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

        let w_fwd = forward_solve_lower(&self.chol_l, &cross)?;
        let schur = new_diag - w_fwd.dot(&w_fwd);
        let gamma_v = solve_spd_lower(&self.chol_l, &cross)?;
        let chol_l = cholesky_extend_spd_lower(&self.chol_l, &cross, new_diag)?;
        let (beta, one_t_beta) = extend_constraint_beta(&self.beta, &cross, &gamma_v, schur)?;

        self.chol_l = chol_l;
        self.beta = beta;
        self.one_t_beta = one_t_beta;
        self.sites.push(site);
        self.prepared.push(prepared_site);
        self.values.push(value);
        self.observation_diagonal.push(obs_var);
        Ok(())
    }

    /// Predict using only a subset of training stations (dual SPD on the local block).
    pub fn predict_subset(
        &self,
        indices: &[usize],
        target: K::Site,
    ) -> Result<Prediction, KrigingError> {
        let k = indices.len();
        if k == 0 {
            return Err(KrigingError::InvalidInput(
                "predict_subset requires at least one station index".to_string(),
            ));
        }
        let prepared_target = self.cov.prepare(target);
        let obs_diag = &self.observation_diagonal;
        let model_diag = self.cov.diagonal();

        let mut local_c = DMatrix::zeros(k, k);
        for ii in 0..k {
            let si = indices[ii];
            for jj in ii..k {
                let sj = indices[jj];
                let cov = if ii == jj {
                    let mut d = model_diag;
                    if let Some(&obs) = obs_diag.get(si) {
                        d += obs;
                    }
                    d
                } else {
                    self.cov.covariance(self.prepared[si], self.prepared[sj])
                };
                local_c[(ii, jj)] = cov;
                local_c[(jj, ii)] = cov;
            }
        }
        let local_l = factor_spd(local_c)?;
        let ones = DVector::from_element(k, 1.0);
        let local_beta = solve_spd_lower(&local_l, &ones)?;
        let one_t_beta = local_beta.sum();

        let mut c0 = DVector::zeros(k);
        for (ii, &si) in indices.iter().enumerate() {
            c0[ii] = self.cov.covariance(self.prepared[si], prepared_target);
        }
        let mut local_values = Vec::with_capacity(k);
        for &si in indices {
            local_values.push(self.values[si]);
        }
        predict_dual_spd(
            &local_l,
            &local_beta,
            one_t_beta,
            &c0,
            &local_values,
            self.cov_at_zero,
        )
    }

    /// Leave-one-out predictions in input order: O(n³) total via Cholesky downdate per hold-out.
    ///
    /// Fast approximate LOO: deletes station `i` from the full `n`-station factorization (same
    /// diagonal jitter as the full fit). Matches fold-refit closely when jitter is size-independent.
    pub fn leave_one_out_predictions(&self) -> Result<Vec<Prediction>, KrigingError> {
        let n = self.len();
        if n < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let predict_hold = |hold: usize| -> Result<Prediction, KrigingError> {
            let l_red = cholesky_delete_index(&self.chol_l, hold)?;
            let beta_red = solve_spd_lower(&l_red, &DVector::from_element(n - 1, 1.0))?;
            let one_t_beta = beta_red.sum();
            let prepared_target = self.cov.prepare(self.sites[hold]);
            let mut c0 = DVector::zeros(n - 1);
            let mut values_red = Vec::with_capacity(n - 1);
            let mut out_i = 0;
            for i in 0..n {
                if i == hold {
                    continue;
                }
                c0[out_i] = self.cov.covariance(self.prepared[i], prepared_target);
                values_red.push(self.values[i]);
                out_i += 1;
            }
            predict_dual_spd(
                &l_red,
                &beta_red,
                one_t_beta,
                &c0,
                &values_red,
                self.cov_at_zero,
            )
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

    fn predict_one(&self, target: K::Site) -> Result<Prediction, KrigingError> {
        let n = self.prepared.len();
        let prepared_target = self.cov.prepare(target);
        let mut c0 = DVector::zeros(n);
        for i in 0..n {
            c0[i] = self.cov.covariance(self.prepared[i], prepared_target);
        }
        self.predict_from_cross_cov_inner(&c0)
    }

    fn predict_from_cross_cov_inner(&self, c0: &DVector<Real>) -> Result<Prediction, KrigingError> {
        predict_dual_spd(
            &self.chol_l,
            &self.beta,
            self.one_t_beta,
            c0,
            &self.values,
            self.cov_at_zero,
        )
    }
}

pub(crate) fn build_covariance<K: PairwiseCovariance>(
    cov: &K,
    prepared: &[K::Prepared],
    obs_extra: &[Real],
) -> Result<DMatrix<Real>, KrigingError> {
    let n = prepared.len();
    let model_diag = cov.diagonal();

    let fill_row = |i: usize| -> Vec<Real> {
        let mut row = Vec::with_capacity(n - i);
        for j in i..n {
            let c = if i == j {
                let mut d = model_diag;
                if let Some(&obs) = obs_extra.get(i) {
                    d += obs;
                }
                d
            } else {
                cov.covariance(prepared[i], prepared[j])
            };
            row.push(c);
        }
        row
    };

    #[cfg(not(target_arch = "wasm32"))]
    let rows: Vec<Vec<Real>> = (0..n).into_par_iter().map(fill_row).collect();
    #[cfg(target_arch = "wasm32")]
    let rows: Vec<Vec<Real>> = (0..n).map(fill_row).collect();

    let mut c = DMatrix::zeros(n, n);
    for (i, row) in rows.into_iter().enumerate() {
        for (off, val) in row.into_iter().enumerate() {
            let j = i + off;
            c[(i, j)] = val;
            c[(j, i)] = val;
        }
    }
    Ok(c)
}

pub(crate) fn factor_spd(c: DMatrix<Real>) -> Result<DMatrix<Real>, KrigingError> {
    c.cholesky()
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
    use crate::kriging::pairwise::{SpaceTimePairwiseCovariance, SpatialPairwiseCovariance};
    use crate::spacetime::coord::SpaceTimeCoord;
    use crate::spacetime::dataset::SpaceTimeDataset;
    use crate::spacetime::kriging::ordinary::SpaceTimeOrdinaryKrigingModel;
    use crate::spacetime::metric::GeoMetric;
    use crate::spacetime::variogram::SpaceTimeVariogram;
    use crate::variogram::models::{VariogramModel, VariogramType};

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

    fn spatial(variogram: VariogramModel) -> SpatialPairwiseCovariance<GeoMetric> {
        SpatialPairwiseCovariance::new(GeoMetric, variogram)
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
            OrdinaryKrigingEngine::fit(spatial(variogram), coords, values).expect("engine fit");
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
            OrdinaryKrigingEngine::fit(spatial(variogram), coords, values).expect("engine fit");
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
            OrdinaryKrigingEngine::fit(spatial(variogram), coords, values).expect("engine fit");
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
            spatial(variogram),
            coords,
            values,
            &extra,
        )
        .expect("engine fit");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_prediction_close(&engine_preds[0], &golden_pred, "extra diagonal");
    }

    #[test]
    fn condition_extends_coords_and_values() {
        let (coords, values, variogram) = sample_fixture();
        let site = GeoCoord::try_new(0.5, 0.5).unwrap();
        let mut engine =
            OrdinaryKrigingEngine::fit(spatial(variogram), coords, values).expect("engine fit");
        engine.append_condition(site, 42.0, 0.0).expect("condition");
        assert_eq!(engine.coords().len(), 4);
        assert_eq!(engine.coords()[3], site);
        assert_eq!(engine.values()[3], 42.0);
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
            SpatialPairwiseCovariance::new(ProjectedMetric::with_anisotropy(anisotropy), variogram),
            coords,
            values,
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

        let mut engine =
            OrdinaryKrigingEngine::fit(spatial(variogram), coords, values).expect("engine fit");
        engine
            .append_condition(new_site, new_value, obs_var)
            .expect("condition");
        let engine_preds = engine.predict(&[target]).expect("engine predict");

        assert_prediction_close(&engine_preds[0], &golden_pred, "after condition");
    }

    fn st_fixture() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>, SpaceTimeVariogram) {
        let coords = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let spatial_v = VariogramModel::new(0.01, 1.0, 300.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let variogram = SpaceTimeVariogram::new_separable(spatial_v, temporal).unwrap();
        (coords, values, variogram)
    }

    #[test]
    fn spacetime_prediction_matches_spacetime_ordinary_model_at_training_site() {
        let (coords, values, variogram) = st_fixture();
        let target = coords[1];
        let dataset = SpaceTimeDataset::new(coords.clone(), values.clone()).unwrap();
        let golden = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, variogram)
            .unwrap()
            .predict(target)
            .unwrap();
        let engine = OrdinaryKrigingEngine::fit(
            SpaceTimePairwiseCovariance::new(GeoMetric, variogram),
            coords,
            values,
        )
        .unwrap();
        let pred = engine.predict(&[target]).unwrap().pop().unwrap();
        assert_prediction_close(&pred, &golden, "training site");
    }

    #[test]
    fn spacetime_condition_then_predict_matches_refit() {
        let (coords, values, variogram) = st_fixture();
        let append = SpaceTimeCoord::new(GeoCoord::try_new(0.5, 0.5).unwrap(), 1.5);
        let append_value = 17.5;
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5);

        let mut engine = OrdinaryKrigingEngine::fit(
            SpaceTimePairwiseCovariance::new(GeoMetric, variogram),
            coords.clone(),
            values.clone(),
        )
        .unwrap();
        engine.append_condition(append, append_value, 0.0).unwrap();
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
