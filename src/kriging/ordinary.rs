use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::distance::{GeoCoord, PreparedGeoCoord, haversine_distance_prepared, prepare_geo_coord};
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::variogram::models::{VariogramModel, VariogramType};

/// Result of a single kriging prediction: the interpolated value and the kriging variance.
#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    /// The predicted (interpolated) value at the target location.
    pub value: Real,
    /// The kriging variance at the target location.
    pub variance: Real,
}

/// Search neighborhood restricting which stations contribute to each prediction.
///
/// - `max_neighbors`: use only the `k` closest stations (by Haversine distance).
/// - `max_radius`: use only stations within this radius (in the same units as
///   [`haversine_distance`](crate::distance::haversine_distance) — kilometers).
///
/// At least one of the two must be set; otherwise the neighborhood is effectively "all
/// stations" and should not be supplied. When both are set, the intersection is used
/// (nearest `k` among those within `max_radius`).
///
/// Enabling a neighborhood switches prediction from a single precomputed LU factorization
/// of the full kriging matrix to a per-target-point system build + solve. This trades
/// some CPU cost for better conditioning and locality (useful for large datasets or
/// non-stationary fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighborhood {
    pub max_neighbors: Option<usize>,
    pub max_radius: Option<Real>,
}

impl Neighborhood {
    /// Neighborhood of the `k` nearest stations, with no radius cap.
    pub fn nearest(k: usize) -> Self {
        Self {
            max_neighbors: Some(k),
            max_radius: None,
        }
    }

    /// Neighborhood of all stations within `radius` (same units as Haversine distance),
    /// with no count cap.
    pub fn within_radius(radius: Real) -> Self {
        Self {
            max_neighbors: None,
            max_radius: Some(radius),
        }
    }

    /// Neighborhood intersection: nearest `k` among those within `radius`.
    pub fn nearest_within(k: usize, radius: Real) -> Self {
        Self {
            max_neighbors: Some(k),
            max_radius: Some(radius),
        }
    }
}

/// Fitted ordinary kriging model for spatial interpolation.
///
/// Build from a [`GeoDataset`] and a [`VariogramModel`]
/// with [`new`](Self::new), then call [`predict`](Self::predict) for a single location or
/// [`predict_batch`](Self::predict_batch) for many. See also the `ordinary_kriging` example.
#[derive(Debug)]
pub struct OrdinaryKrigingModel {
    coords: Vec<GeoCoord>,
    prepared_coords: Vec<PreparedGeoCoord>,
    values: Vec<Real>,
    variogram: VariogramModel,
    cov_at_zero: Real,
    system: DMatrix<Real>,
    /// Shared LU factorization. Wrapped in `Arc` so `Clone` is `O(1)` instead of
    /// re-running the factorization (which is `O(n³)` and the dominant cost at
    /// construction time).
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
    neighborhood: Option<Neighborhood>,
}

impl Clone for OrdinaryKrigingModel {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.clone(),
            prepared_coords: self.prepared_coords.clone(),
            values: self.values.clone(),
            variogram: self.variogram,
            cov_at_zero: self.cov_at_zero,
            system: self.system.clone(),
            system_lu: Arc::clone(&self.system_lu),
            neighborhood: self.neighborhood,
        }
    }
}

impl OrdinaryKrigingModel {
    pub fn new(dataset: GeoDataset, variogram: VariogramModel) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let prepared_coords = coords
            .iter()
            .copied()
            .map(prepare_geo_coord)
            .collect::<Vec<_>>();

        let system = build_ordinary_system(&prepared_coords, variogram);
        let system_lu = Arc::new(system.clone().lu());
        // Validate solvability up front so prediction failures are not deferred.
        let mut probe_rhs = DVector::from_element(coords.len() + 1, 0.0);
        probe_rhs[coords.len()] = 1.0;
        if system_lu.solve(&probe_rhs).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize ordinary kriging system".to_string(),
            ));
        }
        Ok(Self {
            coords,
            prepared_coords,
            values,
            variogram,
            cov_at_zero: variogram.covariance(0.0),
            system,
            system_lu,
            neighborhood: None,
        })
    }

    /// Enable a search neighborhood that restricts which stations are used at each prediction
    /// location. When set, predictions build and solve a smaller local kriging system per target
    /// point instead of using the precomputed full-data LU factorization.
    ///
    /// Pass `None` to clear an existing neighborhood and return to the full-data fast path.
    pub fn with_neighborhood(mut self, neighborhood: Option<Neighborhood>) -> Self {
        self.neighborhood = neighborhood;
        self
    }

    /// In-place variant of [`Self::with_neighborhood`]. Useful when holding the model through
    /// a shared reference (e.g. across FFI boundaries) where consuming `self` is inconvenient.
    pub fn set_neighborhood(&mut self, neighborhood: Option<Neighborhood>) {
        self.neighborhood = neighborhood;
    }

    /// Returns the active search neighborhood, if any.
    pub fn neighborhood(&self) -> Option<Neighborhood> {
        self.neighborhood
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        if self.neighborhood.is_some() {
            return self.predict_local(coord);
        }
        let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
        self.predict_with_rhs(coord, &mut rhs)
    }

    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        if self.neighborhood.is_some() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                return coords
                    .par_iter()
                    .map(|coord| self.predict_local(*coord))
                    .collect();
            }
            #[cfg(target_arch = "wasm32")]
            {
                let mut out = Vec::with_capacity(coords.len());
                for &coord in coords {
                    out.push(self.predict_local(coord)?);
                }
                return Ok(out);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            // `map_init` gives each rayon worker its own RHS buffer that's reused across
            // the targets it processes, avoiding per-prediction `DVector` allocations.
            coords
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(n + 1, 0.0),
                    |rhs, coord| self.predict_with_rhs(*coord, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
            let mut out = Vec::with_capacity(coords.len());
            for &coord in coords {
                out.push(self.predict_with_rhs(coord, &mut rhs)?);
            }
            Ok(out)
        }
    }

    /// GPU-accelerated batch prediction. Returns [`KrigingError::BackendUnavailable`] if the GPU
    /// path fails for any reason (adapter missing, Matérn unsupported, readback failure, etc.).
    /// For automatic CPU fallback use [`predict_batch_gpu_or_cpu`](Self::predict_batch_gpu_or_cpu).
    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        let covariances =
            crate::gpu::build_rhs_covariances_gpu(&self.coords, coords, self.variogram)
                .await
                .map_err(KrigingError::BackendUnavailable)?;
        self.predict_batch_with_covariances(coords, &covariances)
    }

    /// GPU-first batch prediction; falls back to CPU on any GPU error. Use this when you want
    /// silent fallback (e.g. unknown client GPU availability). For strict GPU execution, use
    /// [`predict_batch_gpu`](Self::predict_batch_gpu).
    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu_or_cpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        match crate::gpu::build_rhs_covariances_gpu(&self.coords, coords, self.variogram).await {
            Ok(covariances) => self.predict_batch_with_covariances(coords, &covariances),
            Err(_) => self.predict_batch(coords),
        }
    }

    /// Blocking variant of [`predict_batch_gpu`](Self::predict_batch_gpu). Returns
    /// [`KrigingError::BackendUnavailable`] when the GPU path fails.
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        let covariances =
            crate::gpu::build_rhs_covariances_gpu_blocking(&self.coords, coords, self.variogram)
                .map_err(KrigingError::BackendUnavailable)?;
        self.predict_batch_with_covariances(coords, &covariances)
    }

    /// Blocking variant of [`predict_batch_gpu_or_cpu`](Self::predict_batch_gpu_or_cpu): GPU first,
    /// CPU fallback on any GPU error.
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_or_cpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        match crate::gpu::build_rhs_covariances_gpu_blocking(&self.coords, coords, self.variogram) {
            Ok(covariances) => self.predict_batch_with_covariances(coords, &covariances),
            Err(_) => self.predict_batch(coords),
        }
    }

    fn predict_local(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        let neighborhood = self
            .neighborhood
            .expect("predict_local requires neighborhood");
        let prepared_coord = prepare_geo_coord(coord);
        let n_total = self.prepared_coords.len();

        // Distance from target to every station.
        let mut indexed: Vec<(usize, Real)> = (0..n_total)
            .map(|i| {
                (
                    i,
                    haversine_distance_prepared(self.prepared_coords[i], prepared_coord),
                )
            })
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

        let k = indexed.len();
        if k == 0 {
            return Err(KrigingError::InvalidInput(
                "no stations in search neighborhood for target point".to_string(),
            ));
        }

        // Build the local kriging system on the selected stations.
        let diag_eps = kriging_diagonal_jitter(k, self.variogram);
        let mut a = DMatrix::from_element(k + 1, k + 1, 0.0);
        for i in 0..k {
            let (si, _) = indexed[i];
            for j in i..k {
                let (sj, _) = indexed[j];
                let mut cov = self.variogram.covariance(haversine_distance_prepared(
                    self.prepared_coords[si],
                    self.prepared_coords[sj],
                ));
                if i == j {
                    cov += diag_eps;
                }
                a[(i, j)] = cov;
                a[(j, i)] = cov;
            }
            a[(i, k)] = 1.0;
            a[(k, i)] = 1.0;
        }

        let mut rhs = DVector::from_element(k + 1, 0.0);
        for i in 0..k {
            rhs[i] = self.variogram.covariance(indexed[i].1);
        }
        rhs[k] = 1.0;

        let sol = a.lu().solve(&rhs).ok_or_else(|| {
            KrigingError::MatrixError(
                "could not solve local (neighborhood) kriging system".to_string(),
            )
        })?;
        let mut value: Real = 0.0;
        let mut cov_dot: Real = 0.0;
        for i in 0..k {
            let (si, _) = indexed[i];
            value += sol[i] * self.values[si];
            cov_dot += sol[i] * rhs[i];
        }
        let mu = sol[k];
        let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
        Ok(Prediction { value, variance })
    }

    fn predict_with_rhs(
        &self,
        coord: GeoCoord,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        let n = self.coords.len();
        let prepared_coord = prepare_geo_coord(coord);
        for i in 0..n {
            rhs[i] = self.variogram.covariance(haversine_distance_prepared(
                self.prepared_coords[i],
                prepared_coord,
            ));
        }
        rhs[n] = 1.0;

        let sol = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve ordinary kriging system".to_string())
        })?;
        let mut value = 0.0;
        let mut cov_dot = 0.0;
        for i in 0..n {
            value += sol[i] * self.values[i];
            cov_dot += sol[i] * rhs[i];
        }
        let mu = sol[n];
        let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
        Ok(Prediction { value, variance })
    }

    #[cfg(feature = "gpu")]
    fn predict_batch_with_covariances(
        &self,
        coords: &[GeoCoord],
        covariances: &[Real],
    ) -> Result<Vec<Prediction>, KrigingError> {
        let n = self.coords.len();
        let expected = n.checked_mul(coords.len()).ok_or_else(|| {
            KrigingError::MatrixError("covariance dimensions overflowed".to_string())
        })?;
        if covariances.len() != expected {
            return Err(KrigingError::MatrixError(format!(
                "expected {} covariance entries, got {}",
                expected,
                covariances.len()
            )));
        }
        let mut rhs = DVector::from_element(n + 1, 0.0);
        let mut out = Vec::with_capacity(coords.len());
        for pred_idx in 0..coords.len() {
            for i in 0..n {
                rhs[i] = covariances[pred_idx * n + i];
            }
            rhs[n] = 1.0;
            let sol = self.system_lu.solve(&rhs).ok_or_else(|| {
                KrigingError::MatrixError("could not solve ordinary kriging system".to_string())
            })?;
            let mut value = 0.0;
            let mut cov_dot = 0.0;
            for i in 0..n {
                value += sol[i] * self.values[i];
                cov_dot += sol[i] * rhs[i];
            }
            let mu = sol[n];
            let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
            out.push(Prediction { value, variance });
        }
        Ok(out)
    }
}

/// Diagonal regularization ("jitter") added to the covariance block of a kriging matrix
/// to keep it solvable under `f32` arithmetic.
///
/// Gaussian and cubic covariance are very smooth at the origin, so dense clusters of stations
/// produce matrices whose condition number blows up in `f32`. Stronger jitter is applied for
/// those models; all other models use a small floor.
///
/// The jitter is scaled with `sqrt(n)` (heuristic tracking condition-number growth with
/// sample size) and floored at 1 % of the nugget so that small-nugget fits stay solvable.
/// References: Ababou et al. (1994) Math Geol 26:99–133 (Gaussian is among the worst
/// conditioned), Diamond & Armstrong (1984) Math Geol 16:809–822 (condition number is central
/// to robustness).
pub fn kriging_diagonal_jitter(n_stations: usize, variogram: VariogramModel) -> Real {
    let (nugget, sill, _) = variogram.params();
    let scale = (n_stations as Real).sqrt().max(1.0);
    let nugget_floor: Real = (0.01 * nugget).max(1e-10);
    // Gaussian and Cubic are the notoriously ill-conditioned cases (near-singular when
    // ranges are large relative to station spacing). Exponential/Matérn/Stable are better
    // behaved but still fail when two stations are exactly coincident — a small sill-scaled
    // floor handles that case without materially biasing predictions in well-posed ones.
    match variogram.variogram_type() {
        VariogramType::Gaussian => (1e-5 * sill * scale).max(nugget_floor),
        VariogramType::Cubic => (1e-4 * sill * scale).max(nugget_floor),
        _ => (1e-8 * sill).max(nugget_floor),
    }
}

fn build_ordinary_system(coords: &[PreparedGeoCoord], variogram: VariogramModel) -> DMatrix<Real> {
    let n = coords.len();
    let diag_eps = kriging_diagonal_jitter(n, variogram);

    // Compute only the upper triangle of covariances as a flat buffer indexed by
    // `row_upper_triangle_index(i, j) = i * n - i*(i+1)/2 + j` for i <= j. This halves the
    // work and plays well with parallelism because rows are independent. Native builds use
    // rayon; WASM falls back to sequential.
    let upper_len = n * (n + 1) / 2;
    let fill_row = |i: usize| -> Vec<Real> {
        let mut row = Vec::with_capacity(n - i);
        for j in i..n {
            let mut cov = variogram.covariance(haversine_distance_prepared(coords[i], coords[j]));
            if i == j {
                cov += diag_eps;
            }
            row.push(cov);
        }
        row
    };
    #[cfg(not(target_arch = "wasm32"))]
    let rows: Vec<Vec<Real>> = (0..n).into_par_iter().map(fill_row).collect();
    #[cfg(target_arch = "wasm32")]
    let rows: Vec<Vec<Real>> = (0..n).map(fill_row).collect();
    debug_assert_eq!(rows.iter().map(|r| r.len()).sum::<usize>(), upper_len);

    let mut m = DMatrix::from_element(n + 1, n + 1, 0.0);
    for (i, row) in rows.into_iter().enumerate() {
        for (off, cov) in row.into_iter().enumerate() {
            let j = i + off;
            m[(i, j)] = cov;
            m[(j, i)] = cov;
        }
        m[(i, n)] = 1.0;
        m[(n, i)] = 1.0;
    }
    m[(n, n)] = 0.0;
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_dataset::GeoDataset;
    use crate::variogram::models::VariogramType;

    #[test]
    fn predicts_close_to_training_value_for_collocated_point() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model");
        let pred = model.predict(coords[0]).expect("prediction");
        assert!((pred.value - 10.0).abs() < 1e-3);
        assert!(pred.variance >= 0.0);
    }

    #[test]
    fn neighborhood_matches_full_when_covering_all_stations() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values).unwrap();
        let full = OrdinaryKrigingModel::new(dataset.clone(), variogram).expect("model");
        let local = OrdinaryKrigingModel::new(dataset, variogram)
            .expect("model")
            .with_neighborhood(Some(Neighborhood::nearest(coords.len())));

        let target = GeoCoord::try_new(0.5, 0.5).unwrap();
        let full_pred = full.predict(target).expect("full");
        let local_pred = local.predict(target).expect("local");
        assert!((full_pred.value - local_pred.value).abs() < 1e-3);
        assert!((full_pred.variance - local_pred.variance).abs() < 1e-3);
    }

    #[test]
    fn neighborhood_k1_uses_single_nearest_station() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 10.0).unwrap(),
            GeoCoord::try_new(10.0, 0.0).unwrap(),
        ];
        let values = vec![100.0, 200.0, 300.0];
        let variogram = VariogramModel::new(0.01, 5.0, 1000.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram)
            .expect("model")
            .with_neighborhood(Some(Neighborhood::nearest(1)));
        // Target near station 0: k=1 nearest neighborhood reduces to that single station.
        let pred = model
            .predict(GeoCoord::try_new(0.05, 0.05).unwrap())
            .expect("prediction");
        assert!((pred.value - 100.0).abs() < 1e-4);
    }

    #[test]
    fn neighborhood_rejects_empty_radius() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0];
        let variogram = VariogramModel::new(0.01, 1.0, 100.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram)
            .expect("model")
            .with_neighborhood(Some(Neighborhood::within_radius(1e-9)));
        let err = model
            .predict(GeoCoord::try_new(50.0, 50.0).unwrap())
            .expect_err("should fail with no neighbors");
        match err {
            KrigingError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn predicts_finite_values_with_coincident_stations() {
        // Two stations at the exact same coordinate with different values — the kriging
        // system is still solvable because the diagonal jitter regularizes it. The predicted
        // value at the coincident location should lie between the two observed values and be
        // finite (no NaN/∞).
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 20.0, 30.0];
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model");
        let pred = model.predict(coords[0]).expect("prediction");
        assert!(pred.value.is_finite(), "value must be finite");
        assert!(pred.variance.is_finite() && pred.variance >= 0.0);
        assert!(
            pred.value >= 9.0 && pred.value <= 21.0,
            "predicted value {} should be near the co-located observations",
            pred.value
        );
    }

    #[test]
    fn tiny_nugget_still_conditions_well_for_gaussian_variogram() {
        // A Gaussian variogram with a very small nugget is the classic ill-conditioning
        // case. The kriging_diagonal_jitter helper should add enough regularization for the
        // LU solve to succeed.
        let coords: Vec<GeoCoord> = (0..6)
            .map(|i| GeoCoord::try_new(i as Real * 0.1, i as Real * 0.1).unwrap())
            .collect();
        let values: Vec<Real> = (0..6).map(|i| i as Real).collect();
        let variogram = VariogramModel::new(1e-9, 1.0, 10.0, VariogramType::Gaussian).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model should build");
        let pred = model.predict(coords[2]).expect("prediction");
        assert!(pred.value.is_finite() && pred.variance.is_finite());
        assert!((pred.value - 2.0).abs() < 0.5, "got {}", pred.value);
    }

    #[test]
    fn all_variogram_models_produce_finite_predictions() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let target = GeoCoord::try_new(0.5, 0.5).unwrap();
        // HoleEffect and Power are not covered by the GPU path; validate them only on CPU.
        // Power reinterprets sill/range as slope/exponent so use valid values.
        let models = vec![
            VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Spherical).unwrap(),
            VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap(),
            VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Gaussian).unwrap(),
            VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Cubic).unwrap(),
            VariogramModel::new_with_shape(0.01, 5.0, 300.0, VariogramType::Stable, 1.5).unwrap(),
            VariogramModel::new_with_shape(0.01, 5.0, 300.0, VariogramType::Matern, 1.0).unwrap(),
            VariogramModel::new(0.01, 5.0, 300.0, VariogramType::HoleEffect).unwrap(),
            VariogramModel::new_power(0.01, 0.5, 1.5).unwrap(),
        ];
        for variogram in models {
            let dataset = GeoDataset::new(coords.clone(), values.clone()).unwrap();
            let model = OrdinaryKrigingModel::new(dataset, variogram).unwrap_or_else(|e| {
                panic!("{:?} failed to build: {e:?}", variogram.variogram_type())
            });
            let pred = model.predict(target).unwrap_or_else(|e| {
                panic!("{:?} failed to predict: {e:?}", variogram.variogram_type())
            });
            assert!(
                pred.value.is_finite(),
                "{:?} produced non-finite value",
                variogram.variogram_type()
            );
            assert!(
                pred.variance.is_finite() && pred.variance >= 0.0,
                "{:?} produced invalid variance {}",
                variogram.variogram_type(),
                pred.variance
            );
        }
    }

    #[test]
    fn batch_predictions_match_repeated_single_predictions() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 10.0, 400.0, VariogramType::Gaussian).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model");
        let query_coords = vec![
            GeoCoord::try_new(0.2, 0.3).unwrap(),
            GeoCoord::try_new(0.7, 0.4).unwrap(),
            GeoCoord::try_new(0.5, 0.8).unwrap(),
        ];
        let batch = model.predict_batch(&query_coords).expect("batch");
        let singles = query_coords
            .iter()
            .map(|coord| model.predict(*coord).expect("single"))
            .collect::<Vec<_>>();
        assert_eq!(batch.len(), singles.len());
        for (b, s) in batch.iter().zip(singles.iter()) {
            assert!((b.value - s.value).abs() < 1e-4);
            assert!((b.variance - s.variance).abs() < 1e-4);
        }
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    #[test]
    fn gpu_batch_predictions_match_cpu_batch_predictions() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 14.0, 16.0];
        let variogram = VariogramModel::new(0.01, 10.0, 400.0, VariogramType::Gaussian).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model");
        let query_coords = vec![
            GeoCoord::try_new(0.2, 0.3).unwrap(),
            GeoCoord::try_new(0.7, 0.4).unwrap(),
            GeoCoord::try_new(0.5, 0.8).unwrap(),
        ];
        let cpu = model.predict_batch(&query_coords).expect("cpu batch");
        let gpu = model
            .predict_batch_gpu_blocking(&query_coords)
            .expect("gpu batch");
        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert!((g.value - c.value).abs() < 1e-3);
            assert!((g.variance - c.variance).abs() < 1e-3);
        }
    }

    /// Gaussian variogram: CPU and GPU batch predictions must agree within relative tolerance.
    /// Ensures the same covariance formula and conditioning are used on both paths.
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    #[test]
    fn gaussian_cpu_and_gpu_predictions_agree_within_relative_tolerance() {
        let coords = vec![
            GeoCoord::try_new(37.75, -122.45).unwrap(),
            GeoCoord::try_new(37.76, -122.44).unwrap(),
            GeoCoord::try_new(37.77, -122.43).unwrap(),
            GeoCoord::try_new(37.78, -122.42).unwrap(),
            GeoCoord::try_new(37.79, -122.41).unwrap(),
        ];
        let values = vec![15.0, 16.0, 17.0, 18.0, 19.0];
        let variogram = VariogramModel::new(0.05, 8.0, 6.0, VariogramType::Gaussian).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model");
        let query_coords = vec![
            GeoCoord::try_new(37.765, -122.435).unwrap(),
            GeoCoord::try_new(37.775, -122.425).unwrap(),
        ];
        let cpu = model.predict_batch(&query_coords).expect("cpu batch");
        let gpu = model
            .predict_batch_gpu_blocking(&query_coords)
            .expect("gpu batch");
        assert_eq!(gpu.len(), cpu.len(), "same number of predictions");
        const REL_TOL: f32 = 1e-4;
        const ABS_TOL: f32 = 1e-5;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let rel_value = (g.value - c.value).abs() / (c.value.abs() + ABS_TOL);
            let rel_var = (g.variance - c.variance).abs() / (c.variance + ABS_TOL);
            assert!(
                rel_value < REL_TOL,
                "Gaussian value mismatch at {}: cpu={} gpu={} rel_diff={}",
                i,
                c.value,
                g.value,
                rel_value
            );
            assert!(
                rel_var < REL_TOL,
                "Gaussian variance mismatch at {}: cpu={} gpu={} rel_diff={}",
                i,
                c.variance,
                g.variance,
                rel_var
            );
        }
    }
}
