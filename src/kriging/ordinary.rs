#[cfg(feature = "gpu")]
use nalgebra::DVector;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::spacetime::metric::GeoMetric;
use crate::variogram::models::VariogramModel;

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
/// Enabling a neighborhood builds and solves a smaller local dual-SPD system per target
/// (same formulation as the full engine, restricted to nearby stations). This trades
/// some CPU cost for better conditioning and locality on large datasets.
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
///
/// For per-station observation noise in addition to the nugget and numerical jitter, use
/// [`new_with_extra_diagonal`](Self::new_with_extra_diagonal).
#[derive(Debug, Clone)]
pub struct OrdinaryKrigingModel {
    engine: OrdinaryKrigingEngine<GeoMetric>,
    neighborhood: Option<Neighborhood>,
}

impl OrdinaryKrigingModel {
    pub fn new(dataset: GeoDataset, variogram: VariogramModel) -> Result<Self, KrigingError> {
        Self::new_with_extra_diagonal_internal(dataset, variogram, &[])
    }

    /// Like [`new`](Self::new) but adds `extra` (length `n`) to each main-diagonal
    /// covariance term for that station, modeling observation-specific (non-spatial) noise
    /// on top of the nugget, micro-scale, and [`kriging_diagonal_jitter`].
    ///
    /// Use this for heteroskedastic observation noise, e.g. from binomial / survey sampling
    /// on a transformed (logit) working scale. Each entry must be finite and
    /// non-negative.
    pub fn new_with_extra_diagonal(
        dataset: GeoDataset,
        variogram: VariogramModel,
        extra: Vec<Real>,
    ) -> Result<Self, KrigingError> {
        if !extra.is_empty() && extra.len() != dataset.len() {
            return Err(KrigingError::InvalidInput(
                "extra observation diagonal must be empty (homoscedastic) or the same length as the dataset"
                    .to_string(),
            ));
        }
        for &v in &extra {
            if !v.is_finite() || v < 0.0 {
                return Err(KrigingError::InvalidInput(
                    "observation diagonal entries must be finite and non-negative".to_string(),
                ));
            }
        }
        Self::new_with_extra_diagonal_internal(dataset, variogram, &extra)
    }

    fn new_with_extra_diagonal_internal(
        dataset: GeoDataset,
        variogram: VariogramModel,
        extra: &[Real],
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let engine = OrdinaryKrigingEngine::fit_with_extra_diagonal(
            GeoMetric, coords, values, variogram, extra,
        )?;
        Ok(Self {
            engine,
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

    /// Variogram used when fitting this model.
    pub fn variogram(&self) -> VariogramModel {
        self.engine.variogram()
    }

    /// Number of training stations.
    pub fn len(&self) -> usize {
        self.engine.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<Prediction, KrigingError> {
        if let Some(neighborhood) = self.neighborhood {
            return self.engine.predict_neighborhood(coord, neighborhood);
        }
        self.engine
            .predict(&[coord])
            .map(|mut v| v.pop().expect("single prediction"))
    }

    pub fn predict_batch(&self, coords: &[GeoCoord]) -> Result<Vec<Prediction>, KrigingError> {
        if let Some(neighborhood) = self.neighborhood {
            #[cfg(not(target_arch = "wasm32"))]
            {
                return coords
                    .par_iter()
                    .map(|coord| self.engine.predict_neighborhood(*coord, neighborhood))
                    .collect();
            }
            #[cfg(target_arch = "wasm32")]
            {
                let mut out = Vec::with_capacity(coords.len());
                for &coord in coords {
                    out.push(self.engine.predict_neighborhood(coord, neighborhood)?);
                }
                return Ok(out);
            }
        }
        self.engine.predict(coords)
    }

    /// GPU-accelerated batch prediction. Returns [`KrigingError::BackendUnavailable`] if the GPU
    /// path fails for any reason (adapter missing, Matérn unsupported, readback failure, etc.).
    /// For automatic CPU fallback use [`predict_batch_gpu_or_cpu`](Self::predict_batch_gpu_or_cpu).
    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        let covariances = crate::gpu::build_rhs_covariances_gpu(
            self.engine.coords(),
            coords,
            self.engine.variogram(),
        )
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
        match crate::gpu::build_rhs_covariances_gpu(
            self.engine.coords(),
            coords,
            self.engine.variogram(),
        )
        .await
        {
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
        let covariances = crate::gpu::build_rhs_covariances_gpu_blocking(
            self.engine.coords(),
            coords,
            self.engine.variogram(),
        )
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
        match crate::gpu::build_rhs_covariances_gpu_blocking(
            self.engine.coords(),
            coords,
            self.engine.variogram(),
        ) {
            Ok(covariances) => self.predict_batch_with_covariances(coords, &covariances),
            Err(_) => self.predict_batch(coords),
        }
    }

    #[cfg(feature = "gpu")]
    fn predict_batch_with_covariances(
        &self,
        coords: &[GeoCoord],
        covariances: &[Real],
    ) -> Result<Vec<Prediction>, KrigingError> {
        let n = self.engine.len();
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
        let mut out = Vec::with_capacity(coords.len());
        for pred_idx in 0..coords.len() {
            let c0 = DVector::from_iterator(n, (0..n).map(|i| covariances[pred_idx * n + i]));
            out.push(self.engine.predict_from_cross_cov(c0)?);
        }
        Ok(out)
    }
}

/// Diagonal regularization for kriging covariance blocks (implemented in [`crate::kriging::numerics`]).
pub use crate::kriging::numerics::kriging_diagonal_jitter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_dataset::GeoDataset;
    use crate::variogram::models::VariogramType;

    #[test]
    fn extra_diagonal_nudges_weights_toward_high_trust_sites() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let values = vec![0.0, 0.0, 10.0];
        let variogram = VariogramModel::new(0.01, 5.0, 500.0, VariogramType::Exponential).unwrap();
        let dataset = GeoDataset::new(coords.clone(), values.clone()).unwrap();
        let homo = OrdinaryKrigingModel::new(dataset, variogram).expect("homo");
        // Heavy observation noise on station 2: should pull midpoint prediction down vs homo.
        let extra = vec![0.0, 0.0, 2.0];
        let het = OrdinaryKrigingModel::new_with_extra_diagonal(
            GeoDataset::new(coords, values).unwrap(),
            variogram,
            extra,
        )
        .expect("het");
        let t = GeoCoord::try_new(0.1, 0.1).unwrap();
        let ph = homo.predict(t).expect("h").value;
        let phe = het.predict(t).expect("e").value;
        assert!(
            phe < ph,
            "noisy high-value site should be down-weighted: phe={phe} ph={ph}"
        );
    }

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
        let gpu = match model.predict_batch_gpu_blocking(&query_coords) {
            Ok(v) => v,
            Err(crate::error::KrigingError::BackendUnavailable(msg)) => {
                eprintln!("skipping GPU test: backend unavailable: {msg}");
                return;
            }
            Err(e) => panic!("gpu batch: {e:?}"),
        };
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
        let gpu = match model.predict_batch_gpu_blocking(&query_coords) {
            Ok(v) => v,
            Err(crate::error::KrigingError::BackendUnavailable(msg)) => {
                eprintln!("skipping GPU test: backend unavailable: {msg}");
                return;
            }
            Err(e) => panic!("gpu batch: {e:?}"),
        };
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
