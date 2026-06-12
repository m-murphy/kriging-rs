//! Shared simulation option shapes for the unified WASM seam.

use serde::Deserialize;

use super::cv_options::VariogramParams;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UnifiedSimulateOptions {
    pub geometry: String,
    pub family: String,
    #[serde(default)]
    pub conditioning_lats: Vec<f64>,
    #[serde(default)]
    pub conditioning_lons: Vec<f64>,
    #[serde(default)]
    pub conditioning_xs: Vec<f64>,
    #[serde(default)]
    pub conditioning_ys: Vec<f64>,
    #[serde(default)]
    pub conditioning_times: Vec<f64>,
    #[serde(default)]
    pub conditioning_values: Vec<f64>,
    #[serde(default)]
    pub conditioning_successes: Vec<u32>,
    #[serde(default)]
    pub conditioning_trials: Vec<u32>,
    #[serde(default)]
    pub target_lats: Vec<f64>,
    #[serde(default)]
    pub target_lons: Vec<f64>,
    #[serde(default)]
    pub target_xs: Vec<f64>,
    #[serde(default)]
    pub target_ys: Vec<f64>,
    #[serde(default)]
    pub target_times: Vec<f64>,
    pub variogram: Option<VariogramParams>,
    pub mean: Option<f64>,
    pub trend: Option<String>,
    pub major_angle_deg: Option<f64>,
    pub range_ratio: Option<f64>,
    pub prior_alpha: Option<f64>,
    pub prior_beta: Option<f64>,
    #[serde(default)]
    pub seed: u64,
    pub base_seed: Option<u64>,
    pub n_realizations: Option<u32>,
    pub target_order: Option<Vec<u32>>,
    pub space_time_family: Option<String>,
    pub spatial_type: Option<String>,
    pub spatial_nugget: Option<f64>,
    pub spatial_sill: Option<f64>,
    pub spatial_range: Option<f64>,
    pub spatial_shape: Option<f64>,
    pub temporal_type: Option<String>,
    pub temporal_nugget: Option<f64>,
    pub temporal_sill: Option<f64>,
    pub temporal_range: Option<f64>,
    pub temporal_shape: Option<f64>,
    pub k1: Option<f64>,
    pub k2: Option<f64>,
    pub k3: Option<f64>,
}

impl UnifiedSimulateOptions {
    pub(crate) fn realization_count(&self) -> u32 {
        self.n_realizations.unwrap_or(1).max(1)
    }

    pub(crate) fn is_many(&self) -> bool {
        self.realization_count() > 1
    }

    pub(crate) fn effective_base_seed(&self) -> u64 {
        self.base_seed.unwrap_or(self.seed)
    }
}
