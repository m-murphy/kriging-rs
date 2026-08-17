//! Shared CV option shapes for the unified WASM seam.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VariogramParams {
    pub variogram_type: String,
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
    pub shape: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UnifiedCvOptions {
    pub geometry: String,
    pub family: String,
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub lats: Vec<f64>,
    #[serde(default)]
    pub lons: Vec<f64>,
    #[serde(default)]
    pub xs: Vec<f64>,
    #[serde(default)]
    pub ys: Vec<f64>,
    #[serde(default)]
    pub values: Vec<f64>,
    #[serde(default)]
    pub successes: Vec<u32>,
    #[serde(default)]
    pub trials: Vec<u32>,
    #[serde(default)]
    pub times: Vec<f64>,
    pub mean: Option<f64>,
    pub trend: Option<String>,
    pub major_angle_deg: Option<f64>,
    pub range_ratio: Option<f64>,
    pub variogram: Option<VariogramParams>,
    pub prior_alpha: Option<f64>,
    pub prior_beta: Option<f64>,
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
