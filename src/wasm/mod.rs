// The zero-copy WASM bindings intentionally take flat scalar arguments instead of bundling
// them into serde-deserialized option structs: it avoids a heavy `from_value` deserialization
// step and keeps JS callers from having to construct and garbage-collect a JS object per call.
// This pattern trips clippy's `too_many_arguments` lint.
#![allow(clippy::too_many_arguments)]
// JS interop only accepts `f64`, so this module casts every `Real` value to `f64` at the
// boundary. When the `f64` Cargo feature is on `Real == f64` and clippy flags those casts
// as redundant, but the casts are meaningful in the default `f32` build and removing them
// would break compilation there.
#![allow(clippy::unnecessary_cast)]

use js_sys::{Float64Array, Object, Reflect, Uint32Array};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::aggregate::{PolygonAggregationSummary, polygon_weighted_summaries_batch};
use crate::cv::{
    BinomialCvResidual, BinomialCvSummary, BinomialGeoPredictor, BinomialProjectedPredictor,
};
use crate::distance::GeoCoord;
use crate::geo_dataset::GeoDataset;
#[cfg(feature = "gpu")]
use crate::gpu::detect_gpu_support;
use crate::kriging::binomial::{
    BinomialBuildNotes, BinomialKrigingModel, BinomialObservation, BinomialPrior,
    BinomialStability, HeteroskedasticBinomialConfig,
    build_binomial_observations_dropping_zero_trials, estimate_binomial_prior_from_counts,
    fit_binomial_variogram,
};
use crate::kriging::ordinary::{Neighborhood, OrdinaryKrigingModel};
use crate::kriging::simple::SimpleKrigingModel;
use crate::kriging::universal::{UniversalKrigingModel, UniversalTrend};
use crate::predictor::cv::leave_one_out_cv;
use crate::projected::{
    Anisotropy2D, BinomialProjectedKrigingModel, BinomialTangentPlaneKrigingModel,
    DirectionalConfig, ProjectedBinomialObservation, ProjectedCoord, ProjectedDataset,
    ProjectedKrigingModel, compute_directional_empirical_variogram,
    tangent_plane_reference_centroid,
};
use crate::simulation::{BinomialSimulationManyResult, BinomialSimulationResult};
use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal, VariogramConfig};
use crate::variogram::fitting::fit_variogram;
use crate::variogram::models::{VariogramModel, VariogramType};
use crate::variogram::nested::NestedVariogram;
use crate::{Real, compute_empirical_variogram};
use std::num::NonZeroUsize;

mod cv_dispatch;
mod cv_options;
mod model_cv;
mod model_handle;
mod simulate_dispatch;
mod simulate_options;

pub mod spacetime;

pub use cv_dispatch::wasm_cv;
pub use simulate_dispatch::wasm_simulate;

/// WASM-exposed variogram type enum; maps to crate's VariogramType.
#[wasm_bindgen]
pub enum WasmVariogramType {
    Spherical,
    Exponential,
    Gaussian,
    Cubic,
    Stable,
    Matern,
    Power,
    HoleEffect,
}

impl From<WasmVariogramType> for VariogramType {
    fn from(w: WasmVariogramType) -> Self {
        match w {
            WasmVariogramType::Spherical => VariogramType::Spherical,
            WasmVariogramType::Exponential => VariogramType::Exponential,
            WasmVariogramType::Gaussian => VariogramType::Gaussian,
            WasmVariogramType::Cubic => VariogramType::Cubic,
            WasmVariogramType::Stable => VariogramType::Stable,
            WasmVariogramType::Matern => VariogramType::Matern,
            WasmVariogramType::Power => VariogramType::Power,
            WasmVariogramType::HoleEffect => VariogramType::HoleEffect,
        }
    }
}

pub(super) fn parse_variogram(
    variogram_type: &str,
    nugget: f64,
    sill: f64,
    range: f64,
    shape: Option<f64>,
) -> Result<VariogramModel, JsValue> {
    let vt = match variogram_type.to_ascii_lowercase().as_str() {
        "spherical" => VariogramType::Spherical,
        "exponential" => VariogramType::Exponential,
        "gaussian" => VariogramType::Gaussian,
        "cubic" => VariogramType::Cubic,
        "stable" => VariogramType::Stable,
        "matern" => VariogramType::Matern,
        "power" => VariogramType::Power,
        "holeeffect" | "hole_effect" | "hole-effect" => VariogramType::HoleEffect,
        _ => return Err(coded_err("unknown variogram_type", "unknown_variogram")),
    };
    match (vt, shape) {
        (VariogramType::Stable, Some(s))
        | (VariogramType::Matern, Some(s))
        | (VariogramType::Power, Some(s)) => VariogramModel::new_with_shape(
            nugget as Real,
            sill as Real,
            range as Real,
            vt,
            s as Real,
        )
        .map_err(kriging_err_to_js),
        _ => VariogramModel::new(nugget as Real, sill as Real, range as Real, vt)
            .map_err(kriging_err_to_js),
    }
}

/// Build a row-major grid of `GeoCoord`s spanning `[y_min, y_max] × [x_min, x_max]`. The
/// row axis (outer) is y (latitude), the column axis (inner) is x (longitude). When a
/// cell count is 1 the midpoint of the range is used; otherwise cell centers are spaced
/// uniformly from `min + step/2` to `max − step/2`.
fn build_grid_coords(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x_cells: usize,
    y_cells: usize,
) -> Result<Vec<GeoCoord>, JsValue> {
    if x_cells == 0 || y_cells == 0 {
        return Err(coded_err(
            "xCells and yCells must both be positive",
            "invalid_input",
        ));
    }
    if !(x_min.is_finite() && x_max.is_finite() && y_min.is_finite() && y_max.is_finite()) {
        return Err(coded_err("grid bounds must all be finite", "invalid_input"));
    }
    if x_max <= x_min || y_max <= y_min {
        return Err(coded_err(
            "xMax must exceed xMin and yMax must exceed yMin",
            "invalid_input",
        ));
    }
    let dx = (x_max - x_min) / x_cells as f64;
    let dy = (y_max - y_min) / y_cells as f64;
    let mut coords = Vec::with_capacity(x_cells * y_cells);
    for r in 0..y_cells {
        let lat = y_min + (r as f64 + 0.5) * dy;
        for c in 0..x_cells {
            let lon = x_min + (c as f64 + 0.5) * dx;
            coords.push(GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?);
        }
    }
    Ok(coords)
}

/// Row-major grid of [`ProjectedCoord`] over `[x_min, x_max] × [y_min, y_max]` (planar meters
/// or other projected units). Outer index is **y** (row, low to high `y`), inner is **x**
/// (column, low to high `x`), matching the geographic grid row-major layout (`yCells` × `xCells`).
fn build_projected_grid_coords(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x_cells: usize,
    y_cells: usize,
) -> Result<Vec<ProjectedCoord>, JsValue> {
    if x_cells == 0 || y_cells == 0 {
        return Err(coded_err(
            "xCells and yCells must both be positive",
            "invalid_input",
        ));
    }
    if !(x_min.is_finite() && x_max.is_finite() && y_min.is_finite() && y_max.is_finite()) {
        return Err(coded_err("grid bounds must all be finite", "invalid_input"));
    }
    if x_max <= x_min || y_max <= y_min {
        return Err(coded_err(
            "xMax must exceed xMin and yMax must exceed yMin",
            "invalid_input",
        ));
    }
    let dx = (x_max - x_min) / x_cells as f64;
    let dy = (y_max - y_min) / y_cells as f64;
    let mut coords = Vec::with_capacity(x_cells * y_cells);
    for r in 0..y_cells {
        let y = y_min + (r as f64 + 0.5) * dy;
        for c in 0..x_cells {
            let x = x_min + (c as f64 + 0.5) * dx;
            coords.push(ProjectedCoord::new(x as Real, y as Real));
        }
    }
    Ok(coords)
}

pub(super) fn to_coords(lats: &[f64], lons: &[f64]) -> Result<Vec<GeoCoord>, JsValue> {
    if lats.len() != lons.len() {
        return Err(coded_err(
            "lats and lons must have same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        out.push(GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

/// Map any display-able error to a coded JS error object. Defaults `code` to `invalid_input`.
/// Prefer [`kriging_err_to_js`] for `KrigingError` so the right code is attached.
pub(super) fn err_to_js(err: impl std::fmt::Display) -> JsValue {
    coded_err(&err.to_string(), "invalid_input")
}

/// Map a [`KrigingError`] to a JS `Error`-like object with `message` and a stable `code` field.
pub(super) fn kriging_err_to_js(err: crate::error::KrigingError) -> JsValue {
    let code = error_code_for(&err);
    coded_err(&err.to_string(), code)
}

pub(super) fn coded_err(message: &str, code: &str) -> JsValue {
    let obj = Object::new();
    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("message"),
        &JsValue::from_str(message),
    );
    let _ = Reflect::set(&obj, &JsValue::from_str("code"), &JsValue::from_str(code));
    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("name"),
        &JsValue::from_str("KrigingError"),
    );
    obj.into()
}

fn error_code_for(err: &crate::error::KrigingError) -> &'static str {
    use crate::error::KrigingError;
    match err {
        KrigingError::InsufficientData(_) => "too_few_points",
        KrigingError::DimensionMismatch(_) => "mismatched_arrays",
        KrigingError::InvalidCoordinate { .. } => "invalid_input",
        KrigingError::MatrixError(_) => "singular_covariance",
        KrigingError::FittingError(_) => "invalid_variogram",
        KrigingError::InvalidBinomialData(_) => "invalid_input",
        KrigingError::BackendUnavailable(_) => "backend_unavailable",
        KrigingError::InvalidInput(_) => "invalid_input",
    }
}

/// Geo binomial call sites: drop `trials == 0` rows, keep their original row indices
/// (for [`BinomialKrigingModel::new_with_config`] / build notes).
fn build_observations(
    lats: &[f64],
    lons: &[f64],
    successes: &[u32],
    trials: &[u32],
) -> Result<(Vec<BinomialObservation>, Vec<usize>), JsValue> {
    if lats.len() != lons.len() || lats.len() != successes.len() || lats.len() != trials.len() {
        return Err(coded_err(
            "all input arrays must have same length",
            "mismatched_arrays",
        ));
    }
    let mut coords = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        coords
            .push(GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?);
    }
    build_binomial_observations_dropping_zero_trials(coords, successes, trials)
        .map_err(kriging_err_to_js)
}

fn binomial_cv_slices(observations: &[BinomialObservation]) -> (Vec<GeoCoord>, Vec<u32>, Vec<u32>) {
    let mut cv_coords = Vec::with_capacity(observations.len());
    let mut cv_successes = Vec::with_capacity(observations.len());
    let mut cv_trials = Vec::with_capacity(observations.len());
    for obs in observations {
        cv_coords.push(obs.coord());
        cv_successes.push(obs.successes());
        cv_trials.push(obs.trials());
    }
    (cv_coords, cv_successes, cv_trials)
}

fn binomial_projected_cv_slices(
    observations: &[ProjectedBinomialObservation],
) -> (Vec<ProjectedCoord>, Vec<u32>, Vec<u32>) {
    let mut cv_coords = Vec::with_capacity(observations.len());
    let mut cv_successes = Vec::with_capacity(observations.len());
    let mut cv_trials = Vec::with_capacity(observations.len());
    for obs in observations {
        cv_coords.push(obs.coord());
        cv_successes.push(obs.successes());
        cv_trials.push(obs.trials());
    }
    (cv_coords, cv_successes, cv_trials)
}

#[derive(Debug, Serialize)]
pub(super) struct JsPrediction {
    pub value: f64,
    pub variance: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsFittedVariogram {
    variogram_type: String,
    nugget: f64,
    sill: f64,
    range: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape: Option<f64>,
    residuals: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct JsBinomialPrediction {
    pub prevalence_median: f64,
    pub prevalence_mean: f64,
    pub logit: f64,
    pub logit_variance: f64,
    pub prevalence_variance: f64,
}

type SplitBinomialArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn variogram_type_name(variogram_type: VariogramType) -> &'static str {
    match variogram_type {
        VariogramType::Spherical => "spherical",
        VariogramType::Exponential => "exponential",
        VariogramType::Gaussian => "gaussian",
        VariogramType::Cubic => "cubic",
        VariogramType::Stable => "stable",
        VariogramType::Matern => "matern",
        VariogramType::Power => "power",
        VariogramType::HoleEffect => "holeeffect",
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsVariogramParams {
    variogram_type: String,
    nugget: f64,
    sill: f64,
    range: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape: Option<f64>,
}

fn variogram_to_js_params(vm: VariogramModel) -> JsVariogramParams {
    let vt = vm.variogram_type();
    let (nugget, sill, range) = vm.params();
    JsVariogramParams {
        variogram_type: variogram_type_name(vt).to_string(),
        nugget: nugget as f64,
        sill: sill as f64,
        range: range as f64,
        shape: vm.shape().map(|s| s as f64),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialDiagnosticsLooGeo {
    lats: Vec<f64>,
    lons: Vec<f64>,
    successes: Vec<u32>,
    trials: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialDiagnosticsLooProjected {
    xs: Vec<f64>,
    ys: Vec<f64>,
    successes: Vec<u32>,
    trials: Vec<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsBinomialDiagnosticsOut {
    variogram: JsVariogramParams,
    build_notes: BinomialBuildNotes,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_loo_msdr: Option<f64>,
}

fn binomial_diagnostics_geo_js(
    inner: &BinomialKrigingModel,
    build_notes: &BinomialBuildNotes,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let vm = inner.variogram();
    let variogram = variogram_to_js_params(vm);
    let logit_loo_msdr = if options.is_undefined() || options.is_null() {
        None
    } else {
        let inp: BinomialDiagnosticsLooGeo =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let n = inner.len();
        if inp.lats.len() != n
            || inp.lons.len() != n
            || inp.successes.len() != n
            || inp.trials.len() != n
        {
            return Err(coded_err(
                "loo arrays must match model station count",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords(&inp.lats, &inp.lons)?;
        let residuals = leave_one_out_cv(&BinomialGeoPredictor {
            coords: &coords,
            successes: &inp.successes,
            trials: &inp.trials,
            variogram: vm,
            prior: build_notes.prior,
        })
        .map_err(kriging_err_to_js)?;
        let summary = BinomialCvSummary::from_residuals(&residuals);
        Some(summary.logit.msdr as f64)
    };
    serde_wasm_bindgen::to_value(&JsBinomialDiagnosticsOut {
        variogram,
        build_notes: build_notes.clone(),
        logit_loo_msdr,
    })
    .map_err(err_to_js)
}

fn binomial_diagnostics_projected_js(
    inner: &BinomialProjectedKrigingModel,
    build_notes: &BinomialBuildNotes,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let vm = inner.variogram();
    let variogram = variogram_to_js_params(vm);
    let ani = inner.anisotropy();
    let logit_loo_msdr = if options.is_undefined() || options.is_null() {
        None
    } else {
        let inp: BinomialDiagnosticsLooProjected =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let n = inner.len();
        if inp.xs.len() != n
            || inp.ys.len() != n
            || inp.successes.len() != n
            || inp.trials.len() != n
        {
            return Err(coded_err(
                "loo arrays must match model station count",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = inp
            .xs
            .iter()
            .zip(inp.ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let residuals = leave_one_out_cv(&BinomialProjectedPredictor {
            coords: &coords,
            successes: &inp.successes,
            trials: &inp.trials,
            variogram: vm,
            anisotropy: ani,
            prior: build_notes.prior,
        })
        .map_err(kriging_err_to_js)?;
        let summary = BinomialCvSummary::from_residuals(&residuals);
        Some(summary.logit.msdr as f64)
    };
    serde_wasm_bindgen::to_value(&JsBinomialDiagnosticsOut {
        variogram,
        build_notes: build_notes.clone(),
        logit_loo_msdr,
    })
    .map_err(err_to_js)
}

fn binomial_diagnostics_tangent_geo_js(
    inner: &BinomialTangentPlaneKrigingModel,
    build_notes: &BinomialBuildNotes,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let vm = inner.variogram();
    let variogram = variogram_to_js_params(vm);
    let ani = inner.anisotropy();
    let reference = inner.reference();
    let logit_loo_msdr = if options.is_undefined() || options.is_null() {
        None
    } else {
        let inp: BinomialDiagnosticsLooGeo =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let n = inner.len();
        if inp.lats.len() != n
            || inp.lons.len() != n
            || inp.successes.len() != n
            || inp.trials.len() != n
        {
            return Err(coded_err(
                "loo arrays must match model station count",
                "mismatched_arrays",
            ));
        }
        let mut pcs = Vec::with_capacity(n);
        for i in 0..n {
            let g = GeoCoord::try_new(inp.lats[i] as Real, inp.lons[i] as Real)
                .map_err(kriging_err_to_js)?;
            pcs.push(ProjectedCoord::equirectangular(g, reference));
        }
        let residuals = leave_one_out_cv(&BinomialProjectedPredictor {
            coords: &pcs,
            successes: &inp.successes,
            trials: &inp.trials,
            variogram: vm,
            anisotropy: ani,
            prior: build_notes.prior,
        })
        .map_err(kriging_err_to_js)?;
        let summary = BinomialCvSummary::from_residuals(&residuals);
        Some(summary.logit.msdr as f64)
    };
    serde_wasm_bindgen::to_value(&JsBinomialDiagnosticsOut {
        variogram,
        build_notes: build_notes.clone(),
        logit_loo_msdr,
    })
    .map_err(err_to_js)
}

pub(super) fn map_predictions(out: Vec<crate::kriging::ordinary::Prediction>) -> Vec<JsPrediction> {
    out.into_iter()
        .map(|p| JsPrediction {
            value: p.value as f64,
            variance: p.variance as f64,
        })
        .collect::<Vec<_>>()
}

pub(super) fn split_predictions(
    out: Vec<crate::kriging::ordinary::Prediction>,
) -> (Vec<f64>, Vec<f64>) {
    let mut values = Vec::with_capacity(out.len());
    let mut variances = Vec::with_capacity(out.len());
    for pred in out {
        values.push(pred.value as f64);
        variances.push(pred.variance as f64);
    }
    (values, variances)
}

pub(super) fn map_binomial_predictions(
    out: Vec<crate::kriging::binomial::BinomialPrediction>,
) -> Vec<JsBinomialPrediction> {
    out.into_iter()
        .map(|p| JsBinomialPrediction {
            prevalence_median: p.prevalence_median as f64,
            prevalence_mean: p.prevalence_mean as f64,
            logit: p.logit as f64,
            logit_variance: p.logit_variance as f64,
            prevalence_variance: p.prevalence_variance as f64,
        })
        .collect::<Vec<_>>()
}

pub(super) fn split_binomial_predictions(
    out: Vec<crate::kriging::binomial::BinomialPrediction>,
) -> SplitBinomialArrays {
    let mut prevalence_medians = Vec::with_capacity(out.len());
    let mut prevalence_means = Vec::with_capacity(out.len());
    let mut logits = Vec::with_capacity(out.len());
    let mut logit_variances = Vec::with_capacity(out.len());
    let mut prevalence_variances = Vec::with_capacity(out.len());
    for pred in out {
        prevalence_medians.push(pred.prevalence_median as f64);
        prevalence_means.push(pred.prevalence_mean as f64);
        logits.push(pred.logit as f64);
        logit_variances.push(pred.logit_variance as f64);
        prevalence_variances.push(pred.prevalence_variance as f64);
    }
    (
        prevalence_medians,
        prevalence_means,
        logits,
        logit_variances,
        prevalence_variances,
    )
}

pub(super) fn set_binomial_flat_array_fields(
    result: &Object,
    prevalence_medians: &[f64],
    prevalence_means: &[f64],
    logits: &[f64],
    logit_variances: &[f64],
    prevalence_variances: &[f64],
) -> Result<(), JsValue> {
    set_object_field(
        result,
        "prevalenceMedians",
        &Float64Array::from(prevalence_medians).into(),
    )?;
    set_object_field(
        result,
        "prevalenceMeans",
        &Float64Array::from(prevalence_means).into(),
    )?;
    set_object_field(result, "logitValues", &Float64Array::from(logits).into())?;
    set_object_field(
        result,
        "logitVariances",
        &Float64Array::from(logit_variances).into(),
    )?;
    set_object_field(
        result,
        "prevalenceVariances",
        &Float64Array::from(prevalence_variances).into(),
    )?;
    Ok(())
}

pub(super) fn set_object_field(obj: &Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    match Reflect::set(obj, &JsValue::from_str(key), value) {
        Ok(true) => Ok(()),
        Ok(false) => Err(coded_err(
            &format!("failed to set property '{key}' on result object"),
            "internal_error",
        )),
        Err(e) => Err(e),
    }
}

/// Options for ordinary kriging model construction (JS: single object argument).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OrdinaryKrigingOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    values: Vec<f64>,
    variogram: VariogramParams,
}

/// Variogram parameters (nugget, sill, range, optional shape).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VariogramParams {
    variogram_type: String,
    nugget: f64,
    sill: f64,
    range: f64,
    #[serde(default)]
    shape: Option<f64>,
}

/// Options for binomial kriging model construction (JS: single object argument).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialKrigingOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    successes: Vec<u32>,
    trials: Vec<u32>,
    variogram: VariogramParams,
    #[serde(default)]
    stability: Option<String>,
    #[serde(default)]
    one_step_laplace_observation_variance: Option<bool>,
}

/// Prior parameters for binomial kriging (Beta(alpha, beta)).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialPriorParams {
    alpha: f64,
    beta: f64,
}

/// Options for binomial kriging with prior (JS: single object argument).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialKrigingWithPriorOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    successes: Vec<u32>,
    trials: Vec<u32>,
    variogram: VariogramParams,
    prior: BinomialPriorParams,
    #[serde(default)]
    stability: Option<String>,
    #[serde(default)]
    one_step_laplace_observation_variance: Option<bool>,
}

/// Options for tangent-plane geographic binomial kriging (local equirectangular km plane + 2-D anisotropy).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialTangentPlaneKrigingOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    successes: Vec<u32>,
    trials: Vec<u32>,
    variogram: VariogramParams,
    major_angle_deg: f64,
    range_ratio: f64,
    #[serde(default)]
    tangent_plane_ref_lat: Option<f64>,
    #[serde(default)]
    tangent_plane_ref_lon: Option<f64>,
    #[serde(default)]
    stability: Option<String>,
    #[serde(default)]
    one_step_laplace_observation_variance: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinomialTangentPlaneKrigingWithPriorOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    successes: Vec<u32>,
    trials: Vec<u32>,
    variogram: VariogramParams,
    prior: BinomialPriorParams,
    major_angle_deg: f64,
    range_ratio: f64,
    #[serde(default)]
    tangent_plane_ref_lat: Option<f64>,
    #[serde(default)]
    tangent_plane_ref_lon: Option<f64>,
    #[serde(default)]
    stability: Option<String>,
    #[serde(default)]
    one_step_laplace_observation_variance: Option<bool>,
}

pub(crate) struct WasmOrdinaryKriging {
    inner: OrdinaryKrigingModel,
}

impl WasmOrdinaryKriging {
    pub fn new(options: JsValue) -> Result<WasmOrdinaryKriging, JsValue> {
        let opts: OrdinaryKrigingOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let coords = to_coords(&opts.lats, &opts.lons)?;
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let values_real = opts.values.iter().map(|v| *v as Real).collect::<Vec<_>>();
        let dataset = GeoDataset::new(coords, values_real).map_err(kriging_err_to_js)?;
        let inner = OrdinaryKrigingModel::new(dataset, model).map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    /// Zero-(extra-)copy factory: takes typed arrays directly instead of a JS object, so the
    /// large `lats`/`lons`/`values` slices skip the `serde_wasm_bindgen` deserialization step.
    /// The variogram parameters are passed as scalars.
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmOrdinaryKriging, JsValue> {
        if values.len() != lats.len() {
            return Err(coded_err(
                "values must have the same length as lats/lons",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords(lats, lons)?;
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let values_real = values.iter().map(|v| *v as Real).collect::<Vec<_>>();
        let dataset = GeoDataset::new(coords, values_real).map_err(kriging_err_to_js)?;
        let inner = OrdinaryKrigingModel::new(dataset, model).map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    /// Enable a search neighborhood that restricts which stations are used at each
    /// prediction location. Pass `maxNeighbors` (nearest-k), `maxRadius` (kilometers), or
    /// both (intersection). Pass neither to clear any existing neighborhood and return to
    /// the full-data fast path.
    pub fn set_neighborhood(
        &mut self,
        max_neighbors: Option<usize>,
        max_radius: Option<f64>,
    ) -> Result<(), JsValue> {
        let neighborhood = match (max_neighbors, max_radius) {
            (None, None) => None,
            (k, r) => {
                if let Some(r) = r
                    && (!r.is_finite() || r <= 0.0)
                {
                    return Err(coded_err(
                        "maxRadius must be finite and positive",
                        "invalid_input",
                    ));
                }
                Some(Neighborhood {
                    max_neighbors: k,
                    max_radius: r.map(|r| r as Real),
                })
            }
        };
        self.inner.set_neighborhood(neighborhood);
        Ok(())
    }

    /// Returns the current search neighborhood as `{ maxNeighbors?, maxRadius? }`, or
    /// `null` when no neighborhood is active.
    pub fn neighborhood(&self) -> JsValue {
        match self.inner.neighborhood() {
            None => JsValue::NULL,
            Some(n) => {
                let obj = Object::new();
                if let Some(k) = n.max_neighbors {
                    let _ = set_object_field(&obj, "maxNeighbors", &JsValue::from_f64(k as f64));
                }
                if let Some(r) = n.max_radius {
                    let _ = set_object_field(&obj, "maxRadius", &JsValue::from_f64(r as f64));
                }
                obj.into()
            }
        }
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let values_array = Float64Array::from(values.as_slice());
        let variances_array = Float64Array::from(variances.as_slice());
        let result = Object::new();
        set_object_field(&result, "values", &values_array.into())?;
        set_object_field(&result, "variances", &variances_array.into())?;
        Ok(result.into())
    }

    /// Predict a regular lat/lon grid in a single call, returning flat `Float64Array`s of
    /// length `xCells × yCells` in row-major (y-outer, x-inner) order. Avoids constructing
    /// the lat/lon coordinate arrays in JS.
    pub fn predict_grid_arrays(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_cells: usize,
        y_cells: usize,
    ) -> Result<JsValue, JsValue> {
        let coords = build_grid_coords(x_min, x_max, y_min, y_max, x_cells, y_cells)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        set_object_field(&result, "nRows", &JsValue::from_f64(y_cells as f64))?;
        set_object_field(&result, "nCols", &JsValue::from_f64(x_cells as f64))?;
        Ok(result.into())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch_gpu(&coords)
            .await
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu_or_cpu(
        &self,
        lats: &[f64],
        lons: &[f64],
    ) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch_gpu_or_cpu(&coords)
            .await
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    /// Leave-one-out cross-validation using this model's training data and variogram.
    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        model_cv::ordinary_geo_cv(
            self.inner.coords(),
            self.inner.values(),
            self.inner.variogram(),
            None,
        )
    }

    /// K-fold cross-validation (deterministic round-robin) on this model's training data.
    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::ordinary_geo_cv(
            self.inner.coords(),
            self.inner.values(),
            self.inner.variogram(),
            Some(k),
        )
    }
}

pub(crate) struct WasmBinomialKriging {
    inner: BinomialKrigingModel,
    build_notes: BinomialBuildNotes,
    cv_coords: Vec<GeoCoord>,
    cv_successes: Vec<u32>,
    cv_trials: Vec<u32>,
}

impl WasmBinomialKriging {
    pub fn new(options: JsValue) -> Result<WasmBinomialKriging, JsValue> {
        let opts: BinomialKrigingOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let (observations, zero_trial_drops) =
            build_observations(&opts.lats, &opts.lons, &opts.successes, &opts.trials)?;
        if observations.len() < 2 {
            return Err(coded_err(
                "need at least two non-zero-trial sites after dropping trials==0 rows",
                "too_few_points",
            ));
        }
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(opts.stability.as_deref())?,
            opts.one_step_laplace_observation_variance,
        );
        let (cv_coords, cv_successes, cv_trials) = binomial_cv_slices(&observations);
        let fit = BinomialKrigingModel::new_with_config(
            observations,
            model,
            BinomialPrior::default(),
            hcfg,
            &zero_trial_drops,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords,
            cv_successes,
            cv_trials,
        })
    }

    /// Zero-(extra-)copy factory: takes typed arrays directly instead of a JS object. See
    /// [`WasmOrdinaryKriging::from_arrays`] for the motivation.
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmBinomialKriging, JsValue> {
        let (observations, zero_trial_drops) = build_observations(lats, lons, successes, trials)?;
        if observations.len() < 2 {
            return Err(coded_err(
                "need at least two non-zero-trial sites after dropping trials==0 rows",
                "too_few_points",
            ));
        }
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let (cv_coords, cv_successes, cv_trials) = binomial_cv_slices(&observations);
        let fit = BinomialKrigingModel::new_with_config(
            observations,
            model,
            BinomialPrior::default(),
            hcfg,
            &zero_trial_drops,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords,
            cv_successes,
            cv_trials,
        })
    }

    /// Factory for binomial kriging when the caller already has finite logit values (for
    /// example from an externally fitted mean-field model). Bypasses the empirical-Bayes
    /// shrinkage step used by [`Self::new`] / [`Self::from_arrays`].
    pub fn from_precomputed_logits(
        lats: &[f64],
        lons: &[f64],
        logits: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmBinomialKriging, JsValue> {
        if logits.len() != lats.len() {
            return Err(coded_err(
                "logits must have the same length as lats/lons",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords(lats, lons)?;
        let logits_real: Vec<Real> = logits.iter().map(|v| *v as Real).collect();
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let fit = BinomialKrigingModel::from_precomputed_logits(coords, logits_real, model)
            .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords: Vec::new(),
            cv_successes: Vec::new(),
            cv_trials: Vec::new(),
        })
    }

    /// Pre-computed logits with a per-site **logit observation variance** on the diagonal
    /// (same length as coordinates). Optional `prior_alpha` / `prior_beta` (must be supplied
    /// together) set the prior on build notes; optional `stability` selects the heteroskedastic
    /// inflation preset (`"default"`, `"strict"`, or `"permissive"`, case-insensitive).
    pub fn from_precomputed_logits_with_variances(
        lats: &[f64],
        lons: &[f64],
        logits: &[f64],
        logit_observation_variance: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        prior_alpha: Option<f64>,
        prior_beta: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmBinomialKriging, JsValue> {
        if logits.len() != lats.len() || logits.len() != lons.len() {
            return Err(coded_err(
                "logits must have the same length as lats and lons",
                "mismatched_arrays",
            ));
        }
        if logit_observation_variance.len() != logits.len() {
            return Err(coded_err(
                "logitObservationVariance must have the same length as logits",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords(lats, lons)?;
        let logits_real: Vec<Real> = logits.iter().map(|v| *v as Real).collect();
        let base_var: Vec<Real> = logit_observation_variance
            .iter()
            .map(|v| *v as Real)
            .collect();
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let fit = BinomialKrigingModel::from_precomputed_logits_with_logit_observation_variances(
            coords,
            logits_real,
            model,
            base_var,
            hcfg,
            prior,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords: Vec::new(),
            cv_successes: Vec::new(),
            cv_trials: Vec::new(),
        })
    }

    pub fn new_with_prior(options: JsValue) -> Result<WasmBinomialKriging, JsValue> {
        let opts: BinomialKrigingWithPriorOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let (observations, zero_trial_drops) =
            build_observations(&opts.lats, &opts.lons, &opts.successes, &opts.trials)?;
        if observations.len() < 2 {
            return Err(coded_err(
                "need at least two non-zero-trial sites after dropping trials==0 rows",
                "too_few_points",
            ));
        }
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let prior = BinomialPrior::new(opts.prior.alpha as Real, opts.prior.beta as Real)
            .map_err(kriging_err_to_js)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(opts.stability.as_deref())?,
            opts.one_step_laplace_observation_variance,
        );
        let (cv_coords, cv_successes, cv_trials) = binomial_cv_slices(&observations);
        let fit = BinomialKrigingModel::new_with_config(
            observations,
            model,
            prior,
            hcfg,
            &zero_trial_drops,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords,
            cv_successes,
            cv_trials,
        })
    }

    /// Build diagnostics: calibration version, prior, logit inflation, and dropped
    /// zero-trial input rows (by original array index). See [`BinomialBuildNotes`].
    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.build_notes).map_err(err_to_js)
    }

    /// Variogram parameters, [`BinomialBuildNotes`], and optional leave-one-out logit MSDR.
    ///
    /// Pass a JS object `{ lats, lons, successes, trials }` (same lengths as the training
    /// station count) to compute [`BinomialDiagnostics::logit_loo_msdr`]; omit or pass
    /// `undefined` to skip LOO.
    pub fn get_diagnostics(&self, options: JsValue) -> Result<JsValue, JsValue> {
        binomial_diagnostics_geo_js(&self.inner, &self.build_notes, options)
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsBinomialPrediction {
            prevalence_median: pred.prevalence_median as f64,
            prevalence_mean: pred.prevalence_mean as f64,
            logit: pred.logit as f64,
            logit_variance: pred.logit_variance as f64,
            prevalence_variance: pred.prevalence_variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        Ok(result.into())
    }

    /// Predict a regular lat/lon grid in a single call, returning flat `Float64Array`s in
    /// row-major (y-outer, x-inner) order.
    pub fn predict_grid_arrays(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_cells: usize,
        y_cells: usize,
    ) -> Result<JsValue, JsValue> {
        let coords = build_grid_coords(x_min, x_max, y_min, y_max, x_cells, y_cells)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        set_object_field(&result, "nRows", &JsValue::from_f64(y_cells as f64))?;
        set_object_field(&result, "nCols", &JsValue::from_f64(x_cells as f64))?;
        Ok(result.into())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch_gpu(&coords)
            .await
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu_or_cpu(
        &self,
        lats: &[f64],
        lons: &[f64],
    ) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch_gpu_or_cpu(&coords)
            .await
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        if self.cv_coords.is_empty() {
            return Err(coded_err(
                "leaveOneOut requires count data; build the model from successes/trials, not precomputed logits",
                "invalid_input",
            ));
        }
        model_cv::binomial_geo_cv(
            &self.cv_coords,
            &self.cv_successes,
            &self.cv_trials,
            self.inner.variogram(),
            self.build_notes.prior,
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        if self.cv_coords.is_empty() {
            return Err(coded_err(
                "kFold requires count data; build the model from successes/trials, not precomputed logits",
                "invalid_input",
            ));
        }
        model_cv::binomial_geo_cv(
            &self.cv_coords,
            &self.cv_successes,
            &self.cv_trials,
            self.inner.variogram(),
            self.build_notes.prior,
            Some(k),
        )
    }
}

fn wasm_build_binomial_tangent_plane_inner(
    lats: &[f64],
    lons: &[f64],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    prior: BinomialPrior,
    hcfg: HeteroskedasticBinomialConfig,
    ref_lat: Option<f64>,
    ref_lon: Option<f64>,
    major_angle_deg: f64,
    range_ratio: f64,
) -> Result<WasmBinomialTangentPlaneKriging, JsValue> {
    let (observations, zero_trial_drops) = build_observations(lats, lons, successes, trials)?;
    if observations.len() < 2 {
        return Err(coded_err(
            "need at least two non-zero-trial sites after dropping trials==0 rows",
            "too_few_points",
        ));
    }
    let reference = match (ref_lat, ref_lon) {
        (Some(lat), Some(lon)) => {
            GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?
        }
        (None, None) => {
            tangent_plane_reference_centroid(&observations).map_err(kriging_err_to_js)?
        }
        _ => {
            return Err(coded_err(
                "tangentPlaneRefLat and tangentPlaneRefLon must both be set or both omitted",
                "invalid_input",
            ));
        }
    };
    let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
        .map_err(kriging_err_to_js)?;
    let fit = BinomialTangentPlaneKrigingModel::new_with_prior(
        observations,
        variogram,
        prior,
        hcfg,
        &zero_trial_drops,
        reference,
        anisotropy,
    )
    .map_err(kriging_err_to_js)?;
    Ok(WasmBinomialTangentPlaneKriging {
        inner: fit.model,
        build_notes: fit.notes,
    })
}

pub(crate) struct WasmBinomialTangentPlaneKriging {
    inner: BinomialTangentPlaneKrigingModel,
    build_notes: BinomialBuildNotes,
}

impl WasmBinomialTangentPlaneKriging {
    pub fn new(options: JsValue) -> Result<WasmBinomialTangentPlaneKriging, JsValue> {
        let opts: BinomialTangentPlaneKrigingOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(opts.stability.as_deref())?,
            opts.one_step_laplace_observation_variance,
        );
        wasm_build_binomial_tangent_plane_inner(
            &opts.lats,
            &opts.lons,
            &opts.successes,
            &opts.trials,
            model,
            BinomialPrior::default(),
            hcfg,
            opts.tangent_plane_ref_lat,
            opts.tangent_plane_ref_lon,
            opts.major_angle_deg,
            opts.range_ratio,
        )
    }

    pub fn new_with_prior(options: JsValue) -> Result<WasmBinomialTangentPlaneKriging, JsValue> {
        let opts: BinomialTangentPlaneKrigingWithPriorOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let prior = BinomialPrior::new(opts.prior.alpha as Real, opts.prior.beta as Real)
            .map_err(kriging_err_to_js)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(opts.stability.as_deref())?,
            opts.one_step_laplace_observation_variance,
        );
        wasm_build_binomial_tangent_plane_inner(
            &opts.lats,
            &opts.lons,
            &opts.successes,
            &opts.trials,
            model,
            prior,
            hcfg,
            opts.tangent_plane_ref_lat,
            opts.tangent_plane_ref_lon,
            opts.major_angle_deg,
            opts.range_ratio,
        )
    }

    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        major_angle_deg: f64,
        range_ratio: f64,
        tangent_plane_ref_lat: Option<f64>,
        tangent_plane_ref_lon: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmBinomialTangentPlaneKriging, JsValue> {
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        wasm_build_binomial_tangent_plane_inner(
            lats,
            lons,
            successes,
            trials,
            model,
            BinomialPrior::default(),
            hcfg,
            tangent_plane_ref_lat,
            tangent_plane_ref_lon,
            major_angle_deg,
            range_ratio,
        )
    }

    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.build_notes).map_err(err_to_js)
    }

    /// Same JSON shape as [`WasmBinomialKriging::get_diagnostics`]. LOO uses `{ lats, lons,
    /// successes, trials }` in degrees; coordinates are mapped to the model tangent plane
    /// with the same reference as the fit before projected LOO.
    pub fn get_diagnostics(&self, options: JsValue) -> Result<JsValue, JsValue> {
        binomial_diagnostics_tangent_geo_js(&self.inner, &self.build_notes, options)
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsBinomialPrediction {
            prevalence_median: pred.prevalence_median as f64,
            prevalence_mean: pred.prevalence_mean as f64,
            logit: pred.logit as f64,
            logit_variance: pred.logit_variance as f64,
            prevalence_variance: pred.prevalence_variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        Ok(result.into())
    }

    pub fn predict_grid_arrays(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_cells: usize,
        y_cells: usize,
    ) -> Result<JsValue, JsValue> {
        let coords = build_grid_coords(x_min, x_max, y_min, y_max, x_cells, y_cells)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        set_object_field(&result, "nRows", &JsValue::from_f64(y_cells as f64))?;
        set_object_field(&result, "nCols", &JsValue::from_f64(x_cells as f64))?;
        Ok(result.into())
    }
}

fn parse_estimator(s: Option<&str>) -> Result<EmpiricalEstimator, JsValue> {
    match s.unwrap_or("classical") {
        "classical" => Ok(EmpiricalEstimator::Classical),
        "cressie-hawkins" | "cressie_hawkins" | "cressieHawkins" | "robust" => {
            Ok(EmpiricalEstimator::CressieHawkins)
        }
        other => Err(coded_err(
            &format!(
                "unknown empirical estimator '{other}' (expected 'classical' or 'cressie-hawkins')"
            ),
            "invalid_input",
        )),
    }
}

#[wasm_bindgen(js_name = fitVariogram)]
pub fn wasm_fit_ordinary_variogram(
    sample_lats: &[f64],
    sample_lons: &[f64],
    values: &[f64],
    max_distance: Option<f64>,
    n_bins: usize,
    variogram_type: WasmVariogramType,
    estimator: Option<String>,
) -> Result<JsValue, JsValue> {
    let sample_coords = to_coords(sample_lats, sample_lons)?;
    let n_bins = NonZeroUsize::new(n_bins)
        .ok_or_else(|| coded_err("n_bins must be at least 1", "invalid_bins"))?;
    let max_distance = match max_distance {
        Some(v) if v > 0.0 && v.is_finite() => {
            Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?)
        }
        Some(_) => {
            return Err(coded_err(
                "max_distance must be finite and positive",
                "invalid_input",
            ));
        }
        None => None,
    };
    let estimator = parse_estimator(estimator.as_deref())?;
    let config = VariogramConfig {
        max_distance,
        n_bins,
        estimator,
    };
    let values_real = values.iter().map(|v| *v as Real).collect::<Vec<_>>();
    let dataset = GeoDataset::new(sample_coords, values_real).map_err(kriging_err_to_js)?;
    let empirical = compute_empirical_variogram(&dataset, &config).map_err(kriging_err_to_js)?;
    let crate_type = VariogramType::from(variogram_type);
    let fit = fit_variogram(&empirical, crate_type).map_err(kriging_err_to_js)?;
    let (nugget, sill, range) = fit.model.params();
    serde_wasm_bindgen::to_value(&JsFittedVariogram {
        variogram_type: variogram_type_name(crate_type).to_string(),
        nugget: nugget as f64,
        sill: sill as f64,
        range: range as f64,
        shape: fit.model.shape().map(|s| s as f64),
        residuals: fit.residuals as f64,
    })
    .map_err(err_to_js)
}

/// Fit a parametric variogram to binomial count data using the noise-calibrated empirical
/// variogram on EB-smoothed logits (same path as the binomial kriger). Requires the classical
/// empirical estimator (Cressie–Hawkins is not supported for this path).
#[wasm_bindgen(js_name = fitBinomialVariogram)]
pub fn wasm_fit_binomial_variogram(
    sample_lats: &[f64],
    sample_lons: &[f64],
    successes: &[u32],
    trials: &[u32],
    max_distance: Option<f64>,
    n_bins: usize,
    variogram_type: WasmVariogramType,
    estimator: Option<String>,
    prior_alpha: Option<f64>,
    prior_beta: Option<f64>,
    rel_weight_eps: Option<f64>,
) -> Result<JsValue, JsValue> {
    if sample_lats.len() != sample_lons.len()
        || sample_lats.len() != successes.len()
        || sample_lats.len() != trials.len()
    {
        return Err(coded_err(
            "lats, lons, successes, and trials must have the same length",
            "mismatched_arrays",
        ));
    }
    let sample_coords = to_coords(sample_lats, sample_lons)?;
    let n_bins = NonZeroUsize::new(n_bins)
        .ok_or_else(|| coded_err("n_bins must be at least 1", "invalid_bins"))?;
    let max_distance = match max_distance {
        Some(v) if v > 0.0 && v.is_finite() => {
            Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?)
        }
        Some(_) => {
            return Err(coded_err(
                "max_distance must be finite and positive",
                "invalid_input",
            ));
        }
        None => None,
    };
    let estimator = parse_estimator(estimator.as_deref())?;
    if !matches!(estimator, EmpiricalEstimator::Classical) {
        return Err(coded_err(
            "fitBinomialVariogram requires estimator 'classical' (Cressie-Hawkins is not supported for the calibrated binomial empirical variogram)",
            "invalid_input",
        ));
    }
    let config = VariogramConfig {
        max_distance,
        n_bins,
        estimator,
    };
    let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
    let eps = match rel_weight_eps {
        Some(v) if v.is_finite() && v > 0.0 => v as Real,
        Some(_) => {
            return Err(coded_err(
                "rel_weight_eps must be finite and positive",
                "invalid_input",
            ));
        }
        None => 1e-12 as Real,
    };
    let crate_type = VariogramType::from(variogram_type);
    let fit = fit_binomial_variogram(
        sample_coords,
        successes,
        trials,
        prior,
        &config,
        crate_type,
        eps,
    )
    .map_err(kriging_err_to_js)?;
    let (nugget, sill, range) = fit.model.params();
    serde_wasm_bindgen::to_value(&JsFittedVariogram {
        variogram_type: variogram_type_name(crate_type).to_string(),
        nugget: nugget as f64,
        sill: sill as f64,
        range: range as f64,
        shape: fit.model.shape().map(|s| s as f64),
        residuals: fit.residuals as f64,
    })
    .map_err(err_to_js)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsEstimatedBinomialPrior {
    alpha: f64,
    beta: f64,
}

/// Heuristic Beta(α, β) from pooled station counts (same rule as `prior: "auto"` on the
/// binomial variogram fit and one-shot grid interpolation).
#[wasm_bindgen(js_name = estimateBinomialPrior)]
pub fn wasm_estimate_binomial_prior(successes: &[u32], trials: &[u32]) -> Result<JsValue, JsValue> {
    if successes.len() != trials.len() {
        return Err(coded_err(
            "successes and trials must have the same length",
            "mismatched_arrays",
        ));
    }
    let prior =
        estimate_binomial_prior_from_counts(successes, trials).map_err(kriging_err_to_js)?;
    serde_wasm_bindgen::to_value(&JsEstimatedBinomialPrior {
        alpha: prior.alpha() as f64,
        beta: prior.beta() as f64,
    })
    .map_err(err_to_js)
}

#[cfg(feature = "gpu")]
#[wasm_bindgen(js_name = webgpuAvailable)]
pub async fn wasm_webgpu_available() -> Result<JsValue, JsValue> {
    let support = detect_gpu_support().await;
    serde_wasm_bindgen::to_value(&support.available).map_err(err_to_js)
}

// ---------------------------------------------------------------------------
// Empirical variogram (direct, without fitting)
// ---------------------------------------------------------------------------

fn empirical_to_js(
    empirical: &crate::variogram::empirical::EmpiricalVariogram,
) -> Result<JsValue, JsValue> {
    let distances: Vec<f64> = empirical.distances.iter().map(|v| *v as f64).collect();
    let semivariances: Vec<f64> = empirical.semivariances.iter().map(|v| *v as f64).collect();
    let counts: Vec<u32> = empirical.n_pairs.iter().map(|v| *v as u32).collect();
    let result = Object::new();
    set_object_field(
        &result,
        "distances",
        &Float64Array::from(distances.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "semivariances",
        &Float64Array::from(semivariances.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "counts",
        &Uint32Array::from(counts.as_slice()).into(),
    )?;
    Ok(result.into())
}

/// Compute the empirical (sample) variogram cloud as `{ distances, semivariances, counts }`.
/// Mirrors [`compute_empirical_variogram`] on the Rust side.
#[wasm_bindgen(js_name = computeEmpiricalVariogram)]
pub fn wasm_compute_empirical_variogram(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    max_distance: Option<f64>,
    n_bins: usize,
    estimator: Option<String>,
) -> Result<JsValue, JsValue> {
    let coords = to_coords(lats, lons)?;
    let n_bins = NonZeroUsize::new(n_bins)
        .ok_or_else(|| coded_err("n_bins must be at least 1", "invalid_bins"))?;
    let max_distance = match max_distance {
        Some(v) if v > 0.0 && v.is_finite() => {
            Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?)
        }
        Some(_) => {
            return Err(coded_err(
                "max_distance must be finite and positive",
                "invalid_input",
            ));
        }
        None => None,
    };
    let estimator = parse_estimator(estimator.as_deref())?;
    let config = VariogramConfig {
        max_distance,
        n_bins,
        estimator,
    };
    let values_real = values.iter().map(|v| *v as Real).collect::<Vec<_>>();
    let dataset = GeoDataset::new(coords, values_real).map_err(kriging_err_to_js)?;
    let empirical = compute_empirical_variogram(&dataset, &config).map_err(kriging_err_to_js)?;
    empirical_to_js(&empirical)
}

// ---------------------------------------------------------------------------
// Directional empirical variogram (projected / planar)
// ---------------------------------------------------------------------------

/// Compute a directional empirical variogram on planar `(x, y)` data. Returns the same
/// `{ distances, semivariances, counts }` shape as [`wasm_compute_empirical_variogram`].
#[wasm_bindgen(js_name = computeDirectionalEmpiricalVariogram)]
pub fn wasm_compute_directional_empirical_variogram(
    xs: &[f64],
    ys: &[f64],
    values: &[f64],
    max_distance: f64,
    n_bins: usize,
    azimuth_deg: f64,
    tolerance_deg: f64,
) -> Result<JsValue, JsValue> {
    if xs.len() != ys.len() || xs.len() != values.len() {
        return Err(coded_err(
            "xs, ys and values must have the same length",
            "mismatched_arrays",
        ));
    }
    let coords: Vec<ProjectedCoord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
        .collect();
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let max_distance = PositiveReal::try_new(max_distance as Real).map_err(kriging_err_to_js)?;
    let n_bins = NonZeroUsize::new(n_bins)
        .ok_or_else(|| coded_err("n_bins must be at least 1", "invalid_bins"))?;
    let direction = DirectionalConfig::new(azimuth_deg as Real, tolerance_deg as Real)
        .map_err(kriging_err_to_js)?;
    let empirical = compute_directional_empirical_variogram(
        &coords,
        &values_real,
        max_distance,
        n_bins,
        direction,
    )
    .map_err(kriging_err_to_js)?;
    empirical_to_js(&empirical)
}

// ---------------------------------------------------------------------------
// Simple kriging (known mean)
// ---------------------------------------------------------------------------

pub(crate) struct WasmSimpleKriging {
    inner: SimpleKrigingModel,
}

impl WasmSimpleKriging {
    /// Construct a simple kriging model from typed arrays and a known `mean`.
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        mean: f64,
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmSimpleKriging, JsValue> {
        if values.len() != lats.len() {
            return Err(coded_err(
                "values must have the same length as lats/lons",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords(lats, lons)?;
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
        let dataset = GeoDataset::new(coords, values_real).map_err(kriging_err_to_js)?;
        let inner =
            SimpleKrigingModel::new(dataset, model, mean as Real).map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the known mean used to build the model.
    pub fn mean(&self) -> f64 {
        self.inner.mean() as f64
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        let values = self.inner.values();
        model_cv::simple_geo_cv(
            self.inner.coords(),
            &values,
            self.inner.variogram(),
            self.inner.mean(),
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        let values = self.inner.values();
        model_cv::simple_geo_cv(
            self.inner.coords(),
            &values,
            self.inner.variogram(),
            self.inner.mean(),
            Some(k),
        )
    }
}

// ---------------------------------------------------------------------------
// Universal kriging (trend-based)
// ---------------------------------------------------------------------------

pub(super) fn parse_trend(s: &str) -> Result<UniversalTrend, JsValue> {
    match s {
        "constant" => Ok(UniversalTrend::Constant),
        "linear" => Ok(UniversalTrend::Linear),
        "quadratic" => Ok(UniversalTrend::Quadratic),
        other => Err(coded_err(
            &format!("unknown trend '{other}' (expected 'constant', 'linear', or 'quadratic')"),
            "invalid_input",
        )),
    }
}

pub(crate) struct WasmUniversalKriging {
    inner: UniversalKrigingModel,
}

impl WasmUniversalKriging {
    /// Construct a universal kriging model from typed arrays. `trend` selects the drift
    /// basis: `"constant"`, `"linear"`, or `"quadratic"`.
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        trend: &str,
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmUniversalKriging, JsValue> {
        if values.len() != lats.len() {
            return Err(coded_err(
                "values must have the same length as lats/lons",
                "mismatched_arrays",
            ));
        }
        let trend = parse_trend(trend)?;
        let coords = to_coords(lats, lons)?;
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
        let dataset = GeoDataset::new(coords, values_real).map_err(kriging_err_to_js)?;
        let inner = UniversalKrigingModel::new(dataset, model, trend).map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        model_cv::universal_geo_cv(
            self.inner.coords(),
            self.inner.values(),
            self.inner.variogram(),
            self.inner.trend(),
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::universal_geo_cv(
            self.inner.coords(),
            self.inner.values(),
            self.inner.variogram(),
            self.inner.trend(),
            Some(k),
        )
    }
}

// ---------------------------------------------------------------------------
// Projected (planar) kriging with 2D anisotropy
// ---------------------------------------------------------------------------

pub(crate) struct WasmProjectedKriging {
    inner: ProjectedKrigingModel,
}

impl WasmProjectedKriging {
    /// Construct a projected (planar) ordinary kriging model on `(x, y)` coordinates with
    /// 2D anisotropy. `majorAngleDeg` is the azimuth of the major correlation axis (degrees
    /// CCW from +x); `rangeRatio` is the minor/major range ratio in `(0, 1]`.
    pub fn from_arrays(
        xs: &[f64],
        ys: &[f64],
        values: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        major_angle_deg: f64,
        range_ratio: f64,
    ) -> Result<WasmProjectedKriging, JsValue> {
        if xs.len() != ys.len() || xs.len() != values.len() {
            return Err(coded_err(
                "xs, ys and values must have the same length",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
            .map_err(kriging_err_to_js)?;
        let dataset = ProjectedDataset::new(coords, values_real).map_err(kriging_err_to_js)?;
        let inner =
            ProjectedKrigingModel::new(dataset, model, anisotropy).map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let coord = ProjectedCoord::new(x as Real, y as Real);
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, xs: &[f64], ys: &[f64]) -> Result<JsValue, JsValue> {
        if xs.len() != ys.len() {
            return Err(coded_err(
                "xs and ys must have the same length",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, xs: &[f64], ys: &[f64]) -> Result<JsValue, JsValue> {
        if xs.len() != ys.len() {
            return Err(coded_err(
                "xs and ys must have the same length",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        model_cv::projected_ordinary_cv(
            self.inner.coords(),
            self.inner.values(),
            self.inner.variogram(),
            self.inner.anisotropy(),
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::projected_ordinary_cv(
            self.inner.coords(),
            self.inner.values(),
            self.inner.variogram(),
            self.inner.anisotropy(),
            Some(k),
        )
    }
}

// ---------------------------------------------------------------------------
// Projected binomial kriging
// ---------------------------------------------------------------------------

fn build_projected_binomial_observations(
    xs: &[f64],
    ys: &[f64],
    successes: &[u32],
    trials: &[u32],
) -> Result<(Vec<ProjectedBinomialObservation>, Vec<usize>), JsValue> {
    if xs.len() != ys.len() || xs.len() != successes.len() || xs.len() != trials.len() {
        return Err(coded_err(
            "xs, ys, successes and trials must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::new();
    let mut dropped: Vec<usize> = Vec::new();
    for i in 0..xs.len() {
        if trials[i] == 0 {
            dropped.push(i);
            continue;
        }
        let coord = ProjectedCoord::new(xs[i] as Real, ys[i] as Real);
        out.push(
            ProjectedBinomialObservation::new(coord, successes[i], trials[i])
                .map_err(kriging_err_to_js)?,
        );
    }
    Ok((out, dropped))
}

pub(crate) struct WasmBinomialProjectedKriging {
    inner: BinomialProjectedKrigingModel,
    build_notes: BinomialBuildNotes,
    cv_coords: Vec<ProjectedCoord>,
    cv_successes: Vec<u32>,
    cv_trials: Vec<u32>,
}

impl WasmBinomialProjectedKriging {
    /// Construct a binomial projected kriging model on planar `(x, y)`
    /// coordinates. Mirrors [`WasmBinomialKriging::fromArrays`] but with
    /// 2-D anisotropy (`majorAngleDeg`, `rangeRatio`). Uses
    /// `BinomialPrior::default()` (Beta(1, 1)) for empirical-Bayes
    /// smoothing of the per-station logits.
    pub fn from_arrays(
        xs: &[f64],
        ys: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        major_angle_deg: f64,
        range_ratio: f64,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmBinomialProjectedKriging, JsValue> {
        let (observations, zero_trial_drops) =
            build_projected_binomial_observations(xs, ys, successes, trials)?;
        if observations.len() < 2 {
            return Err(coded_err(
                "need at least two non-zero-trial sites after dropping trials==0 rows",
                "too_few_points",
            ));
        }
        let (cv_coords, cv_successes, cv_trials) = binomial_projected_cv_slices(&observations);
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
            .map_err(kriging_err_to_js)?;
        let prior = BinomialPrior::default();
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let fit = BinomialProjectedKrigingModel::new_with_prior(
            observations,
            model,
            anisotropy,
            prior,
            hcfg,
        )
        .map_err(kriging_err_to_js)?;
        let mut build_notes = fit.notes;
        build_notes.zero_trial_dropped_indices = zero_trial_drops;
        build_notes.zero_trial_dropped_indices.sort_unstable();
        Ok(Self {
            inner: fit.model,
            build_notes,
            cv_coords,
            cv_successes,
            cv_trials,
        })
    }

    /// As [`fromArrays`](Self::from_arrays), with an explicit Beta prior.
    pub fn from_arrays_with_prior(
        xs: &[f64],
        ys: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        major_angle_deg: f64,
        range_ratio: f64,
        prior_alpha: f64,
        prior_beta: f64,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmBinomialProjectedKriging, JsValue> {
        let (observations, zero_trial_drops) =
            build_projected_binomial_observations(xs, ys, successes, trials)?;
        if observations.len() < 2 {
            return Err(coded_err(
                "need at least two non-zero-trial sites after dropping trials==0 rows",
                "too_few_points",
            ));
        }
        let (cv_coords, cv_successes, cv_trials) = binomial_projected_cv_slices(&observations);
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
            .map_err(kriging_err_to_js)?;
        let prior = BinomialPrior::new(prior_alpha as Real, prior_beta as Real)
            .map_err(kriging_err_to_js)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let fit = BinomialProjectedKrigingModel::new_with_prior(
            observations,
            model,
            anisotropy,
            prior,
            hcfg,
        )
        .map_err(kriging_err_to_js)?;
        let mut build_notes = fit.notes;
        build_notes.zero_trial_dropped_indices = zero_trial_drops;
        build_notes.zero_trial_dropped_indices.sort_unstable();
        Ok(Self {
            inner: fit.model,
            build_notes,
            cv_coords,
            cv_successes,
            cv_trials,
        })
    }

    /// Build a projected binomial model from caller-supplied logit values,
    /// bypassing the empirical-Bayes shrinkage step.
    pub fn from_precomputed_logits(
        xs: &[f64],
        ys: &[f64],
        logits: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        major_angle_deg: f64,
        range_ratio: f64,
    ) -> Result<WasmBinomialProjectedKriging, JsValue> {
        if xs.len() != ys.len() || xs.len() != logits.len() {
            return Err(coded_err(
                "xs, ys and logits must have the same length",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let logits_real: Vec<Real> = logits.iter().map(|v| *v as Real).collect();
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
            .map_err(kriging_err_to_js)?;
        let fit = BinomialProjectedKrigingModel::from_precomputed_logits(
            coords,
            logits_real,
            model,
            anisotropy,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords: Vec::new(),
            cv_successes: Vec::new(),
            cv_trials: Vec::new(),
        })
    }

    /// Like [`Self::from_precomputed_logits`], with per-site logit observation variances on the
    /// diagonal (same length as `xs`). Optional `prior_alpha` / `prior_beta` must be supplied
    /// together for build notes.
    pub fn from_precomputed_logits_with_variances(
        xs: &[f64],
        ys: &[f64],
        logits: &[f64],
        logit_observation_variance: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        major_angle_deg: f64,
        range_ratio: f64,
        prior_alpha: Option<f64>,
        prior_beta: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmBinomialProjectedKriging, JsValue> {
        if xs.len() != ys.len() || xs.len() != logits.len() {
            return Err(coded_err(
                "xs, ys and logits must have the same length",
                "mismatched_arrays",
            ));
        }
        if logit_observation_variance.len() != logits.len() {
            return Err(coded_err(
                "logitObservationVariance must have the same length as logits",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let logits_real: Vec<Real> = logits.iter().map(|v| *v as Real).collect();
        let base_var: Vec<Real> = logit_observation_variance
            .iter()
            .map(|v| *v as Real)
            .collect();
        let model = parse_variogram(variogram_type, nugget, sill, range, shape)?;
        let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
            .map_err(kriging_err_to_js)?;
        let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let fit = BinomialProjectedKrigingModel::from_precomputed_logits_with_logit_observation_variances(
            coords,
            logits_real,
            model,
            anisotropy,
            base_var,
            hcfg,
            prior,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner: fit.model,
            build_notes: fit.notes,
            cv_coords: Vec::new(),
            cv_successes: Vec::new(),
            cv_trials: Vec::new(),
        })
    }

    /// Build / conditioning diagnostics for the projected binomial model. See
    /// [`BinomialBuildNotes`].
    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.build_notes).map_err(err_to_js)
    }

    /// Same JSON shape as [`WasmBinomialKriging::get_diagnostics`]. LOO uses `{ xs, ys,
    /// successes, trials }` in the same planar units as the fit.
    pub fn get_diagnostics(&self, options: JsValue) -> Result<JsValue, JsValue> {
        binomial_diagnostics_projected_js(&self.inner, &self.build_notes, options)
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        if self.cv_coords.is_empty() {
            return Err(coded_err(
                "leaveOneOut requires count data; build the model from successes/trials, not precomputed logits",
                "invalid_input",
            ));
        }
        model_cv::binomial_projected_cv(
            &self.cv_coords,
            &self.cv_successes,
            &self.cv_trials,
            self.inner.variogram(),
            self.inner.anisotropy(),
            self.build_notes.prior,
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        if self.cv_coords.is_empty() {
            return Err(coded_err(
                "kFold requires count data; build the model from successes/trials, not precomputed logits",
                "invalid_input",
            ));
        }
        model_cv::binomial_projected_cv(
            &self.cv_coords,
            &self.cv_successes,
            &self.cv_trials,
            self.inner.variogram(),
            self.inner.anisotropy(),
            self.build_notes.prior,
            Some(k),
        )
    }

    pub fn predict(&self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let coord = ProjectedCoord::new(x as Real, y as Real);
        let pred = self.inner.predict(coord).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsBinomialPrediction {
            prevalence_median: pred.prevalence_median as f64,
            prevalence_mean: pred.prevalence_mean as f64,
            logit: pred.logit as f64,
            logit_variance: pred.logit_variance as f64,
            prevalence_variance: pred.prevalence_variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(&self, xs: &[f64], ys: &[f64]) -> Result<JsValue, JsValue> {
        if xs.len() != ys.len() {
            return Err(coded_err(
                "xs and ys must have the same length",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(&self, xs: &[f64], ys: &[f64]) -> Result<JsValue, JsValue> {
        if xs.len() != ys.len() {
            return Err(coded_err(
                "xs and ys must have the same length",
                "mismatched_arrays",
            ));
        }
        let coords: Vec<ProjectedCoord> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
            .collect();
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        Ok(result.into())
    }

    /// Predict a regular `(x, y)` grid in one call; returns flat `Float64Array`s in row-major
    /// order (`y` outer, `x` inner), same layout as geo `WasmBinomialKriging::predict_grid_arrays`.
    pub fn predict_grid_arrays(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_cells: usize,
        y_cells: usize,
    ) -> Result<JsValue, JsValue> {
        let coords = build_projected_grid_coords(x_min, x_max, y_min, y_max, x_cells, y_cells)?;
        let out = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        set_object_field(&result, "nRows", &JsValue::from_f64(y_cells as f64))?;
        set_object_field(&result, "nCols", &JsValue::from_f64(x_cells as f64))?;
        Ok(result.into())
    }
}

// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsCvSummary {
    n: usize,
    mean_error: f64,
    rmse: f64,
    msdr: f64,
}

pub(super) fn cv_result_to_js(residuals: Vec<crate::cv::CvResidual>) -> Result<JsValue, JsValue> {
    let n = residuals.len();
    let summary = crate::cv::CvSummary::from_residuals(&residuals);
    let mut indices: Vec<u32> = Vec::with_capacity(n);
    let mut observed: Vec<f64> = Vec::with_capacity(n);
    let mut predicted: Vec<f64> = Vec::with_capacity(n);
    let mut variances: Vec<f64> = Vec::with_capacity(n);
    for r in &residuals {
        indices.push(r.index as u32);
        observed.push(r.observed as f64);
        predicted.push(r.predicted as f64);
        variances.push(r.variance as f64);
    }
    let result = Object::new();
    set_object_field(
        &result,
        "indices",
        &Uint32Array::from(indices.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "observed",
        &Float64Array::from(observed.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "predicted",
        &Float64Array::from(predicted.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "variances",
        &Float64Array::from(variances.as_slice()).into(),
    )?;
    let summary_js = serde_wasm_bindgen::to_value(&JsCvSummary {
        n: summary.n,
        mean_error: summary.mean_error as f64,
        rmse: summary.rmse as f64,
        msdr: summary.msdr as f64,
    })
    .map_err(err_to_js)?;
    set_object_field(&result, "summary", &summary_js)?;
    Ok(result.into())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsPrevalenceCalibrationBin {
    bin_index: usize,
    predicted_lo: f64,
    predicted_hi: f64,
    n_stations: usize,
    sum_trials: u64,
    sum_successes: u64,
    mean_predicted: f64,
    pooled_observed_prevalence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsBinomialCvSummary {
    n: usize,
    n_evaluated: usize,
    logit: JsCvSummary,
    prevalence: JsCvSummary,
    brier: f64,
    log_score_per_trial: f64,
    calibration_bins: Vec<JsPrevalenceCalibrationBin>,
}

pub(super) fn binomial_cv_result_to_js(
    residuals: Vec<BinomialCvResidual>,
) -> Result<JsValue, JsValue> {
    let n = residuals.len();
    let summary = BinomialCvSummary::from_residuals(&residuals);

    let mut indices: Vec<u32> = Vec::with_capacity(n);
    let mut successes: Vec<u32> = Vec::with_capacity(n);
    let mut trials: Vec<u32> = Vec::with_capacity(n);
    let mut observed_logit: Vec<f64> = Vec::with_capacity(n);
    let mut predicted_logit: Vec<f64> = Vec::with_capacity(n);
    let mut logit_variance: Vec<f64> = Vec::with_capacity(n);
    let mut observed_prevalence: Vec<f64> = Vec::with_capacity(n);
    let mut predicted_prevalence: Vec<f64> = Vec::with_capacity(n);
    let mut prevalence_variance: Vec<f64> = Vec::with_capacity(n);

    for r in &residuals {
        indices.push(r.index as u32);
        successes.push(r.successes);
        trials.push(r.trials);
        observed_logit.push(r.observed_logit as f64);
        predicted_logit.push(r.predicted_logit as f64);
        logit_variance.push(r.logit_variance as f64);
        observed_prevalence.push(r.observed_prevalence as f64);
        predicted_prevalence.push(r.predicted_prevalence as f64);
        prevalence_variance.push(r.prevalence_variance as f64);
    }

    let result = Object::new();
    set_object_field(
        &result,
        "indices",
        &Uint32Array::from(indices.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "successes",
        &Uint32Array::from(successes.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "trials",
        &Uint32Array::from(trials.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "observedLogit",
        &Float64Array::from(observed_logit.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "predictedLogit",
        &Float64Array::from(predicted_logit.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "logitVariance",
        &Float64Array::from(logit_variance.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "observedPrevalence",
        &Float64Array::from(observed_prevalence.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "predictedPrevalence",
        &Float64Array::from(predicted_prevalence.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "prevalenceVariance",
        &Float64Array::from(prevalence_variance.as_slice()).into(),
    )?;

    let summary_js = serde_wasm_bindgen::to_value(&JsBinomialCvSummary {
        n: summary.n,
        n_evaluated: summary.n_evaluated,
        logit: JsCvSummary {
            n: summary.logit.n,
            mean_error: summary.logit.mean_error as f64,
            rmse: summary.logit.rmse as f64,
            msdr: summary.logit.msdr as f64,
        },
        prevalence: JsCvSummary {
            n: summary.prevalence.n,
            mean_error: summary.prevalence.mean_error as f64,
            rmse: summary.prevalence.rmse as f64,
            msdr: summary.prevalence.msdr as f64,
        },
        brier: summary.brier as f64,
        log_score_per_trial: summary.log_score_per_trial as f64,
        calibration_bins: summary
            .calibration_bins
            .iter()
            .map(|b| JsPrevalenceCalibrationBin {
                bin_index: b.bin_index,
                predicted_lo: b.predicted_lo as f64,
                predicted_hi: b.predicted_hi as f64,
                n_stations: b.n_stations,
                sum_trials: b.sum_trials,
                sum_successes: b.sum_successes,
                mean_predicted: b.mean_predicted as f64,
                pooled_observed_prevalence: b.pooled_observed_prevalence as f64,
            })
            .collect(),
    })
    .map_err(err_to_js)?;
    set_object_field(&result, "summary", &summary_js)?;
    Ok(result.into())
}

pub(super) fn parse_binomial_prior(
    prior_alpha: Option<f64>,
    prior_beta: Option<f64>,
) -> Result<BinomialPrior, JsValue> {
    match (prior_alpha, prior_beta) {
        (None, None) => Ok(BinomialPrior::default()),
        (Some(a), Some(b)) => BinomialPrior::new(a as Real, b as Real).map_err(kriging_err_to_js),
        _ => Err(coded_err(
            "priorAlpha and priorBeta must be provided together",
            "invalid_input",
        )),
    }
}

/// Optional `stability` preset (`"default"` / `"strict"` / `"permissive"`) → heteroskedastic
/// build config. Empty or omitted string uses the default preset.
pub(super) fn heteroskedastic_config_from_optional_stability_str(
    stability: Option<&str>,
) -> Result<HeteroskedasticBinomialConfig, JsValue> {
    let s = stability.map(str::trim).filter(|s| !s.is_empty());
    let preset = match s {
        None => BinomialStability::Default,
        Some(x) if x.eq_ignore_ascii_case("default") => BinomialStability::Default,
        Some(x) if x.eq_ignore_ascii_case("strict") => BinomialStability::Strict,
        Some(x) if x.eq_ignore_ascii_case("permissive") => BinomialStability::Permissive,
        Some(other) => {
            return Err(coded_err(
                &format!("stability must be 'default', 'strict', or 'permissive' (got {other:?})"),
                "invalid_input",
            ));
        }
    };
    Ok(preset.to_heteroskedastic_config())
}

pub(super) fn merge_hetero_with_optional_one_step_laplace(
    mut hcfg: HeteroskedasticBinomialConfig,
    one_step: Option<bool>,
) -> HeteroskedasticBinomialConfig {
    if one_step == Some(true) {
        hcfg.one_step_laplace_observation_variance = true;
    }
    hcfg
}

pub(super) fn binomial_simulation_to_js(
    result: BinomialSimulationResult,
) -> Result<JsValue, JsValue> {
    let logit: Vec<f64> = result.logit_samples.into_iter().map(|v| v as f64).collect();
    let prev: Vec<f64> = result
        .prevalence_samples
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let out = Object::new();
    set_object_field(
        &out,
        "logitSamples",
        &Float64Array::from(logit.as_slice()).into(),
    )?;
    set_object_field(
        &out,
        "prevalenceSamples",
        &Float64Array::from(prev.as_slice()).into(),
    )?;
    Ok(out.into())
}

pub(super) fn binomial_many_simulation_to_js(
    result: BinomialSimulationManyResult,
) -> Result<JsValue, JsValue> {
    let logit: Vec<f64> = result.logit_samples.into_iter().map(|v| v as f64).collect();
    let prev: Vec<f64> = result
        .prevalence_samples
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let out = Object::new();
    set_object_field(
        &out,
        "nRealizations",
        &JsValue::from_f64(result.n_realizations as f64),
    )?;
    set_object_field(
        &out,
        "nTargets",
        &JsValue::from_f64(result.n_targets as f64),
    )?;
    set_object_field(
        &out,
        "logitSamples",
        &Float64Array::from(logit.as_slice()).into(),
    )?;
    set_object_field(
        &out,
        "prevalenceSamples",
        &Float64Array::from(prev.as_slice()).into(),
    )?;
    Ok(out.into())
}
// ---------------------------------------------------------------------------
// Nested variograms
// ---------------------------------------------------------------------------

/// Parameters for a single variogram component, used when building nested variograms.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NestedComponent {
    variogram_type: String,
    nugget: f64,
    sill: f64,
    range: f64,
    #[serde(default)]
    shape: Option<f64>,
}

/// Evaluate a nested (additive) variogram at a list of distances. Returns
/// `{ distances, semivariances, covariances }` — semivariance and covariance per lag.
///
/// The `components` argument is an array of `{ variogramType, nugget, sill, range, shape? }`
/// objects. This is a convenience surface for building & validating nested models from JS;
/// composite kriging models will accept nested variograms in a future iteration.
#[wasm_bindgen(js_name = evaluateNestedVariogram)]
pub fn wasm_evaluate_nested_variogram(
    components: JsValue,
    distances: &[f64],
) -> Result<JsValue, JsValue> {
    let comps: Vec<NestedComponent> =
        serde_wasm_bindgen::from_value(components).map_err(err_to_js)?;
    if comps.is_empty() {
        return Err(coded_err(
            "nested variogram requires at least one component",
            "invalid_input",
        ));
    }
    let mut models = Vec::with_capacity(comps.len());
    for c in &comps {
        let m = parse_variogram(&c.variogram_type, c.nugget, c.sill, c.range, c.shape)?;
        models.push(m);
    }
    let nested = NestedVariogram::new(models).map_err(kriging_err_to_js)?;
    let semivariances: Vec<f64> = distances
        .iter()
        .map(|d| nested.semivariance(*d as Real) as f64)
        .collect();
    let covariances: Vec<f64> = distances
        .iter()
        .map(|d| nested.covariance(*d as Real) as f64)
        .collect();
    let result = Object::new();
    set_object_field(&result, "distances", &Float64Array::from(distances).into())?;
    set_object_field(
        &result,
        "semivariances",
        &Float64Array::from(semivariances.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "covariances",
        &Float64Array::from(covariances.as_slice()).into(),
    )?;
    Ok(result.into())
}

// ---------------------------------------------------------------------------
// Polygon aggregation over ensembles
// ---------------------------------------------------------------------------

fn summary_to_js(summary: &PolygonAggregationSummary) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    set_object_field(
        &obj,
        "nRealizations",
        &JsValue::from_f64(summary.n_realizations as f64),
    )?;
    set_object_field(
        &obj,
        "totalWeight",
        &JsValue::from_f64(summary.total_weight as f64),
    )?;
    set_object_field(&obj, "mean", &JsValue::from_f64(summary.mean as f64))?;
    match summary.variance {
        Some(v) => set_object_field(&obj, "variance", &JsValue::from_f64(v as f64))?,
        None => set_object_field(&obj, "variance", &JsValue::NULL)?,
    }
    let probs: Vec<f64> = summary.quantiles.iter().map(|(p, _)| *p as f64).collect();
    let vals: Vec<f64> = summary.quantiles.iter().map(|(_, v)| *v as f64).collect();
    set_object_field(
        &obj,
        "quantileProbabilities",
        &Float64Array::from(probs.as_slice()).into(),
    )?;
    set_object_field(
        &obj,
        "quantileValues",
        &Float64Array::from(vals.as_slice()).into(),
    )?;
    Ok(obj.into())
}

/// Aggregate one or more polygons over a flat row-major ensemble buffer of
/// shape `nRealizations × nTargets`.
///
/// Polygons are passed in CSR-style: `polygonIndices` and `polygonWeights` are
/// the concatenated cell lists, and `polygonOffsets` (length
/// `nPolygons + 1`) marks the start of each polygon's slice. Each polygon must
/// be non-empty, indices must lie in `[0, nTargets)`, weights must be finite
/// and non-negative, and at least one weight per polygon must be positive.
///
/// Returns a JS array of summary objects (one per polygon) with fields:
/// `{ nRealizations, totalWeight, mean, variance | null,
///    quantileProbabilities, quantileValues }`. Quantiles are reported in the
/// order supplied via `quantiles`.
#[wasm_bindgen(js_name = aggregatePolygonsOverEnsemble)]
#[allow(clippy::too_many_arguments)]
pub fn wasm_aggregate_polygons_over_ensemble(
    samples: &[f64],
    n_realizations: u32,
    n_targets: u32,
    polygon_indices: &[u32],
    polygon_weights: &[f64],
    polygon_offsets: &[u32],
    quantiles: &[f64],
) -> Result<JsValue, JsValue> {
    if polygon_offsets.is_empty() {
        return Err(coded_err(
            "polygonOffsets must contain at least [0]",
            "invalid_input",
        ));
    }
    if polygon_offsets[0] != 0 {
        return Err(coded_err("polygonOffsets[0] must be 0", "invalid_input"));
    }
    let total = polygon_offsets[polygon_offsets.len() - 1] as usize;
    if total != polygon_indices.len() || total != polygon_weights.len() {
        return Err(coded_err(
            "polygonIndices and polygonWeights must have length polygonOffsets[last]",
            "mismatched_arrays",
        ));
    }
    for w in polygon_offsets.windows(2) {
        if w[1] < w[0] {
            return Err(coded_err(
                "polygonOffsets must be non-decreasing",
                "invalid_input",
            ));
        }
    }

    // Materialize Real-typed buffers once, then build slice tuples per polygon.
    let samples_real: Vec<Real> = samples.iter().map(|v| *v as Real).collect();
    let weights_real: Vec<Real> = polygon_weights.iter().map(|v| *v as Real).collect();
    let indices_usize: Vec<usize> = polygon_indices.iter().map(|i| *i as usize).collect();
    let probs_real: Vec<Real> = quantiles.iter().map(|v| *v as Real).collect();

    let n_polys = polygon_offsets.len() - 1;
    let polys: Vec<(&[usize], &[Real])> = (0..n_polys)
        .map(|p| {
            let lo = polygon_offsets[p] as usize;
            let hi = polygon_offsets[p + 1] as usize;
            (&indices_usize[lo..hi], &weights_real[lo..hi])
        })
        .collect();

    let summaries = polygon_weighted_summaries_batch(
        &samples_real,
        n_realizations as usize,
        n_targets as usize,
        &polys,
        &probs_real,
    )
    .map_err(kriging_err_to_js)?;

    let arr = js_sys::Array::new_with_length(summaries.len() as u32);
    for (i, s) in summaries.iter().enumerate() {
        let val = summary_to_js(s)?;
        arr.set(i as u32, val);
    }
    Ok(arr.into())
}
