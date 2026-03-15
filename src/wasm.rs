#![cfg(feature = "wasm")]

use js_sys::{Float64Array, Object, Reflect};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::distance::GeoCoord;
use crate::geo_dataset::GeoDataset;
#[cfg(feature = "gpu")]
use crate::gpu::detect_gpu_support;
use crate::kriging::binomial::{BinomialKrigingModel, BinomialObservation, BinomialPrior};
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::variogram::empirical::{PositiveReal, VariogramConfig};
use crate::variogram::fitting::fit_variogram;
use crate::variogram::models::{VariogramModel, VariogramType};
use crate::{Real, compute_empirical_variogram};
use std::num::NonZeroUsize;

/// WASM-exposed variogram type enum; maps to crate's VariogramType.
#[wasm_bindgen]
pub enum WasmVariogramType {
    Spherical,
    Exponential,
    Gaussian,
    Cubic,
    Stable,
    Matern,
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
        }
    }
}

fn parse_variogram(
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
        _ => return Err(JsValue::from_str("unknown variogram_type")),
    };
    match (vt, shape) {
        (VariogramType::Stable, Some(s)) | (VariogramType::Matern, Some(s)) => {
            VariogramModel::new_with_shape(
                nugget as Real,
                sill as Real,
                range as Real,
                vt,
                s as Real,
            )
            .map_err(err_to_js)
        }
        _ => {
            VariogramModel::new(nugget as Real, sill as Real, range as Real, vt).map_err(err_to_js)
        }
    }
}

fn to_coords(lats: &[f64], lons: &[f64]) -> Result<Vec<GeoCoord>, JsValue> {
    if lats.len() != lons.len() {
        return Err(JsValue::from_str("lats and lons must have same length"));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        out.push(GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(err_to_js)?);
    }
    Ok(out)
}

fn err_to_js(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

fn build_observations(
    lats: &[f64],
    lons: &[f64],
    successes: &[u32],
    trials: &[u32],
) -> Result<Vec<BinomialObservation>, JsValue> {
    if lats.len() != lons.len() || lats.len() != successes.len() || lats.len() != trials.len() {
        return Err(JsValue::from_str("all input arrays must have same length"));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let coord = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(err_to_js)?;
        out.push(BinomialObservation::new(coord, successes[i], trials[i]).map_err(err_to_js)?);
    }
    Ok(out)
}

#[derive(Debug, Serialize)]
struct JsPrediction {
    value: f64,
    variance: f64,
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
struct JsBinomialPrediction {
    prevalence: f64,
    logit_value: f64,
    variance: f64,
}

fn variogram_type_name(variogram_type: VariogramType) -> &'static str {
    match variogram_type {
        VariogramType::Spherical => "spherical",
        VariogramType::Exponential => "exponential",
        VariogramType::Gaussian => "gaussian",
        VariogramType::Cubic => "cubic",
        VariogramType::Stable => "stable",
        VariogramType::Matern => "matern",
    }
}

fn map_predictions(out: Vec<crate::kriging::ordinary::Prediction>) -> Vec<JsPrediction> {
    out.into_iter()
        .map(|p| JsPrediction {
            value: p.value as f64,
            variance: p.variance as f64,
        })
        .collect::<Vec<_>>()
}

fn split_predictions(out: Vec<crate::kriging::ordinary::Prediction>) -> (Vec<f64>, Vec<f64>) {
    let mut values = Vec::with_capacity(out.len());
    let mut variances = Vec::with_capacity(out.len());
    for pred in out {
        values.push(pred.value as f64);
        variances.push(pred.variance as f64);
    }
    (values, variances)
}

fn map_binomial_predictions(
    out: Vec<crate::kriging::binomial::BinomialPrediction>,
) -> Vec<JsBinomialPrediction> {
    out.into_iter()
        .map(|p| JsBinomialPrediction {
            prevalence: p.prevalence as f64,
            logit_value: p.logit_value as f64,
            variance: p.variance as f64,
        })
        .collect::<Vec<_>>()
}

fn split_binomial_predictions(
    out: Vec<crate::kriging::binomial::BinomialPrediction>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut prevalences = Vec::with_capacity(out.len());
    let mut logit_values = Vec::with_capacity(out.len());
    let mut variances = Vec::with_capacity(out.len());
    for pred in out {
        prevalences.push(pred.prevalence as f64);
        logit_values.push(pred.logit_value as f64);
        variances.push(pred.variance as f64);
    }
    (prevalences, logit_values, variances)
}

fn set_object_field(obj: &Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    Reflect::set(obj, &JsValue::from_str(key), value).map(|_| ())
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
}

#[wasm_bindgen]
pub struct WasmOrdinaryKriging {
    inner: OrdinaryKrigingModel,
}

#[wasm_bindgen]
impl WasmOrdinaryKriging {
    #[wasm_bindgen(constructor)]
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
        let dataset = GeoDataset::new(coords, values_real).map_err(err_to_js)?;
        let inner = OrdinaryKrigingModel::new(dataset, model).map_err(err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(err_to_js)?;
        let pred = self.inner.predict(coord).map_err(err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self.inner.predict_batch(&coords).map_err(err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self.inner.predict_batch(&coords).map_err(err_to_js)?;
        let (values, variances) = split_predictions(out);
        let values_array = Float64Array::from(values.as_slice());
        let variances_array = Float64Array::from(variances.as_slice());
        let result = Object::new();
        set_object_field(&result, "values", &values_array.into())?;
        set_object_field(&result, "variances", &variances_array.into())?;
        Ok(result.into())
    }

    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = predictBatchGpu)]
    pub async fn predict_batch_gpu(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch_gpu(&coords)
            .await
            .map_err(err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }
}

#[wasm_bindgen]
pub struct WasmBinomialKriging {
    inner: BinomialKrigingModel,
}

#[wasm_bindgen]
impl WasmBinomialKriging {
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<WasmBinomialKriging, JsValue> {
        let opts: BinomialKrigingOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let observations =
            build_observations(&opts.lats, &opts.lons, &opts.successes, &opts.trials)?;
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let inner = BinomialKrigingModel::new(observations, model).map_err(err_to_js)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = newWithPrior)]
    pub fn new_with_prior(options: JsValue) -> Result<WasmBinomialKriging, JsValue> {
        let opts: BinomialKrigingWithPriorOptions =
            serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
        let observations =
            build_observations(&opts.lats, &opts.lons, &opts.successes, &opts.trials)?;
        let model = parse_variogram(
            &opts.variogram.variogram_type,
            opts.variogram.nugget,
            opts.variogram.sill,
            opts.variogram.range,
            opts.variogram.shape,
        )?;
        let prior = BinomialPrior::new(opts.prior.alpha as Real, opts.prior.beta as Real)
            .map_err(err_to_js)?;
        let inner =
            BinomialKrigingModel::new_with_prior(observations, model, prior).map_err(err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(err_to_js)?;
        let pred = self.inner.predict(coord).map_err(err_to_js)?;
        serde_wasm_bindgen::to_value(&JsBinomialPrediction {
            prevalence: pred.prevalence as f64,
            logit_value: pred.logit_value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self.inner.predict_batch(&coords).map_err(err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self.inner.predict_batch(&coords).map_err(err_to_js)?;
        let (prevalences, logit_values, variances) = split_binomial_predictions(out);
        let prevalences_array = Float64Array::from(prevalences.as_slice());
        let logit_values_array = Float64Array::from(logit_values.as_slice());
        let variances_array = Float64Array::from(variances.as_slice());
        let result = Object::new();
        set_object_field(&result, "prevalences", &prevalences_array.into())?;
        set_object_field(&result, "logitValues", &logit_values_array.into())?;
        set_object_field(&result, "variances", &variances_array.into())?;
        Ok(result.into())
    }

    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = predictBatchGpu)]
    pub async fn predict_batch_gpu(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords(lats, lons)?;
        let out = self
            .inner
            .predict_batch_gpu(&coords)
            .await
            .map_err(err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
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
) -> Result<JsValue, JsValue> {
    let sample_coords = to_coords(sample_lats, sample_lons)?;
    let n_bins =
        NonZeroUsize::new(n_bins).ok_or_else(|| JsValue::from_str("n_bins must be at least 1"))?;
    let max_distance = match max_distance {
        Some(v) if v > 0.0 && v.is_finite() => {
            Some(PositiveReal::try_new(v as Real).map_err(err_to_js)?)
        }
        Some(_) => {
            return Err(JsValue::from_str(
                "max_distance must be finite and positive",
            ));
        }
        None => None,
    };
    let config = VariogramConfig {
        max_distance,
        n_bins,
    };
    let values_real = values.iter().map(|v| *v as Real).collect::<Vec<_>>();
    let dataset = GeoDataset::new(sample_coords, values_real).map_err(err_to_js)?;
    let empirical = compute_empirical_variogram(&dataset, &config).map_err(err_to_js)?;
    let crate_type = VariogramType::from(variogram_type);
    let fit = fit_variogram(&empirical, crate_type).map_err(err_to_js)?;
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

#[cfg(feature = "gpu")]
#[wasm_bindgen(js_name = webgpuAvailable)]
pub async fn wasm_webgpu_available() -> Result<JsValue, JsValue> {
    let support = detect_gpu_support().await;
    serde_wasm_bindgen::to_value(&support.available).map_err(err_to_js)
}
