#![cfg(feature = "wasm")]

use js_sys::{Array, Float64Array, Object, Reflect};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::distance::GeoCoord;
#[cfg(feature = "gpu")]
use crate::gpu::detect_gpu_support;
use crate::kriging::binomial::{BinomialKrigingModel, BinomialObservation, BinomialPrior};
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::variogram::empirical::VariogramConfig;
use crate::variogram::fitting::fit_variogram;
use crate::variogram::models::{VariogramModel, VariogramType};
use crate::{Real, compute_empirical_variogram};

fn parse_variogram(
    variogram_type: &str,
    nugget: f64,
    sill: f64,
    range: f64,
) -> Result<VariogramModel, JsValue> {
    let vt = match variogram_type.to_ascii_lowercase().as_str() {
        "spherical" => VariogramType::Spherical,
        "exponential" => VariogramType::Exponential,
        "gaussian" => VariogramType::Gaussian,
        _ => return Err(JsValue::from_str("unknown variogram_type")),
    };
    Ok(match vt {
        VariogramType::Spherical => VariogramModel::Spherical {
            nugget: nugget as Real,
            sill: sill as Real,
            range: range as Real,
        },
        VariogramType::Exponential => VariogramModel::Exponential {
            nugget: nugget as Real,
            sill: sill as Real,
            range: range as Real,
        },
        VariogramType::Gaussian => VariogramModel::Gaussian {
            nugget: nugget as Real,
            sill: sill as Real,
            range: range as Real,
        },
    })
}

fn to_coords(lats: &[f64], lons: &[f64]) -> Result<Vec<GeoCoord>, JsValue> {
    if lats.len() != lons.len() {
        return Err(JsValue::from_str("lats and lons must have same length"));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        out.push(GeoCoord {
            lat: lats[i] as Real,
            lon: lons[i] as Real,
        });
    }
    Ok(out)
}

fn err_to_js(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

fn parse_variogram_types(variogram_types: &Array) -> Result<Vec<VariogramType>, JsValue> {
    let mut out = Vec::with_capacity(variogram_types.length() as usize);
    for value in variogram_types.iter() {
        let text = value
            .as_string()
            .ok_or_else(|| JsValue::from_str("variogram type entries must be strings"))?;
        let vt = match text.to_ascii_lowercase().as_str() {
            "spherical" => VariogramType::Spherical,
            "exponential" => VariogramType::Exponential,
            "gaussian" => VariogramType::Gaussian,
            _ => {
                return Err(JsValue::from_str(
                    "unknown variogram type in variogram_types",
                ));
            }
        };
        out.push(vt);
    }
    Ok(out)
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
        out.push(BinomialObservation {
            coord: GeoCoord {
                lat: lats[i] as Real,
                lon: lons[i] as Real,
            },
            successes: successes[i],
            trials: trials[i],
        });
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

#[wasm_bindgen]
pub struct WasmOrdinaryKriging {
    inner: OrdinaryKrigingModel,
}

#[wasm_bindgen]
impl WasmOrdinaryKriging {
    #[wasm_bindgen(constructor)]
    pub fn new(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        variogram_type: String,
        nugget: f64,
        sill: f64,
        range: f64,
    ) -> Result<WasmOrdinaryKriging, JsValue> {
        let coords = to_coords(lats, lons)?;
        let model = parse_variogram(&variogram_type, nugget, sill, range)?;
        let values_real = values.iter().map(|v| *v as Real).collect::<Vec<_>>();
        let inner = OrdinaryKrigingModel::new(coords, values_real, model).map_err(err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let pred = self
            .inner
            .predict(GeoCoord {
                lat: lat as Real,
                lon: lon as Real,
            })
            .map_err(err_to_js)?;
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
    pub fn new(
        lats: &[f64],
        lons: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram_type: String,
        nugget: f64,
        sill: f64,
        range: f64,
    ) -> Result<WasmBinomialKriging, JsValue> {
        let observations = build_observations(lats, lons, successes, trials)?;
        let model = parse_variogram(&variogram_type, nugget, sill, range)?;
        let inner = BinomialKrigingModel::new(observations, model).map_err(err_to_js)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = newWithPrior)]
    pub fn new_with_prior(
        lats: &[f64],
        lons: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram_type: String,
        nugget: f64,
        sill: f64,
        range: f64,
        alpha: f64,
        beta: f64,
    ) -> Result<WasmBinomialKriging, JsValue> {
        let observations = build_observations(lats, lons, successes, trials)?;
        let model = parse_variogram(&variogram_type, nugget, sill, range)?;
        let prior = BinomialPrior {
            alpha: alpha as Real,
            beta: beta as Real,
        };
        let inner =
            BinomialKrigingModel::new_with_prior(observations, model, prior).map_err(err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64) -> Result<JsValue, JsValue> {
        let pred = self
            .inner
            .predict(GeoCoord {
                lat: lat as Real,
                lon: lon as Real,
            })
            .map_err(err_to_js)?;
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

#[wasm_bindgen(js_name = fitOrdinaryVariogram)]
pub fn wasm_fit_ordinary_variogram(
    sample_lats: &[f64],
    sample_lons: &[f64],
    values: &[f64],
    max_distance: Option<f64>,
    n_bins: usize,
    variogram_types: Array,
) -> Result<JsValue, JsValue> {
    let sample_coords = to_coords(sample_lats, sample_lons)?;
    let config = VariogramConfig {
        max_distance: max_distance.map(|v| v as Real),
        n_bins,
    };
    let values_real = values.iter().map(|v| *v as Real).collect::<Vec<_>>();
    let empirical =
        compute_empirical_variogram(&sample_coords, &values_real, &config).map_err(err_to_js)?;
    let types = parse_variogram_types(&variogram_types)?;
    let mut types_iter = types.iter().copied();
    let first = types_iter
        .next()
        .ok_or_else(|| JsValue::from_str("at least one variogram type must be provided"))?;

    let mut best = fit_variogram(&empirical, first).map_err(err_to_js)?;
    let mut best_type = first;
    for variogram_type in types_iter {
        let candidate = fit_variogram(&empirical, variogram_type).map_err(err_to_js)?;
        if candidate.residuals < best.residuals {
            best = candidate;
            best_type = variogram_type;
        }
    }
    let (nugget, sill, range) = best.model.params();
    serde_wasm_bindgen::to_value(&JsFittedVariogram {
        variogram_type: variogram_type_name(best_type).to_string(),
        nugget: nugget as f64,
        sill: sill as f64,
        range: range as f64,
        residuals: best.residuals as f64,
    })
    .map_err(err_to_js)
}

#[cfg(feature = "gpu")]
#[wasm_bindgen(js_name = webgpuAvailable)]
pub async fn wasm_webgpu_available() -> Result<JsValue, JsValue> {
    let support = detect_gpu_support().await;
    serde_wasm_bindgen::to_value(&support.available).map_err(err_to_js)
}
