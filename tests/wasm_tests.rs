#![cfg(all(feature = "wasm", target_arch = "wasm32"))]

use js_sys::{Array, Float64Array, Reflect};
#[cfg(feature = "gpu")]
use kriging_rs::wasm::wasm_webgpu_available;
use kriging_rs::wasm::{WasmBinomialKriging, WasmOrdinaryKriging};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_ordinary_predict_returns_serializable_object() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let values = vec![1.0, 2.0, 1.5];
    let model = WasmOrdinaryKriging::new(
        &lats,
        &lons,
        &values,
        "exponential".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model.predict(0.2, 0.2).expect("prediction should succeed");
    let value = Reflect::get(&js, &JsValue::from_str("value")).expect("has value field");
    let variance = Reflect::get(&js, &JsValue::from_str("variance")).expect("has variance field");
    assert!(value.as_f64().unwrap_or(f64::NAN).is_finite());
    assert!(variance.as_f64().unwrap_or(f64::NAN).is_finite());
}

#[wasm_bindgen_test]
fn wasm_ordinary_batch_returns_array() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let values = vec![1.0, 2.0, 1.5];
    let model = WasmOrdinaryKriging::new(
        &lats,
        &lons,
        &values,
        "gaussian".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model
        .predict_batch(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch should succeed");
    assert!(Array::is_array(&js));
}

#[wasm_bindgen_test]
fn wasm_binomial_predict_returns_serializable_object() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let successes = vec![1, 3, 4];
    let trials = vec![5, 5, 5];
    let model = WasmBinomialKriging::new(
        &lats,
        &lons,
        &successes,
        &trials,
        "exponential".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model.predict(0.2, 0.2).expect("prediction should succeed");
    let prevalence = Reflect::get(&js, &JsValue::from_str("prevalence")).expect("has prevalence");
    let logit_value = Reflect::get(&js, &JsValue::from_str("logit_value")).expect("has logit");
    let variance = Reflect::get(&js, &JsValue::from_str("variance")).expect("has variance");
    assert!(prevalence.as_f64().unwrap_or(f64::NAN).is_finite());
    assert!(logit_value.as_f64().unwrap_or(f64::NAN).is_finite());
    assert!(variance.as_f64().unwrap_or(f64::NAN).is_finite());
}

#[wasm_bindgen_test]
fn wasm_binomial_batch_returns_array() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let successes = vec![1, 3, 4];
    let trials = vec![5, 5, 5];
    let model = WasmBinomialKriging::new(
        &lats,
        &lons,
        &successes,
        &trials,
        "gaussian".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model
        .predict_batch(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch should succeed");
    assert!(Array::is_array(&js));
}

#[wasm_bindgen_test]
fn wasm_ordinary_batch_arrays_returns_typed_arrays() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let values = vec![1.0, 2.0, 1.5];
    let model = WasmOrdinaryKriging::new(
        &lats,
        &lons,
        &values,
        "exponential".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model
        .predict_batch_arrays(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch arrays should succeed");
    let values = Reflect::get(&js, &JsValue::from_str("values")).expect("has values");
    let variances = Reflect::get(&js, &JsValue::from_str("variances")).expect("has variances");
    assert!(values.is_instance_of::<Float64Array>());
    assert!(variances.is_instance_of::<Float64Array>());
}

#[wasm_bindgen_test]
fn wasm_binomial_batch_arrays_returns_typed_arrays() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let successes = vec![1, 3, 4];
    let trials = vec![5, 5, 5];
    let model = WasmBinomialKriging::new(
        &lats,
        &lons,
        &successes,
        &trials,
        "exponential".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model
        .predict_batch_arrays(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch arrays should succeed");
    let prevalences =
        Reflect::get(&js, &JsValue::from_str("prevalences")).expect("has prevalences");
    let logit_values =
        Reflect::get(&js, &JsValue::from_str("logitValues")).expect("has logit values");
    let variances = Reflect::get(&js, &JsValue::from_str("variances")).expect("has variances");
    assert!(prevalences.is_instance_of::<Float64Array>());
    assert!(logit_values.is_instance_of::<Float64Array>());
    assert!(variances.is_instance_of::<Float64Array>());
}

#[cfg(feature = "gpu")]
#[wasm_bindgen_test(async)]
async fn wasm_webgpu_available_returns_boolean() {
    let js = wasm_webgpu_available()
        .await
        .expect("webgpu availability query should succeed");
    assert!(js.as_bool().is_some());
}

#[cfg(feature = "gpu")]
#[wasm_bindgen_test(async)]
async fn wasm_ordinary_predict_batch_gpu_returns_array_with_cpu_fallback() {
    let lats = vec![0.0, 0.0, 1.0];
    let lons = vec![0.0, 1.0, 0.0];
    let values = vec![1.0, 2.0, 1.5];
    let model = WasmOrdinaryKriging::new(
        &lats,
        &lons,
        &values,
        "gaussian".to_string(),
        0.01,
        2.0,
        300.0,
    )
    .expect("model should construct");
    let js = model
        .predict_batch_gpu(&[0.2, 0.4], &[0.2, 0.4])
        .await
        .expect("gpu batch should succeed");
    assert!(Array::is_array(&js));
}
