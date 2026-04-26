#![cfg(all(feature = "wasm", target_arch = "wasm32"))]

use js_sys::{Array, Float64Array, Object, Reflect};
use kriging_rs::wasm::spacetime::{
    WasmSpaceTimeBinomialKriging, WasmSpaceTimeOrdinaryKriging,
    WasmSpaceTimeOrdinaryProjectedKriging, WasmSpaceTimeSimpleKriging,
    WasmSpaceTimeUniversalKriging, wasm_compute_empirical_spacetime_variogram,
    wasm_fit_spacetime_variogram,
};
#[cfg(feature = "gpu")]
use kriging_rs::wasm::wasm_webgpu_available;
use kriging_rs::wasm::{WasmBinomialKriging, WasmOrdinaryKriging};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn set(obj: &Object, key: &str, value: JsValue) {
    Reflect::set(obj, &JsValue::from_str(key), &value).expect("set property");
}

fn variogram_object(
    variogram_type: &str,
    nugget: f64,
    sill: f64,
    range: f64,
    shape: Option<f64>,
) -> JsValue {
    let vario = Object::new();
    set(&vario, "variogramType", JsValue::from_str(variogram_type));
    set(&vario, "nugget", JsValue::from_f64(nugget));
    set(&vario, "sill", JsValue::from_f64(sill));
    set(&vario, "range", JsValue::from_f64(range));
    if let Some(s) = shape {
        set(&vario, "shape", JsValue::from_f64(s));
    }
    vario.into()
}

fn to_f64_array(slice: &[f64]) -> JsValue {
    let arr = Array::new();
    for v in slice {
        arr.push(&JsValue::from_f64(*v));
    }
    arr.into()
}

fn to_u32_array(slice: &[u32]) -> JsValue {
    let arr = Array::new();
    for v in slice {
        arr.push(&JsValue::from_f64(*v as f64));
    }
    arr.into()
}

fn ordinary_options(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    variogram_type: &str,
    nugget: f64,
    sill: f64,
    range: f64,
) -> JsValue {
    let opts = Object::new();
    set(&opts, "lats", to_f64_array(lats));
    set(&opts, "lons", to_f64_array(lons));
    set(&opts, "values", to_f64_array(values));
    set(
        &opts,
        "variogram",
        variogram_object(variogram_type, nugget, sill, range, None),
    );
    opts.into()
}

fn binomial_options(
    lats: &[f64],
    lons: &[f64],
    successes: &[u32],
    trials: &[u32],
    variogram_type: &str,
    nugget: f64,
    sill: f64,
    range: f64,
) -> JsValue {
    let opts = Object::new();
    set(&opts, "lats", to_f64_array(lats));
    set(&opts, "lons", to_f64_array(lons));
    set(&opts, "successes", to_u32_array(successes));
    set(&opts, "trials", to_u32_array(trials));
    set(
        &opts,
        "variogram",
        variogram_object(variogram_type, nugget, sill, range, None),
    );
    opts.into()
}

#[wasm_bindgen_test]
fn wasm_ordinary_predict_returns_serializable_object() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let values = [1.0, 2.0, 1.5];
    let opts = ordinary_options(&lats, &lons, &values, "exponential", 0.01, 2.0, 300.0);
    let model = WasmOrdinaryKriging::new(opts).expect("model should construct");
    let js = model.predict(0.2, 0.2).expect("prediction should succeed");
    let value = Reflect::get(&js, &JsValue::from_str("value")).expect("has value field");
    let variance = Reflect::get(&js, &JsValue::from_str("variance")).expect("has variance field");
    assert!(value.as_f64().unwrap_or(f64::NAN).is_finite());
    assert!(variance.as_f64().unwrap_or(f64::NAN).is_finite());
}

#[wasm_bindgen_test]
fn wasm_ordinary_batch_returns_array() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let values = [1.0, 2.0, 1.5];
    let opts = ordinary_options(&lats, &lons, &values, "gaussian", 0.01, 2.0, 300.0);
    let model = WasmOrdinaryKriging::new(opts).expect("model should construct");
    let js = model
        .predict_batch(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch should succeed");
    assert!(Array::is_array(&js));
}

#[wasm_bindgen_test]
fn wasm_binomial_predict_returns_serializable_object() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let successes = [1u32, 3, 4];
    let trials = [5u32, 5, 5];
    let opts = binomial_options(
        &lats,
        &lons,
        &successes,
        &trials,
        "exponential",
        0.01,
        2.0,
        300.0,
    );
    let model = WasmBinomialKriging::new(opts).expect("model should construct");
    let js = model.predict(0.2, 0.2).expect("prediction should succeed");
    let prevalence = Reflect::get(&js, &JsValue::from_str("prevalence")).expect("has prevalence");
    let logit_value = Reflect::get(&js, &JsValue::from_str("logitValue")).expect("has logit");
    let variance = Reflect::get(&js, &JsValue::from_str("variance")).expect("has variance");
    let prev_var = Reflect::get(&js, &JsValue::from_str("prevalenceVariance"))
        .expect("has prevalenceVariance");
    assert!(prevalence.as_f64().unwrap_or(f64::NAN).is_finite());
    assert!(logit_value.as_f64().unwrap_or(f64::NAN).is_finite());
    assert!(variance.as_f64().unwrap_or(f64::NAN).is_finite());
    let pv = prev_var.as_f64().unwrap_or(f64::NAN);
    assert!(pv.is_finite() && pv >= 0.0);
}

#[wasm_bindgen_test]
fn wasm_binomial_get_build_notes_includes_version() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let successes = [1u32, 3, 4];
    let trials = [5u32, 5, 5];
    let opts = binomial_options(
        &lats,
        &lons,
        &successes,
        &trials,
        "exponential",
        0.01,
        2.0,
        300.0,
    );
    let model = WasmBinomialKriging::new(opts).expect("model should construct");
    let notes = model.get_build_notes().expect("build notes");
    let v = Reflect::get(&notes, &JsValue::from_str("calibrationVersion"))
        .expect("calibrationVersion")
        .as_f64()
        .unwrap();
    assert!(v >= 1.0);
}

#[wasm_bindgen_test]
fn wasm_binomial_batch_returns_array() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let successes = [1u32, 3, 4];
    let trials = [5u32, 5, 5];
    let opts = binomial_options(
        &lats, &lons, &successes, &trials, "gaussian", 0.01, 2.0, 300.0,
    );
    let model = WasmBinomialKriging::new(opts).expect("model should construct");
    let js = model
        .predict_batch(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch should succeed");
    assert!(Array::is_array(&js));
}

#[wasm_bindgen_test]
fn wasm_ordinary_batch_arrays_returns_typed_arrays() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let values = [1.0, 2.0, 1.5];
    let opts = ordinary_options(&lats, &lons, &values, "exponential", 0.01, 2.0, 300.0);
    let model = WasmOrdinaryKriging::new(opts).expect("model should construct");
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
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let successes = [1u32, 3, 4];
    let trials = [5u32, 5, 5];
    let opts = binomial_options(
        &lats,
        &lons,
        &successes,
        &trials,
        "exponential",
        0.01,
        2.0,
        300.0,
    );
    let model = WasmBinomialKriging::new(opts).expect("model should construct");
    let js = model
        .predict_batch_arrays(&[0.2, 0.4], &[0.2, 0.4])
        .expect("batch arrays should succeed");
    let prevalences =
        Reflect::get(&js, &JsValue::from_str("prevalences")).expect("has prevalences");
    let logit_values =
        Reflect::get(&js, &JsValue::from_str("logitValues")).expect("has logit values");
    let variances = Reflect::get(&js, &JsValue::from_str("variances")).expect("has variances");
    let prevalence_variances = Reflect::get(&js, &JsValue::from_str("prevalenceVariances"))
        .expect("has prevalence variances");
    assert!(prevalences.is_instance_of::<Float64Array>());
    assert!(logit_values.is_instance_of::<Float64Array>());
    assert!(variances.is_instance_of::<Float64Array>());
    assert!(prevalence_variances.is_instance_of::<Float64Array>());
}

#[wasm_bindgen_test]
fn wasm_mismatched_arrays_throws_with_code() {
    let lats = [0.0, 1.0];
    let lons = [0.0, 1.0, 2.0];
    let values = [1.0, 2.0];
    let opts = ordinary_options(&lats, &lons, &values, "exponential", 0.01, 2.0, 300.0);
    let err = WasmOrdinaryKriging::new(opts).expect_err("mismatched arrays should error");
    let code = Reflect::get(&err, &JsValue::from_str("code"))
        .expect("error has code")
        .as_string()
        .unwrap_or_default();
    assert_eq!(code, "mismatched_arrays");
}

// ---------- Space–time bindings ----------

#[wasm_bindgen_test]
fn wasm_spacetime_ordinary_predict_is_finite() {
    let lats = [0.0, 0.0, 1.0, 1.0];
    let lons = [0.0, 1.0, 0.0, 1.0];
    let times = [0.0, 1.0, 2.0, 3.0];
    let values = [1.0, 2.0, 1.5, 2.5];
    let model = WasmSpaceTimeOrdinaryKriging::from_arrays(
        &lats,
        &lons,
        &times,
        &values,
        "separable",
        "exponential",
        0.01,
        1.0,
        300.0,
        None,
        "exponential",
        0.01,
        1.0,
        5.0,
        None,
        None,
        None,
        None,
    )
    .expect("model should construct");
    let js = model
        .predict(0.5, 0.5, 1.5)
        .expect("predict should succeed");
    let value = Reflect::get(&js, &JsValue::from_str("value"))
        .expect("has value")
        .as_f64()
        .unwrap_or(f64::NAN);
    let variance = Reflect::get(&js, &JsValue::from_str("variance"))
        .expect("has variance")
        .as_f64()
        .unwrap_or(f64::NAN);
    assert!(value.is_finite());
    assert!(variance.is_finite() && variance >= 0.0);
}

#[wasm_bindgen_test]
fn wasm_spacetime_ordinary_batch_arrays_returns_typed_arrays() {
    let lats = [0.0, 0.0, 1.0, 1.0];
    let lons = [0.0, 1.0, 0.0, 1.0];
    let times = [0.0, 1.0, 2.0, 3.0];
    let values = [1.0, 2.0, 1.5, 2.5];
    let model = WasmSpaceTimeOrdinaryKriging::from_arrays(
        &lats,
        &lons,
        &times,
        &values,
        "separable",
        "exponential",
        0.01,
        1.0,
        300.0,
        None,
        "exponential",
        0.01,
        1.0,
        5.0,
        None,
        None,
        None,
        None,
    )
    .expect("model should construct");
    let js = model
        .predict_batch_arrays(&[0.2, 0.4], &[0.3, 0.6], &[0.5, 1.5])
        .expect("batch arrays should succeed");
    let values = Reflect::get(&js, &JsValue::from_str("values")).expect("has values");
    let variances = Reflect::get(&js, &JsValue::from_str("variances")).expect("has variances");
    assert!(values.is_instance_of::<Float64Array>());
    assert!(variances.is_instance_of::<Float64Array>());
}

#[wasm_bindgen_test]
fn wasm_spacetime_simple_reverts_to_mean_far_away() {
    let lats = [0.0, 0.0, 1.0, 1.0];
    let lons = [0.0, 1.0, 0.0, 1.0];
    let times = [0.0, 1.0, 2.0, 3.0];
    let values = [1.0, 2.0, 1.5, 2.5];
    let mean = 100.0;
    let model = WasmSpaceTimeSimpleKriging::from_arrays(
        &lats,
        &lons,
        &times,
        &values,
        mean,
        "separable",
        "exponential",
        0.01,
        1.0,
        2.0,
        None,
        "exponential",
        0.01,
        1.0,
        0.5,
        None,
        None,
        None,
        None,
    )
    .expect("model should construct");
    let js = model
        .predict(50.0, 50.0, 500.0)
        .expect("predict should succeed");
    let value = Reflect::get(&js, &JsValue::from_str("value"))
        .expect("has value")
        .as_f64()
        .unwrap();
    assert!((value - mean).abs() < 1.0);
}

#[wasm_bindgen_test]
fn wasm_spacetime_universal_constant_trend_is_finite() {
    let lats = [0.0, 0.0, 1.0, 1.0];
    let lons = [0.0, 1.0, 0.0, 1.0];
    let times = [0.0, 1.0, 2.0, 3.0];
    let values = [1.0, 2.0, 1.5, 2.5];
    let model = WasmSpaceTimeUniversalKriging::from_arrays(
        &lats,
        &lons,
        &times,
        &values,
        "constant",
        "separable",
        "exponential",
        0.05,
        1.0,
        300.0,
        None,
        "exponential",
        0.05,
        1.0,
        5.0,
        None,
        None,
        None,
        None,
    )
    .expect("model should construct");
    let js = model
        .predict(0.5, 0.5, 1.5)
        .expect("predict should succeed");
    let value = Reflect::get(&js, &JsValue::from_str("value"))
        .expect("has value")
        .as_f64()
        .unwrap_or(f64::NAN);
    assert!(value.is_finite());
}

#[wasm_bindgen_test]
fn wasm_spacetime_binomial_returns_unit_interval_prevalence() {
    let lats = [0.0, 0.0, 1.0];
    let lons = [0.0, 1.0, 0.0];
    let times = [0.0, 1.0, 2.0];
    let successes: [u32; 3] = [3, 7, 5];
    let trials: [u32; 3] = [10, 10, 10];
    let model = WasmSpaceTimeBinomialKriging::from_arrays(
        &lats,
        &lons,
        &times,
        &successes,
        &trials,
        "separable",
        "exponential",
        0.05,
        1.0,
        200.0,
        None,
        "exponential",
        0.05,
        1.0,
        5.0,
        None,
        None,
        None,
        None,
    )
    .expect("model should construct");
    let js = model
        .predict(0.5, 0.5, 1.0)
        .expect("predict should succeed");
    let prev = Reflect::get(&js, &JsValue::from_str("prevalence"))
        .expect("has prevalence")
        .as_f64()
        .unwrap();
    assert!(prev > 0.0 && prev < 1.0);
}

#[wasm_bindgen_test]
fn wasm_spacetime_projected_predict_is_finite() {
    let xs = [0.0, 0.0, 1.0, 1.0];
    let ys = [0.0, 1.0, 0.0, 1.0];
    let times = [0.0, 1.0, 2.0, 3.0];
    let values = [1.0, 2.0, 1.5, 2.5];
    let model = WasmSpaceTimeOrdinaryProjectedKriging::from_arrays(
        &xs,
        &ys,
        &times,
        &values,
        0.0,
        1.0,
        "separable",
        "exponential",
        0.01,
        1.0,
        2.0,
        None,
        "exponential",
        0.01,
        1.0,
        3.0,
        None,
        None,
        None,
        None,
    )
    .expect("model should construct");
    let js = model
        .predict(0.5, 0.5, 1.5)
        .expect("predict should succeed");
    let value = Reflect::get(&js, &JsValue::from_str("value"))
        .expect("has value")
        .as_f64()
        .unwrap_or(f64::NAN);
    assert!(value.is_finite());
}

#[wasm_bindgen_test]
fn wasm_compute_empirical_spacetime_variogram_returns_typed_arrays() {
    let lats = [0.0, 0.0, 1.0, 1.0, 0.5];
    let lons = [0.0, 1.0, 0.0, 1.0, 0.5];
    let times = [0.0, 1.0, 2.0, 3.0, 1.5];
    let values = [1.0, 2.0, 1.5, 2.5, 2.0];
    let js = wasm_compute_empirical_spacetime_variogram(
        &lats,
        &lons,
        &times,
        &values,
        None,
        None,
        4,
        4,
        "classical",
    )
    .expect("empirical computation should succeed");
    let semis = Reflect::get(&js, &JsValue::from_str("semivariances")).expect("has semivariances");
    let n_pairs = Reflect::get(&js, &JsValue::from_str("nPairs")).expect("has nPairs");
    assert!(semis.is_instance_of::<Float64Array>());
    assert!(n_pairs.is_instance_of::<Float64Array>());
}

#[wasm_bindgen_test]
fn wasm_fit_spacetime_variogram_returns_fit_object() {
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    let mut times = Vec::new();
    let mut values = Vec::new();
    for i in 0..6 {
        for t in 0..5 {
            lats.push(i as f64 * 0.1);
            lons.push(0.05 * (t as f64));
            times.push(t as f64);
            values.push(((i + t) as f64) * 0.5);
        }
    }
    let js = wasm_fit_spacetime_variogram(
        &lats,
        &lons,
        &times,
        &values,
        None,
        None,
        5,
        5,
        "classical",
        "separable",
        "exponential",
        "exponential",
    )
    .expect("fit should succeed");
    let fit = Reflect::get(&js, &JsValue::from_str("fit")).expect("has fit");
    let family = Reflect::get(&fit, &JsValue::from_str("family"))
        .expect("has family")
        .as_string()
        .unwrap_or_default();
    assert_eq!(family, "separable");
    let residuals = Reflect::get(&fit, &JsValue::from_str("residuals"))
        .expect("has residuals")
        .as_f64()
        .unwrap_or(f64::NAN);
    assert!(residuals.is_finite());
}

#[cfg(feature = "gpu")]
#[wasm_bindgen_test(async)]
async fn wasm_webgpu_available_returns_boolean() {
    let js = wasm_webgpu_available()
        .await
        .expect("webgpu availability query should succeed");
    assert!(js.as_bool().is_some());
}
