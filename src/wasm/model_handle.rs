//! Tagged WASM kriging model handle (ADR-0002).
//!
//! One `WasmKrigingModel` wraps every fitted variant for shared lifecycle and instance CV.
//! Per-family `Wasm*Kriging` types are internal Rust wrappers only (not WASM exports).

use wasm_bindgen::prelude::*;

use super::spacetime::{
    WasmSpaceTimeBinomialKriging, WasmSpaceTimeOrdinaryKriging,
    WasmSpaceTimeOrdinaryProjectedKriging, WasmSpaceTimeSimpleKriging,
    WasmSpaceTimeUniversalKriging,
};
use super::{
    WasmBinomialKriging, WasmBinomialProjectedKriging, WasmBinomialTangentPlaneKriging,
    WasmOrdinaryKriging, WasmProjectedKriging, WasmSimpleKriging, WasmUniversalKriging, coded_err,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub enum KrigingModelTag {
    OrdinaryGeo,
    SimpleGeo,
    UniversalGeo,
    BinomialGeo,
    ProjectedOrdinary,
    BinomialProjected,
    BinomialTangentPlane,
    SpaceTimeOrdinaryGeo,
    SpaceTimeSimpleGeo,
    SpaceTimeUniversalGeo,
    SpaceTimeBinomialGeo,
    SpaceTimeOrdinaryProjected,
}

enum KrigingModelInner {
    OrdinaryGeo(WasmOrdinaryKriging),
    SimpleGeo(WasmSimpleKriging),
    UniversalGeo(WasmUniversalKriging),
    BinomialGeo(WasmBinomialKriging),
    ProjectedOrdinary(WasmProjectedKriging),
    BinomialProjected(WasmBinomialProjectedKriging),
    BinomialTangentPlane(WasmBinomialTangentPlaneKriging),
    SpaceTimeOrdinaryGeo(WasmSpaceTimeOrdinaryKriging),
    SpaceTimeSimpleGeo(WasmSpaceTimeSimpleKriging),
    SpaceTimeUniversalGeo(WasmSpaceTimeUniversalKriging),
    SpaceTimeBinomialGeo(WasmSpaceTimeBinomialKriging),
    SpaceTimeOrdinaryProjected(WasmSpaceTimeOrdinaryProjectedKriging),
}

#[wasm_bindgen]
pub struct WasmKrigingModel {
    inner: KrigingModelInner,
}

#[wasm_bindgen]
impl WasmKrigingModel {
    fn tag(&self) -> KrigingModelTag {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(_) => KrigingModelTag::OrdinaryGeo,
            KrigingModelInner::SimpleGeo(_) => KrigingModelTag::SimpleGeo,
            KrigingModelInner::UniversalGeo(_) => KrigingModelTag::UniversalGeo,
            KrigingModelInner::BinomialGeo(_) => KrigingModelTag::BinomialGeo,
            KrigingModelInner::ProjectedOrdinary(_) => KrigingModelTag::ProjectedOrdinary,
            KrigingModelInner::BinomialProjected(_) => KrigingModelTag::BinomialProjected,
            KrigingModelInner::BinomialTangentPlane(_) => KrigingModelTag::BinomialTangentPlane,
            KrigingModelInner::SpaceTimeOrdinaryGeo(_) => KrigingModelTag::SpaceTimeOrdinaryGeo,
            KrigingModelInner::SpaceTimeSimpleGeo(_) => KrigingModelTag::SpaceTimeSimpleGeo,
            KrigingModelInner::SpaceTimeUniversalGeo(_) => KrigingModelTag::SpaceTimeUniversalGeo,
            KrigingModelInner::SpaceTimeBinomialGeo(_) => KrigingModelTag::SpaceTimeBinomialGeo,
            KrigingModelInner::SpaceTimeOrdinaryProjected(_) => {
                KrigingModelTag::SpaceTimeOrdinaryProjected
            }
        }
    }

    /// Geometry tag: `"geo"`, `"projected"`, or `"spacetime"`.
    #[wasm_bindgen(getter)]
    pub fn geometry(&self) -> String {
        match self.tag() {
            KrigingModelTag::OrdinaryGeo
            | KrigingModelTag::SimpleGeo
            | KrigingModelTag::UniversalGeo
            | KrigingModelTag::BinomialGeo
            | KrigingModelTag::BinomialTangentPlane => "geo".to_string(),
            KrigingModelTag::ProjectedOrdinary | KrigingModelTag::BinomialProjected => {
                "projected".to_string()
            }
            KrigingModelTag::SpaceTimeOrdinaryGeo
            | KrigingModelTag::SpaceTimeSimpleGeo
            | KrigingModelTag::SpaceTimeUniversalGeo
            | KrigingModelTag::SpaceTimeBinomialGeo
            | KrigingModelTag::SpaceTimeOrdinaryProjected => "spacetime".to_string(),
        }
    }

    /// Kriging family tag: `"ordinary"`, `"simple"`, `"universal"`, or `"binomial"`.
    #[wasm_bindgen(getter)]
    pub fn family(&self) -> String {
        match self.tag() {
            KrigingModelTag::OrdinaryGeo
            | KrigingModelTag::ProjectedOrdinary
            | KrigingModelTag::SpaceTimeOrdinaryGeo
            | KrigingModelTag::SpaceTimeOrdinaryProjected => "ordinary".to_string(),
            KrigingModelTag::SimpleGeo | KrigingModelTag::SpaceTimeSimpleGeo => {
                "simple".to_string()
            }
            KrigingModelTag::UniversalGeo | KrigingModelTag::SpaceTimeUniversalGeo => {
                "universal".to_string()
            }
            KrigingModelTag::BinomialGeo
            | KrigingModelTag::BinomialProjected
            | KrigingModelTag::BinomialTangentPlane
            | KrigingModelTag::SpaceTimeBinomialGeo => "binomial".to_string(),
        }
    }

    #[wasm_bindgen(js_name = leaveOneOut)]
    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.leave_one_out(),
            KrigingModelInner::SimpleGeo(m) => m.leave_one_out(),
            KrigingModelInner::UniversalGeo(m) => m.leave_one_out(),
            KrigingModelInner::BinomialGeo(m) => m.leave_one_out(),
            KrigingModelInner::ProjectedOrdinary(m) => m.leave_one_out(),
            KrigingModelInner::BinomialProjected(m) => m.leave_one_out(),
            KrigingModelInner::BinomialTangentPlane(_) => Err(coded_err(
                "leaveOneOut is not available for binomial tangent-plane models",
                "invalid_input",
            )),
            KrigingModelInner::SpaceTimeOrdinaryGeo(m) => m.leave_one_out(),
            KrigingModelInner::SpaceTimeSimpleGeo(m) => m.leave_one_out(),
            KrigingModelInner::SpaceTimeUniversalGeo(m) => m.leave_one_out(),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.leave_one_out(),
            KrigingModelInner::SpaceTimeOrdinaryProjected(m) => m.leave_one_out(),
        }
    }

    #[wasm_bindgen(js_name = kFold)]
    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.k_fold(k),
            KrigingModelInner::SimpleGeo(m) => m.k_fold(k),
            KrigingModelInner::UniversalGeo(m) => m.k_fold(k),
            KrigingModelInner::BinomialGeo(m) => m.k_fold(k),
            KrigingModelInner::ProjectedOrdinary(m) => m.k_fold(k),
            KrigingModelInner::BinomialProjected(m) => m.k_fold(k),
            KrigingModelInner::BinomialTangentPlane(_) => Err(coded_err(
                "kFold is not available for binomial tangent-plane models",
                "invalid_input",
            )),
            KrigingModelInner::SpaceTimeOrdinaryGeo(m) => m.k_fold(k),
            KrigingModelInner::SpaceTimeSimpleGeo(m) => m.k_fold(k),
            KrigingModelInner::SpaceTimeUniversalGeo(m) => m.k_fold(k),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.k_fold(k),
            KrigingModelInner::SpaceTimeOrdinaryProjected(m) => m.k_fold(k),
        }
    }

    /// Geographic ordinary kriging from typed arrays.
    #[wasm_bindgen(js_name = ordinaryGeoFromArrays)]
    pub fn ordinary_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::OrdinaryGeo(WasmOrdinaryKriging::from_arrays(
                lats,
                lons,
                values,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
            )?),
        })
    }

    /// Geographic ordinary kriging from a JSON options object (same shape as `OrdinaryKriging`).
    #[wasm_bindgen(js_name = ordinaryGeoNew)]
    pub fn ordinary_geo_new(options: JsValue) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::OrdinaryGeo(WasmOrdinaryKriging::new(options)?),
        })
    }

    /// Single-point prediction for geo / projected 2-D models (`lat`/`lon` or `x`/`y`).
    pub fn predict(&self, a: f64, b: f64) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.predict(a, b),
            KrigingModelInner::SimpleGeo(m) => m.predict(a, b),
            KrigingModelInner::UniversalGeo(m) => m.predict(a, b),
            KrigingModelInner::BinomialGeo(m) => m.predict(a, b),
            KrigingModelInner::ProjectedOrdinary(m) => m.predict(a, b),
            KrigingModelInner::BinomialProjected(m) => m.predict(a, b),
            KrigingModelInner::BinomialTangentPlane(m) => m.predict(a, b),
            _ => Err(coded_err(
                "predict(lat, lon) requires a 2-D geo or projected model; use predictSpaceTime for spacetime handles",
                "invalid_input",
            )),
        }
    }

    /// Single-point prediction for space-time models (`lat`, `lon`, `time` or `x`, `y`, `time`).
    #[wasm_bindgen(js_name = predictSpaceTime)]
    pub fn predict_spacetime(&self, a: f64, b: f64, time: f64) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::SpaceTimeOrdinaryGeo(m) => m.predict(a, b, time),
            KrigingModelInner::SpaceTimeSimpleGeo(m) => m.predict(a, b, time),
            KrigingModelInner::SpaceTimeUniversalGeo(m) => m.predict(a, b, time),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.predict(a, b, time),
            KrigingModelInner::SpaceTimeOrdinaryProjected(m) => m.predict(a, b, time),
            _ => Err(coded_err(
                "predictSpaceTime requires a spacetime model handle",
                "invalid_input",
            )),
        }
    }

    /// Geographic simple kriging from typed arrays and a known `mean`.
    #[wasm_bindgen(js_name = simpleGeoFromArrays)]
    pub fn simple_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        mean: f64,
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SimpleGeo(WasmSimpleKriging::from_arrays(
                lats,
                lons,
                values,
                mean,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
            )?),
        })
    }

    /// Geographic universal kriging from typed arrays.
    #[wasm_bindgen(js_name = universalGeoFromArrays)]
    pub fn universal_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        values: &[f64],
        trend: &str,
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::UniversalGeo(WasmUniversalKriging::from_arrays(
                lats,
                lons,
                values,
                trend,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
            )?),
        })
    }

    /// Projected ordinary kriging from planar `(x, y)` typed arrays.
    #[wasm_bindgen(js_name = projectedOrdinaryFromArrays)]
    pub fn projected_ordinary_from_arrays(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::ProjectedOrdinary(WasmProjectedKriging::from_arrays(
                xs,
                ys,
                values,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
                major_angle_deg,
                range_ratio,
            )?),
        })
    }

    /// Geographic binomial kriging from a JSON options object.
    #[wasm_bindgen(js_name = binomialGeoNew)]
    pub fn binomial_geo_new(options: JsValue) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialGeo(WasmBinomialKriging::new(options)?),
        })
    }

    /// Geographic binomial kriging from typed arrays.
    #[wasm_bindgen(js_name = binomialGeoFromArrays)]
    pub fn binomial_geo_from_arrays(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialGeo(WasmBinomialKriging::from_arrays(
                lats,
                lons,
                successes,
                trials,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
                stability,
                one_step_laplace_observation_variance,
            )?),
        })
    }

    /// Geographic binomial kriging with an explicit Beta prior.
    #[wasm_bindgen(js_name = binomialGeoNewWithPrior)]
    pub fn binomial_geo_new_with_prior(options: JsValue) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialGeo(WasmBinomialKriging::new_with_prior(options)?),
        })
    }

    #[wasm_bindgen(js_name = binomialGeoFromPrecomputedLogits)]
    pub fn binomial_geo_from_precomputed_logits(
        lats: &[f64],
        lons: &[f64],
        logits: &[f64],
        variogram_type: &str,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialGeo(WasmBinomialKriging::from_precomputed_logits(
                lats,
                lons,
                logits,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
            )?),
        })
    }

    #[wasm_bindgen(js_name = binomialGeoFromPrecomputedLogitsWithVariances)]
    pub fn binomial_geo_from_precomputed_logits_with_variances(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialGeo(
                WasmBinomialKriging::from_precomputed_logits_with_variances(
                    lats,
                    lons,
                    logits,
                    logit_observation_variance,
                    variogram_type,
                    nugget,
                    sill,
                    range,
                    shape,
                    prior_alpha,
                    prior_beta,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = binomialProjectedFromArrays)]
    pub fn binomial_projected_from_arrays(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialProjected(WasmBinomialProjectedKriging::from_arrays(
                xs,
                ys,
                successes,
                trials,
                variogram_type,
                nugget,
                sill,
                range,
                shape,
                major_angle_deg,
                range_ratio,
                stability,
                one_step_laplace_observation_variance,
            )?),
        })
    }

    #[wasm_bindgen(js_name = binomialProjectedFromArraysWithPrior)]
    pub fn binomial_projected_from_arrays_with_prior(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialProjected(
                WasmBinomialProjectedKriging::from_arrays_with_prior(
                    xs,
                    ys,
                    successes,
                    trials,
                    variogram_type,
                    nugget,
                    sill,
                    range,
                    shape,
                    major_angle_deg,
                    range_ratio,
                    prior_alpha,
                    prior_beta,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = binomialProjectedFromPrecomputedLogits)]
    pub fn binomial_projected_from_precomputed_logits(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialProjected(
                WasmBinomialProjectedKriging::from_precomputed_logits(
                    xs,
                    ys,
                    logits,
                    variogram_type,
                    nugget,
                    sill,
                    range,
                    shape,
                    major_angle_deg,
                    range_ratio,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = binomialProjectedFromPrecomputedLogitsWithVariances)]
    pub fn binomial_projected_from_precomputed_logits_with_variances(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialProjected(
                WasmBinomialProjectedKriging::from_precomputed_logits_with_variances(
                    xs,
                    ys,
                    logits,
                    logit_observation_variance,
                    variogram_type,
                    nugget,
                    sill,
                    range,
                    shape,
                    major_angle_deg,
                    range_ratio,
                    prior_alpha,
                    prior_beta,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = binomialTangentPlaneNew)]
    pub fn binomial_tangent_plane_new(options: JsValue) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialTangentPlane(WasmBinomialTangentPlaneKriging::new(
                options,
            )?),
        })
    }

    #[wasm_bindgen(js_name = binomialTangentPlaneNewWithPrior)]
    pub fn binomial_tangent_plane_new_with_prior(
        options: JsValue,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialTangentPlane(
                WasmBinomialTangentPlaneKriging::new_with_prior(options)?,
            ),
        })
    }

    #[wasm_bindgen(js_name = binomialTangentPlaneFromArrays)]
    pub fn binomial_tangent_plane_from_arrays(
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
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::BinomialTangentPlane(
                WasmBinomialTangentPlaneKriging::from_arrays(
                    lats,
                    lons,
                    successes,
                    trials,
                    variogram_type,
                    nugget,
                    sill,
                    range,
                    shape,
                    major_angle_deg,
                    range_ratio,
                    tangent_plane_ref_lat,
                    tangent_plane_ref_lon,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeOrdinaryGeoFromArrays)]
    pub fn spacetime_ordinary_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        values: &[f64],
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeOrdinaryGeo(
                WasmSpaceTimeOrdinaryKriging::from_arrays(
                    lats,
                    lons,
                    times,
                    values,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeSimpleGeoFromArrays)]
    pub fn spacetime_simple_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        values: &[f64],
        mean: f64,
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeSimpleGeo(WasmSpaceTimeSimpleKriging::from_arrays(
                lats,
                lons,
                times,
                values,
                mean,
                family,
                spatial_type,
                spatial_nugget,
                spatial_sill,
                spatial_range,
                spatial_shape,
                temporal_type,
                temporal_nugget,
                temporal_sill,
                temporal_range,
                temporal_shape,
                k1,
                k2,
                k3,
            )?),
        })
    }

    #[wasm_bindgen(js_name = spacetimeUniversalGeoFromArrays)]
    pub fn spacetime_universal_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        values: &[f64],
        trend: &str,
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeUniversalGeo(
                WasmSpaceTimeUniversalKriging::from_arrays(
                    lats,
                    lons,
                    times,
                    values,
                    trend,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeBinomialGeoFromArrays)]
    pub fn spacetime_binomial_geo_from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        successes: &[u32],
        trials: &[u32],
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeBinomialGeo(
                WasmSpaceTimeBinomialKriging::from_arrays(
                    lats,
                    lons,
                    times,
                    successes,
                    trials,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeBinomialGeoFromArraysWithPrior)]
    pub fn spacetime_binomial_geo_from_arrays_with_prior(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        successes: &[u32],
        trials: &[u32],
        prior_alpha: f64,
        prior_beta: f64,
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeBinomialGeo(
                WasmSpaceTimeBinomialKriging::from_arrays_with_prior(
                    lats,
                    lons,
                    times,
                    successes,
                    trials,
                    prior_alpha,
                    prior_beta,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeBinomialGeoFromPrecomputedLogits)]
    pub fn spacetime_binomial_geo_from_precomputed_logits(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        logits: &[f64],
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeBinomialGeo(
                WasmSpaceTimeBinomialKriging::from_precomputed_logits(
                    lats,
                    lons,
                    times,
                    logits,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeBinomialGeoFromPrecomputedLogitsWithVariances)]
    pub fn spacetime_binomial_geo_from_precomputed_logits_with_variances(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        logits: &[f64],
        logit_observation_variance: &[f64],
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
        prior_alpha: Option<f64>,
        prior_beta: Option<f64>,
        stability: Option<String>,
        one_step_laplace_observation_variance: Option<bool>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeBinomialGeo(
                WasmSpaceTimeBinomialKriging::from_precomputed_logits_with_variances(
                    lats,
                    lons,
                    times,
                    logits,
                    logit_observation_variance,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                    prior_alpha,
                    prior_beta,
                    stability,
                    one_step_laplace_observation_variance,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = spacetimeOrdinaryProjectedFromArrays)]
    pub fn spacetime_ordinary_projected_from_arrays(
        xs: &[f64],
        ys: &[f64],
        times: &[f64],
        values: &[f64],
        major_angle_deg: f64,
        range_ratio: f64,
        family: &str,
        spatial_type: &str,
        spatial_nugget: f64,
        spatial_sill: f64,
        spatial_range: f64,
        spatial_shape: Option<f64>,
        temporal_type: &str,
        temporal_nugget: f64,
        temporal_sill: f64,
        temporal_range: f64,
        temporal_shape: Option<f64>,
        k1: Option<f64>,
        k2: Option<f64>,
        k3: Option<f64>,
    ) -> Result<WasmKrigingModel, JsValue> {
        Ok(Self {
            inner: KrigingModelInner::SpaceTimeOrdinaryProjected(
                WasmSpaceTimeOrdinaryProjectedKriging::from_arrays(
                    xs,
                    ys,
                    times,
                    values,
                    major_angle_deg,
                    range_ratio,
                    family,
                    spatial_type,
                    spatial_nugget,
                    spatial_sill,
                    spatial_range,
                    spatial_shape,
                    temporal_type,
                    temporal_nugget,
                    temporal_sill,
                    temporal_range,
                    temporal_shape,
                    k1,
                    k2,
                    k3,
                )?,
            ),
        })
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(&self, a: &[f64], b: &[f64]) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.predict_batch(a, b),
            KrigingModelInner::SimpleGeo(m) => m.predict_batch(a, b),
            KrigingModelInner::UniversalGeo(m) => m.predict_batch(a, b),
            KrigingModelInner::BinomialGeo(m) => m.predict_batch(a, b),
            KrigingModelInner::ProjectedOrdinary(m) => m.predict_batch(a, b),
            KrigingModelInner::BinomialProjected(m) => m.predict_batch(a, b),
            KrigingModelInner::BinomialTangentPlane(m) => m.predict_batch(a, b),
            _ => Err(coded_err(
                "predictBatch requires a 2-D geo or projected model; use predictBatchSpaceTime for spacetime handles",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(&self, a: &[f64], b: &[f64]) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.predict_batch_arrays(a, b),
            KrigingModelInner::SimpleGeo(m) => m.predict_batch_arrays(a, b),
            KrigingModelInner::UniversalGeo(m) => m.predict_batch_arrays(a, b),
            KrigingModelInner::BinomialGeo(m) => m.predict_batch_arrays(a, b),
            KrigingModelInner::ProjectedOrdinary(m) => m.predict_batch_arrays(a, b),
            KrigingModelInner::BinomialProjected(m) => m.predict_batch_arrays(a, b),
            KrigingModelInner::BinomialTangentPlane(m) => m.predict_batch_arrays(a, b),
            _ => Err(coded_err(
                "predictBatchArrays requires a 2-D geo or projected model; use predictBatchArraysSpaceTime for spacetime handles",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = predictGridArrays)]
    pub fn predict_grid_arrays(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_cells: usize,
        y_cells: usize,
    ) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => {
                m.predict_grid_arrays(x_min, x_max, y_min, y_max, x_cells, y_cells)
            }
            KrigingModelInner::BinomialGeo(m) => {
                m.predict_grid_arrays(x_min, x_max, y_min, y_max, x_cells, y_cells)
            }
            KrigingModelInner::BinomialProjected(m) => {
                m.predict_grid_arrays(x_min, x_max, y_min, y_max, x_cells, y_cells)
            }
            KrigingModelInner::BinomialTangentPlane(m) => {
                m.predict_grid_arrays(x_min, x_max, y_min, y_max, x_cells, y_cells)
            }
            _ => Err(coded_err(
                "predictGridArrays requires ordinary geo or a binomial geo/projected/tangent-plane model",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = predictBatchSpaceTime)]
    pub fn predict_batch_spacetime(
        &self,
        a: &[f64],
        b: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::SpaceTimeOrdinaryGeo(m) => m.predict_batch(a, b, times),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.predict_batch(a, b, times),
            KrigingModelInner::SpaceTimeSimpleGeo(_)
            | KrigingModelInner::SpaceTimeUniversalGeo(_)
            | KrigingModelInner::SpaceTimeOrdinaryProjected(_) => Err(coded_err(
                "predictBatchSpaceTime is not available for this spacetime model tag; use predictBatchArraysSpaceTime",
                "invalid_input",
            )),
            _ => Err(coded_err(
                "predictBatchSpaceTime requires a spacetime model handle",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = predictBatchArraysSpaceTime)]
    pub fn predict_batch_arrays_spacetime(
        &self,
        a: &[f64],
        b: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::SpaceTimeOrdinaryGeo(m) => m.predict_batch_arrays(a, b, times),
            KrigingModelInner::SpaceTimeSimpleGeo(m) => m.predict_batch_arrays(a, b, times),
            KrigingModelInner::SpaceTimeUniversalGeo(m) => m.predict_batch_arrays(a, b, times),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.predict_batch_arrays(a, b, times),
            KrigingModelInner::SpaceTimeOrdinaryProjected(m) => m.predict_batch_arrays(a, b, times),
            _ => Err(coded_err(
                "predictBatchArraysSpaceTime requires a spacetime model handle",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = setNeighborhood)]
    pub fn set_neighborhood(
        &mut self,
        max_neighbors: Option<usize>,
        max_radius: Option<f64>,
    ) -> Result<(), JsValue> {
        match &mut self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.set_neighborhood(max_neighbors, max_radius),
            _ => Err(coded_err(
                "setNeighborhood is only available for geographic ordinary kriging models",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = neighborhood)]
    pub fn neighborhood(&self) -> JsValue {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.neighborhood(),
            _ => JsValue::NULL,
        }
    }

    /// Returns the known mean for geographic simple kriging models.
    pub fn mean(&self) -> Result<f64, JsValue> {
        match &self.inner {
            KrigingModelInner::SimpleGeo(m) => Ok(m.mean()),
            _ => Err(coded_err(
                "mean is only available for geographic simple kriging models",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = getBuildNotes)]
    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::BinomialGeo(m) => m.get_build_notes(),
            KrigingModelInner::BinomialProjected(m) => m.get_build_notes(),
            KrigingModelInner::BinomialTangentPlane(m) => m.get_build_notes(),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.get_build_notes(),
            _ => Err(coded_err(
                "getBuildNotes is only available for binomial model handles",
                "invalid_input",
            )),
        }
    }

    #[wasm_bindgen(js_name = getDiagnostics)]
    pub fn get_diagnostics(&self, options: JsValue) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::BinomialGeo(m) => m.get_diagnostics(options),
            KrigingModelInner::BinomialProjected(m) => m.get_diagnostics(options),
            KrigingModelInner::BinomialTangentPlane(m) => m.get_diagnostics(options),
            KrigingModelInner::SpaceTimeBinomialGeo(m) => m.get_diagnostics(options),
            _ => Err(coded_err(
                "getDiagnostics is only available for binomial model handles",
                "invalid_input",
            )),
        }
    }

    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = predictBatchGpu)]
    pub async fn predict_batch_gpu(&self, lats: &[f64], lons: &[f64]) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.predict_batch_gpu(lats, lons).await,
            KrigingModelInner::BinomialGeo(m) => m.predict_batch_gpu(lats, lons).await,
            _ => Err(coded_err(
                "predictBatchGpu is only available for geographic ordinary and binomial models",
                "invalid_input",
            )),
        }
    }

    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = predictBatchGpuOrCpu)]
    pub async fn predict_batch_gpu_or_cpu(
        &self,
        lats: &[f64],
        lons: &[f64],
    ) -> Result<JsValue, JsValue> {
        match &self.inner {
            KrigingModelInner::OrdinaryGeo(m) => m.predict_batch_gpu_or_cpu(lats, lons).await,
            KrigingModelInner::BinomialGeo(m) => m.predict_batch_gpu_or_cpu(lats, lons).await,
            _ => Err(coded_err(
                "predictBatchGpuOrCpu is only available for geographic ordinary and binomial models",
                "invalid_input",
            )),
        }
    }
}
