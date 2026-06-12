//! WebAssembly bindings for the [`crate::spacetime`] module.
//!
//! Layout mirrors the 2-D bindings in [`super`]: one `WasmSpaceTime*Kriging` struct per
//! kriging family, each with `fromArrays` and zero-copy `predictBatchArrays` constructors,
//! plus stand-alone functions for empirical-variogram computation and parametric fitting.

#![allow(clippy::too_many_arguments)]
// JS interop only accepts `f64`. See note in `super` (src/wasm/mod.rs).
#![allow(clippy::unnecessary_cast)]

use js_sys::{Float64Array, Object};
use wasm_bindgen::prelude::*;

use crate::Real;
use crate::cv::{
    SpacetimeBinomialPredictor, SpacetimeOrdinaryPredictor, SpacetimeSimplePredictor,
    SpacetimeUniversalPredictor, k_fold_cv, leave_one_out_cv,
};
use crate::distance::GeoCoord;
use crate::kriging::binomial::BinomialCalibratedResult;
use crate::projected::{Anisotropy2D, ProjectedCoord};
use crate::simulation::{
    SpacetimeBinomialSimulator, SpacetimeOrdinarySimulator, SpacetimeSimpleSimulator,
    SpacetimeUniversalSimulator, sequential_binomial_simulate, sequential_binomial_simulate_many,
    sequential_gaussian_simulate, sequential_gaussian_simulate_many,
};
use crate::spacetime::SpaceTimeBinomialDiagnostics;
use crate::spacetime::{
    EmpiricalSpaceTimeVariogram, GeoMetric, ProjectedMetric, SpaceTimeBinomialKrigingModel,
    SpaceTimeBinomialObservation, SpaceTimeCoord, SpaceTimeDataset, SpaceTimeFitConfig,
    SpaceTimeFitResult, SpaceTimeOrdinaryKrigingModel, SpaceTimeSimpleKrigingModel,
    SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend, SpaceTimeVariogram,
    SpaceTimeVariogramConfig, SpaceTimeVariogramType, compute_empirical_spacetime_variogram,
    fit_spacetime_variogram,
};
use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal};
use crate::variogram::models::{VariogramModel, VariogramType};

use super::cv_options::UnifiedCvOptions;
use super::model_cv;
use super::simulate_options::UnifiedSimulateOptions;
use super::{
    JsBinomialPrediction, JsPrediction, binomial_cv_result_to_js, binomial_many_simulation_to_js,
    binomial_simulation_to_js, coded_err, cv_result_to_js, err_to_js,
    heteroskedastic_config_from_optional_stability_str, kriging_err_to_js,
    map_binomial_predictions, map_predictions, merge_hetero_with_optional_one_step_laplace,
    parse_binomial_prior, parse_variogram, set_binomial_flat_array_fields, set_object_field,
    split_binomial_predictions, split_predictions,
};

const FAMILY_HELP: &str = "family must be 'separable' or 'productSum'";

fn parse_spacetime_variogram(
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
) -> Result<SpaceTimeVariogram, JsValue> {
    let spatial = parse_variogram(
        spatial_type,
        spatial_nugget,
        spatial_sill,
        spatial_range,
        spatial_shape,
    )?;
    let temporal = parse_variogram(
        temporal_type,
        temporal_nugget,
        temporal_sill,
        temporal_range,
        temporal_shape,
    )?;
    match family.to_ascii_lowercase().as_str() {
        "separable" => {
            SpaceTimeVariogram::new_separable(spatial, temporal).map_err(kriging_err_to_js)
        }
        "product_sum" | "productsum" | "product-sum" => {
            let k1 = k1.unwrap_or(1.0) as Real;
            let k2 = k2.unwrap_or(0.0) as Real;
            let k3 = k3.unwrap_or(0.0) as Real;
            SpaceTimeVariogram::new_product_sum(spatial, temporal, k1, k2, k3)
                .map_err(kriging_err_to_js)
        }
        _ => Err(coded_err(FAMILY_HELP, "unknown_family")),
    }
}

fn parse_universal_trend(name: &str) -> Result<SpaceTimeUniversalTrend, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "constant" => SpaceTimeUniversalTrend::Constant,
        "linearintime" | "linear_in_time" => SpaceTimeUniversalTrend::LinearInTime,
        "quadraticintime" | "quadratic_in_time" => SpaceTimeUniversalTrend::QuadraticInTime,
        "linearinspace" | "linear_in_space" => SpaceTimeUniversalTrend::LinearInSpace,
        "linearinspaceandtime" | "linear_in_space_and_time" => {
            SpaceTimeUniversalTrend::LinearInSpaceAndTime
        }
        "quadraticinspaceandtime" | "quadratic_in_space_and_time" => {
            SpaceTimeUniversalTrend::QuadraticInSpaceAndTime
        }
        _ => {
            return Err(coded_err(
                "trend must be 'constant', 'linearInTime', 'quadraticInTime', \
                 'linearInSpace', 'linearInSpaceAndTime', or 'quadraticInSpaceAndTime'",
                "unknown_trend",
            ));
        }
    })
}

fn parse_estimator(name: &str) -> Result<EmpiricalEstimator, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "classical" | "matheron" => EmpiricalEstimator::Classical,
        "cressiehawkins" | "cressie_hawkins" | "cressie-hawkins" => {
            EmpiricalEstimator::CressieHawkins
        }
        _ => {
            return Err(coded_err(
                "estimator must be 'classical' (Matheron) or 'cressieHawkins'",
                "unknown_estimator",
            ));
        }
    })
}

fn parse_family(name: &str) -> Result<SpaceTimeVariogramType, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "separable" => SpaceTimeVariogramType::Separable,
        "product_sum" | "productsum" | "product-sum" => SpaceTimeVariogramType::ProductSum,
        _ => return Err(coded_err(FAMILY_HELP, "unknown_family")),
    })
}

fn build_geo_dataset(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
) -> Result<SpaceTimeDataset<GeoCoord>, JsValue> {
    if lats.len() != lons.len() || lats.len() != times.len() || lats.len() != values.len() {
        return Err(coded_err(
            "lats, lons, times, and values must all have the same length",
            "mismatched_arrays",
        ));
    }
    let mut coords = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let geo = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        coords.push(SpaceTimeCoord::try_new(geo, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    SpaceTimeDataset::new(coords, values_real).map_err(kriging_err_to_js)
}

fn build_geo_targets(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
) -> Result<Vec<SpaceTimeCoord<GeoCoord>>, JsValue> {
    if lats.len() != lons.len() || lats.len() != times.len() {
        return Err(coded_err(
            "lats, lons, and times must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let geo = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        out.push(SpaceTimeCoord::try_new(geo, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

fn build_projected_dataset(
    xs: &[f64],
    ys: &[f64],
    times: &[f64],
    values: &[f64],
) -> Result<SpaceTimeDataset<ProjectedCoord>, JsValue> {
    if xs.len() != ys.len() || xs.len() != times.len() || xs.len() != values.len() {
        return Err(coded_err(
            "xs, ys, times, and values must all have the same length",
            "mismatched_arrays",
        ));
    }
    let mut coords = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let proj = ProjectedCoord::new(xs[i] as Real, ys[i] as Real);
        coords.push(SpaceTimeCoord::try_new(proj, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    SpaceTimeDataset::new(coords, values_real).map_err(kriging_err_to_js)
}

fn build_projected_targets(
    xs: &[f64],
    ys: &[f64],
    times: &[f64],
) -> Result<Vec<SpaceTimeCoord<ProjectedCoord>>, JsValue> {
    if xs.len() != ys.len() || xs.len() != times.len() {
        return Err(coded_err(
            "xs, ys, and times must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let proj = ProjectedCoord::new(xs[i] as Real, ys[i] as Real);
        out.push(SpaceTimeCoord::try_new(proj, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

fn projected_metric(major_angle_deg: f64, range_ratio: f64) -> Result<ProjectedMetric, JsValue> {
    let aniso = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
        .map_err(kriging_err_to_js)?;
    Ok(ProjectedMetric::with_anisotropy(aniso))
}

fn variogram_type_name(vt: VariogramType) -> &'static str {
    match vt {
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

fn st_geo_cv_slices(
    dataset: &SpaceTimeDataset<GeoCoord>,
) -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>) {
    (dataset.coords().to_vec(), dataset.values().to_vec())
}

fn st_projected_cv_slices(
    dataset: &SpaceTimeDataset<ProjectedCoord>,
    anisotropy: Anisotropy2D,
) -> (Vec<SpaceTimeCoord<ProjectedCoord>>, Vec<Real>, Anisotropy2D) {
    (
        dataset.coords().to_vec(),
        dataset.values().to_vec(),
        anisotropy,
    )
}

// ---------- Geo: Ordinary ----------

pub(crate) struct WasmSpaceTimeOrdinaryKriging {
    inner: SpaceTimeOrdinaryKrigingModel<GeoMetric>,
    cv_coords: Vec<SpaceTimeCoord<GeoCoord>>,
    cv_values: Vec<Real>,
}

impl WasmSpaceTimeOrdinaryKriging {
    pub fn from_arrays(
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
    ) -> Result<WasmSpaceTimeOrdinaryKriging, JsValue> {
        let dataset = build_geo_dataset(lats, lons, times, values)?;
        let (cv_coords, cv_values) = st_geo_cv_slices(&dataset);
        let variogram = parse_spacetime_variogram(
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
        )?;
        let inner = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, variogram)
            .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner,
            cv_coords,
            cv_values,
        })
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
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
        model_cv::spacetime_ordinary_cv(
            GeoMetric,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::spacetime_ordinary_cv(
            GeoMetric,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            Some(k),
        )
    }
}

// ---------- Geo: Simple ----------

pub(crate) struct WasmSpaceTimeSimpleKriging {
    inner: SpaceTimeSimpleKrigingModel<GeoMetric>,
    cv_coords: Vec<SpaceTimeCoord<GeoCoord>>,
    cv_values: Vec<Real>,
    cv_mean: Real,
}

impl WasmSpaceTimeSimpleKriging {
    pub fn from_arrays(
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
    ) -> Result<WasmSpaceTimeSimpleKriging, JsValue> {
        let dataset = build_geo_dataset(lats, lons, times, values)?;
        let (cv_coords, cv_values) = st_geo_cv_slices(&dataset);
        let variogram = parse_spacetime_variogram(
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
        )?;
        let cv_mean = mean as Real;
        let inner = SpaceTimeSimpleKrigingModel::new(GeoMetric, dataset, variogram, cv_mean)
            .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner,
            cv_coords,
            cv_values,
            cv_mean,
        })
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
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
        model_cv::spacetime_simple_cv(
            GeoMetric,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            self.cv_mean,
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::spacetime_simple_cv(
            GeoMetric,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            self.cv_mean,
            Some(k),
        )
    }
}

// ---------- Geo: Universal ----------

pub(crate) struct WasmSpaceTimeUniversalKriging {
    inner: SpaceTimeUniversalKrigingModel<GeoMetric>,
    cv_coords: Vec<SpaceTimeCoord<GeoCoord>>,
    cv_values: Vec<Real>,
    cv_trend: SpaceTimeUniversalTrend,
}

impl WasmSpaceTimeUniversalKriging {
    pub fn from_arrays(
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
    ) -> Result<WasmSpaceTimeUniversalKriging, JsValue> {
        let dataset = build_geo_dataset(lats, lons, times, values)?;
        let (cv_coords, cv_values) = st_geo_cv_slices(&dataset);
        let variogram = parse_spacetime_variogram(
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
        )?;
        let trend = parse_universal_trend(trend)?;
        let inner = SpaceTimeUniversalKrigingModel::new(GeoMetric, dataset, variogram, trend)
            .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner,
            cv_coords,
            cv_values,
            cv_trend: trend,
        })
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
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
        model_cv::spacetime_universal_cv(
            GeoMetric,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            self.cv_trend,
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::spacetime_universal_cv(
            GeoMetric,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            self.cv_trend,
            Some(k),
        )
    }
}

// ---------- Geo: Binomial ----------

fn st_binomial_geo_collect_observations(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    successes: &[u32],
    trials: &[u32],
) -> Result<(Vec<SpaceTimeBinomialObservation<GeoCoord>>, Vec<usize>), JsValue> {
    if lats.len() != lons.len()
        || lats.len() != times.len()
        || lats.len() != successes.len()
        || lats.len() != trials.len()
    {
        return Err(coded_err(
            "lats, lons, times, successes, trials must all have the same length",
            "mismatched_arrays",
        ));
    }
    let mut observations = Vec::new();
    let mut zero_trial_drops: Vec<usize> = Vec::new();
    for i in 0..lats.len() {
        if trials[i] == 0 {
            zero_trial_drops.push(i);
            continue;
        }
        let coord =
            GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        let st = SpaceTimeCoord::try_new(coord, times[i] as Real).map_err(kriging_err_to_js)?;
        observations.push(
            SpaceTimeBinomialObservation::new(st, successes[i], trials[i])
                .map_err(kriging_err_to_js)?,
        );
    }
    if observations.len() < 2 {
        return Err(coded_err(
            "need at least two non-zero-trial space-time sites after dropping trials==0",
            "too_few_points",
        ));
    }
    Ok((observations, zero_trial_drops))
}

type StGeoBinomialFit = BinomialCalibratedResult<SpaceTimeBinomialKrigingModel<GeoMetric>>;

pub(crate) struct WasmSpaceTimeBinomialKriging {
    fit: StGeoBinomialFit,
}

fn st_binomial_diagnostics_to_js(d: &SpaceTimeBinomialDiagnostics) -> Result<JsValue, JsValue> {
    let variogram_js = space_time_variogram_diagnostic_js(&d.variogram)?;
    let notes_js = serde_wasm_bindgen::to_value(&d.build_notes).map_err(err_to_js)?;
    let out = Object::new();
    set_object_field(&out, "variogram", &variogram_js)?;
    set_object_field(&out, "buildNotes", &notes_js)?;
    if let Some(m) = d.logit_loo_msdr {
        set_object_field(&out, "logitLooMsdr", &JsValue::from_f64(m as f64))?;
    }
    Ok(out.into())
}

impl WasmSpaceTimeBinomialKriging {
    pub fn from_arrays(
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
    ) -> Result<WasmSpaceTimeBinomialKriging, JsValue> {
        let (observations, zero_trial_drops) =
            st_binomial_geo_collect_observations(lats, lons, times, successes, trials)?;
        let variogram = parse_spacetime_variogram(
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
        )?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let mut fit = SpaceTimeBinomialKrigingModel::new(GeoMetric, observations, variogram, hcfg)
            .map_err(kriging_err_to_js)?;
        fit.notes.zero_trial_dropped_indices = zero_trial_drops;
        fit.notes.zero_trial_dropped_indices.sort_unstable();
        Ok(Self { fit })
    }

    /// Like [`Self::from_arrays`] but with an explicit Beta(`prior_alpha`, `prior_beta`) prior
    /// on prevalence (same semantics as geo `WasmBinomialKriging::new_with_prior`).
    pub fn from_arrays_with_prior(
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
    ) -> Result<WasmSpaceTimeBinomialKriging, JsValue> {
        let (observations, zero_trial_drops) =
            st_binomial_geo_collect_observations(lats, lons, times, successes, trials)?;
        let variogram = parse_spacetime_variogram(
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
        )?;
        let prior = parse_binomial_prior(Some(prior_alpha), Some(prior_beta))?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let mut fit = SpaceTimeBinomialKrigingModel::new_with_prior(
            GeoMetric,
            observations,
            variogram,
            prior,
            hcfg,
        )
        .map_err(kriging_err_to_js)?;
        fit.notes.zero_trial_dropped_indices = zero_trial_drops;
        fit.notes.zero_trial_dropped_indices.sort_unstable();
        Ok(Self { fit })
    }

    /// Build from pre-computed finite logits at each `(lat, lon, time)` (no per-trial observation
    /// variance on the diagonal; see geo `WasmBinomialKriging::from_precomputed_logits`).
    pub fn from_precomputed_logits(
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
    ) -> Result<WasmSpaceTimeBinomialKriging, JsValue> {
        if lats.len() != lons.len() || lats.len() != times.len() || lats.len() != logits.len() {
            return Err(coded_err(
                "lats, lons, times, and logits must all have the same length",
                "mismatched_arrays",
            ));
        }
        if lats.len() < 2 {
            return Err(coded_err(
                "need at least two space-time sites for binomial kriging",
                "too_few_points",
            ));
        }
        let mut coords: Vec<SpaceTimeCoord<GeoCoord>> = Vec::with_capacity(lats.len());
        for i in 0..lats.len() {
            let coord =
                GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
            let st = SpaceTimeCoord::try_new(coord, times[i] as Real).map_err(kriging_err_to_js)?;
            coords.push(st);
        }
        let variogram = parse_spacetime_variogram(
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
        )?;
        let logits_real: Vec<Real> = logits.iter().map(|v| *v as Real).collect();
        let fit = SpaceTimeBinomialKrigingModel::from_precomputed_logits(
            GeoMetric,
            coords,
            logits_real,
            variogram,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self { fit })
    }

    /// Like [`Self::from_precomputed_logits`], with per-site logit observation variances on the
    /// diagonal. Optional `prior_alpha` / `prior_beta` must be supplied together.
    pub fn from_precomputed_logits_with_variances(
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
    ) -> Result<WasmSpaceTimeBinomialKriging, JsValue> {
        if lats.len() != lons.len() || lats.len() != times.len() || lats.len() != logits.len() {
            return Err(coded_err(
                "lats, lons, times, and logits must all have the same length",
                "mismatched_arrays",
            ));
        }
        if logit_observation_variance.len() != logits.len() {
            return Err(coded_err(
                "logitObservationVariance must have the same length as logits",
                "mismatched_arrays",
            ));
        }
        if lats.len() < 2 {
            return Err(coded_err(
                "need at least two space-time sites for binomial kriging",
                "too_few_points",
            ));
        }
        let mut coords: Vec<SpaceTimeCoord<GeoCoord>> = Vec::with_capacity(lats.len());
        for i in 0..lats.len() {
            let coord =
                GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
            let st = SpaceTimeCoord::try_new(coord, times[i] as Real).map_err(kriging_err_to_js)?;
            coords.push(st);
        }
        let variogram = parse_spacetime_variogram(
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
        )?;
        let logits_real: Vec<Real> = logits.iter().map(|v| *v as Real).collect();
        let base_var: Vec<Real> = logit_observation_variance
            .iter()
            .map(|v| *v as Real)
            .collect();
        let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
        let hcfg = merge_hetero_with_optional_one_step_laplace(
            heteroskedastic_config_from_optional_stability_str(stability.as_deref())?,
            one_step_laplace_observation_variance,
        );
        let fit = SpaceTimeBinomialKrigingModel::from_precomputed_logits_with_logit_observation_variances(
            GeoMetric,
            coords,
            logits_real,
            variogram,
            base_var,
            hcfg,
            prior,
        )
        .map_err(kriging_err_to_js)?;
        Ok(Self { fit })
    }

    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.fit.notes).map_err(err_to_js)
    }

    /// `variogram` is a [`SpaceTimeVariogramParams`]-shaped object; `buildNotes` matches
    /// geographic binomial; optional `logitLooMsdr` from retained training counts.
    pub fn get_diagnostics(&self, _options: JsValue) -> Result<JsValue, JsValue> {
        let d = self.fit.diagnostics(GeoMetric).map_err(kriging_err_to_js)?;
        st_binomial_diagnostics_to_js(&d)
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.fit.model.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsBinomialPrediction {
            prevalence_median: pred.prevalence_median as f64,
            prevalence_mean: pred.prevalence_mean as f64,
            logit: pred.logit as f64,
            logit_variance: pred.logit_variance as f64,
            prevalence_variance: pred.prevalence_variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .fit
            .model
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .fit
            .model
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        let (pm, pmean, logits, lv, pv) = split_binomial_predictions(out);
        let result = Object::new();
        set_binomial_flat_array_fields(&result, &pm, &pmean, &logits, &lv, &pv)?;
        Ok(result.into())
    }

    pub fn leave_one_out(&self) -> Result<JsValue, JsValue> {
        let counts = self.fit.training_counts().ok_or_else(|| {
            coded_err(
                "leaveOneOut requires count data; build the model from successes/trials, not precomputed logits",
                "invalid_input",
            )
        })?;
        let coords = self.fit.model.coords();
        model_cv::spacetime_binomial_geo_cv(
            &coords,
            counts.successes(),
            counts.trials(),
            self.fit.model.variogram(),
            self.fit.notes.prior,
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        let counts = self.fit.training_counts().ok_or_else(|| {
            coded_err(
                "kFold requires count data; build the model from successes/trials, not precomputed logits",
                "invalid_input",
            )
        })?;
        let coords = self.fit.model.coords();
        model_cv::spacetime_binomial_geo_cv(
            &coords,
            counts.successes(),
            counts.trials(),
            self.fit.model.variogram(),
            self.fit.notes.prior,
            Some(k),
        )
    }
}

// ---------- Projected: Ordinary ----------

pub(crate) struct WasmSpaceTimeOrdinaryProjectedKriging {
    inner: SpaceTimeOrdinaryKrigingModel<ProjectedMetric>,
    cv_coords: Vec<SpaceTimeCoord<ProjectedCoord>>,
    cv_values: Vec<Real>,
    cv_anisotropy: Anisotropy2D,
}

impl WasmSpaceTimeOrdinaryProjectedKriging {
    pub fn from_arrays(
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
    ) -> Result<WasmSpaceTimeOrdinaryProjectedKriging, JsValue> {
        let dataset = build_projected_dataset(xs, ys, times, values)?;
        let metric = projected_metric(major_angle_deg, range_ratio)?;
        let anisotropy = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
            .map_err(kriging_err_to_js)?;
        let (cv_coords, cv_values, cv_anisotropy) = st_projected_cv_slices(&dataset, anisotropy);
        let variogram = parse_spacetime_variogram(
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
        )?;
        let inner = SpaceTimeOrdinaryKrigingModel::new(metric, dataset, variogram)
            .map_err(kriging_err_to_js)?;
        Ok(Self {
            inner,
            cv_coords,
            cv_values,
            cv_anisotropy,
        })
    }

    pub fn predict(&self, x: f64, y: f64, time: f64) -> Result<JsValue, JsValue> {
        let target =
            SpaceTimeCoord::try_new(ProjectedCoord::new(x as Real, y as Real), time as Real)
                .map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    pub fn predict_batch_arrays(
        &self,
        xs: &[f64],
        ys: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_projected_targets(xs, ys, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
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
        model_cv::spacetime_ordinary_projected_cv(
            self.cv_anisotropy,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            None,
        )
    }

    pub fn k_fold(&self, k: usize) -> Result<JsValue, JsValue> {
        model_cv::spacetime_ordinary_projected_cv(
            self.cv_anisotropy,
            &self.cv_coords,
            &self.cv_values,
            self.inner.variogram(),
            Some(k),
        )
    }
}

// ---------- Empirical + fitting helpers ----------

fn empirical_to_js(emp: &EmpiricalSpaceTimeVariogram) -> Result<JsValue, JsValue> {
    let result = Object::new();
    set_object_field(
        &result,
        "nSpatialBins",
        &JsValue::from_f64(emp.n_spatial_bins as f64),
    )?;
    set_object_field(
        &result,
        "nTemporalBins",
        &JsValue::from_f64(emp.n_temporal_bins as f64),
    )?;
    let spatial: Vec<f64> = emp.spatial_lags.iter().map(|v| *v as f64).collect();
    let temporal: Vec<f64> = emp.temporal_lags.iter().map(|v| *v as f64).collect();
    let semis: Vec<f64> = emp.semivariances.iter().map(|v| *v as f64).collect();
    let counts: Vec<f64> = emp.n_pairs.iter().map(|v| *v as f64).collect();
    set_object_field(
        &result,
        "spatialLags",
        &Float64Array::from(spatial.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "temporalLags",
        &Float64Array::from(temporal.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "semivariances",
        &Float64Array::from(semis.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "nPairs",
        &Float64Array::from(counts.as_slice()).into(),
    )?;
    Ok(result.into())
}

fn variogram_to_js(model: VariogramModel) -> Result<JsValue, JsValue> {
    let (nugget, sill, range) = model.params();
    let obj = Object::new();
    set_object_field(
        &obj,
        "variogramType",
        &JsValue::from_str(variogram_type_name(model.variogram_type())),
    )?;
    set_object_field(&obj, "nugget", &JsValue::from_f64(nugget as f64))?;
    set_object_field(&obj, "sill", &JsValue::from_f64(sill as f64))?;
    set_object_field(&obj, "range", &JsValue::from_f64(range as f64))?;
    if let Some(s) = model.shape() {
        set_object_field(&obj, "shape", &JsValue::from_f64(s as f64))?;
    }
    Ok(obj.into())
}

fn space_time_variogram_diagnostic_js(v: &SpaceTimeVariogram) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    let (family_name, k1, k2, k3) = match v {
        SpaceTimeVariogram::Separable { .. } => ("separable", 1.0_f64, 0.0_f64, 0.0_f64),
        SpaceTimeVariogram::ProductSum { k1, k2, k3, .. } => {
            ("productSum", *k1 as f64, *k2 as f64, *k3 as f64)
        }
    };
    set_object_field(&obj, "family", &JsValue::from_str(family_name))?;
    set_object_field(&obj, "spatial", &variogram_to_js(v.spatial())?)?;
    set_object_field(&obj, "temporal", &variogram_to_js(v.temporal())?)?;
    set_object_field(&obj, "k1", &JsValue::from_f64(k1))?;
    set_object_field(&obj, "k2", &JsValue::from_f64(k2))?;
    set_object_field(&obj, "k3", &JsValue::from_f64(k3))?;
    Ok(obj.into())
}

fn fit_to_js(fit: SpaceTimeFitResult) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    let (family_name, k1, k2, k3) = match fit.model {
        SpaceTimeVariogram::Separable { .. } => ("separable", 1.0 as Real, 0.0, 0.0),
        SpaceTimeVariogram::ProductSum { k1, k2, k3, .. } => ("productSum", k1, k2, k3),
    };
    set_object_field(&obj, "family", &JsValue::from_str(family_name))?;
    set_object_field(&obj, "spatial", &variogram_to_js(fit.model.spatial())?)?;
    set_object_field(&obj, "temporal", &variogram_to_js(fit.model.temporal())?)?;
    set_object_field(&obj, "k1", &JsValue::from_f64(k1 as f64))?;
    set_object_field(&obj, "k2", &JsValue::from_f64(k2 as f64))?;
    set_object_field(&obj, "k3", &JsValue::from_f64(k3 as f64))?;
    set_object_field(&obj, "residuals", &JsValue::from_f64(fit.residuals as f64))?;
    Ok(obj.into())
}

#[wasm_bindgen(js_name = wasmComputeEmpiricalSpaceTimeVariogram)]
pub fn wasm_compute_empirical_spacetime_variogram(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    max_spatial_distance: Option<f64>,
    max_temporal_lag: Option<f64>,
    n_spatial_bins: usize,
    n_temporal_bins: usize,
    estimator: &str,
) -> Result<JsValue, JsValue> {
    let dataset = build_geo_dataset(lats, lons, times, values)?;
    let n_spatial = std::num::NonZeroUsize::new(n_spatial_bins)
        .ok_or_else(|| coded_err("nSpatialBins must be > 0", "invalid_input"))?;
    let n_temporal = std::num::NonZeroUsize::new(n_temporal_bins)
        .ok_or_else(|| coded_err("nTemporalBins must be > 0", "invalid_input"))?;
    let estimator = parse_estimator(estimator)?;
    let max_s = match max_spatial_distance {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let max_t = match max_temporal_lag {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let config = SpaceTimeVariogramConfig {
        max_spatial_distance: max_s,
        max_temporal_lag: max_t,
        n_spatial_bins: n_spatial,
        n_temporal_bins: n_temporal,
        estimator,
    };
    let emp = compute_empirical_spacetime_variogram(&GeoMetric, &dataset, &config)
        .map_err(kriging_err_to_js)?;
    empirical_to_js(&emp)
}

#[wasm_bindgen(js_name = wasmFitSpaceTimeVariogram)]
pub fn wasm_fit_spacetime_variogram(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    max_spatial_distance: Option<f64>,
    max_temporal_lag: Option<f64>,
    n_spatial_bins: usize,
    n_temporal_bins: usize,
    estimator: &str,
    family: &str,
    spatial_model: &str,
    temporal_model: &str,
) -> Result<JsValue, JsValue> {
    let emp_js = wasm_compute_empirical_spacetime_variogram(
        lats,
        lons,
        times,
        values,
        max_spatial_distance,
        max_temporal_lag,
        n_spatial_bins,
        n_temporal_bins,
        estimator,
    )?;
    // Recompute the empirical variogram in Rust form for the actual fit (avoiding a JS
    // round-trip). This duplicates the loop above but keeps the JS API simple.
    let dataset = build_geo_dataset(lats, lons, times, values)?;
    let n_spatial = std::num::NonZeroUsize::new(n_spatial_bins)
        .ok_or_else(|| coded_err("nSpatialBins must be > 0", "invalid_input"))?;
    let n_temporal = std::num::NonZeroUsize::new(n_temporal_bins)
        .ok_or_else(|| coded_err("nTemporalBins must be > 0", "invalid_input"))?;
    let estimator_v = parse_estimator(estimator)?;
    let max_s = match max_spatial_distance {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let max_t = match max_temporal_lag {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let emp = compute_empirical_spacetime_variogram(
        &GeoMetric,
        &dataset,
        &SpaceTimeVariogramConfig {
            max_spatial_distance: max_s,
            max_temporal_lag: max_t,
            n_spatial_bins: n_spatial,
            n_temporal_bins: n_temporal,
            estimator: estimator_v,
        },
    )
    .map_err(kriging_err_to_js)?;
    let family = parse_family(family)?;
    let spatial_model = parse_variogram_type(spatial_model)?;
    let temporal_model = parse_variogram_type(temporal_model)?;
    let config = SpaceTimeFitConfig {
        family,
        spatial_model,
        temporal_model,
    };
    let fit = fit_spacetime_variogram(&emp, config).map_err(kriging_err_to_js)?;
    let result = Object::new();
    set_object_field(&result, "empirical", &emp_js)?;
    set_object_field(&result, "fit", &fit_to_js(fit)?)?;
    Ok(result.into())
}

fn parse_variogram_type(name: &str) -> Result<VariogramType, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "spherical" => VariogramType::Spherical,
        "exponential" => VariogramType::Exponential,
        "gaussian" => VariogramType::Gaussian,
        "cubic" => VariogramType::Cubic,
        "stable" => VariogramType::Stable,
        "matern" => VariogramType::Matern,
        "power" => VariogramType::Power,
        "holeeffect" | "hole_effect" | "hole-effect" => VariogramType::HoleEffect,
        _ => return Err(coded_err("unknown variogram type", "unknown_variogram")),
    })
}

// ---------------------------------------------------------------------------
// Space–time cross-validation
// ---------------------------------------------------------------------------

fn st_variogram_from_cv_options(opts: &UnifiedCvOptions) -> Result<SpaceTimeVariogram, JsValue> {
    let family = opts.space_time_family.as_deref().ok_or_else(|| {
        coded_err(
            "spaceTimeFamily is required for spacetime CV",
            "invalid_input",
        )
    })?;
    let spatial_type = opts
        .spatial_type
        .as_deref()
        .ok_or_else(|| coded_err("spatialType is required for spacetime CV", "invalid_input"))?;
    let temporal_type = opts
        .temporal_type
        .as_deref()
        .ok_or_else(|| coded_err("temporalType is required for spacetime CV", "invalid_input"))?;
    st_parse_spacetime_variogram_all(
        family,
        spatial_type,
        opts.spatial_nugget.unwrap_or(0.0),
        opts.spatial_sill.ok_or_else(|| {
            coded_err("spatialSill is required for spacetime CV", "invalid_input")
        })?,
        opts.spatial_range.ok_or_else(|| {
            coded_err("spatialRange is required for spacetime CV", "invalid_input")
        })?,
        opts.spatial_shape,
        temporal_type,
        opts.temporal_nugget.unwrap_or(0.0),
        opts.temporal_sill.ok_or_else(|| {
            coded_err("temporalSill is required for spacetime CV", "invalid_input")
        })?,
        opts.temporal_range.ok_or_else(|| {
            coded_err(
                "temporalRange is required for spacetime CV",
                "invalid_input",
            )
        })?,
        opts.temporal_shape,
        opts.k1,
        opts.k2,
        opts.k3,
    )
}

pub(super) fn run_spacetime_cv(opts: &UnifiedCvOptions) -> Result<JsValue, JsValue> {
    let coords = st_build_geo_coords(&opts.lats, &opts.lons, &opts.times)?;
    let vg = st_variogram_from_cv_options(opts)?;

    match opts.family.as_str() {
        "ordinary" => {
            if opts.values.len() != opts.lats.len() {
                return Err(coded_err(
                    "values must have the same length as lats/lons/times",
                    "mismatched_arrays",
                ));
            }
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let predictor = SpacetimeOrdinaryPredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values_real,
                variogram: vg,
            };
            st_run_cv_with_predictor(&predictor, opts.k)
        }
        "simple" => {
            if opts.values.len() != opts.lats.len() {
                return Err(coded_err(
                    "values must have the same length as lats/lons/times",
                    "mismatched_arrays",
                ));
            }
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let mean = opts.mean.ok_or_else(|| {
                coded_err("mean is required for simple kriging CV", "invalid_input")
            })?;
            let predictor = SpacetimeSimplePredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values_real,
                variogram: vg,
                mean: mean as Real,
            };
            st_run_cv_with_predictor(&predictor, opts.k)
        }
        "universal" => {
            if opts.values.len() != opts.lats.len() {
                return Err(coded_err(
                    "values must have the same length as lats/lons/times",
                    "mismatched_arrays",
                ));
            }
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let trend_str = opts.trend.as_deref().ok_or_else(|| {
                coded_err(
                    "trend is required for universal kriging CV",
                    "invalid_input",
                )
            })?;
            let trend = parse_universal_trend(trend_str)?;
            let predictor = SpacetimeUniversalPredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values_real,
                variogram: vg,
                trend,
            };
            st_run_cv_with_predictor(&predictor, opts.k)
        }
        "binomial" => {
            if opts.lats.len() != opts.successes.len() || opts.lats.len() != opts.trials.len() {
                return Err(coded_err(
                    "lats, lons, times, successes, and trials must have the same length",
                    "mismatched_arrays",
                ));
            }
            let prior = parse_binomial_prior(opts.prior_alpha, opts.prior_beta)?;
            let predictor = SpacetimeBinomialPredictor {
                metric: GeoMetric,
                coords: &coords,
                successes: &opts.successes,
                trials: &opts.trials,
                variogram: vg,
                prior,
            };
            st_run_binomial_cv_with_predictor(&predictor, opts.k)
        }
        other => Err(coded_err(
            &format!("unsupported spacetime CV family {other:?}"),
            "invalid_input",
        )),
    }
}

fn st_run_cv_with_predictor<P: crate::cv::KrigingPredictor<Residual = crate::cv::CvResidual>>(
    predictor: &P,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let residuals = match k {
        None => leave_one_out_cv(predictor).map_err(kriging_err_to_js)?,
        Some(k) => k_fold_cv(predictor, k).map_err(kriging_err_to_js)?,
    };
    cv_result_to_js(residuals)
}

fn st_run_binomial_cv_with_predictor<
    P: crate::cv::KrigingPredictor<Residual = crate::cv::BinomialCvResidual>,
>(
    predictor: &P,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let residuals = match k {
        None => leave_one_out_cv(predictor).map_err(kriging_err_to_js)?,
        Some(k) => k_fold_cv(predictor, k).map_err(kriging_err_to_js)?,
    };
    binomial_cv_result_to_js(residuals)
}

fn st_simulation_options(opts: &UnifiedSimulateOptions) -> crate::simulation::SimulationOptions {
    crate::simulation::SimulationOptions {
        seed: opts.seed,
        target_order: opts
            .target_order
            .clone()
            .map(|v| v.into_iter().map(|x| x as usize).collect()),
    }
}

fn st_target_order_usize(opts: &UnifiedSimulateOptions) -> Option<Vec<usize>> {
    opts.target_order
        .clone()
        .map(|v| v.into_iter().map(|x| x as usize).collect())
}

pub(super) fn run_spacetime_simulate(opts: &UnifiedSimulateOptions) -> Result<JsValue, JsValue> {
    if opts.conditioning_values.len() != opts.conditioning_lats.len() && opts.family != "binomial" {
        return Err(coded_err(
            "conditioningValues must match conditioning lats/lons/times length",
            "mismatched_arrays",
        ));
    }
    let cond_coords = st_build_geo_coords(
        &opts.conditioning_lats,
        &opts.conditioning_lons,
        &opts.conditioning_times,
    )?;
    let targets = st_build_geo_coords(&opts.target_lats, &opts.target_lons, &opts.target_times)?;
    let family = opts.space_time_family.as_deref().ok_or_else(|| {
        coded_err(
            "spaceTimeFamily is required for spacetime simulation",
            "invalid_input",
        )
    })?;
    let spatial_type = opts.spatial_type.as_deref().ok_or_else(|| {
        coded_err(
            "spatialType is required for spacetime simulation",
            "invalid_input",
        )
    })?;
    let temporal_type = opts.temporal_type.as_deref().ok_or_else(|| {
        coded_err(
            "temporalType is required for spacetime simulation",
            "invalid_input",
        )
    })?;
    let vg = st_parse_spacetime_variogram_all(
        family,
        spatial_type,
        opts.spatial_nugget.unwrap_or(0.0),
        opts.spatial_sill.ok_or_else(|| {
            coded_err(
                "spatialSill is required for spacetime simulation",
                "invalid_input",
            )
        })?,
        opts.spatial_range.ok_or_else(|| {
            coded_err(
                "spatialRange is required for spacetime simulation",
                "invalid_input",
            )
        })?,
        opts.spatial_shape,
        temporal_type,
        opts.temporal_nugget.unwrap_or(0.0),
        opts.temporal_sill.ok_or_else(|| {
            coded_err(
                "temporalSill is required for spacetime simulation",
                "invalid_input",
            )
        })?,
        opts.temporal_range.ok_or_else(|| {
            coded_err(
                "temporalRange is required for spacetime simulation",
                "invalid_input",
            )
        })?,
        opts.temporal_shape,
        opts.k1,
        opts.k2,
        opts.k3,
    )?;

    match opts.family.as_str() {
        "ordinary" => {
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let simulator =
                SpacetimeOrdinarySimulator::new(GeoMetric, &cond_coords, &cond_values, vg)
                    .map_err(kriging_err_to_js)?;
            if opts.is_many() {
                let samples = sequential_gaussian_simulate_many(
                    simulator,
                    &targets,
                    opts.realization_count() as usize,
                    opts.effective_base_seed(),
                    st_target_order_usize(opts),
                )
                .map_err(kriging_err_to_js)?;
                let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
                Ok(Float64Array::from(samples_f64.as_slice()).into())
            } else {
                let samples =
                    sequential_gaussian_simulate(simulator, &targets, st_simulation_options(opts))
                        .map_err(kriging_err_to_js)?;
                let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
                Ok(Float64Array::from(samples_f64.as_slice()).into())
            }
        }
        "simple" => {
            if opts.conditioning_values.len() != opts.conditioning_lats.len() {
                return Err(coded_err(
                    "conditioningValues must match conditioningLats/Lons/Times length",
                    "mismatched_arrays",
                ));
            }
            let mean = opts.mean.ok_or_else(|| {
                coded_err(
                    "mean is required for simple kriging simulation",
                    "invalid_input",
                )
            })?;
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let simulator = SpacetimeSimpleSimulator::new(
                GeoMetric,
                &cond_coords,
                &cond_values,
                vg,
                mean as Real,
            )
            .map_err(kriging_err_to_js)?;
            let samples =
                sequential_gaussian_simulate(simulator, &targets, st_simulation_options(opts))
                    .map_err(kriging_err_to_js)?;
            let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
            Ok(Float64Array::from(samples_f64.as_slice()).into())
        }
        "universal" => {
            if opts.conditioning_values.len() != opts.conditioning_lats.len() {
                return Err(coded_err(
                    "conditioningValues must match conditioningLats/Lons/Times length",
                    "mismatched_arrays",
                ));
            }
            let trend_str = opts.trend.as_deref().ok_or_else(|| {
                coded_err(
                    "trend is required for universal kriging simulation",
                    "invalid_input",
                )
            })?;
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let trend = parse_universal_trend(trend_str)?;
            let simulator =
                SpacetimeUniversalSimulator::new(GeoMetric, &cond_coords, &cond_values, vg, trend)
                    .map_err(kriging_err_to_js)?;
            let samples =
                sequential_gaussian_simulate(simulator, &targets, st_simulation_options(opts))
                    .map_err(kriging_err_to_js)?;
            let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
            Ok(Float64Array::from(samples_f64.as_slice()).into())
        }
        "binomial" => {
            if opts.conditioning_lats.len() != opts.conditioning_successes.len()
                || opts.conditioning_lats.len() != opts.conditioning_trials.len()
            {
                return Err(coded_err(
                    "conditioning lats/lons/times, successes, and trials must have the same length",
                    "mismatched_arrays",
                ));
            }
            let prior = parse_binomial_prior(opts.prior_alpha, opts.prior_beta)?;
            let simulator = SpacetimeBinomialSimulator::new(
                GeoMetric,
                &cond_coords,
                &opts.conditioning_successes,
                &opts.conditioning_trials,
                vg,
                prior,
            )
            .map_err(kriging_err_to_js)?;
            if opts.is_many() {
                let result = sequential_binomial_simulate_many(
                    simulator,
                    &targets,
                    opts.realization_count() as usize,
                    opts.effective_base_seed(),
                    st_target_order_usize(opts),
                )
                .map_err(kriging_err_to_js)?;
                binomial_many_simulation_to_js(result)
            } else {
                let result =
                    sequential_binomial_simulate(simulator, &targets, st_simulation_options(opts))
                        .map_err(kriging_err_to_js)?;
                binomial_simulation_to_js(result)
            }
        }
        other => Err(coded_err(
            &format!("unsupported spacetime simulation family {other:?}"),
            "invalid_input",
        )),
    }
}

fn st_build_geo_coords(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
) -> Result<Vec<SpaceTimeCoord<GeoCoord>>, JsValue> {
    if lats.len() != lons.len() || lats.len() != times.len() {
        return Err(coded_err(
            "lats, lons, and times must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let geo = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        out.push(SpaceTimeCoord::try_new(geo, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

fn st_parse_spacetime_variogram_all(
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
) -> Result<SpaceTimeVariogram, JsValue> {
    parse_spacetime_variogram(
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
    )
}
