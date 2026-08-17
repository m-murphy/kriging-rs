//! Unified cross-validation dispatch for the WASM seam.
//!
//! Replaces the named `leaveOneOut*` / `kFold*` exports with one `cv(options)` entry point
//! keyed by `geometry` + `family`. See ADR-0002.

use wasm_bindgen::prelude::*;

use crate::Real;
use crate::cv::{
    BinomialGeoPredictor, BinomialProjectedPredictor, OrdinaryGeoPredictor,
    ProjectedOrdinaryPredictor, SimpleGeoPredictor, UniversalGeoPredictor, k_fold_cv,
    leave_one_out_cv,
};
use crate::projected::{Anisotropy2D, ProjectedCoord};

use super::cv_options::UnifiedCvOptions;
use super::spacetime::run_spacetime_cv;
use super::{
    binomial_cv_result_to_js, coded_err, cv_result_to_js, err_to_js, kriging_err_to_js,
    parse_binomial_prior, parse_trend, parse_variogram, to_coords,
};

fn invalid_input(msg: impl Into<String>) -> JsValue {
    coded_err(&msg.into(), "invalid_input")
}

fn run_cv_2d(opts: &UnifiedCvOptions) -> Result<JsValue, JsValue> {
    let geometry = opts.geometry.as_str();
    let family = opts.family.as_str();
    let variogram = opts
        .variogram
        .as_ref()
        .ok_or_else(|| invalid_input("variogram is required for 2-D CV"))?;
    let variogram = parse_variogram(
        &variogram.variogram_type,
        variogram.nugget,
        variogram.sill,
        variogram.range,
        variogram.shape,
    )?;

    match (geometry, family) {
        ("geo", "ordinary") => {
            if opts.values.len() != opts.lats.len() {
                return Err(coded_err(
                    "values must have the same length as lats/lons",
                    "mismatched_arrays",
                ));
            }
            let coords = to_coords(&opts.lats, &opts.lons)?;
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let predictor = OrdinaryGeoPredictor {
                coords: &coords,
                values: &values_real,
                variogram,
            };
            run_cv_with_predictor(&predictor, opts.k)
        }
        ("geo", "simple") => {
            let mean = opts
                .mean
                .ok_or_else(|| invalid_input("mean is required for simple kriging CV"))?;
            if opts.values.len() != opts.lats.len() {
                return Err(coded_err(
                    "values must have the same length as lats/lons",
                    "mismatched_arrays",
                ));
            }
            let coords = to_coords(&opts.lats, &opts.lons)?;
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let predictor = SimpleGeoPredictor {
                coords: &coords,
                values: &values_real,
                variogram,
                mean: mean as Real,
            };
            run_cv_with_predictor(&predictor, opts.k)
        }
        ("geo", "universal") => {
            let trend_str = opts
                .trend
                .as_deref()
                .ok_or_else(|| invalid_input("trend is required for universal kriging CV"))?;
            if opts.values.len() != opts.lats.len() {
                return Err(coded_err(
                    "values must have the same length as lats/lons",
                    "mismatched_arrays",
                ));
            }
            let coords = to_coords(&opts.lats, &opts.lons)?;
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let trend = parse_trend(trend_str)?;
            let predictor = UniversalGeoPredictor {
                coords: &coords,
                values: &values_real,
                variogram,
                trend,
            };
            run_cv_with_predictor(&predictor, opts.k)
        }
        ("geo", "binomial") => {
            if opts.lats.len() != opts.lons.len()
                || opts.lats.len() != opts.successes.len()
                || opts.lats.len() != opts.trials.len()
            {
                return Err(coded_err(
                    "lats, lons, successes, and trials must have the same length",
                    "mismatched_arrays",
                ));
            }
            let coords = to_coords(&opts.lats, &opts.lons)?;
            let prior = parse_binomial_prior(opts.prior_alpha, opts.prior_beta)?;
            let predictor = BinomialGeoPredictor {
                coords: &coords,
                successes: &opts.successes,
                trials: &opts.trials,
                variogram,
                prior,
            };
            run_binomial_cv_with_predictor(&predictor, opts.k)
        }
        ("projected", "ordinary") => {
            if opts.xs.len() != opts.ys.len() || opts.xs.len() != opts.values.len() {
                return Err(coded_err(
                    "xs, ys and values must have the same length",
                    "mismatched_arrays",
                ));
            }
            let coords: Vec<ProjectedCoord> = opts
                .xs
                .iter()
                .zip(opts.ys.iter())
                .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
                .collect();
            let values_real: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
            let anisotropy = anisotropy_from_options(opts)?;
            let predictor = ProjectedOrdinaryPredictor {
                coords: &coords,
                values: &values_real,
                variogram,
                anisotropy,
            };
            run_cv_with_predictor(&predictor, opts.k)
        }
        ("projected", "binomial") => {
            if opts.xs.len() != opts.ys.len()
                || opts.xs.len() != opts.successes.len()
                || opts.xs.len() != opts.trials.len()
            {
                return Err(coded_err(
                    "xs, ys, successes and trials must have the same length",
                    "mismatched_arrays",
                ));
            }
            let coords: Vec<ProjectedCoord> = opts
                .xs
                .iter()
                .zip(opts.ys.iter())
                .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
                .collect();
            let anisotropy = anisotropy_from_options(opts)?;
            let prior = parse_binomial_prior(opts.prior_alpha, opts.prior_beta)?;
            let predictor = BinomialProjectedPredictor {
                coords: &coords,
                successes: &opts.successes,
                trials: &opts.trials,
                variogram,
                anisotropy,
                prior,
            };
            run_binomial_cv_with_predictor(&predictor, opts.k)
        }
        (g, f) => Err(invalid_input(format!(
            "unsupported geometry/family pair for 2-D CV: {g}/{f}"
        ))),
    }
}

fn anisotropy_from_options(opts: &UnifiedCvOptions) -> Result<Anisotropy2D, JsValue> {
    let angle = opts.major_angle_deg.unwrap_or(0.0);
    let ratio = opts.range_ratio.unwrap_or(1.0);
    Anisotropy2D::new(angle as Real, ratio as Real).map_err(kriging_err_to_js)
}

fn run_cv_with_predictor<P: crate::cv::KrigingPredictor<Residual = crate::cv::CvResidual>>(
    predictor: &P,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let residuals = match k {
        None => leave_one_out_cv(predictor).map_err(kriging_err_to_js)?,
        Some(k) => k_fold_cv(predictor, k).map_err(kriging_err_to_js)?,
    };
    cv_result_to_js(residuals)
}

fn run_binomial_cv_with_predictor<
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

pub(crate) fn dispatch_cv(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: UnifiedCvOptions = serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    match opts.geometry.as_str() {
        "spacetime" => run_spacetime_cv(&opts),
        "geo" | "projected" => run_cv_2d(&opts),
        other => Err(invalid_input(format!(
            "geometry must be 'geo', 'projected', or 'spacetime' (got {other:?})"
        ))),
    }
}

/// Unified cross-validation. Pass `{ geometry, family, variogram, ... }`; omit `k` for
/// leave-one-out, set `k` for k-fold (deterministic round-robin).
#[wasm_bindgen(js_name = cv)]
pub fn wasm_cv(options: JsValue) -> Result<JsValue, JsValue> {
    dispatch_cv(options)
}

#[cfg(test)]
mod tests {
    use super::super::cv_options::UnifiedCvOptions;

    #[test]
    fn unified_cv_options_has_geometry_and_family() {
        let opts = UnifiedCvOptions {
            geometry: "geo".to_string(),
            family: "ordinary".to_string(),
            k: None,
            lats: vec![0.0, 1.0],
            lons: vec![0.0, 1.0],
            xs: vec![],
            ys: vec![],
            values: vec![1.0, 2.0],
            successes: vec![],
            trials: vec![],
            times: vec![],
            mean: None,
            trend: None,
            major_angle_deg: None,
            range_ratio: None,
            variogram: Some(super::super::cv_options::VariogramParams {
                variogram_type: "spherical".to_string(),
                nugget: 0.0,
                sill: 1.0,
                range: 10.0,
                shape: None,
            }),
            prior_alpha: None,
            prior_beta: None,
            space_time_family: None,
            spatial_type: None,
            spatial_nugget: None,
            spatial_sill: None,
            spatial_range: None,
            spatial_shape: None,
            temporal_type: None,
            temporal_nugget: None,
            temporal_sill: None,
            temporal_range: None,
            temporal_shape: None,
            k1: None,
            k2: None,
            k3: None,
        };
        assert_eq!(opts.geometry, "geo");
        assert_eq!(opts.family, "ordinary");
    }
}
