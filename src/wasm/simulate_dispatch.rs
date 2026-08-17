//! Unified sequential Gaussian simulation dispatch for the WASM seam.

use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use crate::Real;
use crate::geo_dataset::GeoDataset;
use crate::kriging::binomial::{
    BinomialKrigingModel, build_binomial_observations_dropping_zero_trials,
};
use crate::kriging::conditioner::KrigingConditioner;
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::kriging::simple::SimpleKrigingModel;
use crate::kriging::universal::UniversalKrigingModel;
use crate::projected::{
    Anisotropy2D, BinomialProjectedKrigingModel, ProjectedBinomialObservation, ProjectedCoord,
    ProjectedDataset, ProjectedKrigingModel,
};
use crate::simulation::{
    SimulationOptions, sequential_binomial_simulate, sequential_binomial_simulate_many,
    sequential_gaussian_simulate, sequential_gaussian_simulate_many,
};

use super::simulate_options::UnifiedSimulateOptions;
use super::spacetime::run_spacetime_simulate;
use super::{
    binomial_many_simulation_to_js, binomial_simulation_to_js, coded_err, err_to_js,
    kriging_err_to_js, parse_binomial_prior, parse_trend, parse_variogram, to_coords,
};

fn invalid_input(msg: impl Into<String>) -> JsValue {
    coded_err(&msg.into(), "invalid_input")
}

fn anisotropy_from_options(opts: &UnifiedSimulateOptions) -> Result<Anisotropy2D, JsValue> {
    let angle = opts.major_angle_deg.unwrap_or(0.0);
    let ratio = opts.range_ratio.unwrap_or(1.0);
    Anisotropy2D::new(angle as Real, ratio as Real).map_err(kriging_err_to_js)
}

fn simulation_options(opts: &UnifiedSimulateOptions) -> SimulationOptions {
    SimulationOptions {
        seed: opts.seed,
        target_order: opts
            .target_order
            .clone()
            .map(|v| v.into_iter().map(|x| x as usize).collect()),
    }
}

fn target_order_usize(opts: &UnifiedSimulateOptions) -> Option<Vec<usize>> {
    opts.target_order
        .clone()
        .map(|v| v.into_iter().map(|x| x as usize).collect())
}

fn samples_to_js(samples: Vec<Real>) -> JsValue {
    let samples_f64: Vec<f64> = samples.into_iter().map(|v| v as f64).collect();
    Float64Array::from(samples_f64.as_slice()).into()
}

pub(super) fn run_continuous_simulate<S>(
    conditioner: KrigingConditioner<S>,
    targets: &[S],
    opts: &UnifiedSimulateOptions,
) -> Result<JsValue, JsValue>
where
    S: Copy,
{
    let samples = if opts.is_many() {
        sequential_gaussian_simulate_many(
            conditioner,
            targets,
            opts.realization_count() as usize,
            opts.effective_base_seed(),
            target_order_usize(opts),
        )
    } else {
        sequential_gaussian_simulate(conditioner, targets, simulation_options(opts))
    }
    .map_err(kriging_err_to_js)?;
    Ok(samples_to_js(samples))
}

fn run_simulate_2d(opts: &UnifiedSimulateOptions) -> Result<JsValue, JsValue> {
    let geometry = opts.geometry.as_str();
    let family = opts.family.as_str();
    let variogram = opts
        .variogram
        .as_ref()
        .ok_or_else(|| invalid_input("variogram is required for 2-D simulation"))?;
    let variogram = parse_variogram(
        &variogram.variogram_type,
        variogram.nugget,
        variogram.sill,
        variogram.range,
        variogram.shape,
    )?;

    match (geometry, family) {
        ("geo", "ordinary") => {
            if opts.conditioning_values.len() != opts.conditioning_lats.len() {
                return Err(coded_err(
                    "conditioningValues must match conditioningLats/Lons length",
                    "mismatched_arrays",
                ));
            }
            let cond_coords = to_coords(&opts.conditioning_lats, &opts.conditioning_lons)?;
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
            let conditioner = OrdinaryKrigingModel::new(
                GeoDataset::new(cond_coords, cond_values).map_err(kriging_err_to_js)?,
                variogram,
            )
            .and_then(OrdinaryKrigingModel::into_conditioner)
            .map_err(kriging_err_to_js)?;
            run_continuous_simulate(conditioner, &targets, opts)
        }
        ("geo", "simple") => {
            let mean = opts
                .mean
                .ok_or_else(|| invalid_input("mean is required for simple kriging simulation"))?;
            if opts.conditioning_values.len() != opts.conditioning_lats.len() {
                return Err(coded_err(
                    "conditioningValues must match conditioningLats/Lons length",
                    "mismatched_arrays",
                ));
            }
            let cond_coords = to_coords(&opts.conditioning_lats, &opts.conditioning_lons)?;
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
            let conditioner = SimpleKrigingModel::new(
                GeoDataset::new(cond_coords, cond_values).map_err(kriging_err_to_js)?,
                variogram,
                mean as Real,
            )
            .and_then(SimpleKrigingModel::into_conditioner)
            .map_err(kriging_err_to_js)?;
            run_continuous_simulate(conditioner, &targets, opts)
        }
        ("geo", "universal") => {
            let trend_str = opts.trend.as_deref().ok_or_else(|| {
                invalid_input("trend is required for universal kriging simulation")
            })?;
            if opts.conditioning_values.len() != opts.conditioning_lats.len() {
                return Err(coded_err(
                    "conditioningValues must match conditioningLats/Lons length",
                    "mismatched_arrays",
                ));
            }
            let cond_coords = to_coords(&opts.conditioning_lats, &opts.conditioning_lons)?;
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
            let trend = parse_trend(trend_str)?;
            let conditioner = UniversalKrigingModel::new(
                GeoDataset::new(cond_coords, cond_values).map_err(kriging_err_to_js)?,
                variogram,
                trend,
            )
            .and_then(UniversalKrigingModel::into_conditioner)
            .map_err(kriging_err_to_js)?;
            run_continuous_simulate(conditioner, &targets, opts)
        }
        ("geo", "binomial") => {
            if opts.conditioning_lats.len() != opts.conditioning_lons.len()
                || opts.conditioning_lats.len() != opts.conditioning_successes.len()
                || opts.conditioning_lats.len() != opts.conditioning_trials.len()
            {
                return Err(coded_err(
                    "conditioning arrays (lats, lons, successes, trials) must have the same length",
                    "mismatched_arrays",
                ));
            }
            let cond_coords = to_coords(&opts.conditioning_lats, &opts.conditioning_lons)?;
            let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
            let prior = parse_binomial_prior(opts.prior_alpha, opts.prior_beta)?;
            let (observations, _) = build_binomial_observations_dropping_zero_trials(
                cond_coords,
                &opts.conditioning_successes,
                &opts.conditioning_trials,
            )
            .map_err(kriging_err_to_js)?;
            let conditioner = BinomialKrigingModel::new_with_prior(observations, variogram, prior)
                .map(|fit| fit.into_model())
                .and_then(BinomialKrigingModel::into_conditioner)
                .map_err(kriging_err_to_js)?;
            if opts.is_many() {
                let result = sequential_binomial_simulate_many(
                    conditioner,
                    &targets,
                    opts.realization_count() as usize,
                    opts.effective_base_seed(),
                    target_order_usize(opts),
                )
                .map_err(kriging_err_to_js)?;
                binomial_many_simulation_to_js(result)
            } else {
                let result =
                    sequential_binomial_simulate(conditioner, &targets, simulation_options(opts))
                        .map_err(kriging_err_to_js)?;
                binomial_simulation_to_js(result)
            }
        }
        ("projected", "ordinary") => {
            if opts.conditioning_xs.len() != opts.conditioning_ys.len()
                || opts.conditioning_xs.len() != opts.conditioning_values.len()
            {
                return Err(coded_err(
                    "conditioningXs, conditioningYs and conditioningValues must have the same length",
                    "mismatched_arrays",
                ));
            }
            if opts.target_xs.len() != opts.target_ys.len() {
                return Err(coded_err(
                    "targetXs and targetYs must have the same length",
                    "mismatched_arrays",
                ));
            }
            let cond_coords: Vec<ProjectedCoord> = opts
                .conditioning_xs
                .iter()
                .zip(opts.conditioning_ys.iter())
                .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
                .collect();
            let cond_values: Vec<Real> = opts
                .conditioning_values
                .iter()
                .map(|v| *v as Real)
                .collect();
            let targets: Vec<ProjectedCoord> = opts
                .target_xs
                .iter()
                .zip(opts.target_ys.iter())
                .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
                .collect();
            let anisotropy = anisotropy_from_options(opts)?;
            let conditioner = ProjectedKrigingModel::new(
                ProjectedDataset::new(cond_coords, cond_values).map_err(kriging_err_to_js)?,
                variogram,
                anisotropy,
            )
            .and_then(ProjectedKrigingModel::into_conditioner)
            .map_err(kriging_err_to_js)?;
            run_continuous_simulate(conditioner, &targets, opts)
        }
        ("projected", "binomial") => {
            if opts.conditioning_xs.len() != opts.conditioning_ys.len()
                || opts.conditioning_xs.len() != opts.conditioning_successes.len()
                || opts.conditioning_xs.len() != opts.conditioning_trials.len()
            {
                return Err(coded_err(
                    "conditioningXs, conditioningYs, successes and trials must have the same length",
                    "mismatched_arrays",
                ));
            }
            if opts.target_xs.len() != opts.target_ys.len() {
                return Err(coded_err(
                    "targetXs and targetYs must have the same length",
                    "mismatched_arrays",
                ));
            }
            let cond_coords: Vec<ProjectedCoord> = opts
                .conditioning_xs
                .iter()
                .zip(opts.conditioning_ys.iter())
                .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
                .collect();
            let targets: Vec<ProjectedCoord> = opts
                .target_xs
                .iter()
                .zip(opts.target_ys.iter())
                .map(|(&x, &y)| ProjectedCoord::new(x as Real, y as Real))
                .collect();
            let anisotropy = anisotropy_from_options(opts)?;
            let prior = parse_binomial_prior(opts.prior_alpha, opts.prior_beta)?;
            let observations = cond_coords
                .into_iter()
                .zip(opts.conditioning_successes.iter().copied())
                .zip(opts.conditioning_trials.iter().copied())
                .filter(|(_, trials)| *trials > 0)
                .map(|((coord, successes), trials)| {
                    ProjectedBinomialObservation::new(coord, successes, trials)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(kriging_err_to_js)?;
            let conditioner = BinomialProjectedKrigingModel::new_with_prior(
                observations,
                variogram,
                anisotropy,
                prior,
                Default::default(),
            )
            .map(|fit| fit.into_model())
            .and_then(BinomialProjectedKrigingModel::into_conditioner)
            .map_err(kriging_err_to_js)?;
            if opts.is_many() {
                let result = sequential_binomial_simulate_many(
                    conditioner,
                    &targets,
                    opts.realization_count() as usize,
                    opts.effective_base_seed(),
                    target_order_usize(opts),
                )
                .map_err(kriging_err_to_js)?;
                binomial_many_simulation_to_js(result)
            } else {
                let result =
                    sequential_binomial_simulate(conditioner, &targets, simulation_options(opts))
                        .map_err(kriging_err_to_js)?;
                binomial_simulation_to_js(result)
            }
        }
        (g, f) => Err(invalid_input(format!(
            "unsupported geometry/family pair for 2-D simulation: {g}/{f}"
        ))),
    }
}

pub(crate) fn dispatch_simulate(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: UnifiedSimulateOptions =
        serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    match opts.geometry.as_str() {
        "spacetime" => run_spacetime_simulate(&opts),
        "geo" | "projected" => run_simulate_2d(&opts),
        other => Err(invalid_input(format!(
            "geometry must be 'geo', 'projected', or 'spacetime' (got {other:?})"
        ))),
    }
}

/// Unified sequential Gaussian simulation. Pass `{ geometry, family, variogram, conditioning…,
/// target… }`. Set `nRealizations > 1` for ensemble output (uses `baseSeed` or `seed`).
#[wasm_bindgen(js_name = simulate)]
pub fn wasm_simulate(options: JsValue) -> Result<JsValue, JsValue> {
    dispatch_simulate(options)
}
