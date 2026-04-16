//! Minimal example: fit a separable space-time variogram and predict at a new
//! `(lat, lon, time)` location using ordinary space-time kriging.
//!
//! Run with `cargo run --example spacetime_kriging`.

use std::num::NonZeroUsize;

use kriging_rs::spacetime::{
    GeoMetric, SpaceTimeCoord, SpaceTimeDataset, SpaceTimeFitConfig, SpaceTimeOrdinaryKrigingModel,
    SpaceTimeVariogramConfig, SpaceTimeVariogramType, compute_empirical_spacetime_variogram,
    fit_spacetime_variogram,
};
use kriging_rs::variogram::empirical::EmpiricalEstimator;
use kriging_rs::variogram::models::VariogramType;
use kriging_rs::{GeoCoord, Real};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic 4 x 4 space-time grid with a smooth trend in both axes.
    let mut coords = Vec::with_capacity(16);
    let mut values = Vec::with_capacity(16);
    for i in 0..4 {
        for t in 0..4 {
            let lat = 37.70 + 0.01 * i as Real;
            let lon = -122.40 + 0.01 * i as Real;
            let time = t as Real;
            coords.push(SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon)?, time)?);
            values.push(15.0 + 0.5 * i as Real + 0.3 * time);
        }
    }
    let dataset = SpaceTimeDataset::new(coords, values)?;

    // Fit a separable exponential × exponential space-time variogram.
    let n_spatial = NonZeroUsize::new(4).unwrap();
    let n_temporal = NonZeroUsize::new(4).unwrap();
    let emp = compute_empirical_spacetime_variogram(
        &GeoMetric,
        &dataset,
        &SpaceTimeVariogramConfig {
            max_spatial_distance: None,
            max_temporal_lag: None,
            n_spatial_bins: n_spatial,
            n_temporal_bins: n_temporal,
            estimator: EmpiricalEstimator::Classical,
        },
    )?;
    let fit = fit_spacetime_variogram(
        &emp,
        SpaceTimeFitConfig {
            family: SpaceTimeVariogramType::Separable,
            spatial_model: VariogramType::Exponential,
            temporal_model: VariogramType::Exponential,
        },
    )?;
    println!(
        "fit residuals: {:.4}  (family: separable, spatial+temporal = exponential)",
        fit.residuals
    );

    // Build an ordinary ST kriging model and predict at an interior location.
    let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, fit.model)?;
    let target = SpaceTimeCoord::try_new(GeoCoord::try_new(37.715, -122.395)?, 1.5 as Real)?;
    let pred = model.predict(target)?;
    println!(
        "predicted value: {:.3}   kriging variance: {:.4}",
        pred.value, pred.variance
    );
    Ok(())
}
