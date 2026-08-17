//! Disease prevalence mapping cookbook (Rust edition).
//!
//! This example mirrors the TypeScript "Disease prevalence mapping cookbook"
//! from `npm/kriging-rs-wasm/README.md` and walks through the recommended
//! workflow for the package's primary use case:
//!
//!   1. Build a small set of `(lat, lon, successes, trials)` observations.
//!   2. Smooth the empirical proportions with a Beta prior, fit a variogram on
//!      the EB-smoothed logits.
//!   3. Cross-validate the fitted variogram (both logit and prevalence scales)
//!      to sanity-check calibration before producing maps.
//!   4. Build a reusable [`BinomialKrigingModel`] and predict the posterior
//!      prevalence over a small regular grid.
//!   5. Run a many-realization Sequential Gaussian Simulation (SGS) ensemble
//!      with [`conditional_simulate_many_binomial`], which amortizes the
//!      conditioning factorization across realizations.
//!   6. Roll the ensemble up to two illustrative "districts" with
//!      [`polygon_weighted_summary`] to obtain population-weighted posterior
//!      mean / variance / credible intervals.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example prevalence_mapping
//! ```

use std::num::NonZeroUsize;

use kriging_rs::aggregate::polygon_weighted_summary;
use kriging_rs::cv::{BinomialCvSummary, BinomialGeoPredictor, k_fold_cv};
use kriging_rs::simulation::sequential_binomial_simulate_many;
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, EmpiricalEstimator, GeoCoord,
    GeoDataset, Real, VariogramConfig, VariogramType, compute_empirical_variogram, logit_clamped,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------------
    // 1. Synthetic survey data: 12 villages with (successes / trials).
    //    A real workflow would load this from a CSV, GeoJSON, or DB query.
    // ---------------------------------------------------------------------
    let raw: &[(Real, Real, u32, u32)] = &[
        (40.71, -74.00, 25, 50),
        (40.72, -73.99, 35, 50),
        (40.70, -74.02, 20, 50),
        (40.74, -73.97, 40, 60),
        (40.69, -74.04, 12, 40),
        (40.73, -74.01, 28, 55),
        (40.715, -73.985, 30, 50),
        (40.705, -74.015, 18, 45),
        (40.725, -73.995, 33, 50),
        (40.735, -73.98, 38, 55),
        (40.7, -74.0, 22, 50),
        (40.745, -73.99, 41, 60),
    ];
    let coords: Vec<GeoCoord> = raw
        .iter()
        .map(|(lat, lon, _, _)| GeoCoord::try_new(*lat, *lon).unwrap())
        .collect();
    let successes: Vec<u32> = raw.iter().map(|(_, _, s, _)| *s).collect();
    let trials: Vec<u32> = raw.iter().map(|(_, _, _, n)| *n).collect();

    // `BinomialPrior::default()` in this crate is **Beta(1, 1)** (uniform on prevalence),
    // which matches the calibrated default binomial kriging path. For Jeffreys `Beta(½, ½)`
    // (stronger mass near 0/1 for small `n`), use e.g. `BinomialPrior::new(0.5, 0.5)?`.
    let prior = BinomialPrior::new(1.0, 1.0)?;

    // ---------------------------------------------------------------------
    // 2. Empirical variogram on EB-smoothed logits, then fit an exponential.
    //    `BinomialObservation::smoothed_logit_with_prior` returns the same
    //    logit transform that `BinomialKrigingModel` consumes internally.
    // ---------------------------------------------------------------------
    let smoothed_logits: Vec<Real> = raw
        .iter()
        .map(|(lat, lon, s, n)| {
            let coord = GeoCoord::try_new(*lat, *lon).unwrap();
            BinomialObservation::new(coord, *s, *n)
                .unwrap()
                .smoothed_logit_with_prior(prior)
        })
        .collect();
    let dataset = GeoDataset::new(coords.clone(), smoothed_logits.clone())?;
    let empirical = compute_empirical_variogram(
        &dataset,
        &VariogramConfig {
            max_distance: None,
            n_bins: NonZeroUsize::new(8).unwrap(),
            estimator: EmpiricalEstimator::CressieHawkins,
        },
    )?;
    let fit = fit_variogram(&empirical, VariogramType::Exponential)?;
    let variogram = fit.model;
    let (nugget, sill, range) = variogram.params();
    println!(
        "fitted variogram: type={:?} nugget={:.4} sill={:.4} range={:.4}",
        VariogramType::Exponential,
        nugget,
        sill,
        range,
    );

    // ---------------------------------------------------------------------
    // 3. K-fold CV on both scales. A logit-scale MSDR ≈ 1 indicates the
    //    nugget/sill split is calibrated to the held-out residuals.
    // ---------------------------------------------------------------------
    let residuals = k_fold_cv(
        &BinomialGeoPredictor {
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram,
            prior,
        },
        4,
    )?;
    let cv = BinomialCvSummary::from_residuals(&residuals);
    println!(
        "CV (k=4): n={} evaluated={} logit_rmse={:.4} prevalence_rmse={:.4} logit_msdr={:.3}",
        cv.n, cv.n_evaluated, cv.logit.rmse, cv.prevalence.rmse, cv.logit.msdr,
    );

    // ---------------------------------------------------------------------
    // 4. Build the reusable model and predict a small 5×5 grid.
    //    For larger grids prefer `interpolateBinomialToGrid` from the WASM
    //    bindings, or batch the targets manually here.
    // ---------------------------------------------------------------------
    let observations: Vec<BinomialObservation> = raw
        .iter()
        .map(|(lat, lon, s, n)| {
            BinomialObservation::new(GeoCoord::try_new(*lat, *lon).unwrap(), *s, *n).unwrap()
        })
        .collect();
    let model = BinomialKrigingModel::new_with_prior(observations, variogram, prior)?;

    let (west, south, east, north) = (-74.05 as Real, 40.69 as Real, -73.95 as Real, 40.75 as Real);
    let (x_cells, y_cells) = (5usize, 5usize);
    let dx = (east - west) / (x_cells as Real - 1.0);
    let dy = (north - south) / (y_cells as Real - 1.0);
    let mut targets: Vec<GeoCoord> = Vec::with_capacity(x_cells * y_cells);
    for j in 0..y_cells {
        for i in 0..x_cells {
            let lat = south + (j as Real) * dy;
            let lon = west + (i as Real) * dx;
            targets.push(GeoCoord::try_new(lat, lon)?);
        }
    }
    let mut grid_prevalence = vec![0.0 as Real; targets.len()];
    for (k, target) in targets.iter().enumerate() {
        let pred = model.predict(*target)?;
        grid_prevalence[k] = pred.prevalence_median;
    }
    let pred_min = grid_prevalence
        .iter()
        .copied()
        .fold(Real::INFINITY, Real::min);
    let pred_max = grid_prevalence
        .iter()
        .copied()
        .fold(Real::NEG_INFINITY, Real::max);
    println!(
        "point predictions on {}x{} grid: min={:.4} max={:.4}",
        x_cells, y_cells, pred_min, pred_max
    );

    // Sanity-check that the smoothed logits round-trip (the variogram was
    // fit on these): the sample mean of the smoothed logits should be in
    // the same neighborhood as the modeled grid mean on the logit scale.
    let mean_smoothed_logit =
        smoothed_logits.iter().copied().sum::<Real>() / smoothed_logits.len() as Real;
    let mean_grid_logit = grid_prevalence
        .iter()
        .copied()
        .map(logit_clamped)
        .sum::<Real>()
        / grid_prevalence.len() as Real;
    println!(
        "sanity: mean smoothed logit={:.3}  mean grid logit={:.3}",
        mean_smoothed_logit, mean_grid_logit
    );

    // ---------------------------------------------------------------------
    // 5. SGS ensemble (50 realizations) over the same grid.
    //    `_many_*` variants reuse the conditioning factorization across
    //    realizations and are the recommended path whenever you need any
    //    aggregation (district means, exceedance maps, ...).
    // ---------------------------------------------------------------------
    let n_realizations = 50usize;
    let ensemble = sequential_binomial_simulate_many(
        model.into_model().into_conditioner()?,
        &targets,
        n_realizations,
        /* base_seed = */ 42,
        /* target_order = */ None,
    )?;
    println!(
        "ensemble: realizations={} targets={} prevalence_buf_len={}",
        ensemble.n_realizations,
        ensemble.n_targets,
        ensemble.prevalence_samples.len(),
    );

    // ---------------------------------------------------------------------
    // 6. Polygon roll-ups: two illustrative "districts" each backed by a
    //    handful of grid cells with synthetic population weights. In
    //    practice these come from a population raster intersected with
    //    administrative boundaries.
    // ---------------------------------------------------------------------
    let district_a_indices: Vec<usize> = vec![0, 1, 2, 5, 6, 7]; // north-west corner
    let district_a_weights: Vec<Real> = vec![1.0, 2.0, 1.0, 2.0, 3.0, 2.0];
    let district_b_indices: Vec<usize> = vec![17, 18, 19, 22, 23, 24]; // south-east corner
    let district_b_weights: Vec<Real> = vec![1.0, 1.5, 1.0, 1.0, 2.0, 1.5];

    for (label, indices, weights) in [
        ("district A", &district_a_indices, &district_a_weights),
        ("district B", &district_b_indices, &district_b_weights),
    ] {
        let summary = polygon_weighted_summary(
            &ensemble.prevalence_samples,
            ensemble.n_realizations,
            ensemble.n_targets,
            indices,
            weights,
            /* quantile_probs = */ &[0.025, 0.5, 0.975],
        )?;
        let q = &summary.quantiles;
        let variance = summary.variance.unwrap_or(0.0);
        println!(
            "{:>10}: mean={:.4} sd={:.4} 95% CI=({:.4}, {:.4}) median={:.4} pop_weight={:.1}",
            label,
            summary.mean,
            variance.sqrt(),
            q[0].1,
            q[2].1,
            q[1].1,
            summary.total_weight,
        );
    }

    Ok(())
}
