use criterion::{Criterion, criterion_group, criterion_main};
use kriging_rs::spacetime::{
    GeoMetric, SpaceTimeCoord, SpaceTimeDataset, SpaceTimeOrdinaryKrigingModel, SpaceTimeVariogram,
    SpaceTimeVariogramConfig, compute_empirical_spacetime_variogram,
};
use kriging_rs::variogram::empirical::{VariogramConfig, compute_empirical_variogram};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::VariogramType;
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord, GeoDataset,
    OrdinaryKrigingModel, Real, VariogramModel,
};

fn build_model(point_count: usize) -> OrdinaryKrigingModel {
    let coords = (0..point_count)
        .map(|i| GeoCoord::try_new(35.0 + (i as Real) * 0.01, -120.0 + (i as Real) * 0.01).unwrap())
        .collect::<Vec<_>>();
    let values = (0..point_count)
        .map(|i| 10.0 + (i as Real) * 0.2)
        .collect::<Vec<_>>();
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    OrdinaryKrigingModel::new(
        dataset,
        VariogramModel::new(0.05, 4.0, 20.0, VariogramType::Exponential).unwrap(),
    )
    .expect("benchmark model should build")
}

fn build_observations(point_count: usize) -> Vec<BinomialObservation> {
    (0..point_count)
        .map(|i| {
            let p = 0.15 + ((i % 17) as Real) * 0.04;
            let trials = 100;
            let successes = (p.min(0.95) * trials as Real).round() as u32;
            BinomialObservation::new(
                GeoCoord::try_new(35.0 + (i as Real) * 0.01, -120.0 + (i as Real) * 0.01).unwrap(),
                successes,
                trials,
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn build_prediction_grid(count: usize, step: Real) -> Vec<GeoCoord> {
    (0..count)
        .map(|i| GeoCoord::try_new(35.0 + (i as Real) * step, -120.0 + (i as Real) * step).unwrap())
        .collect::<Vec<_>>()
}

/// Representative of a large **visual** web workload: ~400 conditioning stations and
/// 200×200 = 40_000 target cells. Models are pre-built; benches measure `predict_batch` only.
/// See `benches/BROWSER_BENCHMARKS.md`.
fn build_browser_binomial_400() -> (Vec<BinomialObservation>, VariogramModel) {
    let p_at = |i: usize| (0.12f32 + (i % 23) as f32 * 0.003).min(0.92) as Real;
    let mut obs = Vec::with_capacity(400);
    for row in 0..20u32 {
        for col in 0..20u32 {
            let i = (row * 20 + col) as usize;
            // Mixed denominators: every other site has few trials.
            let trials: u32 = if i.is_multiple_of(2) { 8 } else { 180 };
            let p = p_at(i);
            let successes = ((p * trials as Real).round() as u32).min(trials);
            let lat = 35.0 + (row as Real) * 0.02;
            let lon = -120.0 + (col as Real) * 0.02;
            obs.push(
                BinomialObservation::new(GeoCoord::try_new(lat, lon).unwrap(), successes, trials)
                    .unwrap(),
            );
        }
    }
    // Fixed, smooth exponential — matches typical fitted shapes without benching the fitter.
    let var = VariogramModel::new(0.04, 3.5, 0.4, VariogramType::Exponential).unwrap();
    (obs, var)
}

fn build_200x200_target_grid() -> Vec<GeoCoord> {
    let n = 200usize;
    let (south, west) = (35.0f32, -120.0f32);
    let (north, east) = (39.0f32, -116.0f32);
    let dlat = (north - south) / (n - 1) as f32;
    let dlon = (east - west) / (n - 1) as f32;
    let mut out = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            out.push(
                GeoCoord::try_new(
                    (south + dlat * r as f32) as Real,
                    (west + dlon * c as f32) as Real,
                )
                .unwrap(),
            );
        }
    }
    out
}

fn bench_binomial_browser_representative(c: &mut Criterion) {
    let (ref observations, variogram) = build_browser_binomial_400();
    let grid_40k = build_200x200_target_grid();
    assert_eq!(grid_40k.len(), 40_000);
    let prior = BinomialPrior::default();
    let m = BinomialKrigingModel::new_with_prior(observations.clone(), variogram, prior)
        .expect("binomial 400");
    c.bench_function("browser_binomial_pred_only_40k_cal_400st", |b| {
        b.iter(|| {
            let _ = m.predict_batch(&grid_40k);
        });
    });
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    {
        c.bench_function("browser_binomial_pred_only_40k_cal_400st_gpu", |b| {
            b.iter(|| {
                let _ = m.predict_batch_gpu_blocking(&grid_40k).expect("gpu");
            });
        });
    }
}

fn run_ordinary_pipeline(
    coords: &[GeoCoord],
    values: &[Real],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_type: VariogramType,
) {
    let dataset = GeoDataset::new(coords.to_vec(), values.to_vec()).expect("dataset");
    let empirical =
        compute_empirical_variogram(&dataset, variogram_config).expect("empirical variogram");
    let fit = fit_variogram(&empirical, variogram_type).expect("fit variogram");
    let model = OrdinaryKrigingModel::new(dataset, fit.model).expect("ordinary pipeline model");
    let _ = model
        .predict_batch(prediction_coords)
        .expect("ordinary predict");
}

#[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
fn run_ordinary_pipeline_gpu(
    coords: &[GeoCoord],
    values: &[Real],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_type: VariogramType,
) {
    let dataset = GeoDataset::new(coords.to_vec(), values.to_vec()).expect("dataset");
    let empirical =
        compute_empirical_variogram(&dataset, variogram_config).expect("empirical variogram");
    let fit = fit_variogram(&empirical, variogram_type).expect("fit variogram");
    let model = OrdinaryKrigingModel::new(dataset, fit.model).expect("ordinary pipeline model");
    let _ = model
        .predict_batch_gpu_blocking(prediction_coords)
        .expect("ordinary gpu predict");
}

fn run_binomial_pipeline(
    observations: &[BinomialObservation],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_type: VariogramType,
    prior: BinomialPrior,
) {
    let logits: Vec<_> = observations
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect();
    let obs_coords = observations.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let dataset = GeoDataset::new(obs_coords, logits).expect("dataset");
    let empirical = compute_empirical_variogram(&dataset, variogram_config).expect("empirical");
    let fit = fit_variogram(&empirical, variogram_type).expect("fit variogram");
    let model = BinomialKrigingModel::new_with_prior(observations.to_vec(), fit.model, prior)
        .expect("binomial pipeline model");
    let _ = model
        .predict_batch(prediction_coords)
        .expect("binomial predict");
}

#[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
fn run_binomial_pipeline_gpu(
    observations: &[BinomialObservation],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_type: VariogramType,
    prior: BinomialPrior,
) {
    let logits: Vec<_> = observations
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect();
    let obs_coords = observations.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let dataset = GeoDataset::new(obs_coords, logits).expect("dataset");
    let empirical = compute_empirical_variogram(&dataset, variogram_config).expect("empirical");
    let fit = fit_variogram(&empirical, variogram_type).expect("fit variogram");
    let model = BinomialKrigingModel::new_with_prior(observations.to_vec(), fit.model, prior)
        .expect("binomial pipeline model");
    let _ = model
        .predict_batch_gpu_blocking(prediction_coords)
        .expect("binomial gpu predict");
}

fn bench_model_build(c: &mut Criterion) {
    c.bench_function("ordinary_model_build_50pts", |b| {
        b.iter(|| {
            let _ = build_model(50);
        })
    });
    c.bench_function("ordinary_model_build_350pts", |b| {
        b.iter(|| {
            let _ = build_model(350);
        })
    });
    c.bench_function("ordinary_model_build_1000pts", |b| {
        b.iter(|| {
            let _ = build_model(1000);
        })
    });
}

fn bench_single_prediction(c: &mut Criterion) {
    let model = build_model(50);
    c.bench_function("ordinary_predict_single", |b| {
        b.iter(|| {
            let _ = model.predict(GeoCoord::try_new(35.25, -119.75).unwrap());
        })
    });

    let model_1000 = build_model(1000);
    c.bench_function("ordinary_predict_single_with_1000pts", |b| {
        b.iter(|| {
            let _ = model_1000.predict(GeoCoord::try_new(35.25, -119.75).unwrap());
        })
    });
}

fn bench_batch_prediction(c: &mut Criterion) {
    let small_model = build_model(50);
    let small_grid = (0..250)
        .map(|i| {
            GeoCoord::try_new(35.0 + (i as Real) * 0.002, -120.0 + (i as Real) * 0.002).unwrap()
        })
        .collect::<Vec<_>>();
    c.bench_function("ordinary_predict_batch_250", |b| {
        b.iter(|| {
            let _ = small_model.predict_batch(&small_grid);
        })
    });

    let model_350 = build_model(350);
    let grid_1296 = (0..1296)
        .map(|i| {
            GeoCoord::try_new(35.0 + (i as Real) * 0.001, -120.0 + (i as Real) * 0.001).unwrap()
        })
        .collect::<Vec<_>>();
    c.bench_function("ordinary_predict_batch_1296_with_350pts", |b| {
        b.iter(|| {
            let _ = model_350.predict_batch(&grid_1296);
        })
    });

    let model_1000 = build_model(1000);
    let grid_1000 = build_prediction_grid(1000, 0.0005);
    c.bench_function("ordinary_predict_batch_1000_with_1000pts", |b| {
        b.iter(|| {
            let _ = model_1000.predict_batch(&grid_1000);
        })
    });
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    c.bench_function("ordinary_predict_batch_1000_with_1000pts_gpu", |b| {
        b.iter(|| {
            let _ = model_1000.predict_batch_gpu_blocking(&grid_1000);
        })
    });
}

fn bench_binomial(c: &mut Criterion) {
    c.bench_function("binomial_model_build_350pts", |b| {
        b.iter(|| {
            let observations = build_observations(350);
            let _ = BinomialKrigingModel::new(
                observations,
                VariogramModel::new(0.05, 4.0, 20.0, VariogramType::Exponential).unwrap(),
            );
        })
    });
    c.bench_function("binomial_model_build_1000pts", |b| {
        b.iter(|| {
            let observations = build_observations(1000);
            let _ = BinomialKrigingModel::new(
                observations,
                VariogramModel::new(0.05, 4.0, 20.0, VariogramType::Exponential).unwrap(),
            );
        })
    });

    let observations = build_observations(1000);
    let model = BinomialKrigingModel::new(
        observations,
        VariogramModel::new(0.05, 4.0, 20.0, VariogramType::Exponential).unwrap(),
    )
    .expect("binomial benchmark model should build");
    let grid_1000 = build_prediction_grid(1000, 0.0005);
    c.bench_function("binomial_predict_batch_1000_with_1000pts", |b| {
        b.iter(|| {
            let _ = model.predict_batch(&grid_1000);
        })
    });
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    c.bench_function("binomial_predict_batch_1000_with_1000pts_gpu", |b| {
        b.iter(|| {
            let _ = model.predict_batch_gpu_blocking(&grid_1000);
        })
    });
}

fn bench_quick_phase_profile(c: &mut Criterion) {
    let variogram_config = VariogramConfig::default();
    let variogram_type = VariogramType::Exponential;

    let coords_1000 = build_prediction_grid(1000, 0.003);
    let values_1000 = (0..1000)
        .map(|i| ((i as Real) * 0.017).sin() + ((i as Real) * 0.011).cos())
        .collect::<Vec<_>>();
    let pred_1000 = build_prediction_grid(1000, 0.0025);
    let observations_1000 = build_observations(1000);

    let dataset_1000 = GeoDataset::new(coords_1000.clone(), values_1000.clone()).expect("dataset");
    c.bench_function("quick_ordinary_empirical_1000", |b| {
        b.iter(|| {
            let _ = compute_empirical_variogram(&dataset_1000, &variogram_config);
        })
    });
    c.bench_function("quick_ordinary_fit_1000", |b| {
        let empirical = compute_empirical_variogram(&dataset_1000, &variogram_config)
            .expect("empirical variogram");
        b.iter(|| {
            let _ = fit_variogram(&empirical, variogram_type).expect("fit");
        })
    });
    c.bench_function("quick_ordinary_model_build_1000", |b| {
        let empirical = compute_empirical_variogram(&dataset_1000, &variogram_config)
            .expect("empirical variogram");
        let fit = fit_variogram(&empirical, variogram_type).expect("fit");
        b.iter(|| {
            let dataset =
                GeoDataset::new(coords_1000.clone(), values_1000.clone()).expect("dataset");
            let _ = OrdinaryKrigingModel::new(dataset, fit.model);
        })
    });
    c.bench_function("quick_ordinary_predict_1000x1000", |b| {
        let empirical = compute_empirical_variogram(&dataset_1000, &variogram_config)
            .expect("empirical variogram");
        let fit = fit_variogram(&empirical, variogram_type).expect("fit");
        let model = OrdinaryKrigingModel::new(
            GeoDataset::new(coords_1000.clone(), values_1000.clone()).expect("dataset"),
            fit.model,
        )
        .expect("ordinary model");
        b.iter(|| {
            let _ = model.predict_batch(&pred_1000);
        })
    });
    c.bench_function("pipeline_ordinary_end_to_end_1000", |b| {
        b.iter(|| {
            run_ordinary_pipeline(
                &coords_1000,
                &values_1000,
                &pred_1000,
                &variogram_config,
                variogram_type,
            );
        })
    });
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    c.bench_function("pipeline_ordinary_end_to_end_1000_gpu", |b| {
        b.iter(|| {
            run_ordinary_pipeline_gpu(
                &coords_1000,
                &values_1000,
                &pred_1000,
                &variogram_config,
                variogram_type,
            );
        })
    });

    let obs_coords_1000: Vec<_> = observations_1000.iter().map(|o| o.coord()).collect();
    let logits_1000: Vec<_> = observations_1000
        .iter()
        .map(|o| o.smoothed_logit_with_prior(BinomialPrior::default()))
        .collect();
    let binomial_dataset_1000 = GeoDataset::new(obs_coords_1000, logits_1000).expect("dataset");
    c.bench_function("quick_binomial_empirical_1000", |b| {
        b.iter(|| {
            let _ = compute_empirical_variogram(&binomial_dataset_1000, &variogram_config);
        })
    });
    c.bench_function("quick_binomial_fit_1000", |b| {
        let empirical = compute_empirical_variogram(&binomial_dataset_1000, &variogram_config)
            .expect("empirical variogram");
        b.iter(|| {
            let _ = fit_variogram(&empirical, variogram_type).expect("fit");
        })
    });
    c.bench_function("quick_binomial_model_build_1000", |b| {
        let empirical = compute_empirical_variogram(&binomial_dataset_1000, &variogram_config)
            .expect("empirical variogram");
        let fit = fit_variogram(&empirical, variogram_type).expect("fit");
        b.iter(|| {
            let _ = BinomialKrigingModel::new_with_prior(
                observations_1000.clone(),
                fit.model,
                BinomialPrior::default(),
            );
        })
    });
    c.bench_function("quick_binomial_predict_1000x1000", |b| {
        let empirical = compute_empirical_variogram(&binomial_dataset_1000, &variogram_config)
            .expect("empirical variogram");
        let fit = fit_variogram(&empirical, variogram_type).expect("fit");
        let model = BinomialKrigingModel::new_with_prior(
            observations_1000.clone(),
            fit.model,
            BinomialPrior::default(),
        )
        .expect("binomial model");
        b.iter(|| {
            let _ = model.predict_batch(&pred_1000);
        })
    });
    c.bench_function("pipeline_binomial_end_to_end_1000", |b| {
        b.iter(|| {
            run_binomial_pipeline(
                &observations_1000,
                &pred_1000,
                &variogram_config,
                variogram_type,
                BinomialPrior::default(),
            );
        })
    });
    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    c.bench_function("pipeline_binomial_end_to_end_1000_gpu", |b| {
        b.iter(|| {
            run_binomial_pipeline_gpu(
                &observations_1000,
                &pred_1000,
                &variogram_config,
                variogram_type,
                BinomialPrior::default(),
            );
        })
    });
}

fn build_spacetime_dataset(spatial_side: usize, time_steps: usize) -> SpaceTimeDataset<GeoCoord> {
    let mut coords = Vec::with_capacity(spatial_side * spatial_side * time_steps);
    let mut values = Vec::with_capacity(spatial_side * spatial_side * time_steps);
    for i in 0..spatial_side {
        for j in 0..spatial_side {
            for k in 0..time_steps {
                let lat = 35.0 + (i as Real) * 0.05;
                let lon = -120.0 + (j as Real) * 0.05;
                let t = k as Real;
                let coord = SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon).unwrap(), t)
                    .expect("st coord");
                coords.push(coord);
                values.push(
                    10.0 + 2.0 * lat
                        + 1.5 * lon
                        + 0.5 * t
                        + ((i + j + k) as Real * 0.1).sin() * 0.25,
                );
            }
        }
    }
    SpaceTimeDataset::new(coords, values).expect("spacetime dataset")
}

fn build_spacetime_variogram() -> SpaceTimeVariogram {
    let spatial = VariogramModel::new(0.05, 2.0, 10.0, VariogramType::Exponential)
        .expect("spatial variogram");
    let temporal = VariogramModel::new(0.05, 1.0, 3.0, VariogramType::Exponential)
        .expect("temporal variogram");
    SpaceTimeVariogram::new_separable(spatial, temporal).expect("separable variogram")
}

fn build_spacetime_targets(count: usize) -> Vec<SpaceTimeCoord<GeoCoord>> {
    (0..count)
        .map(|i| {
            let lat = 35.1 + (i as Real) * 0.001;
            let lon = -119.9 + (i as Real) * 0.001;
            let t = ((i % 5) as Real) * 0.5;
            SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon).unwrap(), t).unwrap()
        })
        .collect()
}

fn bench_spacetime(c: &mut Criterion) {
    // ~10x10 spatial x 4 time steps = 400 points.
    let dataset_400 = build_spacetime_dataset(10, 4);
    let variogram = build_spacetime_variogram();

    c.bench_function("spacetime_ordinary_model_build_400pts", |b| {
        let variogram = build_spacetime_variogram();
        let dataset = dataset_400.clone();
        b.iter(|| {
            let _ = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset.clone(), variogram)
                .expect("st model");
        })
    });

    let model_400 = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset_400.clone(), variogram)
        .expect("st model");
    let single_target =
        SpaceTimeCoord::try_new(GeoCoord::try_new(35.25, -119.75).unwrap(), 1.0).unwrap();
    c.bench_function("spacetime_ordinary_predict_single_400pts", |b| {
        b.iter(|| {
            let _ = model_400.predict(single_target).expect("st predict");
        })
    });

    let targets_250 = build_spacetime_targets(250);
    c.bench_function("spacetime_ordinary_predict_batch_250_with_400pts", |b| {
        b.iter(|| {
            let _ = model_400
                .predict_batch(&targets_250)
                .expect("st predict batch");
        })
    });

    let variogram_config = SpaceTimeVariogramConfig::default();
    c.bench_function("spacetime_empirical_variogram_400pts", |b| {
        b.iter(|| {
            let _ =
                compute_empirical_spacetime_variogram(&GeoMetric, &dataset_400, &variogram_config)
                    .expect("st empirical");
        })
    });
}

criterion_group!(
    performance,
    bench_model_build,
    bench_single_prediction,
    bench_batch_prediction,
    bench_binomial,
    bench_binomial_browser_representative,
    bench_spacetime,
    bench_quick_phase_profile
);
criterion_main!(performance);
