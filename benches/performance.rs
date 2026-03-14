use criterion::{Criterion, criterion_group, criterion_main};
use kriging_rs::variogram::empirical::{VariogramConfig, compute_empirical_variogram};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::VariogramType;
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord, OrdinaryKrigingModel, Real,
    VariogramModel,
};

fn build_model(point_count: usize) -> OrdinaryKrigingModel {
    let coords = (0..point_count)
        .map(|i| GeoCoord {
            lat: 35.0 + (i as Real) * 0.01,
            lon: -120.0 + (i as Real) * 0.01,
        })
        .collect::<Vec<_>>();
    let values = (0..point_count)
        .map(|i| 10.0 + (i as Real) * 0.2)
        .collect::<Vec<_>>();
    OrdinaryKrigingModel::new(
        coords,
        values,
        VariogramModel::Exponential {
            nugget: 0.05,
            sill: 4.0,
            range: 20.0,
        },
    )
    .expect("benchmark model should build")
}

fn build_observations(point_count: usize) -> Vec<BinomialObservation> {
    (0..point_count)
        .map(|i| {
            let p = 0.15 + ((i % 17) as Real) * 0.04;
            let trials = 100;
            let successes = (p.min(0.95) * trials as Real).round() as u32;
            BinomialObservation {
                coord: GeoCoord {
                    lat: 35.0 + (i as Real) * 0.01,
                    lon: -120.0 + (i as Real) * 0.01,
                },
                successes,
                trials,
            }
        })
        .collect::<Vec<_>>()
}

fn build_prediction_grid(count: usize, step: Real) -> Vec<GeoCoord> {
    (0..count)
        .map(|i| GeoCoord {
            lat: 35.0 + (i as Real) * step,
            lon: -120.0 + (i as Real) * step,
        })
        .collect::<Vec<_>>()
}

fn pick_best_fit(
    empirical: &kriging_rs::variogram::empirical::EmpiricalVariogram,
    variogram_types: &[VariogramType],
) -> kriging_rs::FitResult {
    let mut iter = variogram_types.iter().copied();
    let first = iter.next().expect("at least one variogram type");
    let mut best = fit_variogram(empirical, first).expect("fit first variogram");
    for vt in iter {
        let candidate = fit_variogram(empirical, vt).expect("fit candidate variogram");
        if candidate.residuals < best.residuals {
            best = candidate;
        }
    }
    best
}

fn run_ordinary_pipeline(
    coords: &[GeoCoord],
    values: &[Real],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_types: &[VariogramType],
) {
    let empirical =
        compute_empirical_variogram(coords, values, variogram_config).expect("empirical variogram");
    let best = pick_best_fit(&empirical, variogram_types);
    let model = OrdinaryKrigingModel::new(coords.to_vec(), values.to_vec(), best.model)
        .expect("ordinary pipeline model");
    let _ = model.predict_batch(prediction_coords).expect("ordinary predict");
}

#[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
fn run_ordinary_pipeline_gpu(
    coords: &[GeoCoord],
    values: &[Real],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_types: &[VariogramType],
) {
    let empirical =
        compute_empirical_variogram(coords, values, variogram_config).expect("empirical variogram");
    let best = pick_best_fit(&empirical, variogram_types);
    let model = OrdinaryKrigingModel::new(coords.to_vec(), values.to_vec(), best.model)
        .expect("ordinary pipeline model");
    let _ = model
        .predict_batch_gpu_blocking(prediction_coords)
        .expect("ordinary gpu predict");
}

fn run_binomial_pipeline(
    observations: &[BinomialObservation],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_types: &[VariogramType],
    prior: BinomialPrior,
) {
    let logits = observations
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect::<Result<Vec<_>, _>>()
        .expect("binomial logits");
    let obs_coords = observations.iter().map(|o| o.coord).collect::<Vec<_>>();
    let empirical =
        compute_empirical_variogram(&obs_coords, &logits, variogram_config).expect("empirical");
    let best = pick_best_fit(&empirical, variogram_types);
    let model = BinomialKrigingModel::new_with_prior(observations.to_vec(), best.model, prior)
        .expect("binomial pipeline model");
    let _ = model.predict_batch(prediction_coords).expect("binomial predict");
}

#[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
fn run_binomial_pipeline_gpu(
    observations: &[BinomialObservation],
    prediction_coords: &[GeoCoord],
    variogram_config: &VariogramConfig,
    variogram_types: &[VariogramType],
    prior: BinomialPrior,
) {
    let logits = observations
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect::<Result<Vec<_>, _>>()
        .expect("binomial logits");
    let obs_coords = observations.iter().map(|o| o.coord).collect::<Vec<_>>();
    let empirical =
        compute_empirical_variogram(&obs_coords, &logits, variogram_config).expect("empirical");
    let best = pick_best_fit(&empirical, variogram_types);
    let model = BinomialKrigingModel::new_with_prior(observations.to_vec(), best.model, prior)
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
            let _ = model.predict(GeoCoord {
                lat: 35.25,
                lon: -119.75,
            });
        })
    });

    let model_1000 = build_model(1000);
    c.bench_function("ordinary_predict_single_with_1000pts", |b| {
        b.iter(|| {
            let _ = model_1000.predict(GeoCoord {
                lat: 35.25,
                lon: -119.75,
            });
        })
    });
}

fn bench_batch_prediction(c: &mut Criterion) {
    let small_model = build_model(50);
    let small_grid = (0..250)
        .map(|i| GeoCoord {
            lat: 35.0 + (i as Real) * 0.002,
            lon: -120.0 + (i as Real) * 0.002,
        })
        .collect::<Vec<_>>();
    c.bench_function("ordinary_predict_batch_250", |b| {
        b.iter(|| {
            let _ = small_model.predict_batch(&small_grid);
        })
    });

    let model_350 = build_model(350);
    let grid_1296 = (0..1296)
        .map(|i| GeoCoord {
            lat: 35.0 + (i as Real) * 0.001,
            lon: -120.0 + (i as Real) * 0.001,
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
                VariogramModel::Exponential {
                    nugget: 0.05,
                    sill: 4.0,
                    range: 20.0,
                },
            );
        })
    });
    c.bench_function("binomial_model_build_1000pts", |b| {
        b.iter(|| {
            let observations = build_observations(1000);
            let _ = BinomialKrigingModel::new(
                observations,
                VariogramModel::Exponential {
                    nugget: 0.05,
                    sill: 4.0,
                    range: 20.0,
                },
            );
        })
    });

    let observations = build_observations(1000);
    let model = BinomialKrigingModel::new(
        observations,
        VariogramModel::Exponential {
            nugget: 0.05,
            sill: 4.0,
            range: 20.0,
        },
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
    let variogram_types = [
        VariogramType::Spherical,
        VariogramType::Exponential,
        VariogramType::Gaussian,
    ];

    let coords_1000 = build_prediction_grid(1000, 0.003);
    let values_1000 = (0..1000)
        .map(|i| ((i as Real) * 0.017).sin() + ((i as Real) * 0.011).cos())
        .collect::<Vec<_>>();
    let pred_1000 = build_prediction_grid(1000, 0.0025);
    let observations_1000 = build_observations(1000);

    c.bench_function("quick_ordinary_empirical_1000", |b| {
        b.iter(|| {
            let _ = compute_empirical_variogram(&coords_1000, &values_1000, &variogram_config);
        })
    });
    c.bench_function("quick_ordinary_fit_1000", |b| {
        let empirical = compute_empirical_variogram(&coords_1000, &values_1000, &variogram_config)
            .expect("empirical variogram");
        b.iter(|| {
            let _ = pick_best_fit(&empirical, &variogram_types);
        })
    });
    c.bench_function("quick_ordinary_model_build_1000", |b| {
        let empirical = compute_empirical_variogram(&coords_1000, &values_1000, &variogram_config)
            .expect("empirical variogram");
        let best = pick_best_fit(&empirical, &variogram_types);
        b.iter(|| {
            let _ = OrdinaryKrigingModel::new(coords_1000.clone(), values_1000.clone(), best.model);
        })
    });
    c.bench_function("quick_ordinary_predict_1000x1000", |b| {
        let empirical = compute_empirical_variogram(&coords_1000, &values_1000, &variogram_config)
            .expect("empirical variogram");
        let best = pick_best_fit(&empirical, &variogram_types);
        let model = OrdinaryKrigingModel::new(coords_1000.clone(), values_1000.clone(), best.model)
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
                &variogram_types,
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
                &variogram_types,
            );
        })
    });

    c.bench_function("quick_binomial_empirical_1000", |b| {
        let logits = observations_1000
            .iter()
            .map(|o| o.smoothed_logit_with_prior(BinomialPrior::default()))
            .collect::<Result<Vec<_>, _>>()
            .expect("binomial logits");
        let obs_coords = observations_1000
            .iter()
            .map(|o| o.coord)
            .collect::<Vec<_>>();
        b.iter(|| {
            let _ = compute_empirical_variogram(&obs_coords, &logits, &variogram_config);
        })
    });
    c.bench_function("quick_binomial_fit_1000", |b| {
        let logits = observations_1000
            .iter()
            .map(|o| o.smoothed_logit_with_prior(BinomialPrior::default()))
            .collect::<Result<Vec<_>, _>>()
            .expect("binomial logits");
        let obs_coords = observations_1000
            .iter()
            .map(|o| o.coord)
            .collect::<Vec<_>>();
        let empirical = compute_empirical_variogram(&obs_coords, &logits, &variogram_config)
            .expect("empirical variogram");
        b.iter(|| {
            let _ = pick_best_fit(&empirical, &variogram_types);
        })
    });
    c.bench_function("quick_binomial_model_build_1000", |b| {
        let logits = observations_1000
            .iter()
            .map(|o| o.smoothed_logit_with_prior(BinomialPrior::default()))
            .collect::<Result<Vec<_>, _>>()
            .expect("binomial logits");
        let obs_coords = observations_1000
            .iter()
            .map(|o| o.coord)
            .collect::<Vec<_>>();
        let empirical = compute_empirical_variogram(&obs_coords, &logits, &variogram_config)
            .expect("empirical variogram");
        let best = pick_best_fit(&empirical, &variogram_types);
        b.iter(|| {
            let _ = BinomialKrigingModel::new_with_prior(
                observations_1000.clone(),
                best.model,
                BinomialPrior::default(),
            );
        })
    });
    c.bench_function("quick_binomial_predict_1000x1000", |b| {
        let logits = observations_1000
            .iter()
            .map(|o| o.smoothed_logit_with_prior(BinomialPrior::default()))
            .collect::<Result<Vec<_>, _>>()
            .expect("binomial logits");
        let obs_coords = observations_1000
            .iter()
            .map(|o| o.coord)
            .collect::<Vec<_>>();
        let empirical = compute_empirical_variogram(&obs_coords, &logits, &variogram_config)
            .expect("empirical variogram");
        let best = pick_best_fit(&empirical, &variogram_types);
        let model = BinomialKrigingModel::new_with_prior(
            observations_1000.clone(),
            best.model,
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
                &variogram_types,
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
                &variogram_types,
                BinomialPrior::default(),
            );
        })
    });
}

criterion_group!(
    performance,
    bench_model_build,
    bench_single_prediction,
    bench_batch_prediction,
    bench_binomial,
    bench_quick_phase_profile
);
criterion_main!(performance);
