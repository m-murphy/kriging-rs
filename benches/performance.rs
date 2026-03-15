use criterion::{Criterion, criterion_group, criterion_main};
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

criterion_group!(
    performance,
    bench_model_build,
    bench_single_prediction,
    bench_batch_prediction,
    bench_binomial,
    bench_quick_phase_profile
);
criterion_main!(performance);
