use kriging_rs::variogram::empirical::{
    PositiveReal, VariogramConfig, compute_empirical_variogram,
};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::{VariogramModel, VariogramType};
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord, GeoDataset,
    OrdinaryKrigingModel,
};
use std::num::NonZeroUsize;

fn coord(lat: f32, lon: f32) -> GeoCoord {
    GeoCoord::try_new(lat, lon).unwrap()
}

fn fit_variogram_for_type(
    dataset: &GeoDataset,
    config: &VariogramConfig,
    variogram_type: VariogramType,
) -> VariogramModel {
    let empirical = compute_empirical_variogram(dataset, config).expect("empirical variogram");
    fit_variogram(&empirical, variogram_type)
        .expect("fit variogram")
        .model
}

#[test]
fn ordinary_kriging_predicts_midpoint_between_linear_samples() {
    let coords = vec![coord(0.0, 0.0), coord(0.0, 1.0), coord(0.0, 2.0)];
    let values = vec![10.0, 20.0, 30.0];
    let variogram = VariogramModel::new(0.01, 100.0, 500.0, VariogramType::Exponential).unwrap();
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model should build");
    let pred = model
        .predict(coord(0.0, 1.0))
        .expect("prediction should succeed");
    assert!((pred.value - 20.0).abs() < 1e-4);
}

#[test]
fn ordinary_pipeline_with_variogram_fit_runs_end_to_end() {
    let coords = vec![coord(0.0, 0.0), coord(0.5, 0.5), coord(1.0, 1.0)];
    let values = vec![2.0, 3.0, 4.0];
    let predictions = vec![coord(0.75, 0.75)];
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    let model = fit_variogram_for_type(
        &dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let ordinary = OrdinaryKrigingModel::new(dataset, model).expect("ordinary model should build");
    let out = ordinary
        .predict_batch(&predictions)
        .expect("ordinary batch predict");
    assert_eq!(out.len(), 1);
    assert!(out[0].value.is_finite());
    assert!(out[0].variance.is_finite());
}

#[test]
fn binomial_pipeline_with_variogram_fit_runs_end_to_end() {
    let obs = vec![
        BinomialObservation::new(coord(0.0, 0.0), 3, 10).unwrap(),
        BinomialObservation::new(coord(0.0, 1.0), 6, 10).unwrap(),
        BinomialObservation::new(coord(1.0, 0.5), 8, 10).unwrap(),
    ];
    let coords = obs.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let logits = obs
        .iter()
        .map(BinomialObservation::smoothed_logit)
        .collect::<Vec<_>>();
    let dataset = GeoDataset::new(coords, logits).expect("dataset");
    let model = fit_variogram_for_type(
        &dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let binomial = BinomialKrigingModel::new(obs, model).expect("binomial model should build");
    let out = binomial
        .predict_batch(&[coord(0.5, 0.5)])
        .expect("binomial batch predict");
    assert_eq!(out.len(), 1);
    assert!((0.0..=1.0).contains(&out[0].prevalence));
}

#[test]
fn explicit_variogram_config_accepts_custom_settings() {
    let coords = vec![
        coord(0.0, 0.0),
        coord(0.5, 0.5),
        coord(1.0, 1.0),
        coord(1.0, 0.0),
    ];
    let values = vec![2.0, 3.0, 4.0, 3.5];
    let predictions = vec![coord(0.75, 0.75)];
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    let model = fit_variogram_for_type(
        &dataset,
        &VariogramConfig {
            max_distance: Some(PositiveReal::try_new(200.0).unwrap()),
            n_bins: NonZeroUsize::new(4).unwrap(),
        },
        VariogramType::Gaussian,
    );
    let ordinary = OrdinaryKrigingModel::new(dataset, model).expect("ordinary model should build");
    let out = ordinary
        .predict_batch(&predictions)
        .expect("ordinary batch predict");
    assert_eq!(out.len(), 1);
    assert!(out[0].value.is_finite());
    assert!(out[0].variance.is_finite());
}

#[test]
fn binomial_prior_changes_prevalence_with_fitted_model() {
    let obs = vec![
        BinomialObservation::new(coord(0.0, 0.0), 0, 5).unwrap(),
        BinomialObservation::new(coord(0.0, 1.0), 2, 5).unwrap(),
        BinomialObservation::new(coord(1.0, 0.0), 5, 5).unwrap(),
    ];
    let pred_coord = [coord(0.25, 0.25)];
    let coords = obs.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let default_prior = BinomialPrior::default();
    let default_logits = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(default_prior))
        .collect::<Vec<_>>();
    let default_dataset = GeoDataset::new(coords.clone(), default_logits).expect("dataset");
    let default_model = fit_variogram_for_type(
        &default_dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let out_default =
        BinomialKrigingModel::new_with_prior(obs.clone(), default_model, default_prior)
            .expect("default prior model")
            .predict_batch(&pred_coord)
            .expect("default prior prediction");

    let heavy_prior = BinomialPrior::new(5.0, 5.0).unwrap();
    let heavy_logits = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(heavy_prior))
        .collect::<Vec<_>>();
    let heavy_dataset = GeoDataset::new(coords, heavy_logits).expect("dataset");
    let heavy_model = fit_variogram_for_type(
        &heavy_dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let out_heavy = BinomialKrigingModel::new_with_prior(obs, heavy_model, heavy_prior)
        .expect("heavy prior model")
        .predict_batch(&pred_coord)
        .expect("heavy prior prediction");
    assert_eq!(out_default.len(), 1);
    assert_eq!(out_heavy.len(), 1);
    assert!((out_default[0].prevalence - out_heavy[0].prevalence).abs() > 1e-6);
}

#[test]
fn binomial_predict_batch_matches_repeated_predict() {
    let obs = vec![
        BinomialObservation::new(coord(0.0, 0.0), 1, 5).unwrap(),
        BinomialObservation::new(coord(0.0, 1.0), 2, 5).unwrap(),
        BinomialObservation::new(coord(1.0, 0.0), 4, 5).unwrap(),
        BinomialObservation::new(coord(1.0, 1.0), 5, 5).unwrap(),
    ];
    let model = BinomialKrigingModel::new(
        obs,
        VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential).unwrap(),
    )
    .expect("binomial model");
    let coords = vec![coord(0.2, 0.2), coord(0.4, 0.7), coord(0.8, 0.3)];
    let batch = model.predict_batch(&coords).expect("batch predict");
    let singles = coords
        .iter()
        .map(|coord| model.predict(*coord).expect("single predict"))
        .collect::<Vec<_>>();
    assert_eq!(batch.len(), singles.len());
    for (batch_pred, single_pred) in batch.iter().zip(singles.iter()) {
        assert!((batch_pred.prevalence - single_pred.prevalence).abs() < 1e-12);
        assert!((batch_pred.logit_value - single_pred.logit_value).abs() < 1e-12);
        assert!((batch_pred.variance - single_pred.variance).abs() < 1e-12);
    }
}
