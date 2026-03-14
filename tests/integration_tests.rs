use kriging_rs::variogram::empirical::{VariogramConfig, compute_empirical_variogram};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::{VariogramModel, VariogramType};
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord, OrdinaryKrigingModel,
};

fn fit_best_variogram(
    coords: &[GeoCoord],
    values: &[f32],
    config: &VariogramConfig,
    variogram_types: &[VariogramType],
) -> VariogramModel {
    let empirical = compute_empirical_variogram(coords, values, config).expect("empirical variogram");
    let mut types = variogram_types.iter().copied();
    let first = types.next().expect("at least one variogram type");
    let mut best = fit_variogram(&empirical, first).expect("fit first variogram");
    for model_type in types {
        let candidate = fit_variogram(&empirical, model_type).expect("fit candidate variogram");
        if candidate.residuals < best.residuals {
            best = candidate;
        }
    }
    best.model
}

#[test]
fn ordinary_kriging_predicts_midpoint_between_linear_samples() {
    let coords = vec![
        GeoCoord { lat: 0.0, lon: 0.0 },
        GeoCoord { lat: 0.0, lon: 1.0 },
        GeoCoord { lat: 0.0, lon: 2.0 },
    ];
    let values = vec![10.0, 20.0, 30.0];
    let variogram = VariogramModel::Exponential {
        nugget: 0.01,
        sill: 100.0,
        range: 500.0,
    };
    let model = OrdinaryKrigingModel::new(coords, values, variogram).expect("model should build");
    let pred = model
        .predict(GeoCoord { lat: 0.0, lon: 1.0 })
        .expect("prediction should succeed");
    assert!((pred.value - 20.0).abs() < 1e-4);
}

#[test]
fn ordinary_pipeline_with_variogram_fit_runs_end_to_end() {
    let coords = vec![
        GeoCoord { lat: 0.0, lon: 0.0 },
        GeoCoord { lat: 0.5, lon: 0.5 },
        GeoCoord { lat: 1.0, lon: 1.0 },
    ];
    let values = vec![2.0, 3.0, 4.0];
    let predictions = vec![GeoCoord {
        lat: 0.75,
        lon: 0.75,
    }];
    let model = fit_best_variogram(&coords, &values, &VariogramConfig::default(), &[VariogramType::Spherical, VariogramType::Exponential, VariogramType::Gaussian]);
    let ordinary = OrdinaryKrigingModel::new(coords, values, model).expect("ordinary model should build");
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
        BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 0.0 },
            successes: 3,
            trials: 10,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 1.0 },
            successes: 6,
            trials: 10,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 1.0, lon: 0.5 },
            successes: 8,
            trials: 10,
        },
    ];
    let coords = obs.iter().map(|o| o.coord).collect::<Vec<_>>();
    let logits = obs
        .iter()
        .map(BinomialObservation::smoothed_logit)
        .collect::<Result<Vec<_>, _>>()
        .expect("smoothed logits");
    let model = fit_best_variogram(&coords, &logits, &VariogramConfig::default(), &[VariogramType::Spherical, VariogramType::Exponential, VariogramType::Gaussian]);
    let binomial = BinomialKrigingModel::new(obs, model).expect("binomial model should build");
    let out = binomial
        .predict_batch(&[GeoCoord { lat: 0.5, lon: 0.5 }])
        .expect("binomial batch predict");
    assert_eq!(out.len(), 1);
    assert!((0.0..=1.0).contains(&out[0].prevalence));
}

#[test]
fn explicit_variogram_config_accepts_custom_settings() {
    let coords = vec![
        GeoCoord { lat: 0.0, lon: 0.0 },
        GeoCoord { lat: 0.5, lon: 0.5 },
        GeoCoord { lat: 1.0, lon: 1.0 },
        GeoCoord { lat: 1.0, lon: 0.0 },
    ];
    let values = vec![2.0, 3.0, 4.0, 3.5];
    let predictions = vec![GeoCoord {
        lat: 0.75,
        lon: 0.75,
    }];
    let model = fit_best_variogram(
        &coords,
        &values,
        &VariogramConfig {
            max_distance: Some(200.0),
            n_bins: 4,
        },
        &[VariogramType::Gaussian],
    );
    let ordinary = OrdinaryKrigingModel::new(coords, values, model).expect("ordinary model should build");
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
        BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 0.0 },
            successes: 0,
            trials: 5,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 1.0 },
            successes: 2,
            trials: 5,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 1.0, lon: 0.0 },
            successes: 5,
            trials: 5,
        },
    ];
    let pred_coord = [GeoCoord {
        lat: 0.25,
        lon: 0.25,
    }];
    let coords = obs.iter().map(|o| o.coord).collect::<Vec<_>>();
    let default_prior = BinomialPrior::default();
    let default_logits = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(default_prior))
        .collect::<Result<Vec<_>, _>>()
        .expect("default smoothed logits");
    let default_model = fit_best_variogram(
        &coords,
        &default_logits,
        &VariogramConfig::default(),
        &[VariogramType::Exponential],
    );
    let out_default = BinomialKrigingModel::new_with_prior(obs.clone(), default_model, default_prior)
        .expect("default prior model")
        .predict_batch(&pred_coord)
        .expect("default prior prediction");

    let heavy_prior = BinomialPrior {
        alpha: 5.0,
        beta: 5.0,
    };
    let heavy_logits = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(heavy_prior))
        .collect::<Result<Vec<_>, _>>()
        .expect("heavy smoothed logits");
    let heavy_model = fit_best_variogram(
        &coords,
        &heavy_logits,
        &VariogramConfig::default(),
        &[VariogramType::Exponential],
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
        BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 0.0 },
            successes: 1,
            trials: 5,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 1.0 },
            successes: 2,
            trials: 5,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 1.0, lon: 0.0 },
            successes: 4,
            trials: 5,
        },
        BinomialObservation {
            coord: GeoCoord { lat: 1.0, lon: 1.0 },
            successes: 5,
            trials: 5,
        },
    ];
    let model = BinomialKrigingModel::new(
        obs,
        VariogramModel::Exponential {
            nugget: 0.01,
            sill: 2.0,
            range: 300.0,
        },
    )
    .expect("binomial model");
    let coords = vec![
        GeoCoord { lat: 0.2, lon: 0.2 },
        GeoCoord { lat: 0.4, lon: 0.7 },
        GeoCoord { lat: 0.8, lon: 0.3 },
    ];
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
