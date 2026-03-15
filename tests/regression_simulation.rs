use kriging_rs::GeoDataset;
use kriging_rs::variogram::empirical::{VariogramConfig, compute_empirical_variogram};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::VariogramType;
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord, OrdinaryKrigingModel, Real,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::num::NonZeroUsize;

const SEEDS: [u64; 4] = [7, 17, 29, 53];
const TRAIN_POINTS: usize = 36;
const TEST_POINTS: usize = 18;
const ORDINARY_MAE_MAX: Real = 0.09;
const ORDINARY_RMSE_MAX: Real = 0.14;
const BINOMIAL_MAE_MAX: Real = 0.07;

#[derive(Clone, Copy)]
struct SamplePoint {
    coord: GeoCoord,
    x: Real,
    y: Real,
}

fn logistic(x: Real) -> Real {
    1.0 / (1.0 + (-x).exp())
}

fn to_coord(x: Real, y: Real) -> GeoCoord {
    GeoCoord::try_new(35.0 + x, -120.0 + y).unwrap()
}

fn ordinary_field(x: Real, y: Real) -> Real {
    (6.0 * x).sin() + 0.7 * (5.0 * y).cos() + 0.2 * x * y
}

fn binomial_logit_field(x: Real, y: Real) -> Real {
    -0.3 + 1.2 * (std::f32::consts::TAU * x).sin() * (std::f32::consts::TAU * y).cos()
}

fn sample_points(rng: &mut StdRng, count: usize) -> Vec<SamplePoint> {
    (0..count)
        .map(|_| {
            let x = rng.random::<f32>();
            let y = rng.random::<f32>();
            SamplePoint {
                coord: to_coord(x, y),
                x,
                y,
            }
        })
        .collect()
}

fn binomial_draws(rng: &mut StdRng, trials: u32, p: Real) -> u32 {
    (0..trials).map(|_| (rng.random::<f32>() < p) as u32).sum()
}

fn sample_grid_points(rows: usize, cols: usize) -> Vec<SamplePoint> {
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let x = row as Real / (rows - 1) as Real;
        for col in 0..cols {
            let y = col as Real / (cols - 1) as Real;
            out.push(SamplePoint {
                coord: to_coord(x, y),
                x,
                y,
            });
        }
    }
    out
}

fn fit_variogram_for_type(
    dataset: &GeoDataset,
    config: &VariogramConfig,
    variogram_type: VariogramType,
) -> kriging_rs::VariogramModel {
    let empirical = compute_empirical_variogram(dataset, config).expect("empirical");
    fit_variogram(&empirical, variogram_type)
        .expect("fit variogram")
        .model
}

#[test]
fn ordinary_pipeline_meets_regression_error_budget() {
    let mut all_abs_errors = Vec::new();
    let mut all_squared_errors = Vec::new();

    for seed in SEEDS {
        let mut rng = StdRng::seed_from_u64(seed);
        let train = sample_points(&mut rng, TRAIN_POINTS);
        let test = sample_points(&mut rng, TEST_POINTS);

        let train_coords: Vec<_> = train.iter().map(|p| p.coord).collect();
        let train_values: Vec<_> = train.iter().map(|p| ordinary_field(p.x, p.y)).collect();
        let test_coords: Vec<_> = test.iter().map(|p| p.coord).collect();
        let test_true: Vec<_> = test.iter().map(|p| ordinary_field(p.x, p.y)).collect();

        let train_dataset = GeoDataset::new(train_coords, train_values).expect("dataset");
        let best_model = fit_variogram_for_type(
            &train_dataset,
            &VariogramConfig::default(),
            VariogramType::Exponential,
        );
        let ordinary =
            OrdinaryKrigingModel::new(train_dataset, best_model).expect("ordinary model");
        let preds = ordinary
            .predict_batch(&test_coords)
            .expect("ordinary batch predict");
        assert_eq!(preds.len(), TEST_POINTS);

        for (pred, truth) in preds.iter().zip(test_true.iter()) {
            assert!(pred.value.is_finite());
            assert!(pred.variance.is_finite());
            let err = pred.value - truth;
            all_abs_errors.push(err.abs());
            all_squared_errors.push(err * err);
        }
    }

    let mae = all_abs_errors.iter().sum::<Real>() / all_abs_errors.len() as Real;
    let rmse = (all_squared_errors.iter().sum::<Real>() / all_squared_errors.len() as Real).sqrt();

    assert!(mae < ORDINARY_MAE_MAX, "mae={mae}");
    assert!(rmse < ORDINARY_RMSE_MAX, "rmse={rmse}");
}

#[test]
fn binomial_pipeline_meets_regression_error_budget() {
    let mut all_abs_errors = Vec::new();

    for seed in SEEDS {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let train = sample_points(&mut rng, TRAIN_POINTS);
        let test = sample_points(&mut rng, TEST_POINTS);

        let train_obs = train
            .iter()
            .map(|point| {
                let p = logistic(binomial_logit_field(point.x, point.y));
                BinomialObservation::new(point.coord, binomial_draws(&mut rng, 40, p), 40).unwrap()
            })
            .collect::<Vec<_>>();
        let test_coords: Vec<_> = test.iter().map(|p| p.coord).collect();
        let test_true_p: Vec<_> = test
            .iter()
            .map(|p| logistic(binomial_logit_field(p.x, p.y)))
            .collect();

        let train_coords: Vec<_> = train_obs.iter().map(|o| o.coord()).collect();
        let logits = train_obs
            .iter()
            .map(BinomialObservation::smoothed_logit)
            .collect::<Vec<_>>();
        let train_dataset = GeoDataset::new(train_coords, logits).expect("dataset");
        let best_model = fit_variogram_for_type(
            &train_dataset,
            &VariogramConfig::default(),
            VariogramType::Exponential,
        );
        let model = BinomialKrigingModel::new(train_obs, best_model).expect("binomial model");
        let preds = model
            .predict_batch(&test_coords)
            .expect("binomial batch predict");
        assert_eq!(preds.len(), TEST_POINTS);

        for (pred, truth_p) in preds.iter().zip(test_true_p.iter()) {
            assert!(pred.prevalence.is_finite());
            assert!(pred.logit_value.is_finite());
            assert!(pred.variance.is_finite());
            all_abs_errors.push((pred.prevalence - truth_p).abs());
        }
    }

    let mae = all_abs_errors.iter().sum::<Real>() / all_abs_errors.len() as Real;
    assert!(mae < BINOMIAL_MAE_MAX, "mae={mae}");
}

#[test]
fn ordinary_pipeline_350_point_batch_matches_single_predictions() {
    let mut rng = StdRng::seed_from_u64(0xBADC_0FFE);
    let train = sample_grid_points(14, 25);
    let test = sample_points(&mut rng, 40);

    let train_coords: Vec<_> = train.iter().map(|p| p.coord).collect();
    let train_values: Vec<_> = train.iter().map(|p| ordinary_field(p.x, p.y)).collect();
    let test_coords: Vec<_> = test.iter().map(|p| p.coord).collect();

    let train_dataset = GeoDataset::new(train_coords, train_values).expect("dataset");
    let best_model = fit_variogram_for_type(
        &train_dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let ordinary = OrdinaryKrigingModel::new(train_dataset, best_model).expect("ordinary model");
    let batch_preds = ordinary.predict_batch(&test_coords).expect("batch");
    let single_preds = test_coords
        .iter()
        .map(|coord| ordinary.predict(*coord).expect("single"))
        .collect::<Vec<_>>();
    assert_eq!(batch_preds.len(), single_preds.len());

    for (batch, single) in batch_preds.iter().zip(single_preds.iter()) {
        assert!((batch.value - single.value).abs() < 1e-4);
        assert!((batch.variance - single.variance).abs() < 1e-4);
    }
}

#[test]
fn binomial_pipeline_350_point_accuracy_regression_budget() {
    let mut rng = StdRng::seed_from_u64(0xFACE_FEED);
    let train = sample_grid_points(14, 25);
    let test = sample_points(&mut rng, 60);

    let train_obs = train
        .iter()
        .map(|point| {
            let p = logistic(binomial_logit_field(point.x, point.y));
            BinomialObservation::new(point.coord, binomial_draws(&mut rng, 50, p), 50).unwrap()
        })
        .collect::<Vec<_>>();
    let test_coords: Vec<_> = test.iter().map(|p| p.coord).collect();
    let test_true_p: Vec<_> = test
        .iter()
        .map(|p| logistic(binomial_logit_field(p.x, p.y)))
        .collect();

    let train_coords: Vec<_> = train_obs.iter().map(|o| o.coord()).collect();
    let logits = train_obs
        .iter()
        .map(BinomialObservation::smoothed_logit)
        .collect::<Vec<_>>();
    let train_dataset = GeoDataset::new(train_coords, logits).expect("dataset");
    let best_model = fit_variogram_for_type(
        &train_dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let model = BinomialKrigingModel::new(train_obs, best_model).expect("binomial model");
    let preds = model.predict_batch(&test_coords).expect("binomial batch");
    assert_eq!(preds.len(), test_coords.len());
    let mae = preds
        .iter()
        .zip(test_true_p.iter())
        .map(|(pred, truth)| (pred.prevalence - truth).abs())
        .sum::<Real>()
        / preds.len() as Real;
    assert!(mae < 0.085, "350-point binomial mae={mae}");
}

#[test]
fn binomial_pipeline_matches_manual_pipeline_at_fixed_seed() {
    let mut rng = StdRng::seed_from_u64(0x1234_5678_9ABC_DEF0);
    let train = sample_points(&mut rng, 64);
    let test = sample_points(&mut rng, 24);
    let observations = train
        .iter()
        .map(|point| {
            let p = logistic(binomial_logit_field(point.x, point.y));
            BinomialObservation::new(point.coord, binomial_draws(&mut rng, 40, p), 40).unwrap()
        })
        .collect::<Vec<_>>();
    let test_coords = test.iter().map(|p| p.coord).collect::<Vec<_>>();
    let config = VariogramConfig {
        max_distance: None,
        n_bins: NonZeroUsize::new(10).unwrap(),
    };
    let variogram_type = VariogramType::Exponential;
    let prior = BinomialPrior::default();

    let coords = observations.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let logits = observations
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect::<Vec<_>>();
    let dataset = GeoDataset::new(coords, logits).expect("dataset");
    let best_model = fit_variogram_for_type(&dataset, &config, variogram_type);
    let pipeline = BinomialKrigingModel::new_with_prior(observations.clone(), best_model, prior)
        .expect("pipeline model")
        .predict_batch(&test_coords)
        .expect("pipeline predictions");

    let empirical = compute_empirical_variogram(&dataset, &config).expect("empirical");
    let best = fit_variogram(&empirical, variogram_type).expect("fit variogram");
    let model = BinomialKrigingModel::new_with_prior(observations.clone(), best.model, prior)
        .expect("model");
    let manual = model.predict_batch(&test_coords).expect("manual predict");

    assert_eq!(pipeline.len(), manual.len());
    for (q, m) in pipeline.iter().zip(manual.iter()) {
        assert!((q.prevalence - m.prevalence).abs() < 1e-4);
        assert!((q.logit_value - m.logit_value).abs() < 1e-4);
        assert!((q.variance - m.variance).abs() < 1e-4);
    }
}
