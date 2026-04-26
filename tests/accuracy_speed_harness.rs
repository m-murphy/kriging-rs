//! Deterministic checks for the same synthetic logit field + pipeline used by
//! `examples/accuracy_speed_report` (calibrated binomial MAE/RMSE on known prevalence).

use kriging_rs::GeoDataset;
use kriging_rs::variogram::empirical::{VariogramConfig, compute_empirical_variogram};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::VariogramType;
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord,
    HeteroskedasticBinomialConfig, Real, VariogramModel,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const TAU: Real = std::f64::consts::TAU as Real;
const SEED: u64 = 91;

/// Convert `Real` to `f64` for scalar stats without clippy `unnecessary_cast` under `f64`.
#[inline]
fn real_to_f64(x: Real) -> f64 {
    #[cfg(feature = "f64")]
    {
        x
    }
    #[cfg(not(feature = "f64"))]
    {
        x as f64
    }
}

fn logistic(x: Real) -> Real {
    1.0 / (1.0 + (-x).exp())
}

fn to_coord(x: Real, y: Real) -> GeoCoord {
    GeoCoord::try_new(35.0 + x, -120.0 + y).unwrap()
}

fn binomial_logit_field(x: Real, y: Real) -> Real {
    -0.3 + 1.2 * (TAU * x).sin() * (TAU * y).cos()
}

fn true_p(x: Real, y: Real) -> Real {
    logistic(binomial_logit_field(x, y))
}

fn sample(n: usize, rng: &mut StdRng) -> Vec<(Real, Real, GeoCoord)> {
    (0..n)
        .map(|_| {
            let x = rng.random::<f32>() as Real;
            let y = rng.random::<f32>() as Real;
            (x, y, to_coord(x, y))
        })
        .collect()
}

fn make_uniform(
    points: &[(Real, Real, GeoCoord)],
    trials: u32,
    rng: &mut StdRng,
) -> Vec<BinomialObservation> {
    points
        .iter()
        .map(|(x, y, c)| {
            let p0 = true_p(*x, *y);
            let s = (0..trials)
                .map(|_| (rng.random::<f32>() as Real) < p0)
                .map(u32::from)
                .sum();
            BinomialObservation::new(*c, s, trials).expect("obs")
        })
        .collect()
}

fn mae_rmse(m: &BinomialKrigingModel, test: &[(Real, Real, GeoCoord)]) -> (f64, f64) {
    let n = test.len() as f64;
    let mut s_abs = 0.0f64;
    let mut s_sq = 0.0f64;
    for (x, y, c) in test {
        let t0 = real_to_f64(true_p(*x, *y));
        let p = real_to_f64(m.predict(*c).expect("pred").prevalence);
        let e = p - t0;
        s_abs += e.abs();
        s_sq += e * e;
    }
    (s_abs / n, (s_sq / n).sqrt())
}

#[test]
fn homo_and_hetero_match_accuracy_gate_on_synthetic_field() {
    const N_TR: usize = 35;
    const N_TE: usize = 8;
    let prior = BinomialPrior::default();
    let hcfg = HeteroskedasticBinomialConfig::default();

    let mut r = StdRng::seed_from_u64(SEED);
    let train = sample(N_TR, &mut r);
    let test = sample(N_TE, &mut r);
    let obs = make_uniform(&train, 45, &mut r);

    let train_coords: Vec<GeoCoord> = train.iter().map(|t| t.2).collect();
    let log: Vec<Real> = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(prior))
        .collect();
    let dataset = GeoDataset::new(train_coords.clone(), log).expect("dataset");
    let em = compute_empirical_variogram(&dataset, &VariogramConfig::default()).expect("emp");
    let v: VariogramModel = fit_variogram(&em, VariogramType::Exponential)
        .expect("fit")
        .model;

    let m = BinomialKrigingModel::new_with_config(obs, v, prior, hcfg, &[]).expect("build");

    let (mae, rmse) = mae_rmse(&m, &test);
    for (label, a, s) in [("mae", mae, 0.18f64), ("rmse", rmse, 0.22f64)] {
        assert!(a < s, "{label} a={a} s={s}");
    }

    let tcoords: Vec<GeoCoord> = test.iter().map(|t| t.2).collect();
    let pb = m.predict_batch(&tcoords).expect("batch");
    assert_eq!(pb.len(), N_TE);
}
