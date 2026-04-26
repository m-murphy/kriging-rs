//! Synthetic accuracy and timing: homo vs heteroskedastic binomial on a known latent
//! logit field (regression `tests/regression_simulation.rs` is related).
//!
//! Run (release for meaningful timings):
//!   cargo run --release --example accuracy_speed_report
//!   cargo run --release --example accuracy_speed_report -- --out-dir target/accuracy_sweep
//! Plots: install deps (`pip install -r scripts/requirements-plot.txt` in a venv) then
//!   python3 scripts/plot_accuracy_speed.py --tsv target/accuracy_sweep/summary.tsv --out target/accuracy_sweep/figs

use kriging_rs::GeoDataset;
use kriging_rs::variogram::empirical::{VariogramConfig, compute_empirical_variogram};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::VariogramType;
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord,
    HeteroskedasticBinomialConfig, Real, VariogramModel,
};

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use std::io::Write;
use std::time::Instant;

const TAU: Real = std::f64::consts::TAU as Real;

/// Convert `Real` to `f64` for scalar I/O without clippy `unnecessary_cast` under `f64`.
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

fn fit_v(coords: &[GeoCoord], logits: &[Real]) -> (VariogramModel, f64) {
    let t0 = Instant::now();
    let dataset = GeoDataset::new(coords.to_vec(), logits.to_vec()).expect("dataset");
    let em = compute_empirical_variogram(&dataset, &VariogramConfig::default()).expect("emp");
    let v = fit_variogram(&em, VariogramType::Exponential)
        .expect("fit")
        .model;
    (v, t0.elapsed().as_secs_f64() * 1000.0)
}

fn make_uniform(
    points: &[(Real, Real, GeoCoord)],
    trials: u32,
    rng: &mut impl Rng,
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

fn make_mixed(
    points: &[(Real, Real, GeoCoord)],
    lo: u32,
    hi: u32,
    rng: &mut impl Rng,
) -> Vec<BinomialObservation> {
    points
        .iter()
        .enumerate()
        .map(|(i, (x, y, c))| {
            let t = if i.is_multiple_of(2) { lo } else { hi };
            let p0 = true_p(*x, *y);
            let s = (0..t)
                .map(|_| (rng.random::<f32>() as Real) < p0)
                .map(u32::from)
                .sum();
            BinomialObservation::new(*c, s, t).expect("obs")
        })
        .collect()
}

fn logit_vec(o: &[BinomialObservation], prior: BinomialPrior) -> Vec<Real> {
    o.iter()
        .map(|x| x.smoothed_logit_with_prior(prior))
        .collect()
}

fn sample(rng: &mut StdRng, n: usize) -> Vec<(Real, Real, GeoCoord)> {
    (0..n)
        .map(|_| {
            let x = rng.random::<f32>() as Real;
            let y = rng.random::<f32>() as Real;
            (x, y, to_coord(x, y))
        })
        .collect()
}

fn mae_rmse(m: &BinomialKrigingModel, test: &[(Real, Real, GeoCoord)]) -> (f64, f64) {
    let n = test.len() as f64;
    let mut sum_abs = 0.0;
    let mut sum_sq = 0.0;
    for (x, y, c) in test {
        let t0 = real_to_f64(true_p(*x, *y));
        let p = real_to_f64(m.predict(*c).expect("pred").prevalence);
        let e = p - t0;
        sum_abs += e.abs();
        sum_sq += e * e;
    }
    (sum_abs / n, (sum_sq / n).sqrt())
}

fn time_batch_ms(m: &BinomialKrigingModel, coords: &[GeoCoord]) -> f64 {
    let t0 = Instant::now();
    let _ = m.predict_batch(coords).expect("batch");
    t0.elapsed().as_secs_f64() * 1000.0
}

struct Row {
    scenario: String,
    path: String,
    n_train: usize,
    n_test: usize,
    n_grid: usize,
    fit_ms: f64,
    build_ms: f64,
    pred_test_ms: f64,
    pred_grid_ms: f64,
    mae: f64,
    rmse: f64,
}

#[allow(clippy::too_many_arguments)]
fn one_row(
    scenario: &str,
    path: &str,
    train: &[(Real, Real, GeoCoord)],
    test: &[(Real, Real, GeoCoord)],
    grid: &[GeoCoord],
    obs: &[BinomialObservation],
    v: VariogramModel,
    fit_ms: f64,
    prior: BinomialPrior,
    hcfg: HeteroskedasticBinomialConfig,
) -> Row {
    let n_train = train.len();
    let n_test = test.len();
    let n_grid = grid.len();
    let (build_ms, m) = {
        let t0 = Instant::now();
        // Calibrated default: homo/hetero TSV labels are kept for the sweep script; both
        // use the same `new_with_config` pipeline.
        let m = BinomialKrigingModel::new_with_config(obs.to_vec(), v, prior, hcfg, &[])
            .expect("build");
        (t0.elapsed().as_secs_f64() * 1000.0, m)
    };
    let (ma, rm) = mae_rmse(&m, test);
    let test_cc: Vec<GeoCoord> = test.iter().map(|t| t.2).collect();
    let pred_test_ms = {
        let t0 = Instant::now();
        m.predict_batch(&test_cc).expect("b");
        t0.elapsed().as_secs_f64() * 1000.0
    };
    let pred_grid_ms = time_batch_ms(&m, grid);
    Row {
        scenario: scenario.to_string(),
        path: path.to_string(),
        n_train,
        n_test,
        n_grid,
        fit_ms,
        build_ms,
        pred_test_ms,
        pred_grid_ms,
        mae: ma,
        rmse: rm,
    }
}

fn grid(n_side: usize) -> Vec<GeoCoord> {
    if n_side < 2 {
        return vec![];
    }
    let mut out = Vec::with_capacity(n_side * n_side);
    for i in 0..n_side {
        let x = i as Real / (n_side - 1) as Real;
        for j in 0..n_side {
            let y = j as Real / (n_side - 1) as Real;
            out.push(to_coord(x, y));
        }
    }
    out
}

fn out_dir() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from("target/accuracy_sweep");
    for (i, a) in std::env::args().enumerate() {
        if a == "--out-dir" && i + 1 < std::env::args().count() {
            p = std::path::PathBuf::from(std::env::args().nth(i + 1).unwrap());
        }
    }
    p
}

fn main() {
    let out = out_dir();
    let _ = std::fs::create_dir_all(&out);
    let dest = out.join("summary.tsv");
    let prior = BinomialPrior::default();
    let hcfg = HeteroskedasticBinomialConfig::default();

    const N_TR: usize = 40;
    const N_TE: usize = 20;
    const N_SIDE: usize = 32;
    const SEED: u64 = 42;

    let g = grid(N_SIDE);
    let mut rows: Vec<Row> = Vec::new();

    for (label, use_mixed) in [("S40_mixed5_200", true), ("S40_uniform50", false)] {
        let mut r = StdRng::seed_from_u64(SEED);
        let train = sample(&mut r, N_TR);
        let test = sample(&mut r, N_TE);
        let obs = if use_mixed {
            make_mixed(&train, 5, 200, &mut r)
        } else {
            make_uniform(&train, 50, &mut r)
        };
        let train_coords: Vec<GeoCoord> = train.iter().map(|t| t.2).collect();
        let log = logit_vec(&obs, prior);
        let (v, fit_ms) = fit_v(&train_coords, &log);
        for path in ["homo", "hetero"] {
            rows.push(one_row(
                label, path, &train, &test, &g, &obs, v, fit_ms, prior, hcfg,
            ));
        }
    }

    let mut f = std::fs::File::create(&dest).expect("open");
    writeln!(
        f,
        "scenario\tpath\tn_train\tn_test\tn_grid\tfit_variogram_ms\tbuild_model_ms\tpred_test_ms\tpred_grid_ms\tmae_prevalence\trmse_prevalence"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t{:.6}",
            r.scenario,
            r.path,
            r.n_train,
            r.n_test,
            r.n_grid,
            r.fit_ms,
            r.build_ms,
            r.pred_test_ms,
            r.pred_grid_ms,
            r.mae,
            r.rmse
        )
        .unwrap();
    }
    eprintln!("Wrote {} rows to {}", rows.len(), dest.display());
    for r in &rows {
        eprintln!(
            "{}/{}: MAE={:.4} build={:.1}ms pred_test={:.1}ms pred_grid={:.1}ms",
            r.scenario, r.path, r.mae, r.build_ms, r.pred_test_ms, r.pred_grid_ms
        );
    }
}
