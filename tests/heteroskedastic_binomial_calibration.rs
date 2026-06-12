//! Integration tests for **experimental** heteroskedastic binomial kriging: monotonic
//! logit observation variance, build diagnostics, and k-fold logit **MSDR** (mean
//! squared deviation ratio) on mixed trial counts.

use kriging_rs::cv::{
    BinomialCvSummary, BinomialGeoPredictor, CvSummary, OrdinaryGeoPredictor, k_fold_cv,
};
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord,
    HeteroskedasticBinomialConfig, Real, VariogramModel, VariogramType, logit_clamped,
    logit_observation_variance_empirical_bayes,
};

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

#[test]
fn more_trials_imply_smaller_empirical_bayes_logit_var_at_same_p() {
    let p = BinomialPrior::default();
    let a = logit_observation_variance_empirical_bayes(p, 1, 10);
    let b = logit_observation_variance_empirical_bayes(p, 10, 100);
    assert!(a > b, "a={a} b={b}");
    assert!(a > 0.0 && b > 0.0);
    assert!(a.is_finite() && b.is_finite());
}

#[test]
fn hetero_build_succeeds_with_mixed_and_reports_inflation() {
    let p = BinomialPrior::default();
    let v = VariogramModel::new(0.04, 3.5, 0.3, VariogramType::Exponential).unwrap();
    let mut obs = Vec::new();
    for k in 0u32..24 {
        let lat = 40.0 + (k as Real) * 0.05;
        let lon = -75.0 + (k as Real) * 0.05;
        let t = 8u32 + (k % 5) * 18;
        let s = t / 2;
        obs.push(
            BinomialObservation::new(GeoCoord::try_new(lat, lon).unwrap(), s, t).expect("obs"),
        );
    }
    let fit = BinomialKrigingModel::new_with_config(
        obs,
        v,
        p,
        HeteroskedasticBinomialConfig::default(),
        &[],
    )
    .expect("hetero build");
    assert_eq!(fit.notes.logit_inflation, 1.0, "no inflation when solvable");
    let x = fit
        .model
        .predict(GeoCoord::try_new(40.5, -74.2).unwrap())
        .expect("p");
    assert!(x.prevalence_median.is_finite() && x.logit.is_finite());
}

#[test]
fn mixed_trials_kfold_msdar_hetero_and_homo_sane() {
    let prior = BinomialPrior::default();
    let v = VariogramModel::new(0.05, 2.0, 0.25, VariogramType::Exponential).unwrap();
    let (coords, successes, trials) = mixed_40();
    // Homoskedastic binomial CV
    let h_res = k_fold_cv(
        &BinomialGeoPredictor {
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram: v,
            prior,
        },
        5,
    )
    .expect("h cv");
    let h = BinomialCvSummary::from_residuals(&h_res);
    if h.n_evaluated > 0 {
        let d_homo = (h.logit.msdr - 1.0).abs();
        assert!(d_homo < 5.0, "homoskedastic |MSDR-1|={d_homo}");
    }
    // Hetero: same folds, manual, compare MSDR
    let n = coords.len();
    let mut rat = 0.0f64;
    let mut msd_n = 0u32;
    for fold in 0..5 {
        let tr: Vec<usize> = (0..n).filter(|i| i % 5 != fold).collect();
        let te: Vec<usize> = (0..n).filter(|i| i % 5 == fold).collect();
        if tr.len() < 2 || te.is_empty() {
            continue;
        }
        let otrain: Vec<BinomialObservation> = tr
            .iter()
            .map(|&i| {
                BinomialObservation::new(coords[i], successes[i], trials[i]).expect("train obs")
            })
            .collect();
        let m = BinomialKrigingModel::new_with_config(
            otrain,
            v,
            prior,
            HeteroskedasticBinomialConfig::default(),
            &[],
        )
        .expect("fold het model");
        for &i in &te {
            let p = m.predict(coords[i]).expect("pred");
            if !p.logit_variance.is_finite() || p.logit_variance <= 0.0 {
                continue;
            }
            let t = trials[i];
            if t == 0 {
                continue;
            }
            let p_hat = successes[i] as f64 / t as f64;
            let obs = real_to_f64(logit_clamped(p_hat as Real));
            let e = obs - real_to_f64(p.logit);
            rat += (e * e) / real_to_f64(p.logit_variance);
            msd_n += 1;
        }
    }
    assert!(msd_n > 0, "need at least one k-fold point");
    let msdr_het = rat / msd_n as f64;
    assert!(
        msdr_het.is_finite() && msdr_het > 0.01 && msdr_het < 20.0,
        "msdr_het={msdr_het}"
    );
}

#[test]
fn homo_logit_kfold_msdr_finite() {
    let prior = BinomialPrior::default();
    let v = VariogramModel::new(0.05, 2.0, 0.25, VariogramType::Exponential).unwrap();
    let (coords, suc, tr) = mixed_40();
    let logits: Vec<Real> = (0..coords.len())
        .map(|i| {
            let o = BinomialObservation::new(coords[i], suc[i], tr[i]).unwrap();
            o.smoothed_logit_with_prior(prior)
        })
        .collect();
    let r = k_fold_cv(
        &OrdinaryGeoPredictor {
            coords: &coords,
            values: &logits,
            variogram: v,
        },
        5,
    )
    .expect("homo logit kfold");
    let s = CvSummary::from_residuals(&r);
    assert!(s.n > 0);
    assert!(s.msdr.is_finite());
}

fn mixed_40() -> (Vec<GeoCoord>, Vec<u32>, Vec<u32>) {
    let mut coords = Vec::new();
    let mut s = Vec::new();
    let mut t = Vec::new();
    for k in 0..40 {
        let row = (k / 8) as Real;
        let col = (k % 8) as Real;
        coords.push(GeoCoord::try_new(40.0 + row * 0.08, -75.0 + col * 0.08).expect("c"));
        let ntrials: u32 = if k % 2 == 0 { 5 } else { 120 };
        let su = ntrials / 2;
        t.push(ntrials);
        s.push(su);
    }
    (coords, s, t)
}
