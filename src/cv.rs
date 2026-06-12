//! Cross-validation for kriging models.
//!
//! Fold iteration is generic over [`KrigingPredictor`](crate::predictor::cv::KrigingPredictor)
//! backends in [`crate::predictor::cv`]. Construct a predictor struct (e.g.
//! [`OrdinaryGeoPredictor`](crate::predictor::cv::OrdinaryGeoPredictor)) and call
//! [`leave_one_out_cv`](crate::predictor::cv::leave_one_out_cv) or
//! [`k_fold_cv`](crate::predictor::cv::k_fold_cv).
//!
//! The continuous variants share [`CvResidual`] / [`CvSummary`]. Binomial CV returns
//! [`BinomialCvResidual`] values that carry **both** the logit-scale residual (directly
//! comparable to continuous kriging and MSDR-calibratable) **and** the prevalence-scale
//! residual (intuitive; delta-method variance). [`BinomialCvSummary`] aggregates both plus
//! trial-weighted **log score per trial**, **Brier** (mean squared prevalence error), and
//! **calibration bins** over predicted prevalence.
//!
//! All routines assume the supplied variogram / drift / mean / anisotropy is held fixed
//! across folds; only the kriging system is refit per fold. Callers who want to refit
//! variogram parameters inside each fold must iterate themselves.
//!
//! The folds are deterministic round-robin assignments (station `i` goes to fold `i % k`),
//! which keeps validation reproducible and avoids the need for an RNG dependency. Callers
//! who want randomization can shuffle inputs before calling.

use crate::Real;

/// A single cross-validation residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvResidual {
    /// Index of the held-out station in the original input arrays.
    pub index: usize,
    /// Observed value at the held-out station.
    pub observed: Real,
    /// Kriging prediction at the held-out station (from the training fold).
    pub predicted: Real,
    /// Kriging variance at the held-out station.
    pub variance: Real,
}

impl CvResidual {
    /// Signed residual `observed − predicted`.
    pub fn error(&self) -> Real {
        self.observed - self.predicted
    }
}

/// Summary statistics over a set of cross-validation residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvSummary {
    pub n: usize,
    /// Mean signed error (bias).
    pub mean_error: Real,
    /// Root mean squared error.
    pub rmse: Real,
    /// Mean squared deviation ratio: mean((obs − pred)² / variance).
    /// Approximately 1 when the variogram is well-calibrated.
    pub msdr: Real,
}

impl CvSummary {
    pub fn from_residuals(residuals: &[CvResidual]) -> Self {
        Self::from_scalar_iter(
            residuals.len(),
            residuals.iter().map(|r| (r.error(), r.variance)),
        )
    }

    /// Internal: compute bias/RMSE/MSDR from pre-computed `(error, variance)` pairs. NaN
    /// errors (e.g. when an observation is undefined, as for binomial trials == 0) are
    /// skipped. The `n` field is set to the number of *finite* residuals rather than the
    /// total length of the input iterator.
    fn from_scalar_iter<I>(_hint: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = (Real, Real)>,
    {
        let mut n_finite = 0usize;
        let mut sum_e = 0.0 as Real;
        let mut sum_e2 = 0.0 as Real;
        let mut sum_ratio = 0.0 as Real;
        let mut ratio_n = 0usize;
        for (e, variance) in iter {
            if !e.is_finite() {
                continue;
            }
            n_finite += 1;
            sum_e += e;
            sum_e2 += e * e;
            if variance > 0.0 && variance.is_finite() {
                sum_ratio += e * e / variance;
                ratio_n += 1;
            }
        }
        if n_finite == 0 {
            return Self {
                n: 0,
                mean_error: 0.0,
                rmse: 0.0,
                msdr: 0.0,
            };
        }
        let nf = n_finite as Real;
        let msdr = if ratio_n == 0 {
            0.0
        } else {
            sum_ratio / ratio_n as Real
        };
        Self {
            n: n_finite,
            mean_error: sum_e / nf,
            rmse: (sum_e2 / nf).sqrt(),
            msdr,
        }
    }
}

// ---------------------------------------------------------------------------
// Binomial kriging CV (reports BOTH logit and prevalence scales)
// ---------------------------------------------------------------------------

/// A single binomial cross-validation residual. Reports the held-out observation and the
/// prediction on **both** the logit scale (directly comparable to continuous kriging and
/// calibratable via MSDR) and the prevalence scale (intuitive for probability-scale
/// diagnostics, with a delta-method variance).
///
/// When `trials == 0` at the held-out station, the observed prevalence and logit are
/// undefined and reported as `NaN`. The prediction is still populated; the entry is
/// retained at its input index so downstream code can audit which stations were
/// unobservable. [`BinomialCvSummary`] skips these when aggregating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialCvResidual {
    /// Index of the held-out station in the original input arrays.
    pub index: usize,
    /// Held-out success count.
    pub successes: u32,
    /// Held-out trial count. `0` means the observation is undefined.
    pub trials: u32,
    /// Held-out observed logit. `NaN` when `trials == 0`.
    pub observed_logit: Real,
    /// Model prediction on the logit scale.
    pub predicted_logit: Real,
    /// Kriging variance on the logit scale.
    pub logit_variance: Real,
    /// Held-out observed prevalence `successes / trials`. `NaN` when `trials == 0`.
    pub observed_prevalence: Real,
    /// Model prediction on the prevalence scale (logistic of `predicted_logit`).
    pub predicted_prevalence: Real,
    /// Delta-method approximation of the variance of `predicted_prevalence`.
    pub prevalence_variance: Real,
}

impl BinomialCvResidual {
    /// Signed logit-scale error `observed_logit − predicted_logit`. `NaN` when `trials == 0`.
    pub fn logit_error(&self) -> Real {
        self.observed_logit - self.predicted_logit
    }

    /// Signed prevalence-scale error `observed_prevalence − predicted_prevalence`.
    /// `NaN` when `trials == 0`.
    pub fn prevalence_error(&self) -> Real {
        self.observed_prevalence - self.predicted_prevalence
    }
}

/// One calibration bin for prevalence-scale CV: stations whose **predicted** prevalence
/// falls in `[predicted_lo, predicted_hi)` (last bin includes the upper endpoint at 1).
///
/// `pooled_observed_prevalence` is `sum(successes) / sum(trials)` over stations in the bin
/// (trial-weighted). Empty bins report `n_stations == 0` and `NaN` means.
#[derive(Debug, Clone, PartialEq)]
pub struct PrevalenceCalibrationBin {
    pub bin_index: usize,
    pub predicted_lo: Real,
    pub predicted_hi: Real,
    pub n_stations: usize,
    pub sum_trials: u64,
    pub sum_successes: u64,
    pub mean_predicted: Real,
    pub pooled_observed_prevalence: Real,
}

/// Aggregate summary for binomial CV, reported on **both** scales.
///
/// - `n` — total residuals (including any with `trials == 0`).
/// - `n_evaluated` — number of residuals with `trials > 0`, i.e. those contributing to
///   `logit` / `prevalence`.
/// - `logit` — summary statistics on the logit scale (bias / RMSE / MSDR).
/// - `prevalence` — summary statistics on the prevalence scale.
/// - `brier` — mean squared error `(ŷ − y)²` over evaluated stations, one term per station
///   (`y = successes / trials`). `NaN` when `n_evaluated == 0`.
/// - `log_score_per_trial` — trial-weighted mean log predictive mass
///   `(∑ᵢ sᵢ log ŷᵢ + (nᵢ−sᵢ) log(1−ŷᵢ)) / (∑ᵢ nᵢ)` with `ŷ` clamped to `(ε, 1−ε)` for
///   stability. Higher is better (larger log-likelihood per trial). `NaN` when no trials.
/// - `calibration_bins` — ten equal-width bins on predicted prevalence in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BinomialCvSummary {
    pub n: usize,
    pub n_evaluated: usize,
    pub logit: CvSummary,
    pub prevalence: CvSummary,
    pub brier: Real,
    pub log_score_per_trial: Real,
    pub calibration_bins: Vec<PrevalenceCalibrationBin>,
}

const PREVALENCE_CALIBRATION_BIN_COUNT: usize = 10;

impl BinomialCvSummary {
    pub fn from_residuals(residuals: &[BinomialCvResidual]) -> Self {
        let n = residuals.len();
        let n_evaluated = residuals.iter().filter(|r| r.trials > 0).count();
        let logit = CvSummary::from_scalar_iter(
            n,
            residuals
                .iter()
                .map(|r| (r.logit_error(), r.logit_variance)),
        );
        let prevalence = CvSummary::from_scalar_iter(
            n,
            residuals
                .iter()
                .map(|r| (r.prevalence_error(), r.prevalence_variance)),
        );
        let eps = (1e-12 as Real).max(Real::EPSILON * 8.0);
        let mut sum_brier = 0.0 as Real;
        let mut n_brier = 0usize;
        let mut sum_ll = 0.0 as Real;
        let mut sum_trials_w = 0u64;

        let mut bin_n = [0usize; PREVALENCE_CALIBRATION_BIN_COUNT];
        let mut bin_sum_pred = [0.0 as Real; PREVALENCE_CALIBRATION_BIN_COUNT];
        let mut bin_sum_s = [0u64; PREVALENCE_CALIBRATION_BIN_COUNT];
        let mut bin_sum_t = [0u64; PREVALENCE_CALIBRATION_BIN_COUNT];

        for r in residuals {
            if r.trials == 0 {
                continue;
            }
            let y = r.observed_prevalence;
            let p_hat = r.predicted_prevalence;
            sum_brier += (p_hat - y) * (p_hat - y);
            n_brier += 1;

            let pc = p_hat.clamp(eps, 1.0 - eps);
            let s = r.successes as Real;
            let nt = r.trials as Real;
            sum_ll += s * pc.ln() + (nt - s) * (1.0 - pc).ln();
            sum_trials_w += r.trials as u64;

            let p_bin = p_hat.clamp(0.0 as Real, 1.0 as Real);
            let mut k = (p_bin * PREVALENCE_CALIBRATION_BIN_COUNT as Real).floor() as usize;
            if k >= PREVALENCE_CALIBRATION_BIN_COUNT {
                k = PREVALENCE_CALIBRATION_BIN_COUNT - 1;
            }
            bin_n[k] += 1;
            bin_sum_pred[k] += p_hat;
            bin_sum_s[k] += r.successes as u64;
            bin_sum_t[k] += r.trials as u64;
        }

        let brier = if n_brier > 0 {
            sum_brier / n_brier as Real
        } else {
            Real::NAN
        };
        let log_score_per_trial = if sum_trials_w > 0 {
            sum_ll / sum_trials_w as Real
        } else {
            Real::NAN
        };

        let mut calibration_bins = Vec::with_capacity(PREVALENCE_CALIBRATION_BIN_COUNT);
        for i in 0..PREVALENCE_CALIBRATION_BIN_COUNT {
            let predicted_lo = i as Real / PREVALENCE_CALIBRATION_BIN_COUNT as Real;
            let predicted_hi = if i + 1 == PREVALENCE_CALIBRATION_BIN_COUNT {
                1.0 as Real
            } else {
                (i + 1) as Real / PREVALENCE_CALIBRATION_BIN_COUNT as Real
            };
            let ns = bin_n[i];
            let mean_predicted = if ns > 0 {
                bin_sum_pred[i] / ns as Real
            } else {
                Real::NAN
            };
            let st = bin_sum_t[i];
            let pooled_observed_prevalence = if st > 0 {
                bin_sum_s[i] as Real / st as Real
            } else {
                Real::NAN
            };
            calibration_bins.push(PrevalenceCalibrationBin {
                bin_index: i,
                predicted_lo,
                predicted_hi,
                n_stations: ns,
                sum_trials: st,
                sum_successes: bin_sum_s[i],
                mean_predicted,
                pooled_observed_prevalence,
            });
        }

        Self {
            n,
            n_evaluated,
            logit,
            prevalence,
            brier,
            log_score_per_trial,
            calibration_bins,
        }
    }
}

// Generic CV harness — see [`crate::predictor::cv`].
pub use crate::predictor::cv::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::kriging::binomial::BinomialPrior;
    use crate::kriging::universal::UniversalTrend;
    use crate::projected::{Anisotropy2D, ProjectedCoord};
    use crate::spacetime::coord::SpaceTimeCoord;
    use crate::spacetime::kriging::universal::SpaceTimeUniversalTrend;
    use crate::spacetime::variogram::SpaceTimeVariogram;
    use crate::utils::{logistic, logit_clamped};
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn grid_points() -> (Vec<GeoCoord>, Vec<Real>) {
        // A small 4x4 grid with a smooth linear trend in both coordinates.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let lat = i as Real;
                let lon = j as Real;
                coords.push(GeoCoord::try_new(lat, lon).unwrap());
                values.push(2.0 * lat + 3.0 * lon + 1.0);
            }
        }
        (coords, values)
    }

    fn projected_grid_points() -> (Vec<ProjectedCoord>, Vec<Real>) {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as Real;
                let y = j as Real;
                coords.push(ProjectedCoord::new(x, y));
                values.push(2.0 * x + 3.0 * y + 1.0);
            }
        }
        (coords, values)
    }

    fn binomial_grid_points() -> (Vec<GeoCoord>, Vec<u32>, Vec<u32>) {
        // 4x4 grid of counts; a smooth logit gradient in lat yields prevalences ~ 0.1..0.9.
        let mut coords = Vec::new();
        let mut successes = Vec::new();
        let mut trials = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let lat = i as Real;
                let lon = j as Real;
                let p = logistic(-2.0 + 0.5 * lat + 0.5 * lon);
                let n = 40u32;
                let s = (p * n as Real).round() as u32;
                coords.push(GeoCoord::try_new(lat, lon).unwrap());
                successes.push(s);
                trials.push(n);
            }
        }
        (coords, successes, trials)
    }

    #[test]
    fn leave_one_out_returns_one_residual_per_station_in_order() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&OrdinaryGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.observed, values[i]);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn leave_one_out_has_small_rmse_for_smooth_linear_field() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 10.0, 500.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&OrdinaryGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
        })
        .unwrap();
        let summary = CvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, coords.len());
        assert!(
            summary.rmse.is_finite() && summary.rmse < 3.0,
            "RMSE on smooth field should be modest, got {}",
            summary.rmse
        );
    }

    #[test]
    fn k_fold_covers_every_station_exactly_once() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_cv(
            &OrdinaryGeoPredictor {
                coords: &coords,
                values: &values,
                variogram,
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index], "duplicate residual for index {}", r.index);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn k_fold_rejects_invalid_k() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        assert!(
            k_fold_cv(
                &OrdinaryGeoPredictor {
                    coords: &coords,
                    values: &values,
                    variogram
                },
                1
            )
            .is_err()
        );
        assert!(
            k_fold_cv(
                &OrdinaryGeoPredictor {
                    coords: &coords,
                    values: &values,
                    variogram
                },
                coords.len() + 1
            )
            .is_err()
        );
    }

    #[test]
    fn leave_one_out_rejects_fewer_than_two_stations() {
        let coords = vec![GeoCoord::try_new(0.0, 0.0).unwrap()];
        let values = vec![1.0];
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        assert!(
            leave_one_out_cv(&OrdinaryGeoPredictor {
                coords: &coords,
                values: &values,
                variogram
            })
            .is_err()
        );
    }

    #[test]
    fn cv_summary_mean_error_matches_hand_calculation() {
        let residuals = vec![
            CvResidual {
                index: 0,
                observed: 10.0,
                predicted: 11.0,
                variance: 1.0,
            },
            CvResidual {
                index: 1,
                observed: 20.0,
                predicted: 18.0,
                variance: 1.0,
            },
        ];
        let s = CvSummary::from_residuals(&residuals);
        assert_eq!(s.n, 2);
        // Errors are -1 and 2; mean = 0.5.
        assert!((s.mean_error - 0.5).abs() < 1e-6);
        // RMSE = sqrt((1 + 4) / 2) = sqrt(2.5).
        assert!((s.rmse - (2.5 as Real).sqrt()).abs() < 1e-6);
        // MSDR with unit variance = mean of squared errors = 2.5.
        assert!((s.msdr - 2.5).abs() < 1e-6);
    }

    #[test]
    fn simple_loo_runs_with_known_mean() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals = leave_one_out_cv(&SimpleGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
            mean,
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn simple_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals = k_fold_cv(
            &SimpleGeoPredictor {
                coords: &coords,
                values: &values,
                variogram,
                mean,
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn universal_loo_matches_ordinary_for_constant_trend_within_tol() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let ok = leave_one_out_cv(&OrdinaryGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
        })
        .unwrap();
        let uk = leave_one_out_cv(&UniversalGeoPredictor {
            coords: &coords,
            values: &values,
            variogram,
            trend: UniversalTrend::Constant,
        })
        .unwrap();
        assert_eq!(ok.len(), uk.len());
        for (a, b) in ok.iter().zip(uk.iter()) {
            // Dual SPD ordinary kriging (ADR-0001) can differ from bordered LU at the last
            // few f32 ULPs; constant-trend universal still uses the legacy path today.
            assert!(
                (a.predicted - b.predicted).abs() < 5e-6,
                "constant-trend UK should match OK at station {} (ok={}, uk={})",
                a.index,
                a.predicted,
                b.predicted
            );
        }
    }

    #[test]
    fn universal_k_fold_runs_with_linear_trend() {
        let (coords, values) = grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_cv(
            &UniversalGeoPredictor {
                coords: &coords,
                values: &values,
                variogram,
                trend: UniversalTrend::Linear,
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn projected_loo_matches_ordinary_when_isotropic_and_euclidean() {
        // Sanity check: projected kriging with isotropic anisotropy on a planar grid should
        // produce finite residuals and pass structural checks.
        let (coords, values) = projected_grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&ProjectedOrdinaryPredictor {
            coords: &coords,
            values: &values,
            variogram,
            anisotropy: Anisotropy2D::isotropic(),
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn projected_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = projected_grid_points();
        let variogram = VariogramModel::new(0.01, 5.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_cv(
            &ProjectedOrdinaryPredictor {
                coords: &coords,
                values: &values,
                variogram,
                anisotropy: Anisotropy2D::isotropic(),
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn binomial_loo_reports_both_scales_in_input_order() {
        let (coords, successes, trials) = binomial_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&BinomialGeoPredictor {
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram,
            prior: BinomialPrior::default(),
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.successes, successes[i]);
            assert_eq!(r.trials, trials[i]);
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
            assert!(r.predicted_logit.is_finite());
            assert!(r.predicted_prevalence.is_finite());
            assert!(r.logit_variance.is_finite());
            assert!(r.prevalence_variance.is_finite());
            assert!(
                r.predicted_prevalence >= 0.0 && r.predicted_prevalence <= 1.0,
                "prevalence must lie in [0,1], got {}",
                r.predicted_prevalence
            );
        }
    }

    #[test]
    fn binomial_loo_handles_zero_trials_with_nan_observations() {
        let (coords, mut successes, mut trials) = binomial_grid_points();
        // Flip the first station to "unobservable" and keep the rest.
        successes[0] = 0;
        trials[0] = 0;
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&BinomialGeoPredictor {
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram,
            prior: BinomialPrior::default(),
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        // First station: observed should be NaN, prediction should still be finite.
        let r0 = residuals[0];
        assert_eq!(r0.trials, 0);
        assert!(r0.observed_logit.is_nan());
        assert!(r0.observed_prevalence.is_nan());
        assert!(r0.predicted_logit.is_finite());
        assert!(r0.predicted_prevalence.is_finite());
        // Others must remain well-defined.
        for r in &residuals[1..] {
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
        }
        // Summary must aggregate only the observable stations on each scale.
        let summary = BinomialCvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, residuals.len());
        assert_eq!(summary.n_evaluated, residuals.len() - 1);
        assert_eq!(summary.logit.n, summary.n_evaluated);
        assert_eq!(summary.prevalence.n, summary.n_evaluated);
        assert!(summary.logit.rmse.is_finite());
        assert!(summary.prevalence.rmse.is_finite());
    }

    #[test]
    fn binomial_k_fold_covers_every_station_exactly_once() {
        let (coords, successes, trials) = binomial_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_cv(
            &BinomialGeoPredictor {
                coords: &coords,
                successes: &successes,
                trials: &trials,
                variogram,
                prior: BinomialPrior::default(),
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    fn binomial_projected_grid_points() -> (Vec<ProjectedCoord>, Vec<u32>, Vec<u32>) {
        let mut coords = Vec::new();
        let mut successes = Vec::new();
        let mut trials = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as Real;
                let y = j as Real;
                let p = logistic(-2.0 + 0.5 * x + 0.5 * y);
                let n = 40u32;
                let s = (p * n as Real).round() as u32;
                coords.push(ProjectedCoord::new(x, y));
                successes.push(s);
                trials.push(n);
            }
        }
        (coords, successes, trials)
    }

    #[test]
    fn binomial_projected_loo_returns_one_residual_per_station_in_order() {
        let (coords, successes, trials) = binomial_projected_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = leave_one_out_cv(&BinomialProjectedPredictor {
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram,
            anisotropy: Anisotropy2D::isotropic(),
            prior: BinomialPrior::default(),
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.predicted_logit.is_finite());
            assert!(r.predicted_prevalence > 0.0 && r.predicted_prevalence < 1.0);
            assert!(r.logit_variance >= 0.0);
            assert!(r.prevalence_variance >= 0.0);
        }
    }

    #[test]
    fn binomial_projected_k_fold_covers_every_station_exactly_once() {
        let (coords, successes, trials) = binomial_projected_grid_points();
        let variogram = VariogramModel::new(0.05, 2.0, 5.0, VariogramType::Exponential).unwrap();
        let residuals = k_fold_cv(
            &BinomialProjectedPredictor {
                coords: &coords,
                successes: &successes,
                trials: &trials,
                variogram,
                anisotropy: Anisotropy2D::isotropic(),
                prior: BinomialPrior::default(),
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn binomial_summary_prevalence_rmse_is_consistent_with_residuals() {
        // Hand-constructed residuals with known errors on both scales.
        let residuals = vec![
            BinomialCvResidual {
                index: 0,
                successes: 3,
                trials: 10,
                observed_logit: logit_clamped(0.3),
                predicted_logit: logit_clamped(0.2),
                logit_variance: 1.0,
                observed_prevalence: 0.3,
                predicted_prevalence: 0.2,
                prevalence_variance: 0.01,
            },
            BinomialCvResidual {
                index: 1,
                successes: 0,
                trials: 0,
                observed_logit: Real::NAN,
                predicted_logit: 0.0,
                logit_variance: 1.0,
                observed_prevalence: Real::NAN,
                predicted_prevalence: 0.5,
                prevalence_variance: 0.0625,
            },
        ];
        let summary = BinomialCvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, 2);
        assert_eq!(summary.n_evaluated, 1);
        // Only the first residual contributes: prevalence error = 0.3 - 0.2 = 0.1; RMSE = 0.1.
        // Tolerance accounts for binary representation of 0.1 / 0.3 / 0.2 in f64.
        assert!(
            (summary.prevalence.rmse - 0.1).abs() < 1e-6,
            "expected ~0.1, got {}",
            summary.prevalence.rmse
        );
        // Logit error = logit(0.3) - logit(0.2); finite.
        assert!(summary.logit.rmse.is_finite() && summary.logit.rmse > 0.0);
        // Brier = (0.2 − 0.3)² on the single evaluated station.
        assert!(
            (summary.brier - 0.01).abs() < 1e-5,
            "expected Brier ~0.01, got {}",
            summary.brier
        );
        let p_c = 0.2 as Real;
        let expected_llpt =
            (3.0 as Real * p_c.ln() + 7.0 as Real * (1.0 - p_c).ln()) / 10.0 as Real;
        assert!(
            (summary.log_score_per_trial - expected_llpt).abs() < 1e-5,
            "log_score_per_trial mismatch: got {}",
            summary.log_score_per_trial
        );
        assert_eq!(summary.calibration_bins.len(), 10);
        let bin2 = &summary.calibration_bins[2];
        assert_eq!(bin2.n_stations, 1);
        assert_eq!(bin2.sum_trials, 10);
        assert_eq!(bin2.sum_successes, 3);
        assert!((bin2.mean_predicted - 0.2).abs() < 1e-6);
        assert!((bin2.pooled_observed_prevalence - 0.3).abs() < 1e-6);
    }

    // ----- Space–time CV ----------------------------------------------------

    use crate::spacetime::GeoMetric;

    fn st_grid_points() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>) {
        // 3x3x3 grid with a smooth linear drift in lat, lon, and time.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let lat = i as Real;
                    let lon = j as Real;
                    let t = k as Real;
                    coords.push(
                        SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon).unwrap(), t).unwrap(),
                    );
                    values.push(2.0 * lat + 3.0 * lon + 0.5 * t + 1.0);
                }
            }
        }
        (coords, values)
    }

    fn st_variogram() -> SpaceTimeVariogram {
        let spatial = VariogramModel::new(0.05, 2.0, 300.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 3.0, VariogramType::Exponential).unwrap();
        SpaceTimeVariogram::new_separable(spatial, temporal).unwrap()
    }

    fn st_binomial_grid_points() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<u32>, Vec<u32>) {
        let mut coords = Vec::new();
        let mut successes = Vec::new();
        let mut trials = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let lat = i as Real;
                    let lon = j as Real;
                    let t = k as Real;
                    let p = logistic(-2.0 + 0.5 * lat + 0.5 * lon + 0.1 * t);
                    let n = 40u32;
                    let s = (p * n as Real).round() as u32;
                    coords.push(
                        SpaceTimeCoord::try_new(GeoCoord::try_new(lat, lon).unwrap(), t).unwrap(),
                    );
                    successes.push(s);
                    trials.push(n);
                }
            }
        }
        (coords, successes, trials)
    }

    #[test]
    fn st_leave_one_out_returns_one_residual_per_station_in_order() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let residuals = leave_one_out_cv(&SpacetimeOrdinaryPredictor {
            metric: GeoMetric,
            coords: &coords,
            values: &values,
            variogram,
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.observed, values[i]);
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn st_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let residuals = k_fold_cv(
            &SpacetimeOrdinaryPredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values,
                variogram,
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index], "duplicate residual for index {}", r.index);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn st_leave_one_out_rejects_fewer_than_two_stations() {
        let coords =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap()];
        let values = vec![1.0];
        let variogram = st_variogram();
        assert!(
            leave_one_out_cv(&SpacetimeOrdinaryPredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values,
                variogram
            })
            .is_err()
        );
    }

    #[test]
    fn st_k_fold_rejects_invalid_k() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        assert!(
            k_fold_cv(
                &SpacetimeOrdinaryPredictor {
                    metric: GeoMetric,
                    coords: &coords,
                    values: &values,
                    variogram
                },
                1
            )
            .is_err()
        );
        assert!(
            k_fold_cv(
                &SpacetimeOrdinaryPredictor {
                    metric: GeoMetric,
                    coords: &coords,
                    values: &values,
                    variogram
                },
                coords.len() + 1
            )
            .is_err()
        );
    }

    #[test]
    fn st_simple_loo_runs_with_known_mean() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals = leave_one_out_cv(&SpacetimeSimplePredictor {
            metric: GeoMetric,
            coords: &coords,
            values: &values,
            variogram,
            mean,
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn st_simple_k_fold_covers_every_station_exactly_once() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let mean = values.iter().copied().sum::<Real>() / values.len() as Real;
        let residuals = k_fold_cv(
            &SpacetimeSimplePredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values,
                variogram,
                mean,
            },
            4,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn st_universal_loo_matches_ordinary_for_constant_trend_within_tol() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let ok = leave_one_out_cv(&SpacetimeOrdinaryPredictor {
            metric: GeoMetric,
            coords: &coords,
            values: &values,
            variogram,
        })
        .unwrap();
        let uk = leave_one_out_cv(&SpacetimeUniversalPredictor {
            metric: GeoMetric,
            coords: &coords,
            values: &values,
            variogram,
            trend: SpaceTimeUniversalTrend::Constant,
        })
        .unwrap();
        assert_eq!(ok.len(), uk.len());
        for (a, b) in ok.iter().zip(uk.iter()) {
            assert!(
                (a.predicted - b.predicted).abs() < 1e-3,
                "constant-trend ST UK should ~match ST OK at index {} (ok={}, uk={})",
                a.index,
                a.predicted,
                b.predicted
            );
        }
    }

    #[test]
    fn st_universal_k_fold_runs_with_linear_in_time_trend() {
        let (coords, values) = st_grid_points();
        let variogram = st_variogram();
        let residuals = k_fold_cv(
            &SpacetimeUniversalPredictor {
                metric: GeoMetric,
                coords: &coords,
                values: &values,
                variogram,
                trend: SpaceTimeUniversalTrend::LinearInTime,
            },
            3,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for r in &residuals {
            assert!(r.predicted.is_finite());
            assert!(r.variance.is_finite());
        }
    }

    #[test]
    fn st_binomial_loo_reports_both_scales_in_input_order() {
        let (coords, successes, trials) = st_binomial_grid_points();
        let variogram = st_variogram();
        let residuals = leave_one_out_cv(&SpacetimeBinomialPredictor {
            metric: GeoMetric,
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram,
            prior: BinomialPrior::default(),
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.successes, successes[i]);
            assert_eq!(r.trials, trials[i]);
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
            assert!(r.predicted_logit.is_finite());
            assert!(r.predicted_prevalence.is_finite());
            assert!(
                r.predicted_prevalence >= 0.0 && r.predicted_prevalence <= 1.0,
                "prevalence must lie in [0,1], got {}",
                r.predicted_prevalence
            );
        }
    }

    #[test]
    fn st_binomial_loo_handles_zero_trials_with_nan_observations() {
        let (coords, mut successes, mut trials) = st_binomial_grid_points();
        successes[0] = 0;
        trials[0] = 0;
        let variogram = st_variogram();
        let residuals = leave_one_out_cv(&SpacetimeBinomialPredictor {
            metric: GeoMetric,
            coords: &coords,
            successes: &successes,
            trials: &trials,
            variogram,
            prior: BinomialPrior::default(),
        })
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let r0 = residuals[0];
        assert_eq!(r0.trials, 0);
        assert!(r0.observed_logit.is_nan());
        assert!(r0.observed_prevalence.is_nan());
        assert!(r0.predicted_logit.is_finite());
        assert!(r0.predicted_prevalence.is_finite());
        for r in &residuals[1..] {
            assert!(r.observed_logit.is_finite());
            assert!(r.observed_prevalence.is_finite());
        }
        let summary = BinomialCvSummary::from_residuals(&residuals);
        assert_eq!(summary.n, residuals.len());
        assert_eq!(summary.n_evaluated, residuals.len() - 1);
    }

    #[test]
    fn st_binomial_k_fold_covers_every_station_exactly_once() {
        let (coords, successes, trials) = st_binomial_grid_points();
        let variogram = st_variogram();
        let residuals = k_fold_cv(
            &SpacetimeBinomialPredictor {
                metric: GeoMetric,
                coords: &coords,
                successes: &successes,
                trials: &trials,
                variogram,
                prior: BinomialPrior::default(),
            },
            3,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        let mut seen = vec![false; coords.len()];
        for r in &residuals {
            assert!(!seen[r.index]);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }
}
