//! Empirical space–time variogram.
//!
//! Bins pairs of observations by their spatial lag `h_s` and temporal lag `h_t` into a 2-D
//! grid and reports the weighted mean lag, empirical semivariance (via the chosen
//! [`EmpiricalEstimator`]), and pair count in each bin. Unpopulated bins are carried
//! through as zeros with `n_pairs == 0` so callers can mask them during fitting.

use std::num::NonZeroUsize;

use crate::Real;
use crate::error::KrigingError;
use crate::spacetime::coord::temporal_distance;
use crate::spacetime::dataset::SpaceTimeDataset;
use crate::spacetime::metric::SpatialMetric;
use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal};

/// 2-D (spatial × temporal) binned empirical variogram.
///
/// Storage is row-major: entry at spatial bin `i` and temporal bin `j` is at flat index
/// `i * n_temporal_bins + j`. Bins with `n_pairs == 0` have undefined `spatial_lags`,
/// `temporal_lags`, and `semivariances` entries (they're initialized to zero but should be
/// filtered by count).
#[derive(Debug, Clone)]
pub struct EmpiricalSpaceTimeVariogram {
    pub n_spatial_bins: usize,
    pub n_temporal_bins: usize,
    /// Mean spatial lag inside each bin (row-major, length `n_spatial_bins * n_temporal_bins`).
    pub spatial_lags: Vec<Real>,
    /// Mean temporal lag inside each bin (row-major, same length).
    pub temporal_lags: Vec<Real>,
    /// Empirical semivariance inside each bin (row-major, same length).
    pub semivariances: Vec<Real>,
    /// Number of pairs contributing to each bin (row-major, same length).
    pub n_pairs: Vec<usize>,
}

impl EmpiricalSpaceTimeVariogram {
    /// Flat index for (spatial bin `i`, temporal bin `j`).
    #[inline]
    pub fn index(&self, i: usize, j: usize) -> usize {
        i * self.n_temporal_bins + j
    }

    /// Total number of pair contributions across all bins.
    pub fn total_pairs(&self) -> usize {
        self.n_pairs.iter().sum()
    }
}

/// Binning and estimator configuration for a space–time empirical variogram.
///
/// If `max_spatial_distance` or `max_temporal_lag` is `None`, the corresponding limit is
/// inferred from the data with a two-pass sweep (mirrors
/// [`compute_empirical_variogram`](crate::compute_empirical_variogram)).
#[derive(Debug, Clone)]
pub struct SpaceTimeVariogramConfig {
    pub max_spatial_distance: Option<PositiveReal>,
    pub max_temporal_lag: Option<PositiveReal>,
    pub n_spatial_bins: NonZeroUsize,
    pub n_temporal_bins: NonZeroUsize,
    pub estimator: EmpiricalEstimator,
}

impl Default for SpaceTimeVariogramConfig {
    fn default() -> Self {
        Self {
            max_spatial_distance: None,
            max_temporal_lag: None,
            n_spatial_bins: NonZeroUsize::new(10).expect("10 != 0"),
            n_temporal_bins: NonZeroUsize::new(8).expect("8 != 0"),
            estimator: EmpiricalEstimator::Classical,
        }
    }
}

/// Compute the binned space–time empirical variogram for `dataset` using `metric` for the
/// spatial distance and the supplied binning config.
pub fn compute_empirical_spacetime_variogram<M: SpatialMetric>(
    metric: &M,
    dataset: &SpaceTimeDataset<M::Coord>,
    config: &SpaceTimeVariogramConfig,
) -> Result<EmpiricalSpaceTimeVariogram, KrigingError> {
    let coords = dataset.coords();
    let values = dataset.values();
    let n = coords.len();
    if n < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    let prepared: Vec<M::Prepared> = coords.iter().map(|c| metric.prepare(c.spatial)).collect();

    let (max_spatial, max_temporal) = resolve_max_lags(
        &prepared,
        coords,
        metric,
        config.max_spatial_distance,
        config.max_temporal_lag,
    )?;

    let n_s = config.n_spatial_bins.get();
    let n_t = config.n_temporal_bins.get();
    let bin_s = max_spatial / n_s as Real;
    let bin_t = max_temporal / n_t as Real;
    let len = n_s * n_t;
    let mut dist_s_sum = vec![0.0 as Real; len];
    let mut dist_t_sum = vec![0.0 as Real; len];
    let mut value_sums = vec![0.0 as Real; len];
    let mut counts = vec![0usize; len];
    let robust = matches!(config.estimator, EmpiricalEstimator::CressieHawkins);

    for i in 0..n {
        for j in (i + 1)..n {
            let hs = metric.distance(prepared[i], prepared[j]);
            if hs > max_spatial {
                continue;
            }
            let ht = temporal_distance(coords[i].time, coords[j].time);
            if ht > max_temporal {
                continue;
            }
            let bi = clamp_bin(hs, bin_s, n_s);
            let bj = clamp_bin(ht, bin_t, n_t);
            let idx = bi * n_t + bj;
            let dz = (values[i] - values[j]).abs();
            let g = if robust { dz.sqrt() } else { 0.5 * dz * dz };
            dist_s_sum[idx] += hs;
            dist_t_sum[idx] += ht;
            value_sums[idx] += g;
            counts[idx] += 1;
        }
    }

    let mut spatial_lags = vec![0.0 as Real; len];
    let mut temporal_lags = vec![0.0 as Real; len];
    let mut semivariances = vec![0.0 as Real; len];
    for idx in 0..len {
        if counts[idx] == 0 {
            continue;
        }
        let n_i = counts[idx] as Real;
        let g = if robust {
            // Cressie–Hawkins: γ̂(h) = 0.5·(mean(|Δz|^{1/2}))⁴ / (0.457 + 0.494/N + 0.045/N²).
            let mean_sqrt = value_sums[idx] / n_i;
            let numer = mean_sqrt.powi(4);
            let denom = 0.457 + 0.494 / n_i + 0.045 / (n_i * n_i);
            0.5 * numer / denom
        } else {
            value_sums[idx] / n_i
        };
        spatial_lags[idx] = dist_s_sum[idx] / n_i;
        temporal_lags[idx] = dist_t_sum[idx] / n_i;
        semivariances[idx] = g;
    }

    if counts.iter().all(|c| *c == 0) {
        return Err(KrigingError::FittingError(
            "no pairs in selected space-time lag range".to_string(),
        ));
    }

    Ok(EmpiricalSpaceTimeVariogram {
        n_spatial_bins: n_s,
        n_temporal_bins: n_t,
        spatial_lags,
        temporal_lags,
        semivariances,
        n_pairs: counts,
    })
}

fn clamp_bin(value: Real, bin_width: Real, n_bins: usize) -> usize {
    let mut bin = (value / bin_width.max(Real::MIN_POSITIVE)).floor() as usize;
    if bin >= n_bins {
        bin = n_bins - 1;
    }
    bin
}

fn resolve_max_lags<M: SpatialMetric>(
    prepared: &[M::Prepared],
    coords: &[crate::spacetime::SpaceTimeCoord<M::Coord>],
    metric: &M,
    supplied_spatial: Option<PositiveReal>,
    supplied_temporal: Option<PositiveReal>,
) -> Result<(Real, Real), KrigingError> {
    let mut max_spatial = supplied_spatial.map(|v| v.get()).unwrap_or(0.0);
    let mut max_temporal = supplied_temporal.map(|v| v.get()).unwrap_or(0.0);
    if supplied_spatial.is_none() || supplied_temporal.is_none() {
        for i in 0..prepared.len() {
            for j in (i + 1)..prepared.len() {
                if supplied_spatial.is_none() {
                    let hs = metric.distance(prepared[i], prepared[j]);
                    if hs > max_spatial {
                        max_spatial = hs;
                    }
                }
                if supplied_temporal.is_none() {
                    let ht = temporal_distance(coords[i].time, coords[j].time);
                    if ht > max_temporal {
                        max_temporal = ht;
                    }
                }
            }
        }
    }
    if max_spatial <= 0.0 {
        return Err(KrigingError::FittingError(
            "max spatial distance must be positive".to_string(),
        ));
    }
    if max_temporal <= 0.0 {
        return Err(KrigingError::FittingError(
            "max temporal lag must be positive".to_string(),
        ));
    }
    Ok((max_spatial, max_temporal))
}

/// Extract the spatial marginal 1-D empirical variogram: the row of bins with the smallest
/// temporal lag (bin `j = 0`). Used by [`fit_spacetime_variogram`](crate::spacetime::fitting::fit_spacetime_variogram).
pub fn spatial_marginal(
    empirical: &EmpiricalSpaceTimeVariogram,
) -> crate::variogram::empirical::EmpiricalVariogram {
    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for i in 0..empirical.n_spatial_bins {
        let idx = empirical.index(i, 0);
        if empirical.n_pairs[idx] == 0 {
            continue;
        }
        distances.push(empirical.spatial_lags[idx]);
        semivariances.push(empirical.semivariances[idx]);
        n_pairs.push(empirical.n_pairs[idx]);
    }
    crate::variogram::empirical::EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    }
}

/// Extract the temporal marginal 1-D empirical variogram: the column of bins with the
/// smallest spatial lag (bin `i = 0`). Used by
/// [`fit_spacetime_variogram`](crate::spacetime::fitting::fit_spacetime_variogram).
pub fn temporal_marginal(
    empirical: &EmpiricalSpaceTimeVariogram,
) -> crate::variogram::empirical::EmpiricalVariogram {
    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for j in 0..empirical.n_temporal_bins {
        let idx = empirical.index(0, j);
        if empirical.n_pairs[idx] == 0 {
            continue;
        }
        distances.push(empirical.temporal_lags[idx]);
        semivariances.push(empirical.semivariances[idx]);
        n_pairs.push(empirical.n_pairs[idx]);
    }
    crate::variogram::empirical::EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::SpaceTimeCoord;
    use crate::spacetime::metric::GeoMetric;

    fn make_dataset() -> SpaceTimeDataset<GeoCoord> {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..5 {
            for t in 0..4 {
                coords.push(SpaceTimeCoord::new(
                    GeoCoord::try_new(i as Real * 0.1, 0.0).unwrap(),
                    t as Real,
                ));
                values.push((i as Real) + 0.5 * (t as Real));
            }
        }
        SpaceTimeDataset::new(coords, values).unwrap()
    }

    #[test]
    fn default_config_produces_non_empty_variogram() {
        let ds = make_dataset();
        let config = SpaceTimeVariogramConfig::default();
        let emp = compute_empirical_spacetime_variogram(&GeoMetric, &ds, &config).unwrap();
        assert_eq!(emp.n_spatial_bins, 10);
        assert_eq!(emp.n_temporal_bins, 8);
        assert!(emp.total_pairs() > 0);
        // γ̂(h_s, 0) should increase with h_s for this linearly-increasing-in-space field.
    }

    #[test]
    fn pair_accounting_is_consistent_with_explicit_loop() {
        let ds = make_dataset();
        let max_s = PositiveReal::try_new(100.0).unwrap();
        let max_t = PositiveReal::try_new(10.0).unwrap();
        let config = SpaceTimeVariogramConfig {
            max_spatial_distance: Some(max_s),
            max_temporal_lag: Some(max_t),
            n_spatial_bins: NonZeroUsize::new(4).unwrap(),
            n_temporal_bins: NonZeroUsize::new(4).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let emp = compute_empirical_spacetime_variogram(&GeoMetric, &ds, &config).unwrap();

        let n = ds.len();
        let mut expected = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let hs = crate::distance::haversine_distance(
                    ds.coords()[i].spatial,
                    ds.coords()[j].spatial,
                );
                let ht = (ds.coords()[i].time - ds.coords()[j].time).abs();
                if hs <= max_s.get() && ht <= max_t.get() {
                    expected += 1;
                }
            }
        }
        assert_eq!(emp.total_pairs(), expected);
    }

    #[test]
    fn cressie_hawkins_is_robust_to_outlier() {
        // Same structure as the 1-D test: smooth linear field with one outlier.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..6 {
            for t in 0..4 {
                coords.push(SpaceTimeCoord::new(
                    GeoCoord::try_new(i as Real * 0.1, 0.0).unwrap(),
                    t as Real,
                ));
                values.push((i + t) as Real);
            }
        }
        *values.last_mut().unwrap() = 1000.0;
        let ds = SpaceTimeDataset::new(coords, values).unwrap();
        let config_classical = SpaceTimeVariogramConfig {
            estimator: EmpiricalEstimator::Classical,
            n_spatial_bins: NonZeroUsize::new(5).unwrap(),
            n_temporal_bins: NonZeroUsize::new(4).unwrap(),
            ..Default::default()
        };
        let config_robust = SpaceTimeVariogramConfig {
            estimator: EmpiricalEstimator::CressieHawkins,
            ..config_classical.clone()
        };
        let classical =
            compute_empirical_spacetime_variogram(&GeoMetric, &ds, &config_classical).unwrap();
        let robust =
            compute_empirical_spacetime_variogram(&GeoMetric, &ds, &config_robust).unwrap();
        let max_classical = classical
            .semivariances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        let max_robust = robust
            .semivariances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        assert!(
            max_robust < max_classical * 0.8,
            "robust max {} should be notably smaller than classical max {}",
            max_robust,
            max_classical
        );
    }

    #[test]
    fn spatial_marginal_monotonic_for_linear_in_space_field() {
        // z(i, t) = i + 0.5*t. Along a row with t ≈ 0 the variogram should increase with
        // spatial lag; the first-temporal-bin extraction is our proxy for the spatial
        // marginal.
        let ds = make_dataset();
        let config = SpaceTimeVariogramConfig {
            n_spatial_bins: NonZeroUsize::new(4).unwrap(),
            n_temporal_bins: NonZeroUsize::new(4).unwrap(),
            ..Default::default()
        };
        let emp = compute_empirical_spacetime_variogram(&GeoMetric, &ds, &config).unwrap();
        let marginal = spatial_marginal(&emp);
        assert!(!marginal.distances.is_empty());
        let g: Vec<_> = marginal.semivariances.to_vec();
        assert!(
            g.windows(2).all(|w| w[1] >= w[0] - 1e-6),
            "spatial marginal should be non-decreasing, got {:?}",
            g
        );
    }
}
