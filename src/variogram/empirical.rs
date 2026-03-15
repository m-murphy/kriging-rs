use std::num::NonZeroUsize;

use crate::geo_dataset::GeoDataset;
use crate::Real;
use crate::distance::haversine_distance;
use crate::error::KrigingError;

/// A positive real number (> 0), enforced at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveReal(Real);

impl PositiveReal {
    pub fn try_new(x: Real) -> Result<Self, KrigingError> {
        if x > 0.0 && x.is_finite() {
            Ok(Self(x))
        } else {
            Err(KrigingError::FittingError(
                "value must be finite and positive".to_string(),
            ))
        }
    }

    #[inline]
    pub fn get(self) -> Real {
        self.0
    }
}

impl std::ops::Deref for PositiveReal {
    type Target = Real;
    #[inline]
    fn deref(&self) -> &Real {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct EmpiricalVariogram {
    pub distances: Vec<Real>,
    pub semivariances: Vec<Real>,
    pub n_pairs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct VariogramConfig {
    pub max_distance: Option<PositiveReal>,
    pub n_bins: NonZeroUsize,
}

impl Default for VariogramConfig {
    fn default() -> Self {
        Self {
            max_distance: None,
            n_bins: NonZeroUsize::new(12).expect("12 != 0"),
        }
    }
}

pub fn compute_empirical_variogram(
    dataset: &GeoDataset,
    config: &VariogramConfig,
) -> Result<EmpiricalVariogram, KrigingError> {
    let coords = dataset.coords();
    let values = dataset.values();
    let n = coords.len();
    let n_bins = config.n_bins.get();
    let mut dist_sums = vec![0.0; n_bins];
    let mut semi_sums = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];

    if let Some(max_dist) = config.max_distance {
        let max_dist = max_dist.get();
        let bin_width = max_dist / n_bins as Real;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = haversine_distance(coords[i], coords[j]);
                if d > max_dist {
                    continue;
                }
                let g = 0.5 * (values[i] - values[j]).powi(2);
                let mut bin = (d / bin_width).floor() as usize;
                if bin >= n_bins {
                    bin = n_bins - 1;
                }
                dist_sums[bin] += d;
                semi_sums[bin] += g;
                counts[bin] += 1;
            }
        }
    } else {
        let mut pairs = Vec::new();
        let mut max_observed: Real = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = haversine_distance(coords[i], coords[j]);
                let g = 0.5 * (values[i] - values[j]).powi(2);
                max_observed = max_observed.max(d);
                pairs.push((d, g));
            }
        }
        if max_observed <= 0.0 {
            return Err(KrigingError::FittingError(
                "max distance must be positive".to_string(),
            ));
        }
        let bin_width = max_observed / n_bins as Real;
        for (d, g) in pairs {
            let mut bin = (d / bin_width).floor() as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            dist_sums[bin] += d;
            semi_sums[bin] += g;
            counts[bin] += 1;
        }
    }

    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for i in 0..n_bins {
        if counts[i] == 0 {
            continue;
        }
        distances.push(dist_sums[i] / counts[i] as Real);
        semivariances.push(semi_sums[i] / counts[i] as Real);
        n_pairs.push(counts[i]);
    }

    if distances.is_empty() {
        return Err(KrigingError::FittingError(
            "no pairs in selected distance range".to_string(),
        ));
    }

    Ok(EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;

    #[test]
    fn empirical_variogram_has_non_empty_bins() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 2.0, 3.0];
        let dataset = GeoDataset::new(coords, values).unwrap();
        let out = compute_empirical_variogram(
            &dataset,
            &VariogramConfig {
                max_distance: None,
                n_bins: NonZeroUsize::new(6).unwrap(),
            },
        )
        .expect("empirical variogram should compute");
        assert!(!out.distances.is_empty());
        assert_eq!(out.distances.len(), out.semivariances.len());
        assert_eq!(out.distances.len(), out.n_pairs.len());
    }

    #[test]
    fn empirical_variogram_preserves_pair_accounting_with_fixed_max_distance() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 0.5).unwrap(),
            GeoCoord::try_new(0.5, 0.0).unwrap(),
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let max_distance = PositiveReal::try_new(500.0).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let out = compute_empirical_variogram(
            &dataset,
            &VariogramConfig {
                max_distance: Some(max_distance),
                n_bins: NonZeroUsize::new(8).unwrap(),
            },
        )
        .expect("empirical variogram should compute");

        let mut expected_pair_count = 0usize;
        let coords = dataset.coords();
        for i in 0..coords.len() {
            for j in (i + 1)..coords.len() {
                if haversine_distance(coords[i], coords[j]) <= max_distance.get() {
                    expected_pair_count += 1;
                }
            }
        }
        let observed_pair_count = out.n_pairs.iter().sum::<usize>();
        assert_eq!(observed_pair_count, expected_pair_count);
        assert_eq!(out.distances.len(), out.semivariances.len());
        assert_eq!(out.distances.len(), out.n_pairs.len());
        assert!(
            out.distances
                .iter()
                .all(|d| *d >= 0.0 && *d <= max_distance.get())
        );
        assert!(out.semivariances.iter().all(|g| g.is_finite() && *g >= 0.0));
    }
}
