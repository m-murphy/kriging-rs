use crate::Real;
use crate::distance::{GeoCoord, haversine_distance};
use crate::error::KrigingError;

#[derive(Debug, Clone)]
pub struct EmpiricalVariogram {
    pub distances: Vec<Real>,
    pub semivariances: Vec<Real>,
    pub n_pairs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct VariogramConfig {
    pub max_distance: Option<Real>,
    pub n_bins: usize,
}

impl Default for VariogramConfig {
    fn default() -> Self {
        Self {
            max_distance: None,
            n_bins: 12,
        }
    }
}

pub fn compute_empirical_variogram(
    coords: &[GeoCoord],
    values: &[Real],
    config: &VariogramConfig,
) -> Result<EmpiricalVariogram, KrigingError> {
    if coords.len() != values.len() {
        return Err(KrigingError::DimensionMismatch(
            "coords and values length must match".to_string(),
        ));
    }
    if coords.len() < 2 {
        return Err(KrigingError::InsufficientData(2));
    }
    if config.n_bins == 0 {
        return Err(KrigingError::FittingError(
            "n_bins must be at least 1".to_string(),
        ));
    }

    let n = coords.len();
    let mut dist_sums = vec![0.0; config.n_bins];
    let mut semi_sums = vec![0.0; config.n_bins];
    let mut counts = vec![0usize; config.n_bins];

    if let Some(max_dist) = config.max_distance {
        if max_dist <= 0.0 {
            return Err(KrigingError::FittingError(
                "max distance must be positive".to_string(),
            ));
        }
        let bin_width = max_dist / config.n_bins as Real;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = haversine_distance(coords[i], coords[j]);
                if d > max_dist {
                    continue;
                }
                let g = 0.5 * (values[i] - values[j]).powi(2);
                let mut bin = (d / bin_width).floor() as usize;
                if bin >= config.n_bins {
                    bin = config.n_bins - 1;
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
        let bin_width = max_observed / config.n_bins as Real;
        for (d, g) in pairs {
            let mut bin = (d / bin_width).floor() as usize;
            if bin >= config.n_bins {
                bin = config.n_bins - 1;
            }
            dist_sums[bin] += d;
            semi_sums[bin] += g;
            counts[bin] += 1;
        }
    }

    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for i in 0..config.n_bins {
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

    #[test]
    fn empirical_variogram_has_non_empty_bins() {
        let coords = vec![
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord { lat: 0.0, lon: 1.0 },
            GeoCoord { lat: 1.0, lon: 0.0 },
            GeoCoord { lat: 1.0, lon: 1.0 },
        ];
        let values = vec![1.0, 2.0, 2.0, 3.0];
        let out = compute_empirical_variogram(
            &coords,
            &values,
            &VariogramConfig {
                max_distance: None,
                n_bins: 6,
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
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord { lat: 0.0, lon: 0.5 },
            GeoCoord { lat: 0.5, lon: 0.0 },
            GeoCoord { lat: 0.5, lon: 0.5 },
            GeoCoord { lat: 1.0, lon: 1.0 },
        ];
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let max_distance = 500.0;
        let n_bins = 8;
        let out = compute_empirical_variogram(
            &coords,
            &values,
            &VariogramConfig {
                max_distance: Some(max_distance),
                n_bins,
            },
        )
        .expect("empirical variogram should compute");

        let mut expected_pair_count = 0usize;
        for i in 0..coords.len() {
            for j in (i + 1)..coords.len() {
                if haversine_distance(coords[i], coords[j]) <= max_distance {
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
                .all(|d| *d >= 0.0 && *d <= max_distance)
        );
        assert!(out.semivariances.iter().all(|g| g.is_finite() && *g >= 0.0));
    }
}
