//! Aggregation helpers over conditional simulation ensembles.
//!
//! These helpers are designed to be applied to the **flat row-major** ensemble
//! buffers produced by [`crate::simulation::conditional_simulate_many`] and
//! [`crate::simulation::conditional_simulate_many_binomial`] (rows = realizations,
//! columns = targets, length = `n_realizations * n_targets`).
//!
//! The motivating use case is **polygon-level reporting** for disease prevalence
//! mapping: a regulator typically wants the posterior distribution of
//! "prevalence inside this district" rather than 10 000 per-cell distributions.
//! Given a set of cell weights `(idx_i, w_i)` representing a polygon, this
//! module computes
//!
//! ```text
//!     P_k = Σ_i (w_i · x[k, idx_i]) / Σ_i w_i        for k = 0 .. n_realizations
//! ```
//!
//! and then reduces the resulting `n_realizations`-length distribution to the
//! requested point estimates and quantiles.
//!
//! Weights may be unnormalized (we divide by `Σ w_i`); they must be finite,
//! non-negative, and sum to a strictly positive value.

use crate::Real;
use crate::error::KrigingError;

/// Per-realization weighted means produced by
/// [`polygon_weighted_mean_per_realization`].
pub type PolygonRealizations = Vec<Real>;

/// Compute the polygon-weighted mean of every realization in a flat row-major
/// ensemble buffer.
///
/// The returned vector has length `n_realizations` and entry `k` equals
/// `Σ_i (w_i · samples[k * n_targets + idx_i]) / Σ_i w_i`.
///
/// # Errors
///
/// - `DimensionMismatch` if `samples.len() != n_realizations * n_targets` or if
///   `indices.len() != weights.len()`.
/// - `InsufficientData(1)` if the polygon has no cells.
/// - `InvalidInput` if any index is out of range, any weight is non-finite or
///   negative, or the total weight is not strictly positive.
pub fn polygon_weighted_mean_per_realization(
    samples: &[Real],
    n_realizations: usize,
    n_targets: usize,
    indices: &[usize],
    weights: &[Real],
) -> Result<PolygonRealizations, KrigingError> {
    validate_ensemble_shape(samples.len(), n_realizations, n_targets)?;
    let total = validate_polygon(indices, weights, n_targets)?;
    let inv = (1.0 as Real) / total;
    let mut out = vec![0.0 as Real; n_realizations];
    for (k, slot) in out.iter_mut().enumerate() {
        let row_off = k * n_targets;
        let mut acc: Real = 0.0;
        for (idx, w) in indices.iter().zip(weights.iter()) {
            acc += *w * samples[row_off + *idx];
        }
        *slot = acc * inv;
    }
    Ok(out)
}

/// Summary of a polygon's posterior distribution across an ensemble.
///
/// `quantiles[i]` holds the value at probability `quantile_probs[i]` (in the
/// caller-supplied order). All scalars are reported in the **scale of the
/// supplied ensemble buffer** (logit-scale buffers produce logit-scale
/// summaries, prevalence buffers produce prevalence summaries, etc.).
#[derive(Debug, Clone)]
pub struct PolygonAggregationSummary {
    /// Number of realizations summarized.
    pub n_realizations: usize,
    /// Sum of polygon weights (useful for weighting downstream multi-polygon
    /// roll-ups, e.g. population-weighted national rates).
    pub total_weight: Real,
    /// Empirical mean of the per-realization weighted means.
    pub mean: Real,
    /// Empirical sample variance (denominator `n_realizations - 1`). `None`
    /// when `n_realizations < 2` (variance is undefined for a single sample).
    pub variance: Option<Real>,
    /// Quantile values in the same order as the requested probabilities.
    pub quantiles: Vec<(Real, Real)>,
}

/// Reduce a polygon's per-realization weighted means to a point-estimate
/// summary. See [`polygon_weighted_mean_per_realization`] for the underlying
/// per-realization computation.
///
/// `quantile_probs` may be empty (no quantiles will be reported). All
/// probabilities must satisfy `0 <= p <= 1`. Quantiles are estimated with the
/// "linear" interpolation rule (numpy's default, R's type 7).
///
/// # Errors
///
/// In addition to the errors propagated from
/// [`polygon_weighted_mean_per_realization`], `InvalidInput` is returned if any
/// probability lies outside `[0, 1]` or is non-finite.
pub fn polygon_weighted_summary(
    samples: &[Real],
    n_realizations: usize,
    n_targets: usize,
    indices: &[usize],
    weights: &[Real],
    quantile_probs: &[Real],
) -> Result<PolygonAggregationSummary, KrigingError> {
    let realizations = polygon_weighted_mean_per_realization(
        samples,
        n_realizations,
        n_targets,
        indices,
        weights,
    )?;
    let total_weight = weights.iter().copied().sum::<Real>();

    let mean = if realizations.is_empty() {
        0.0
    } else {
        realizations.iter().copied().sum::<Real>() / (realizations.len() as Real)
    };

    let variance = if realizations.len() >= 2 {
        let inv = (1.0 as Real) / ((realizations.len() - 1) as Real);
        let v = realizations
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<Real>()
            * inv;
        Some(v)
    } else {
        None
    };

    let quantiles = if quantile_probs.is_empty() {
        Vec::new()
    } else {
        compute_quantiles_linear(&realizations, quantile_probs)?
    };

    Ok(PolygonAggregationSummary {
        n_realizations,
        total_weight,
        mean,
        variance,
        quantiles,
    })
}

/// Compute polygon-weighted summaries for many polygons over the same ensemble
/// buffer in a single call. Returns one summary per polygon, in the same order
/// as `polygons`. Each polygon is `(indices, weights)`; see
/// [`polygon_weighted_summary`] for per-polygon error semantics.
pub fn polygon_weighted_summaries_batch(
    samples: &[Real],
    n_realizations: usize,
    n_targets: usize,
    polygons: &[(&[usize], &[Real])],
    quantile_probs: &[Real],
) -> Result<Vec<PolygonAggregationSummary>, KrigingError> {
    validate_ensemble_shape(samples.len(), n_realizations, n_targets)?;
    let mut out = Vec::with_capacity(polygons.len());
    for (indices, weights) in polygons.iter() {
        let s = polygon_weighted_summary(
            samples,
            n_realizations,
            n_targets,
            indices,
            weights,
            quantile_probs,
        )?;
        out.push(s);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

fn validate_ensemble_shape(
    sample_len: usize,
    n_realizations: usize,
    n_targets: usize,
) -> Result<(), KrigingError> {
    if n_realizations == 0 {
        return Err(KrigingError::InsufficientData(1));
    }
    if n_targets == 0 {
        return Err(KrigingError::InsufficientData(1));
    }
    let expected = n_realizations
        .checked_mul(n_targets)
        .ok_or_else(|| KrigingError::DimensionMismatch("ensemble size overflows usize".into()))?;
    if sample_len != expected {
        return Err(KrigingError::DimensionMismatch(format!(
            "ensemble buffer length ({}) must equal n_realizations * n_targets ({})",
            sample_len, expected
        )));
    }
    Ok(())
}

fn validate_polygon(
    indices: &[usize],
    weights: &[Real],
    n_targets: usize,
) -> Result<Real, KrigingError> {
    if indices.is_empty() {
        return Err(KrigingError::InsufficientData(1));
    }
    if indices.len() != weights.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "polygon indices ({}) and weights ({}) must have the same length",
            indices.len(),
            weights.len()
        )));
    }
    let mut total: Real = 0.0;
    for (i, w) in indices.iter().zip(weights.iter()) {
        if *i >= n_targets {
            return Err(KrigingError::InvalidInput(format!(
                "polygon cell index {} is out of range (n_targets = {})",
                i, n_targets
            )));
        }
        if !w.is_finite() {
            return Err(KrigingError::InvalidInput(
                "polygon weights must be finite".into(),
            ));
        }
        if *w < 0.0 {
            return Err(KrigingError::InvalidInput(
                "polygon weights must be non-negative".into(),
            ));
        }
        total += *w;
    }
    if total <= 0.0 {
        return Err(KrigingError::InvalidInput(
            "polygon weights must sum to a strictly positive value".into(),
        ));
    }
    Ok(total)
}

fn compute_quantiles_linear(
    values: &[Real],
    probs: &[Real],
) -> Result<Vec<(Real, Real)>, KrigingError> {
    for p in probs {
        if !p.is_finite() || *p < 0.0 || *p > 1.0 {
            return Err(KrigingError::InvalidInput(format!(
                "quantile probabilities must be in [0, 1] (got {})",
                p
            )));
        }
    }
    if values.is_empty() {
        return Err(KrigingError::InsufficientData(1));
    }
    let mut sorted: Vec<Real> = values.to_vec();
    // Total ordering for floats; ensemble samples should always be finite, but
    // partial_cmp would panic on NaN — reject NaNs explicitly for clarity.
    if sorted.iter().any(|v| v.is_nan()) {
        return Err(KrigingError::InvalidInput(
            "ensemble values contained NaN; cannot compute quantiles".into(),
        ));
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("checked: no NaN"));
    let last_idx = (sorted.len() - 1) as Real;
    let mut out = Vec::with_capacity(probs.len());
    for p in probs {
        let pos = last_idx * *p;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(sorted.len() - 1);
        let frac = pos - (lo as Real);
        let v = if hi == lo {
            sorted[lo]
        } else {
            sorted[lo] + (sorted[hi] - sorted[lo]) * frac
        };
        out.push((*p, v));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(rows: usize, cols: usize, f: impl Fn(usize, usize) -> Real) -> Vec<Real> {
        let mut v = Vec::with_capacity(rows * cols);
        for k in 0..rows {
            for j in 0..cols {
                v.push(f(k, j));
            }
        }
        v
    }

    #[test]
    fn polygon_weighted_mean_uniform_weights_matches_simple_mean() {
        // 3 realizations, 4 targets; polygon = all 4 cells equally weighted.
        let samples = buf(3, 4, |k, j| (k * 10 + j) as Real);
        let indices = vec![0, 1, 2, 3];
        let weights = vec![1.0; 4];
        let per =
            polygon_weighted_mean_per_realization(&samples, 3, 4, &indices, &weights).unwrap();
        // Row k = [10k+0, 10k+1, 10k+2, 10k+3] -> mean = 10k + 1.5.
        assert_eq!(per.len(), 3);
        assert!((per[0] - 1.5).abs() < 1e-6);
        assert!((per[1] - 11.5).abs() < 1e-6);
        assert!((per[2] - 21.5).abs() < 1e-6);
    }

    #[test]
    fn polygon_weighted_mean_respects_unnormalized_weights() {
        // Two cells: weight (1, 3) -> 0.25 * x_0 + 0.75 * x_1.
        let samples = vec![0.0, 4.0, 1.0, 5.0]; // 2 realizations × 2 targets
        let per =
            polygon_weighted_mean_per_realization(&samples, 2, 2, &[0, 1], &[1.0, 3.0]).unwrap();
        assert!((per[0] - 3.0).abs() < 1e-6); // 0.25*0 + 0.75*4
        assert!((per[1] - 4.0).abs() < 1e-6); // 0.25*1 + 0.75*5
    }

    #[test]
    fn polygon_weighted_summary_reports_mean_variance_and_quantiles() {
        // 5 realizations, 2 targets. Polygon uses only target 0.
        // Per-realization values (target 0): 0, 1, 2, 3, 4.
        let samples: Vec<Real> = (0..10).map(|i| (i / 2) as Real).collect();
        let s = polygon_weighted_summary(&samples, 5, 2, &[0], &[1.0], &[0.0, 0.5, 1.0]).unwrap();
        assert_eq!(s.n_realizations, 5);
        assert!((s.total_weight - 1.0).abs() < 1e-6);
        assert!((s.mean - 2.0).abs() < 1e-6);
        // Sample variance of [0,1,2,3,4] with n-1 denom = 10/4 = 2.5
        let var = s.variance.expect("variance with 5 samples");
        assert!((var - 2.5).abs() < 1e-6);
        assert_eq!(s.quantiles.len(), 3);
        assert!((s.quantiles[0].1 - 0.0).abs() < 1e-6);
        assert!((s.quantiles[1].1 - 2.0).abs() < 1e-6);
        assert!((s.quantiles[2].1 - 4.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_weighted_summary_handles_single_realization() {
        let s = polygon_weighted_summary(&[7.0, 9.0], 1, 2, &[0, 1], &[1.0, 1.0], &[0.5]).unwrap();
        assert!((s.mean - 8.0).abs() < 1e-6);
        assert!(s.variance.is_none());
        assert!((s.quantiles[0].1 - 8.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_weighted_summary_rejects_bad_inputs() {
        let samples = vec![0.0; 6]; // 3 × 2
        // wrong shape
        assert!(polygon_weighted_summary(&samples, 3, 3, &[0], &[1.0], &[]).is_err());
        // empty polygon
        assert!(polygon_weighted_summary(&samples, 3, 2, &[], &[], &[]).is_err());
        // index out of range
        assert!(polygon_weighted_summary(&samples, 3, 2, &[2], &[1.0], &[]).is_err());
        // negative weight
        assert!(polygon_weighted_summary(&samples, 3, 2, &[0], &[-1.0], &[]).is_err());
        // zero total weight
        assert!(polygon_weighted_summary(&samples, 3, 2, &[0, 1], &[0.0, 0.0], &[]).is_err());
        // bad quantile
        assert!(polygon_weighted_summary(&samples, 3, 2, &[0], &[1.0], &[1.5]).is_err());
    }

    #[test]
    fn polygon_weighted_summaries_batch_matches_individual_calls() {
        let samples = buf(4, 3, |k, j| (k as Real) - (j as Real));
        let polys: &[(&[usize], &[Real])] = &[
            (&[0, 2], &[1.0, 1.0]),
            (&[1], &[2.0]),
            (&[0, 1, 2], &[1.0, 2.0, 3.0]),
        ];
        let probs = [0.25, 0.75];
        let batch = polygon_weighted_summaries_batch(&samples, 4, 3, polys, &probs).unwrap();
        for (i, (idx, w)) in polys.iter().enumerate() {
            let single = polygon_weighted_summary(&samples, 4, 3, idx, w, &probs).unwrap();
            assert!((batch[i].mean - single.mean).abs() < 1e-9);
            for q in 0..probs.len() {
                assert!((batch[i].quantiles[q].1 - single.quantiles[q].1).abs() < 1e-9);
            }
        }
    }
}
