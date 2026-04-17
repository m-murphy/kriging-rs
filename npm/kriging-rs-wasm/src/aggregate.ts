/**
 * Generic aggregators over flat row-major ensemble buffers produced by the
 * `conditionalSimulateMany*` family.
 *
 * Every function in this module accepts a row-major `Float64Array` of length
 * `nRealizations * nTargets` (where row `k` = realization `k` over all targets)
 * and reduces along the realization axis to produce per-target summaries:
 *
 * - {@link ensembleMean} — point-wise mean.
 * - {@link ensembleVariance} — point-wise sample variance (unbiased, `n - 1` denominator).
 * - {@link ensembleQuantiles} — selected quantiles via linear interpolation, columnwise.
 * - {@link ensembleExceedanceProbability} — P(value > threshold), columnwise.
 *
 * These helpers are pure JavaScript so they work identically in the Node test
 * suite and in browser bundles. They are deliberately allocation-light and reuse
 * a single temporary `Float64Array` per column when sorting is required.
 *
 * @module
 */

import { KrigingError } from "./errors.js";

function requirePositiveInt(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new KrigingError(`${label} must be a positive integer`, {
      code: "invalid_input",
    });
  }
  return value;
}

function requireDimensions(
  samples: Float64Array,
  nRealizations: number,
  nTargets: number
): void {
  if (samples.length !== nRealizations * nTargets) {
    throw new KrigingError(
      `samples length (${samples.length}) must equal nRealizations * nTargets (${nRealizations * nTargets})`,
      { code: "mismatched_arrays" }
    );
  }
}

/**
 * Compute the per-target mean across realizations.
 *
 * @param samples Row-major `Float64Array(nRealizations * nTargets)`.
 * @param nRealizations Number of realizations (rows).
 * @param nTargets Number of targets per realization (columns).
 * @returns A `Float64Array(nTargets)` with the per-column mean.
 */
export function ensembleMean(
  samples: Float64Array,
  nRealizations: number,
  nTargets: number
): Float64Array {
  requirePositiveInt(nRealizations, "nRealizations");
  requirePositiveInt(nTargets, "nTargets");
  requireDimensions(samples, nRealizations, nTargets);
  const out = new Float64Array(nTargets);
  for (let k = 0; k < nRealizations; k++) {
    const off = k * nTargets;
    for (let j = 0; j < nTargets; j++) {
      out[j] += samples[off + j];
    }
  }
  const inv = 1 / nRealizations;
  for (let j = 0; j < nTargets; j++) out[j] *= inv;
  return out;
}

/**
 * Compute the per-target unbiased sample variance across realizations
 * (denominator `nRealizations - 1`). Requires at least 2 realizations.
 */
export function ensembleVariance(
  samples: Float64Array,
  nRealizations: number,
  nTargets: number
): Float64Array {
  requirePositiveInt(nRealizations, "nRealizations");
  requirePositiveInt(nTargets, "nTargets");
  if (nRealizations < 2) {
    throw new KrigingError(
      "ensembleVariance requires nRealizations >= 2 (unbiased estimator)",
      { code: "too_few_points" }
    );
  }
  requireDimensions(samples, nRealizations, nTargets);
  const mean = ensembleMean(samples, nRealizations, nTargets);
  const out = new Float64Array(nTargets);
  for (let k = 0; k < nRealizations; k++) {
    const off = k * nTargets;
    for (let j = 0; j < nTargets; j++) {
      const d = samples[off + j] - mean[j];
      out[j] += d * d;
    }
  }
  const inv = 1 / (nRealizations - 1);
  for (let j = 0; j < nTargets; j++) out[j] *= inv;
  return out;
}

/**
 * Compute the per-target empirical exceedance probability `P(value > threshold)`
 * across realizations. Returned values lie in `[0, 1]`.
 */
export function ensembleExceedanceProbability(
  samples: Float64Array,
  nRealizations: number,
  nTargets: number,
  threshold: number
): Float64Array {
  requirePositiveInt(nRealizations, "nRealizations");
  requirePositiveInt(nTargets, "nTargets");
  requireDimensions(samples, nRealizations, nTargets);
  if (!Number.isFinite(threshold)) {
    throw new KrigingError("threshold must be a finite number", {
      code: "invalid_input",
    });
  }
  const out = new Float64Array(nTargets);
  for (let k = 0; k < nRealizations; k++) {
    const off = k * nTargets;
    for (let j = 0; j < nTargets; j++) {
      if (samples[off + j] > threshold) out[j] += 1;
    }
  }
  const inv = 1 / nRealizations;
  for (let j = 0; j < nTargets; j++) out[j] *= inv;
  return out;
}

/**
 * Compute one or more per-target quantiles across realizations using linear
 * interpolation between adjacent realizations (numpy-style "linear" type 7).
 *
 * @param samples Row-major `Float64Array(nRealizations * nTargets)`.
 * @param nRealizations Number of realizations.
 * @param nTargets Number of targets per realization.
 * @param quantiles One or more probabilities in `[0, 1]`.
 * @returns A row-major `Float64Array(quantiles.length * nTargets)` where row
 *   `q` holds the requested quantile across all targets, in the order supplied.
 */
export function ensembleQuantiles(
  samples: Float64Array,
  nRealizations: number,
  nTargets: number,
  quantiles: ArrayLike<number>
): Float64Array {
  requirePositiveInt(nRealizations, "nRealizations");
  requirePositiveInt(nTargets, "nTargets");
  requireDimensions(samples, nRealizations, nTargets);
  const nq = quantiles.length;
  if (nq === 0) {
    throw new KrigingError("quantiles must contain at least one probability", {
      code: "invalid_input",
    });
  }
  for (let q = 0; q < nq; q++) {
    const p = quantiles[q];
    if (!(p >= 0 && p <= 1)) {
      throw new KrigingError(
        `quantile probabilities must be in [0, 1] (got ${p} at index ${q})`,
        { code: "invalid_input" }
      );
    }
  }

  const out = new Float64Array(nq * nTargets);
  // Reusable scratch column.
  const col = new Float64Array(nRealizations);
  // Precompute (n - 1) * p once per quantile.
  const positions = new Float64Array(nq);
  const lows = new Int32Array(nq);
  const fracs = new Float64Array(nq);
  const lastIdx = nRealizations - 1;
  for (let q = 0; q < nq; q++) {
    const pos = lastIdx * quantiles[q];
    positions[q] = pos;
    const lo = Math.floor(pos);
    lows[q] = lo;
    fracs[q] = pos - lo;
  }

  for (let j = 0; j < nTargets; j++) {
    for (let k = 0; k < nRealizations; k++) {
      col[k] = samples[k * nTargets + j];
    }
    // Sort in place (Float64Array.prototype.sort is numeric by default).
    col.sort();
    for (let q = 0; q < nq; q++) {
      const lo = lows[q];
      const hi = lo + 1;
      const f = fracs[q];
      const v = hi <= lastIdx ? col[lo] + (col[hi] - col[lo]) * f : col[lo];
      out[q * nTargets + j] = v;
    }
  }
  return out;
}
