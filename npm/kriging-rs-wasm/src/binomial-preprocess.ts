/**
 * Count-data preprocessing for binomial kriging: EB-smoothed logits and Laplace
 * (inverse-Fisher) logit observation variances, matching the default geographic /
 * projected / space–time binomial build in Rust.
 *
 * @module
 */

import type { BinomialPriorParams, IntegerArrayInput } from "./types.js";
import {
  resolveBinomialPriorOrDefault,
  smoothedLogits,
} from "./internal/prior.js";

function logitObservationVarianceLaplace(
  successes: number,
  trials: number,
  prior: BinomialPriorParams
): number {
  if (trials === 0) {
    return 0;
  }
  const s = successes;
  const n = trials;
  const p = (s + prior.alpha) / (n + prior.alpha + prior.beta);
  const denom = n * p * (1 - p);
  if (!(denom > 0) || !Number.isFinite(denom)) {
    return 0;
  }
  return Math.max(0, 1 / denom);
}

export interface BinomialPreprocessOptions {
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  prior?: BinomialPriorParams;
}

export interface BinomialPreprocessResult {
  logits: Float64Array;
  logitVariances: Float64Array;
}

/**
 * Smoothed logits plus Laplace logit observation variances for each station.
 * Use with {@link fitVariogram} on `logits` when building from precomputed logits.
 */
export function binomialPreprocess(
  options: BinomialPreprocessOptions
): BinomialPreprocessResult {
  const prior = resolveBinomialPriorOrDefault(options.prior);
  const logits = smoothedLogits(
    options.successes,
    options.trials,
    prior
  );
  const n = logits.length;
  const logitVariances = new Float64Array(n);
  const sArr = Array.from(options.successes as ArrayLike<number>);
  const tArr = Array.from(options.trials as ArrayLike<number>);
  if (tArr.length !== n) {
    throw new Error(
      `binomialPreprocess: successes (${n}) and trials (${tArr.length}) must have the same length`
    );
  }
  for (let i = 0; i < n; i++) {
    logitVariances[i] = logitObservationVarianceLaplace(
      sArr[i]!,
      tArr[i]!,
      prior
    );
  }
  return { logits, logitVariances };
}
