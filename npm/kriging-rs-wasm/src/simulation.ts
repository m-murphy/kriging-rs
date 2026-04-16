/**
 * Conditional simulation (Sequential Gaussian Simulation) for every kriging variant.
 *
 * The functions in this module let callers draw single realizations of the underlying
 * Gaussian random field that honor observed data. Each variant mirrors the corresponding
 * kriging API:
 *
 * - {@link conditionalSimulate} — ordinary kriging (no trend).
 * - {@link conditionalSimulateSimple} — simple kriging (known `mean`).
 * - {@link conditionalSimulateUniversal} — universal kriging (polynomial drift).
 * - {@link conditionalSimulateProjected} — planar `(x, y)` with 2-D anisotropy.
 * - {@link conditionalSimulateBinomial} — count data on the logit scale; result carries
 *   both logit and prevalence samples.
 *
 * All variants are deterministic for a given `seed`. Visiting order defaults to input order;
 * pass `targetOrder` to override (must be a permutation of `0..nTargets`). Sampled values at
 * each target are appended to the conditioning pool before the next target is visited.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "./errors.js";
import {
  requireFloat64Array,
  toFloat64Array,
  toUint32Array,
} from "./internal/convert.js";
import { requireLoadedModule } from "./internal/module.js";
import {
  packSpaceTimeVariogram,
  requireSpaceTimeUniversalTrend,
} from "./internal/spacetime.js";
import type {
  BinomialSimulationResult,
  ConditionalSimulateBinomialOptions,
  ConditionalSimulateOptions,
  ConditionalSimulateProjectedOptions,
  ConditionalSimulateSimpleOptions,
  ConditionalSimulateSpaceTimeBinomialOptions,
  ConditionalSimulateSpaceTimeOptions,
  ConditionalSimulateSpaceTimeSimpleOptions,
  ConditionalSimulateSpaceTimeUniversalOptions,
  ConditionalSimulateUniversalOptions,
} from "./types.js";

function unavailable(name: string): never {
  throw new KrigingError(`${name} is not available; rebuild the WASM package`, {
    code: "backend_unavailable",
  });
}

function normalizeSeed(seed: number | bigint | undefined): bigint {
  return typeof seed === "bigint" ? seed : BigInt(seed ?? 0);
}

function normalizeTargetOrder(
  order: ArrayLike<number> | Uint32Array | undefined
): Uint32Array | undefined {
  if (!order) return undefined;
  return order instanceof Uint32Array
    ? order
    : Uint32Array.from(order as ArrayLike<number>);
}

/**
 * Sequential Gaussian simulation conditioned on observed stations using ordinary kriging.
 *
 * Returns a `Float64Array` of length `targetLats.length` with one sampled value per target
 * in input order. Deterministic for a given `seed`.
 */
export function conditionalSimulate(
  options: ConditionalSimulateOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulate !== "function") {
    unavailable("conditionalSimulate");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  try {
    const out = mod.conditionalSimulate(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Sequential Gaussian simulation using simple kriging with known `mean`.
 *
 * Useful when a long-run mean is known a priori — e.g. from regional climatology or a large
 * out-of-sample reference dataset. Simple kriging uses the mean directly rather than
 * estimating it from the conditioning set, reducing bias for small neighborhoods.
 */
export function conditionalSimulateSimple(
  options: ConditionalSimulateSimpleOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateSimple !== "function") {
    unavailable("conditionalSimulateSimple");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  try {
    const out = mod.conditionalSimulateSimple(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      options.mean,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Sequential Gaussian simulation using universal kriging with polynomial drift.
 *
 * Trend coefficients are re-estimated at each simulation step against the growing
 * conditioning pool. Requires at least `p + 1` conditioning stations, where `p` is the
 * basis size (1 for `"constant"`, 3 for `"linear"`, 6 for `"quadratic"`).
 */
export function conditionalSimulateUniversal(
  options: ConditionalSimulateUniversalOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateUniversal !== "function") {
    unavailable("conditionalSimulateUniversal");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  try {
    const out = mod.conditionalSimulateUniversal(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      options.trend,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Sequential Gaussian simulation on projected (planar) coordinates with optional 2-D
 * geometric anisotropy.
 *
 * Use `rangeRatio = 1` for isotropic simulation (angle is then ignored). Coordinates are in
 * the user's chosen projection units; distances use Euclidean geometry.
 */
export function conditionalSimulateProjected(
  options: ConditionalSimulateProjectedOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateProjected !== "function") {
    unavailable("conditionalSimulateProjected");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  try {
    const out = mod.conditionalSimulateProjected(
      toFloat64Array(options.conditioningXs),
      toFloat64Array(options.conditioningYs),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetXs),
      toFloat64Array(options.targetYs),
      options.majorAngleDeg,
      options.rangeRatio,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

function validateBinomialPrior(
  priorAlpha: number | undefined,
  priorBeta: number | undefined
): void {
  const hasAlpha = priorAlpha !== undefined;
  const hasBeta = priorBeta !== undefined;
  if (hasAlpha !== hasBeta) {
    throw new KrigingError(
      "priorAlpha and priorBeta must be provided together (or both omitted)",
      { code: "invalid_input" }
    );
  }
}

/**
 * Sequential Gaussian simulation for binomial (count) data.
 *
 * Simulation happens on the **logit** scale, where the Gaussian assumption is natural, and
 * the result is returned on **both** the logit and prevalence scales. Stations with
 * `trials === 0` are dropped from the initial conditioning pool (they carry no
 * information); the simulator still requires at least 2 remaining valid stations.
 *
 * Each target step:
 * 1. Fits a binomial kriging model against the current pool (initial smoothed logits +
 *    previously simulated logits).
 * 2. Samples `logit ~ N(logit̂, σ²̂)`.
 * 3. Appends the simulated logit to the pool for subsequent targets.
 * 4. Back-transforms via the logistic function to report prevalence.
 */
export function conditionalSimulateBinomial(
  options: ConditionalSimulateBinomialOptions
): BinomialSimulationResult {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateBinomial !== "function") {
    unavailable("conditionalSimulateBinomial");
  }
  validateBinomialPrior(options.priorAlpha, options.priorBeta);
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  try {
    const raw = mod.conditionalSimulateBinomial(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toUint32Array(options.successes),
      toUint32Array(options.trials),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      options.priorAlpha,
      options.priorBeta,
      seed,
      targetOrder
    ) as { logitSamples: unknown; prevalenceSamples: unknown };
    return {
      logitSamples: requireFloat64Array(raw.logitSamples),
      prevalenceSamples: requireFloat64Array(raw.prevalenceSamples),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}

// ---------------------------------------------------------------------------
// Space-time conditional simulation
// ---------------------------------------------------------------------------

/**
 * Sequential Gaussian simulation for space-time ordinary kriging on geographic
 * coordinates. Returns a `Float64Array` with one sampled value per target in input
 * order. Deterministic for a given `seed`.
 */
export function conditionalSimulateSpaceTime(
  options: ConditionalSimulateSpaceTimeOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateSpaceTime !== "function") {
    unavailable("conditionalSimulateSpaceTime");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.conditionalSimulateSpaceTime(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningTimes),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      toFloat64Array(options.targetTimes),
      packed.family,
      packed.spatialType,
      packed.spatialNugget,
      packed.spatialSill,
      packed.spatialRange,
      packed.spatialShape,
      packed.temporalType,
      packed.temporalNugget,
      packed.temporalSill,
      packed.temporalRange,
      packed.temporalShape,
      packed.k1,
      packed.k2,
      packed.k3,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** SGS for space-time simple kriging with a known `mean`. */
export function conditionalSimulateSpaceTimeSimple(
  options: ConditionalSimulateSpaceTimeSimpleOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateSpaceTimeSimple !== "function") {
    unavailable("conditionalSimulateSpaceTimeSimple");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.conditionalSimulateSpaceTimeSimple(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningTimes),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      toFloat64Array(options.targetTimes),
      options.mean,
      packed.family,
      packed.spatialType,
      packed.spatialNugget,
      packed.spatialSill,
      packed.spatialRange,
      packed.spatialShape,
      packed.temporalType,
      packed.temporalNugget,
      packed.temporalSill,
      packed.temporalRange,
      packed.temporalShape,
      packed.k1,
      packed.k2,
      packed.k3,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * SGS for space-time universal kriging. Trend coefficients are re-estimated at each
 * simulation step against the growing conditioning pool; requires `p + 1` initial
 * conditioning stations, where `p` is the trend basis size.
 */
export function conditionalSimulateSpaceTimeUniversal(
  options: ConditionalSimulateSpaceTimeUniversalOptions
): Float64Array {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateSpaceTimeUniversal !== "function") {
    unavailable("conditionalSimulateSpaceTimeUniversal");
  }
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  const packed = packSpaceTimeVariogram(options.variogram);
  const trend = requireSpaceTimeUniversalTrend(options.trend);
  try {
    const out = mod.conditionalSimulateSpaceTimeUniversal(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningTimes),
      toFloat64Array(options.conditioningValues),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      toFloat64Array(options.targetTimes),
      trend,
      packed.family,
      packed.spatialType,
      packed.spatialNugget,
      packed.spatialSill,
      packed.spatialRange,
      packed.spatialShape,
      packed.temporalType,
      packed.temporalNugget,
      packed.temporalSill,
      packed.temporalRange,
      packed.temporalShape,
      packed.k1,
      packed.k2,
      packed.k3,
      seed,
      targetOrder
    );
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * SGS for space-time binomial kriging. Simulation happens on the logit scale; the
 * result carries both logit and prevalence samples. See {@link conditionalSimulateBinomial}
 * for full semantics.
 */
export function conditionalSimulateSpaceTimeBinomial(
  options: ConditionalSimulateSpaceTimeBinomialOptions
): BinomialSimulationResult {
  const mod = requireLoadedModule();
  if (typeof mod.conditionalSimulateSpaceTimeBinomial !== "function") {
    unavailable("conditionalSimulateSpaceTimeBinomial");
  }
  validateBinomialPrior(options.priorAlpha, options.priorBeta);
  const seed = normalizeSeed(options.seed);
  const targetOrder = normalizeTargetOrder(options.targetOrder);
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const raw = mod.conditionalSimulateSpaceTimeBinomial(
      toFloat64Array(options.conditioningLats),
      toFloat64Array(options.conditioningLons),
      toFloat64Array(options.conditioningTimes),
      toUint32Array(options.successes),
      toUint32Array(options.trials),
      toFloat64Array(options.targetLats),
      toFloat64Array(options.targetLons),
      toFloat64Array(options.targetTimes),
      packed.family,
      packed.spatialType,
      packed.spatialNugget,
      packed.spatialSill,
      packed.spatialRange,
      packed.spatialShape,
      packed.temporalType,
      packed.temporalNugget,
      packed.temporalSill,
      packed.temporalRange,
      packed.temporalShape,
      packed.k1,
      packed.k2,
      packed.k3,
      options.priorAlpha,
      options.priorBeta,
      seed,
      targetOrder
    ) as { logitSamples: unknown; prevalenceSamples: unknown };
    return {
      logitSamples: requireFloat64Array(raw.logitSamples),
      prevalenceSamples: requireFloat64Array(raw.prevalenceSamples),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}
