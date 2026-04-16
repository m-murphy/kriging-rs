/**
 * Cross-validation entry points for every kriging variant shipped by this package.
 *
 * - Continuous variants — {@link leaveOneOut}, {@link kFold} (ordinary),
 *   {@link leaveOneOutSimple}, {@link kFoldSimple} (known mean),
 *   {@link leaveOneOutUniversal}, {@link kFoldUniversal} (polynomial drift), and
 *   {@link leaveOneOutProjected}, {@link kFoldProjected} (planar + 2-D anisotropy) — all
 *   return {@link CvResult} with `observed`, `predicted`, and kriging variance per
 *   held-out station, plus summary statistics.
 * - Binomial CV — {@link leaveOneOutBinomial}, {@link kFoldBinomial} — returns
 *   {@link BinomialCvResult} with residuals on **both** the logit scale (directly
 *   comparable to continuous kriging and MSDR-calibratable) and the prevalence scale
 *   (intuitive; delta-method variance). Stations with `trials === 0` carry `NaN` for
 *   observed fields; summaries skip them.
 *
 * Folds are deterministic round-robin (station `i` → fold `i % k`); shuffle inputs
 * before calling for randomized validation.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "./errors.js";
import { toFloat64Array, toUint32Array } from "./internal/convert.js";
import { mapBinomialCvOutput, mapCvOutput } from "./internal/mappers.js";
import { requireLoadedModule } from "./internal/module.js";
import {
  packSpaceTimeVariogram,
  requireSpaceTimeUniversalTrend,
} from "./internal/spacetime.js";
import type {
  BinomialCvResult,
  CvResult,
  KFoldBinomialOptions,
  KFoldOptions,
  KFoldProjectedOptions,
  KFoldSimpleOptions,
  KFoldSpaceTimeBinomialOptions,
  KFoldSpaceTimeOptions,
  KFoldSpaceTimeSimpleOptions,
  KFoldSpaceTimeUniversalOptions,
  KFoldUniversalOptions,
  LeaveOneOutBinomialOptions,
  LeaveOneOutOptions,
  LeaveOneOutProjectedOptions,
  LeaveOneOutSimpleOptions,
  LeaveOneOutSpaceTimeBinomialOptions,
  LeaveOneOutSpaceTimeOptions,
  LeaveOneOutSpaceTimeSimpleOptions,
  LeaveOneOutSpaceTimeUniversalOptions,
  LeaveOneOutUniversalOptions,
} from "./types.js";

function unavailable(method: string): never {
  throw new KrigingError(
    `${method} is not available; rebuild the WASM package`,
    {
      code: "backend_unavailable",
    }
  );
}

/**
 * Leave-one-out cross-validation: for each station, predict its value from the other
 * `n − 1` stations using ordinary kriging with the supplied variogram. Returns
 * residuals in input order together with aggregate summary statistics.
 */
export function leaveOneOut(options: LeaveOneOutOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOut !== "function") unavailable("leaveOneOut");
  try {
    const out = mod.leaveOneOut(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.values),
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * K-fold cross-validation over ordinary kriging. Folds are deterministic round-robin
 * (station `i` → fold `i % k`); shuffle inputs before calling for randomized folds.
 * `k` must satisfy `2 ≤ k ≤ n`.
 */
export function kFold(options: KFoldOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFold !== "function") unavailable("kFold");
  try {
    const out = mod.kFold(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.values),
      options.k,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Leave-one-out CV over simple kriging. The supplied `mean` is treated as known and is
 * held fixed across folds.
 */
export function leaveOneOutSimple(options: LeaveOneOutSimpleOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutSimple !== "function")
    unavailable("leaveOneOutSimple");
  try {
    const out = mod.leaveOneOutSimple(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.values),
      options.mean,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV over simple kriging. See {@link leaveOneOutSimple} for `mean` semantics. */
export function kFoldSimple(options: KFoldSimpleOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldSimple !== "function") unavailable("kFoldSimple");
  try {
    const out = mod.kFoldSimple(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.values),
      options.mean,
      options.k,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Leave-one-out CV over universal kriging with the given polynomial drift. Trend
 * coefficients are re-estimated inside each fold from the training stations, so the
 * trend contributes no in-sample leakage.
 */
export function leaveOneOutUniversal(
  options: LeaveOneOutUniversalOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutUniversal !== "function") {
    unavailable("leaveOneOutUniversal");
  }
  try {
    const out = mod.leaveOneOutUniversal(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.values),
      options.trend,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV over universal kriging with the given polynomial drift. */
export function kFoldUniversal(options: KFoldUniversalOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldUniversal !== "function") unavailable("kFoldUniversal");
  try {
    const out = mod.kFoldUniversal(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.values),
      options.trend,
      options.k,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Leave-one-out CV over projected kriging on planar `(x, y)` coordinates. Euclidean
 * distances (optionally anisotropy-deformed) are used; pass `rangeRatio = 1` for
 * isotropic models.
 */
export function leaveOneOutProjected(
  options: LeaveOneOutProjectedOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutProjected !== "function") {
    unavailable("leaveOneOutProjected");
  }
  try {
    const out = mod.leaveOneOutProjected(
      toFloat64Array(options.xs),
      toFloat64Array(options.ys),
      toFloat64Array(options.values),
      options.majorAngleDeg,
      options.rangeRatio,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV over projected kriging. See {@link leaveOneOutProjected} for coord semantics. */
export function kFoldProjected(options: KFoldProjectedOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldProjected !== "function") unavailable("kFoldProjected");
  try {
    const out = mod.kFoldProjected(
      toFloat64Array(options.xs),
      toFloat64Array(options.ys),
      toFloat64Array(options.values),
      options.majorAngleDeg,
      options.rangeRatio,
      options.k,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

function validateBinomialPrior(options: LeaveOneOutBinomialOptions): void {
  const hasAlpha = typeof options.priorAlpha === "number";
  const hasBeta = typeof options.priorBeta === "number";
  if (hasAlpha !== hasBeta) {
    throw new KrigingError(
      "priorAlpha and priorBeta must be provided together",
      { code: "invalid_input" }
    );
  }
}

/**
 * Leave-one-out CV over binomial kriging. Returns residuals on **both** the logit scale
 * (directly comparable to continuous kriging; MSDR-calibratable) and the prevalence
 * scale (delta-method variance). Stations with `trials === 0` get `NaN` observed fields;
 * `summary.logit` and `summary.prevalence` skip them automatically.
 */
export function leaveOneOutBinomial(
  options: LeaveOneOutBinomialOptions
): BinomialCvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutBinomial !== "function") {
    unavailable("leaveOneOutBinomial");
  }
  validateBinomialPrior(options);
  try {
    const out = mod.leaveOneOutBinomial(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toUint32Array(options.successes),
      toUint32Array(options.trials),
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      options.priorAlpha,
      options.priorBeta
    );
    return mapBinomialCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV over binomial kriging. See {@link leaveOneOutBinomial} for result shape. */
export function kFoldBinomial(options: KFoldBinomialOptions): BinomialCvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldBinomial !== "function") unavailable("kFoldBinomial");
  validateBinomialPrior(options);
  try {
    const out = mod.kFoldBinomial(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toUint32Array(options.successes),
      toUint32Array(options.trials),
      options.k,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      options.priorAlpha,
      options.priorBeta
    );
    return mapBinomialCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

// ---------------------------------------------------------------------------
// Space-time cross-validation
// ---------------------------------------------------------------------------

/**
 * Leave-one-out CV for space-time ordinary kriging. Coordinates are geographic
 * `(lat, lon)` in degrees; `times` is a scalar axis with user-chosen units.
 */
export function leaveOneOutSpaceTime(
  options: LeaveOneOutSpaceTimeOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutSpaceTime !== "function") {
    unavailable("leaveOneOutSpaceTime");
  }
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.leaveOneOutSpaceTime(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
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
      packed.k3
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV for space-time ordinary kriging. */
export function kFoldSpaceTime(options: KFoldSpaceTimeOptions): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldSpaceTime !== "function") unavailable("kFoldSpaceTime");
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.kFoldSpaceTime(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
      options.k,
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
      packed.k3
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** Leave-one-out CV for space-time simple kriging with a known `mean`. */
export function leaveOneOutSpaceTimeSimple(
  options: LeaveOneOutSpaceTimeSimpleOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutSpaceTimeSimple !== "function") {
    unavailable("leaveOneOutSpaceTimeSimple");
  }
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.leaveOneOutSpaceTimeSimple(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
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
      packed.k3
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV for space-time simple kriging with a known `mean`. */
export function kFoldSpaceTimeSimple(
  options: KFoldSpaceTimeSimpleOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldSpaceTimeSimple !== "function") {
    unavailable("kFoldSpaceTimeSimple");
  }
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.kFoldSpaceTimeSimple(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
      options.mean,
      options.k,
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
      packed.k3
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Leave-one-out CV for space-time universal kriging. Trend coefficients are
 * re-estimated inside each fold from the training stations.
 */
export function leaveOneOutSpaceTimeUniversal(
  options: LeaveOneOutSpaceTimeUniversalOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutSpaceTimeUniversal !== "function") {
    unavailable("leaveOneOutSpaceTimeUniversal");
  }
  const packed = packSpaceTimeVariogram(options.variogram);
  const trend = requireSpaceTimeUniversalTrend(options.trend);
  try {
    const out = mod.leaveOneOutSpaceTimeUniversal(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
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
      packed.k3
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV for space-time universal kriging. */
export function kFoldSpaceTimeUniversal(
  options: KFoldSpaceTimeUniversalOptions
): CvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldSpaceTimeUniversal !== "function") {
    unavailable("kFoldSpaceTimeUniversal");
  }
  const packed = packSpaceTimeVariogram(options.variogram);
  const trend = requireSpaceTimeUniversalTrend(options.trend);
  try {
    const out = mod.kFoldSpaceTimeUniversal(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
      trend,
      options.k,
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
      packed.k3
    );
    return mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

function validateSpaceTimeBinomialPrior(
  priorAlpha: number | undefined,
  priorBeta: number | undefined
): void {
  const hasAlpha = priorAlpha !== undefined;
  const hasBeta = priorBeta !== undefined;
  if (hasAlpha !== hasBeta) {
    throw new KrigingError(
      "priorAlpha and priorBeta must be provided together",
      { code: "invalid_input" }
    );
  }
}

/**
 * Leave-one-out CV for space-time binomial kriging. Returns residuals on **both** the
 * logit and prevalence scales (see {@link leaveOneOutBinomial}).
 */
export function leaveOneOutSpaceTimeBinomial(
  options: LeaveOneOutSpaceTimeBinomialOptions
): BinomialCvResult {
  const mod = requireLoadedModule();
  if (typeof mod.leaveOneOutSpaceTimeBinomial !== "function") {
    unavailable("leaveOneOutSpaceTimeBinomial");
  }
  validateSpaceTimeBinomialPrior(options.priorAlpha, options.priorBeta);
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.leaveOneOutSpaceTimeBinomial(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toUint32Array(options.successes),
      toUint32Array(options.trials),
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
      options.priorBeta
    );
    return mapBinomialCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** K-fold CV for space-time binomial kriging. */
export function kFoldSpaceTimeBinomial(
  options: KFoldSpaceTimeBinomialOptions
): BinomialCvResult {
  const mod = requireLoadedModule();
  if (typeof mod.kFoldSpaceTimeBinomial !== "function") {
    unavailable("kFoldSpaceTimeBinomial");
  }
  validateSpaceTimeBinomialPrior(options.priorAlpha, options.priorBeta);
  const packed = packSpaceTimeVariogram(options.variogram);
  try {
    const out = mod.kFoldSpaceTimeBinomial(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toUint32Array(options.successes),
      toUint32Array(options.trials),
      options.k,
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
      options.priorBeta
    );
    return mapBinomialCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}
