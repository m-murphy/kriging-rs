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
import type {
  BinomialCvResult,
  CvResult,
  KFoldBinomialOptions,
  KFoldOptions,
  KFoldProjectedOptions,
  KFoldSimpleOptions,
  KFoldUniversalOptions,
  LeaveOneOutBinomialOptions,
  LeaveOneOutOptions,
  LeaveOneOutProjectedOptions,
  LeaveOneOutSimpleOptions,
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
