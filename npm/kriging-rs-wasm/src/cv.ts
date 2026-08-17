/**
 * Unified cross-validation entry point for every kriging variant.
 *
 * Pass `{ geometry, family, variogram, … }` and optionally `k` for k-fold CV.
 * Omit `k` for leave-one-out. Folds are deterministic round-robin when `k` is set.
 *
 * ## When to use stateless CV vs model methods
 *
 * **Use `cv` / `leaveOneOut` / `kFold` here** when validating from raw arrays, comparing
 * variogram settings before building a model, or when you do not want to hold a WASM
 * handle. Each fold refits from the arrays in your options object.
 *
 * **Use `model.leaveOneOut()` / `model.kFold(k)`** on a fitted kriging class when you
 * already built the model for prediction and want diagnostics on that exact fit. See
 * {@link modelLeaveOneOut} for the full decision guide.
 *
 * @module
 */

import { wrapThrown } from "./errors.js";
import { mapBinomialCvOutput, mapCvOutput } from "./internal/mappers.js";
import { requireLoadedModule } from "./internal/module.js";
import { packCvOptions } from "./internal/unified-boundary.js";
import type { BinomialCvResult, CvOptionsInput, CvResult, LeaveOneOutOptions } from "./types.js";

/**
 * Cross-validation keyed by `geometry` and `family`.
 *
 * - Continuous families return {@link CvResult}.
 * - Binomial families return {@link BinomialCvResult} with logit- and prevalence-scale residuals.
 */
export function cv(
  options: CvOptionsInput
): CvResult | BinomialCvResult {
  const mod = requireLoadedModule();
  const family = options.family ?? "ordinary";
  try {
    const out = mod.cv(packCvOptions(options));
    return family === "binomial"
      ? mapBinomialCvOutput(out)
      : mapCvOutput(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** Leave-one-out CV — convenience wrapper around {@link cv} without `k`. */
export function leaveOneOut(
  options: LeaveOneOutOptions
): CvResult | BinomialCvResult {
  return cv(options as CvOptionsInput);
}

/** K-fold CV — convenience wrapper around {@link cv} with required `k`. */
export function kFold(
  options: CvOptionsInput & { k: number }
): CvResult | BinomialCvResult {
  return cv(options);
}
