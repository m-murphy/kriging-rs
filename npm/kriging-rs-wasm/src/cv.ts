/**
 * Unified cross-validation entry point for every kriging variant.
 *
 * Pass `{ geometry, family, variogram, … }` and optionally `k` for k-fold CV.
 * Omit `k` for leave-one-out. Folds are deterministic round-robin when `k` is set.
 *
 * @module
 */

import { wrapThrown } from "./errors.js";
import { mapBinomialCvOutput, mapCvOutput } from "./internal/mappers.js";
import { requireLoadedModule } from "./internal/module.js";
import { packCvOptions } from "./internal/unified-boundary.js";
import type { BinomialCvResult, CvOptions, CvResult } from "./types.js";

/**
 * Cross-validation keyed by `geometry` and `family`.
 *
 * - Continuous families return {@link CvResult}.
 * - Binomial families return {@link BinomialCvResult} with logit- and prevalence-scale residuals.
 */
export function cv(
  options: CvOptions & Partial<Pick<CvOptions, "geometry" | "family">>
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
  options: Omit<CvOptions, "k"> & Partial<Pick<CvOptions, "geometry" | "family">>
): CvResult | BinomialCvResult {
  return cv(options);
}

/** K-fold CV — convenience wrapper around {@link cv} with required `k`. */
export function kFold(
  options: (CvOptions & { k: number }) &
    Partial<Pick<CvOptions, "geometry" | "family">>
): CvResult | BinomialCvResult {
  return cv(options);
}
