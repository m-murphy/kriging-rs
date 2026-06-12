/**
 * Instance cross-validation on fitted WASM model handles.
 *
 * ## When to use model CV vs stateless {@link cv}
 *
 * **Use `model.leaveOneOut()` / `model.kFold(k)`** when you already built a model for
 * prediction and want diagnostics on the **same** training stations and variogram that
 * model holds. No extra option object — call the method on the handle you are about to
 * use (or just used) for `predict` / `predictGrid`.
 *
 * **Use {@link cv} / {@link leaveOneOut} / {@link kFold}** when you have raw arrays and
 * have **not** built a model yet, when you want CV without holding WASM state, or when
 * you are comparing variogram candidates before committing to a fit. Stateless CV refits
 * a fresh model on each fold from the arrays you pass in.
 *
 * Both paths use the same fold logic and return the same result shapes; they differ only
 * in whether a fitted handle already exists.
 *
 * @module
 */

import { wrapThrown } from "../errors.js";
import { mapBinomialCvOutput, mapCvOutput } from "./mappers.js";
import type { BinomialCvResult, CvResult, KrigingFamily } from "../types.js";

export interface WasmModelCvInstance {
  leaveOneOut(): unknown;
  kFold(k: number): unknown;
}

function mapModelCvOutput(
  raw: unknown,
  family: KrigingFamily
): CvResult | BinomialCvResult {
  return family === "binomial"
    ? mapBinomialCvOutput(raw)
    : mapCvOutput(raw);
}

/**
 * Leave-one-out CV on a fitted model handle (same training data and variogram).
 * Prefer {@link leaveOneOut} when validating from raw arrays before building a model.
 */
export function modelLeaveOneOut(
  inner: WasmModelCvInstance,
  family: KrigingFamily
): CvResult | BinomialCvResult {
  try {
    return mapModelCvOutput(inner.leaveOneOut(), family);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * K-fold CV on a fitted model handle (deterministic round-robin folds).
 * Prefer {@link kFold} when validating from raw arrays before building a model.
 */
export function modelKFold(
  inner: WasmModelCvInstance,
  k: number,
  family: KrigingFamily
): CvResult | BinomialCvResult {
  try {
    return mapModelCvOutput(inner.kFold(k), family);
  } catch (e) {
    throw wrapThrown(e);
  }
}
