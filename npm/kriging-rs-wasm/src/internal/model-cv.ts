/**
 * Instance cross-validation on fitted WASM model handles.
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

/** Leave-one-out CV on a fitted model (uses its training data and variogram). */
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

/** K-fold CV on a fitted model (deterministic round-robin folds). */
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
