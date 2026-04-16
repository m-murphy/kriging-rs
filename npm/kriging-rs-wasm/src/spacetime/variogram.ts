/**
 * Empirical space-time variogram computation and parametric fitting.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import {
  mapEmpiricalSpaceTimeVariogram,
  mapFitSpaceTimeVariogramResult,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import type {
  ComputeEmpiricalSpaceTimeVariogramOptions,
  EmpiricalSpaceTimeVariogramResult,
  FitSpaceTimeVariogramOptions,
  FitSpaceTimeVariogramResult,
} from "../types.js";

function defaultEstimator(
  estimator: ComputeEmpiricalSpaceTimeVariogramOptions["estimator"]
): string {
  return estimator ?? "classical";
}

/**
 * Compute a 2-D empirical space-time variogram on geographic coordinates.
 *
 * Both `nSpatialBins` and `nTemporalBins` must be ≥ 1; if `maxSpatialDistance` or
 * `maxTemporalLag` are omitted, half the maximum observed pair lag is used for each
 * axis.
 *
 * @throws {@link KrigingError} for invalid inputs (mismatched arrays, zero bins,
 *   non-positive max-distance/lag, etc.).
 */
export function computeEmpiricalSpaceTimeVariogram(
  options: ComputeEmpiricalSpaceTimeVariogramOptions
): EmpiricalSpaceTimeVariogramResult {
  const mod = requireLoadedModule();
  const fn = mod.wasmComputeEmpiricalSpaceTimeVariogram;
  if (typeof fn !== "function") {
    throw new KrigingError(
      "wasmComputeEmpiricalSpaceTimeVariogram is not available; rebuild the WASM package",
      { code: "backend_unavailable" }
    );
  }
  try {
    const out = fn(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
      options.maxSpatialDistance,
      options.maxTemporalLag,
      options.nSpatialBins,
      options.nTemporalBins,
      defaultEstimator(options.estimator)
    );
    return mapEmpiricalSpaceTimeVariogram(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Fit a parametric space-time variogram (separable or product-sum) to the empirical
 * space-time variogram of the supplied data. The fit uses sequential optimization:
 * marginals are fit independently via weighted least squares, and for `product_sum`
 * the three non-negative coefficients `(k1, k2, k3)` are fit by non-negative least
 * squares against the full 2-D empirical surface.
 */
export function fitSpaceTimeVariogram(
  options: FitSpaceTimeVariogramOptions
): FitSpaceTimeVariogramResult {
  const mod = requireLoadedModule();
  const fn = mod.wasmFitSpaceTimeVariogram;
  if (typeof fn !== "function") {
    throw new KrigingError(
      "wasmFitSpaceTimeVariogram is not available; rebuild the WASM package",
      { code: "backend_unavailable" }
    );
  }
  try {
    const out = fn(
      toFloat64Array(options.lats),
      toFloat64Array(options.lons),
      toFloat64Array(options.times),
      toFloat64Array(options.values),
      options.maxSpatialDistance,
      options.maxTemporalLag,
      options.nSpatialBins,
      options.nTemporalBins,
      defaultEstimator(options.estimator),
      options.family,
      options.spatialModel,
      options.temporalModel
    );
    return mapFitSpaceTimeVariogramResult(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}
