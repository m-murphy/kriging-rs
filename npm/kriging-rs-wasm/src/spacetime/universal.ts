/**
 * Space-time universal kriging with polynomial drift bases in space and/or time.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import {
  mapOrdinaryBatchArrayOutput,
  mapOrdinaryPrediction,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import { modelKFold, modelLeaveOneOut } from "../internal/model-cv.js";
import {
  fittedToSpaceTimeVariogramParams,
  packSpaceTimeVariogram,
  requireSpaceTimeUniversalTrend,
} from "../internal/spacetime.js";
import type { WasmKrigingModelHandle } from "../internal/wasm-shapes.js";
import type {
  CvResult,
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  SpaceTimeUniversalKrigingFromFittedOptions,
  SpaceTimeUniversalKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeUniversalKriging model has been freed";

/**
 * Space-time universal kriging model with a polynomial drift in the spatial and/or
 * temporal axes (see {@link SpaceTimeUniversalTrend} for available bases).
 */
export class SpaceTimeUniversalKriging {
  private inner: WasmKrigingModelHandle | null;

  constructor(options: SpaceTimeUniversalKrigingOptions) {
    const mod = requireLoadedModule();
    const factory = mod.WasmKrigingModel?.spacetimeUniversalGeoFromArrays;
    if (!factory) {
      throw new KrigingError(
        "SpaceTimeUniversalKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const packed = packSpaceTimeVariogram(options.variogram);
    const trend = requireSpaceTimeUniversalTrend(options.trend);
    try {
      this.inner = factory.call(
        mod.WasmKrigingModel,
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
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmKrigingModelHandle {
    if (this.inner === null) {
      throw new KrigingError(FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /** Build a universal-kriging model from a fitted space-time variogram and a trend. */
  static fromFitted(
    options: SpaceTimeUniversalKrigingFromFittedOptions
  ): SpaceTimeUniversalKriging {
    return new SpaceTimeUniversalKriging({
      lats: options.lats,
      lons: options.lons,
      times: options.times,
      values: options.values,
      trend: options.trend,
      variogram: fittedToSpaceTimeVariogramParams(options.fittedVariogram),
    });
  }

  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  predict(lat: number, lon: number, time: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predictSpaceTime(lat, lon, time));
  }

  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArraysSpaceTime(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }

  leaveOneOut(): CvResult {
    return modelLeaveOneOut(this.requireInner(), "universal") as CvResult;
  }

  kFold(k: number): CvResult {
    return modelKFold(this.requireInner(), k, "universal") as CvResult;
  }
}
