/**
 * Space-time binomial kriging: prevalence/proportion surfaces from count data
 * that vary in space and time.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array, toUint32Array } from "../internal/convert.js";
import { buildGridLatsLons, reshapeFlatToGrid } from "../internal/grid.js";
import {
  mapBinomialBatchArrayOutput,
  mapBinomialPrediction,
  mapBinomialPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import {
  fittedToSpaceTimeVariogramParams,
  packSpaceTimeVariogram,
} from "../internal/spacetime.js";
import type { WasmSpaceTimeBinomialInstance } from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialGridOutput,
  BinomialPrediction,
  NumericArrayInput,
  PredictGridAtTimeOptions,
  SpaceTimeBinomialKrigingFromFittedOptions,
  SpaceTimeBinomialKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeBinomialKriging model has been freed";

/**
 * Space-time binomial kriging model. Observations are `(successes, trials)` pairs
 * located at `(lat, lon, time)`; predictions are reported on both the prevalence
 * (`[0, 1]`) and logit scales.
 */
export class SpaceTimeBinomialKriging {
  private inner: WasmSpaceTimeBinomialInstance | null;

  constructor(options: SpaceTimeBinomialKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeBinomialKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeBinomialKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const packed = packSpaceTimeVariogram(options.variogram);
    try {
      this.inner = ctor.fromArrays(
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
        packed.k3
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmSpaceTimeBinomialInstance {
    if (this.inner === null) {
      throw new KrigingError(FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /** Build a binomial-kriging model from sample counts plus a fitted space-time variogram. */
  static fromFitted(
    options: SpaceTimeBinomialKrigingFromFittedOptions
  ): SpaceTimeBinomialKriging {
    return new SpaceTimeBinomialKriging({
      lats: options.lats,
      lons: options.lons,
      times: options.times,
      successes: options.successes,
      trials: options.trials,
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

  predict(lat: number, lon: number, time: number): BinomialPrediction {
    return mapBinomialPrediction(this.requireInner().predict(lat, lon, time));
  }

  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): BinomialPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapBinomialPredictionArray(out);
  }

  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapBinomialBatchArrayOutput(out);
  }

  /**
   * Predict prevalence on a rectangular lat/lon grid at a fixed `time` slice.
   * Builds cell-center coordinates in row-major order, fills a `time` array
   * with the same scalar for every cell, and reshapes results into 2D arrays
   * of shape `[yCells][xCells]`.
   */
  predictGridAtTime(options: PredictGridAtTimeOptions): BinomialGridOutput {
    const inner = this.requireInner();
    const nRows = Math.max(1, Math.floor(options.yCells));
    const nCols = Math.max(1, Math.floor(options.xCells));
    const { lats, lons } = buildGridLatsLons(options);
    const times = new Float64Array(lats.length);
    times.fill(options.time);
    const out = inner.predictBatchArrays(lats, lons, times);
    const {
      prevalences: prevFlat,
      logitValues: logitFlat,
      variances: varFlat,
      prevalenceVariances: prevVarFlat,
    } = mapBinomialBatchArrayOutput(out);
    return {
      prevalences: reshapeFlatToGrid(prevFlat, nRows, nCols),
      logitValues: reshapeFlatToGrid(logitFlat, nRows, nCols),
      variances: reshapeFlatToGrid(varFlat, nRows, nCols),
      prevalenceVariances: reshapeFlatToGrid(prevVarFlat, nRows, nCols),
    };
  }
}
