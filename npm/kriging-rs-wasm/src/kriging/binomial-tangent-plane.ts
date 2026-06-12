/**
 * Geographic binomial kriging on a **local tangent plane** (equirectangular km coordinates)
 * with optional 2-D geometric anisotropy. Input sites remain lat/lon; the variogram range is
 * in **kilometers** (same as default geographic binomial).
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array, toUint32Array } from "../internal/convert.js";
import { reshapeFlatToGrid } from "../internal/grid.js";
import {
  mapBinomialBatchArrayOutput,
  mapBinomialBuildNotes,
  mapBinomialDiagnostics,
  mapBinomialPrediction,
  mapBinomialPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import type {
  BinomialTangentPlaneKrigingWithPriorOptionsWasm,
  WasmBinomialInstance,
} from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialDiagnostics,
  BinomialGridOutput,
  BinomialPrediction,
  BinomialTangentPlaneKrigingOptions,
  BinomialTangentPlaneKrigingWithPriorOptions,
  IntegerArrayInput,
  NumericArrayInput,
  PredictGridOptions,
} from "../types.js";

const FREED = "BinomialTangentPlaneKriging model has been freed";

function toTangentPlaneWithPriorOptionsWasm(
  opts: BinomialTangentPlaneKrigingWithPriorOptions
): BinomialTangentPlaneKrigingWithPriorOptionsWasm {
  return {
    lats: Array.from(toFloat64Array(opts.lats)),
    lons: Array.from(toFloat64Array(opts.lons)),
    successes: Array.from(toUint32Array(opts.successes)),
    trials: Array.from(toUint32Array(opts.trials)),
    variogram: {
      variogramType: opts.variogram.variogramType,
      nugget: opts.variogram.nugget,
      sill: opts.variogram.sill,
      range: opts.variogram.range,
      shape: opts.variogram.shape,
    },
    prior: { alpha: opts.prior.alpha, beta: opts.prior.beta },
    majorAngleDeg: opts.majorAngleDeg,
    rangeRatio: opts.rangeRatio,
    ...(opts.tangentPlaneRefLat !== undefined
      ? { tangentPlaneRefLat: opts.tangentPlaneRefLat }
      : {}),
    ...(opts.tangentPlaneRefLon !== undefined
      ? { tangentPlaneRefLon: opts.tangentPlaneRefLon }
      : {}),
    ...(opts.stability !== undefined ? { stability: opts.stability } : {}),
    ...(opts.oneStepLaplaceObservationVariance === true
      ? { oneStepLaplaceObservationVariance: true }
      : {}),
  };
}

/**
 * Binomial kriging on a local equirectangular tangent plane with 2-D anisotropy.
 * Prefer {@link BinomialKriging} when isotropic Haversine distances are adequate.
 */
export class BinomialTangentPlaneKriging {
  private inner: WasmBinomialInstance | null;

  constructor(options: BinomialTangentPlaneKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialTangentPlaneKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialTangentPlaneKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toUint32Array(options.successes),
        toUint32Array(options.trials),
        options.variogram.variogramType,
        options.variogram.nugget,
        options.variogram.sill,
        options.variogram.range,
        options.variogram.shape,
        options.majorAngleDeg,
        options.rangeRatio,
        options.tangentPlaneRefLat,
        options.tangentPlaneRefLon,
        options.stability,
        options.oneStepLaplaceObservationVariance
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmBinomialInstance {
    if (this.inner === null) {
      throw new KrigingError(FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  static newWithPrior(
    options: BinomialTangentPlaneKrigingWithPriorOptions
  ): BinomialTangentPlaneKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialTangentPlaneKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialTangentPlaneKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const instance = Object.create(
      BinomialTangentPlaneKriging.prototype
    ) as BinomialTangentPlaneKriging;
    try {
      (instance as unknown as { inner: WasmBinomialInstance | null }).inner =
        ctor.newWithPrior(toTangentPlaneWithPriorOptionsWasm(options));
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
    this.inner = null;
  }

  [Symbol.dispose](): void {
    this.free();
  }

  get buildNotes(): BinomialBuildNotes {
    try {
      return mapBinomialBuildNotes(this.requireInner().getBuildNotes());
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Same shape as {@link BinomialKriging.diagnostics}: variogram in km-range units, build
   * notes, and optional LOO MSDR from `{ lats, lons, successes, trials }` (degrees).
   */
  diagnostics(counts?: {
    lats: NumericArrayInput;
    lons: NumericArrayInput;
    successes: IntegerArrayInput;
    trials: IntegerArrayInput;
  }): BinomialDiagnostics {
    try {
      const inner = this.requireInner();
      const getDiagnostics = inner.getDiagnostics;
      if (typeof getDiagnostics !== "function") {
        throw new KrigingError(
          "BinomialTangentPlaneKriging.diagnostics requires WASM getDiagnostics",
          { code: "internal_error" }
        );
      }
      const opts: unknown =
        counts === undefined
          ? undefined
          : {
              lats: toFloat64Array(counts.lats),
              lons: toFloat64Array(counts.lons),
              successes: toUint32Array(counts.successes),
              trials: toUint32Array(counts.trials),
            };
      return mapBinomialDiagnostics(getDiagnostics.call(inner, opts));
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  predict(lat: number, lon: number): BinomialPrediction {
    return mapBinomialPrediction(this.requireInner().predict(lat, lon));
  }

  predictBatch(lats: NumericArrayInput, lons: NumericArrayInput): BinomialPrediction[] {
    const out = this
      .requireInner()
      .predictBatch(toFloat64Array(lats), toFloat64Array(lons));
    return mapBinomialPredictionArray(out);
  }

  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this
      .requireInner()
      .predictBatchArrays(toFloat64Array(lats), toFloat64Array(lons));
    return mapBinomialBatchArrayOutput(out);
  }

  predictGrid(options: PredictGridOptions): BinomialGridOutput {
    const inner = this.requireInner();
    const nRows = Math.max(1, Math.floor(options.yCells));
    const nCols = Math.max(1, Math.floor(options.xCells));
    const out = inner.predictGridArrays(
      options.west,
      options.east,
      options.south,
      options.north,
      nCols,
      nRows
    );
    const {
      prevalenceMedians: pmFlat,
      prevalenceMeans: pmeanFlat,
      logitValues: lFlat,
      logitVariances: lvFlat,
      prevalenceVariances: pvFlat,
    } = mapBinomialBatchArrayOutput(out);
    return {
      prevalenceMedians: reshapeFlatToGrid(pmFlat, nRows, nCols),
      prevalenceMeans: reshapeFlatToGrid(pmeanFlat, nRows, nCols),
      logitValues: reshapeFlatToGrid(lFlat, nRows, nCols),
      logitVariances: reshapeFlatToGrid(lvFlat, nRows, nCols),
      prevalenceVariances: reshapeFlatToGrid(pvFlat, nRows, nCols),
    };
  }
}
