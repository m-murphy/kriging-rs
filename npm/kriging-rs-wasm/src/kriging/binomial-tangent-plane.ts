/**
 * Geographic binomial kriging on a **local tangent plane** (equirectangular km coordinates)
 * with optional 2-D geometric anisotropy.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array, toUint32Array } from "../internal/convert.js";
import {
  attachBinomialHandle,
  binomialKFold,
  binomialLeaveOneOut,
  freeBinomialHandle,
  getBinomialBuildNotes,
  getBinomialDiagnostics2d,
  packGeoDiagnosticsOpts,
  predictBatchArraysBinomialGeo,
  predictBatchBinomialGeo,
  predictBinomialGeo,
  predictGridBinomialGeo,
  requireBinomialHandle,
  type GeoBinomialDiagnosticsCounts,
} from "../internal/binomial-model-shared.js";
import { requireLoadedModule } from "../internal/module.js";
import type {
  BinomialTangentPlaneKrigingWithPriorOptionsWasm,
  WasmKrigingModelHandle,
} from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResult,
  BinomialDiagnostics,
  BinomialGridOutput,
  BinomialPrediction,
  BinomialTangentPlaneKrigingOptions,
  BinomialTangentPlaneKrigingWithPriorOptions,
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

function requireTangentFactory(
  name: string
): (...args: unknown[]) => WasmKrigingModelHandle {
  const mod = requireLoadedModule();
  const factory = mod.WasmKrigingModel?.[name as keyof typeof mod.WasmKrigingModel];
  if (typeof factory !== "function") {
    throw new KrigingError(
      "BinomialTangentPlaneKriging is not available; rebuild the WASM package",
      { code: "backend_unavailable" }
    );
  }
  return factory as (...args: unknown[]) => WasmKrigingModelHandle;
}

/**
 * Binomial kriging on a local equirectangular tangent plane with 2-D anisotropy.
 */
export class BinomialTangentPlaneKriging {
  private inner: WasmKrigingModelHandle | null;

  constructor(options: BinomialTangentPlaneKrigingOptions) {
    const factory = requireTangentFactory("binomialTangentPlaneFromArrays");
    try {
      this.inner = factory.call(
        requireLoadedModule().WasmKrigingModel,
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

  static newWithPrior(
    options: BinomialTangentPlaneKrigingWithPriorOptions
  ): BinomialTangentPlaneKriging {
    const factory = requireTangentFactory("binomialTangentPlaneNewWithPrior");
    try {
      return attachBinomialHandle(
        BinomialTangentPlaneKriging.prototype,
        factory.call(
          requireLoadedModule().WasmKrigingModel,
          toTangentPlaneWithPriorOptionsWasm(options)
        )
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  free(): void {
    this.inner = freeBinomialHandle(this.inner);
  }

  [Symbol.dispose](): void {
    this.free();
  }

  get buildNotes(): BinomialBuildNotes {
    return getBinomialBuildNotes(
      requireBinomialHandle(this.inner, FREED),
      FREED
    );
  }

  diagnostics(counts?: GeoBinomialDiagnosticsCounts): BinomialDiagnostics {
    return getBinomialDiagnostics2d(
      requireBinomialHandle(this.inner, FREED),
      FREED,
      counts,
      packGeoDiagnosticsOpts
    );
  }

  predict(lat: number, lon: number): BinomialPrediction {
    return predictBinomialGeo(requireBinomialHandle(this.inner, FREED), lat, lon);
  }

  predictBatch(lats: NumericArrayInput, lons: NumericArrayInput): BinomialPrediction[] {
    return predictBatchBinomialGeo(
      requireBinomialHandle(this.inner, FREED),
      lats,
      lons
    );
  }

  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialBatchArrayOutput {
    return predictBatchArraysBinomialGeo(
      requireBinomialHandle(this.inner, FREED),
      lats,
      lons
    );
  }

  predictGrid(options: PredictGridOptions): BinomialGridOutput {
    return predictGridBinomialGeo(
      requireBinomialHandle(this.inner, FREED),
      options
    );
  }

  leaveOneOut(): BinomialCvResult {
    return binomialLeaveOneOut(requireBinomialHandle(this.inner, FREED), FREED);
  }

  kFold(k: number): BinomialCvResult {
    return binomialKFold(requireBinomialHandle(this.inner, FREED), FREED, k);
  }
}
