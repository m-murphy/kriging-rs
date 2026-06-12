/**
 * Binomial kriging: prevalence / proportion surfaces from count data (successes/trials).
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
  predictBatchGpuBinomialGeo,
  predictBatchGpuOrCpuBinomialGeo,
  predictBinomialGeo,
  predictGridBinomialGeo,
  requireBinomialHandle,
  type GeoBinomialDiagnosticsCounts,
} from "../internal/binomial-model-shared.js";
import { requireLoadedModule } from "../internal/module.js";
import type {
  BinomialKrigingOptionsWasm,
  BinomialKrigingWithPriorOptionsWasm,
  WasmKrigingModelHandle,
} from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResult,
  BinomialDiagnostics,
  BinomialFromPrecomputedLogitsOptions,
  BinomialFromPrecomputedLogitsWithVariancesOptions,
  BinomialGridOutput,
  BinomialKrigingFromFittedVariogramOptions,
  BinomialKrigingFromFittedVariogramWithPriorOptions,
  BinomialKrigingOptions,
  BinomialKrigingWithPriorOptions,
  BinomialPrediction,
  NumericArrayInput,
  PredictGridOptions,
} from "../types.js";

const BINOMIAL_FREED = "BinomialKriging model has been freed";

function toBinomialOptionsWasm(
  opts: BinomialKrigingOptions
): BinomialKrigingOptionsWasm {
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
    ...(opts.stability !== undefined ? { stability: opts.stability } : {}),
    ...(opts.oneStepLaplaceObservationVariance === true
      ? { oneStepLaplaceObservationVariance: true }
      : {}),
  };
}

function toBinomialWithPriorOptionsWasm(
  opts: BinomialKrigingWithPriorOptions
): BinomialKrigingWithPriorOptionsWasm {
  return {
    ...toBinomialOptionsWasm({
      lats: opts.lats,
      lons: opts.lons,
      successes: opts.successes,
      trials: opts.trials,
      variogram: opts.variogram,
      stability: opts.stability,
    }),
    prior: { alpha: opts.prior.alpha, beta: opts.prior.beta },
  };
}

/**
 * Binomial kriging model for prevalence (proportion) surfaces from count data (successes/trials).
 * Coordinates are in degrees; distances use Haversine. Use {@link BinomialKriging.newWithPrior}
 * to supply a Beta(alpha, beta) prior for stabilization. For building from a fitted variogram,
 * use {@link BinomialKriging.fromFittedVariogram} or {@link BinomialKriging.fromFittedVariogramWithPrior}.
 *
 * @throws {KrigingError} When the WASM module is not loaded, or when inputs are invalid.
 */
export class BinomialKriging {
  private inner: WasmKrigingModelHandle | null;

  constructor(options: BinomialKrigingOptions) {
    const mod = requireLoadedModule();
    try {
      this.inner = mod.WasmKrigingModel.binomialGeoFromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toUint32Array(options.successes),
        toUint32Array(options.trials),
        options.variogram.variogramType,
        options.variogram.nugget,
        options.variogram.sill,
        options.variogram.range,
        options.variogram.shape,
        options.stability,
        options.oneStepLaplaceObservationVariance
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static fromPrecomputedLogits(
    options: BinomialFromPrecomputedLogitsOptions
  ): BinomialKriging {
    const mod = requireLoadedModule();
    const factory = mod.WasmKrigingModel;
    try {
      return attachBinomialHandle(
        BinomialKriging.prototype,
        factory.binomialGeoFromPrecomputedLogits(
          toFloat64Array(options.lats),
          toFloat64Array(options.lons),
          toFloat64Array(options.logits),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape
        )
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static fromPrecomputedLogitsWithVariances(
    options: BinomialFromPrecomputedLogitsWithVariancesOptions
  ): BinomialKriging {
    const mod = requireLoadedModule();
    const factory = mod.WasmKrigingModel;
    const priorAlpha = options.prior?.alpha;
    const priorBeta = options.prior?.beta;
    try {
      return attachBinomialHandle(
        BinomialKriging.prototype,
        factory.binomialGeoFromPrecomputedLogitsWithVariances(
          toFloat64Array(options.lats),
          toFloat64Array(options.lons),
          toFloat64Array(options.logits),
          toFloat64Array(options.logitObservationVariance),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape,
          priorAlpha,
          priorBeta,
          options.stability,
          options.oneStepLaplaceObservationVariance
        )
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static newWithPrior(
    options: BinomialKrigingWithPriorOptions
  ): BinomialKriging {
    const mod = requireLoadedModule();
    try {
      return attachBinomialHandle(
        BinomialKriging.prototype,
        mod.WasmKrigingModel.binomialGeoNewWithPrior(
          toBinomialWithPriorOptionsWasm(options)
        )
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static fromFittedVariogram(
    options: BinomialKrigingFromFittedVariogramOptions
  ): BinomialKriging {
    return new BinomialKriging({
      lats: options.lats,
      lons: options.lons,
      successes: options.successes,
      trials: options.trials,
      variogram: {
        variogramType: options.fittedVariogram.variogramType,
        nugget: options.fittedVariogram.nugget,
        sill: options.fittedVariogram.sill,
        range: options.fittedVariogram.range,
        shape: options.fittedVariogram.shape,
      },
      ...(options.stability !== undefined ? { stability: options.stability } : {}),
      ...(options.oneStepLaplaceObservationVariance === true
        ? { oneStepLaplaceObservationVariance: true }
        : {}),
    });
  }

  static fromFittedVariogramWithPrior(
    options: BinomialKrigingFromFittedVariogramWithPriorOptions
  ): BinomialKriging {
    return BinomialKriging.newWithPrior({
      lats: options.lats,
      lons: options.lons,
      successes: options.successes,
      trials: options.trials,
      variogram: {
        variogramType: options.fittedVariogram.variogramType,
        nugget: options.fittedVariogram.nugget,
        sill: options.fittedVariogram.sill,
        range: options.fittedVariogram.range,
        shape: options.fittedVariogram.shape,
      },
      prior: options.prior,
      ...(options.stability !== undefined ? { stability: options.stability } : {}),
      ...(options.oneStepLaplaceObservationVariance === true
        ? { oneStepLaplaceObservationVariance: true }
        : {}),
    });
  }

  free(): void {
    this.inner = freeBinomialHandle(this.inner);
  }

  [Symbol.dispose](): void {
    this.free();
  }

  get buildNotes(): BinomialBuildNotes {
    return getBinomialBuildNotes(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      BINOMIAL_FREED
    );
  }

  diagnostics(counts?: GeoBinomialDiagnosticsCounts): BinomialDiagnostics {
    return getBinomialDiagnostics2d(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      BINOMIAL_FREED,
      counts,
      packGeoDiagnosticsOpts
    );
  }

  predict(lat: number, lon: number): BinomialPrediction {
    return predictBinomialGeo(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      lat,
      lon
    );
  }

  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialPrediction[] {
    return predictBatchBinomialGeo(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      lats,
      lons
    );
  }

  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialBatchArrayOutput {
    return predictBatchArraysBinomialGeo(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      lats,
      lons
    );
  }

  predictGrid(options: PredictGridOptions): BinomialGridOutput {
    return predictGridBinomialGeo(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      options
    );
  }

  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<BinomialPrediction[]> {
    return predictBatchGpuBinomialGeo(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      lats,
      lons
    );
  }

  async predictBatchGpuOrCpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<BinomialPrediction[]> {
    const inner = requireBinomialHandle(this.inner, BINOMIAL_FREED);
    return predictBatchGpuOrCpuBinomialGeo(inner, lats, lons, () =>
      this.predictBatch(lats, lons)
    );
  }

  leaveOneOut(): BinomialCvResult {
    return binomialLeaveOneOut(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      BINOMIAL_FREED
    );
  }

  kFold(k: number): BinomialCvResult {
    return binomialKFold(
      requireBinomialHandle(this.inner, BINOMIAL_FREED),
      BINOMIAL_FREED,
      k
    );
  }
}
