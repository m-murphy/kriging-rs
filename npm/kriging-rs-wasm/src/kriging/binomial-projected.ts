/**
 * Projected (planar) binomial kriging on `(x, y)` coordinates with 2D anisotropy.
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
  packProjectedDiagnosticsOpts,
  predictBatchArraysBinomialProjected,
  predictBatchBinomialProjected,
  predictBinomialProjected,
  predictGridBinomialProjected,
  requireBinomialHandle,
  type ProjectedBinomialDiagnosticsCounts,
} from "../internal/binomial-model-shared.js";
import { requireLoadedModule } from "../internal/module.js";
import type { WasmKrigingModelHandle } from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResult,
  BinomialDiagnostics,
  BinomialGridOutput,
  BinomialPrediction,
  BinomialProjectedFromPrecomputedLogitsOptions,
  BinomialProjectedFromPrecomputedLogitsWithVariancesOptions,
  BinomialProjectedKrigingFromFittedVariogramOptions,
  BinomialProjectedKrigingFromFittedVariogramWithPriorOptions,
  BinomialProjectedKrigingOptions,
  BinomialProjectedKrigingWithPriorOptions,
  NumericArrayInput,
  PredictProjectedGridOptions,
} from "../types.js";

const PROJECTED_BINOMIAL_FREED =
  "BinomialProjectedKriging model has been freed";

function requireProjectedFactory(
  name: string
): (...args: unknown[]) => WasmKrigingModelHandle {
  const mod = requireLoadedModule();
  const factory = mod.WasmKrigingModel?.[name as keyof typeof mod.WasmKrigingModel];
  if (typeof factory !== "function") {
    throw new KrigingError(
      "BinomialProjectedKriging is not available; rebuild the WASM package",
      { code: "backend_unavailable" }
    );
  }
  return factory as (...args: unknown[]) => WasmKrigingModelHandle;
}

/**
 * Projected / planar binomial kriging on `(x, y)` coordinates with 2-D anisotropy.
 */
export class BinomialProjectedKriging {
  private inner: WasmKrigingModelHandle | null;

  constructor(options: BinomialProjectedKrigingOptions) {
    const factory = requireProjectedFactory("binomialProjectedFromArrays");
    try {
      this.inner = factory.call(
        requireLoadedModule().WasmKrigingModel,
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toUint32Array(options.successes),
        toUint32Array(options.trials),
        options.variogram.variogramType,
        options.variogram.nugget,
        options.variogram.sill,
        options.variogram.range,
        options.variogram.shape,
        options.majorAngleDeg,
        options.rangeRatio,
        options.stability,
        options.oneStepLaplaceObservationVariance
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static newWithPrior(
    options: BinomialProjectedKrigingWithPriorOptions
  ): BinomialProjectedKriging {
    const factory = requireProjectedFactory("binomialProjectedFromArraysWithPrior");
    try {
      return attachBinomialHandle(
        BinomialProjectedKriging.prototype,
        factory.call(
          requireLoadedModule().WasmKrigingModel,
          toFloat64Array(options.xs),
          toFloat64Array(options.ys),
          toUint32Array(options.successes),
          toUint32Array(options.trials),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape,
          options.majorAngleDeg,
          options.rangeRatio,
          options.prior.alpha,
          options.prior.beta,
          options.stability,
          options.oneStepLaplaceObservationVariance
        )
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static fromPrecomputedLogits(
    options: BinomialProjectedFromPrecomputedLogitsOptions
  ): BinomialProjectedKriging {
    const factory = requireProjectedFactory("binomialProjectedFromPrecomputedLogits");
    try {
      return attachBinomialHandle(
        BinomialProjectedKriging.prototype,
        factory.call(
          requireLoadedModule().WasmKrigingModel,
          toFloat64Array(options.xs),
          toFloat64Array(options.ys),
          toFloat64Array(options.logits),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape,
          options.majorAngleDeg,
          options.rangeRatio
        )
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  static fromPrecomputedLogitsWithVariances(
    options: BinomialProjectedFromPrecomputedLogitsWithVariancesOptions
  ): BinomialProjectedKriging {
    const factory = requireProjectedFactory(
      "binomialProjectedFromPrecomputedLogitsWithVariances"
    );
    const priorAlpha = options.prior?.alpha;
    const priorBeta = options.prior?.beta;
    try {
      return attachBinomialHandle(
        BinomialProjectedKriging.prototype,
        factory.call(
          requireLoadedModule().WasmKrigingModel,
          toFloat64Array(options.xs),
          toFloat64Array(options.ys),
          toFloat64Array(options.logits),
          toFloat64Array(options.logitObservationVariance),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape,
          options.majorAngleDeg,
          options.rangeRatio,
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

  static fromFittedVariogram(
    options: BinomialProjectedKrigingFromFittedVariogramOptions
  ): BinomialProjectedKriging {
    const v = options.fittedVariogram;
    return new BinomialProjectedKriging({
      xs: options.xs,
      ys: options.ys,
      successes: options.successes,
      trials: options.trials,
      variogram: {
        variogramType: v.variogramType,
        nugget: v.nugget,
        sill: v.sill,
        range: v.range,
        shape: v.shape,
      },
      majorAngleDeg: options.majorAngleDeg,
      rangeRatio: options.rangeRatio,
      ...(options.stability !== undefined ? { stability: options.stability } : {}),
      ...(options.oneStepLaplaceObservationVariance === true
        ? { oneStepLaplaceObservationVariance: true }
        : {}),
    });
  }

  static fromFittedVariogramWithPrior(
    options: BinomialProjectedKrigingFromFittedVariogramWithPriorOptions
  ): BinomialProjectedKriging {
    const v = options.fittedVariogram;
    return BinomialProjectedKriging.newWithPrior({
      xs: options.xs,
      ys: options.ys,
      successes: options.successes,
      trials: options.trials,
      variogram: {
        variogramType: v.variogramType,
        nugget: v.nugget,
        sill: v.sill,
        range: v.range,
        shape: v.shape,
      },
      majorAngleDeg: options.majorAngleDeg,
      rangeRatio: options.rangeRatio,
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
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      PROJECTED_BINOMIAL_FREED
    );
  }

  diagnostics(counts?: ProjectedBinomialDiagnosticsCounts): BinomialDiagnostics {
    return getBinomialDiagnostics2d(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      PROJECTED_BINOMIAL_FREED,
      counts,
      packProjectedDiagnosticsOpts
    );
  }

  predict(x: number, y: number): BinomialPrediction {
    return predictBinomialProjected(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      x,
      y
    );
  }

  predictBatch(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): BinomialPrediction[] {
    return predictBatchBinomialProjected(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      xs,
      ys
    );
  }

  predictBatchArrays(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): BinomialBatchArrayOutput {
    return predictBatchArraysBinomialProjected(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      xs,
      ys
    );
  }

  predictGrid(options: PredictProjectedGridOptions): BinomialGridOutput {
    return predictGridBinomialProjected(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      options
    );
  }

  /**
   * Leave-one-out CV on **this fitted model** (same training data and variogram).
   * Prefer {@link leaveOneOut} when validating from raw arrays before building a model.
   */
  leaveOneOut(): BinomialCvResult {
    return binomialLeaveOneOut(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      PROJECTED_BINOMIAL_FREED
    );
  }

  /**
   * K-fold CV on **this fitted model** (deterministic round-robin folds).
   * Prefer {@link kFold} when validating from raw arrays before building a model.
   */
  kFold(k: number): BinomialCvResult {
    return binomialKFold(
      requireBinomialHandle(this.inner, PROJECTED_BINOMIAL_FREED),
      PROJECTED_BINOMIAL_FREED,
      k
    );
  }
}
