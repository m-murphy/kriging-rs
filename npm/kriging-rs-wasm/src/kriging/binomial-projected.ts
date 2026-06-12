/**
 * Projected (planar) binomial kriging on `(x, y)` coordinates with 2D anisotropy.
 *
 * The planar / anisotropic counterpart of {@link BinomialKriging}: count data
 * (`successes`, `trials`) is mapped to a smoothed logit scale (default
 * Beta(1, 1) prior, matching Rust) and ordinary kriging runs on those logits using
 * Euclidean (optionally anisotropy-deformed) distances. Predictions are
 * back-transformed to prevalences via the logistic function with a delta-method
 * variance approximation. See {@link BinomialKriging} for the geographic
 * (Haversine) flavor.
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
import type { WasmBinomialProjectedInstance } from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialDiagnostics,
  BinomialGridOutput,
  BinomialPrediction,
  BinomialProjectedFromPrecomputedLogitsOptions,
  BinomialProjectedFromPrecomputedLogitsWithVariancesOptions,
  BinomialProjectedKrigingFromFittedVariogramOptions,
  BinomialProjectedKrigingFromFittedVariogramWithPriorOptions,
  BinomialProjectedKrigingOptions,
  BinomialProjectedKrigingWithPriorOptions,
  IntegerArrayInput,
  NumericArrayInput,
  PredictProjectedGridOptions,
} from "../types.js";

const PROJECTED_BINOMIAL_FREED =
  "BinomialProjectedKriging model has been freed";

/**
 * Projected / planar binomial kriging on `(x, y)` coordinates with 2-D anisotropy.
 * Fits ordinary kriging on the (smoothed) logit scale and back-transforms via
 * the logistic; suitable for prevalence mapping when the data are already
 * projected (e.g. meters in a local equal-area projection) and a directional
 * range matters.
 */
export class BinomialProjectedKriging {
  private inner: WasmBinomialProjectedInstance | null;

  /**
   * Build a projected binomial kriging model with default Beta(1, 1) prior.
   * Use {@link BinomialProjectedKriging.newWithPrior} for an explicit
   * Beta prior, or {@link BinomialProjectedKriging.fromPrecomputedLogits} when
   * the caller already has finite logit estimates.
   *
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are
   *   invalid (mismatched array lengths, `trials[i] === 0`,
   *   `rangeRatio ∉ (0, 1]`, …), or when the WASM package was built without
   *   this class (code `backend_unavailable`).
   */
  constructor(options: BinomialProjectedKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
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

  private requireInner(): WasmBinomialProjectedInstance {
    if (this.inner === null) {
      throw new KrigingError(PROJECTED_BINOMIAL_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Build a projected binomial kriging model with an explicit Beta(`alpha`, `beta`)
   * prior on prevalence. Useful when counts are small.
   */
  static newWithPrior(
    options: BinomialProjectedKrigingWithPriorOptions
  ): BinomialProjectedKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const instance = Object.create(
      BinomialProjectedKriging.prototype
    ) as BinomialProjectedKriging;
    try {
      (
        instance as unknown as {
          inner: WasmBinomialProjectedInstance | null;
        }
      ).inner = ctor.fromArraysWithPrior(
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
      );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Build a projected binomial kriging model from caller-supplied logit values
   * (bypasses the empirical-Bayes shrinkage). All `logits` must be finite.
   */
  static fromPrecomputedLogits(
    options: BinomialProjectedFromPrecomputedLogitsOptions
  ): BinomialProjectedKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const instance = Object.create(
      BinomialProjectedKriging.prototype
    ) as BinomialProjectedKriging;
    try {
      (
        instance as unknown as {
          inner: WasmBinomialProjectedInstance | null;
        }
      ).inner = ctor.fromPrecomputedLogits(
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
      );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Like {@link BinomialProjectedKriging.fromPrecomputedLogits}, with per-site logit observation
   * variances on the diagonal.
   */
  static fromPrecomputedLogitsWithVariances(
    options: BinomialProjectedFromPrecomputedLogitsWithVariancesOptions
  ): BinomialProjectedKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const instance = Object.create(
      BinomialProjectedKriging.prototype
    ) as BinomialProjectedKriging;
    const priorAlpha = options.prior?.alpha;
    const priorBeta = options.prior?.beta;
    try {
      (
        instance as unknown as {
          inner: WasmBinomialProjectedInstance | null;
        }
      ).inner = ctor.fromPrecomputedLogitsWithVariances(
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
      );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Build from count data and a {@link FittedVariogram} (e.g. from {@link fitVariogram} on
   * planar sample coordinates and logits), without manually spreading nugget/sill/range.
   */
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

  /**
   * Like {@link BinomialProjectedKriging.fromFittedVariogram} with an explicit Beta prior
   * on prevalence.
   */
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

  /** Release WASM-held resources. Safe to call multiple times. */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /** Build-time diagnostics (prior, dropped rows, inflation, warnings, …). */
  get buildNotes(): BinomialBuildNotes {
    try {
      return mapBinomialBuildNotes(this.requireInner().getBuildNotes());
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Variogram, {@link BinomialBuildNotes}, and optional LOO logit MSDR. LOO counts use
   * `{ xs, ys, successes, trials }` in the same planar units as the fit.
   */
  diagnostics(counts?: {
    xs: NumericArrayInput;
    ys: NumericArrayInput;
    successes: IntegerArrayInput;
    trials: IntegerArrayInput;
  }): BinomialDiagnostics {
    try {
      const inner = this.requireInner();
      const getDiagnostics = inner.getDiagnostics;
      if (typeof getDiagnostics !== "function") {
        throw new KrigingError(
          "BinomialProjectedKriging.diagnostics requires WASM getDiagnostics",
          { code: "internal_error" }
        );
      }
      const opts: unknown =
        counts === undefined
          ? undefined
          : {
              xs: toFloat64Array(counts.xs),
              ys: toFloat64Array(counts.ys),
              successes: toUint32Array(counts.successes),
              trials: toUint32Array(counts.trials),
            };
      return mapBinomialDiagnostics(getDiagnostics.call(inner, opts));
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /** Single-point prevalence prediction at planar `(x, y)`. */
  predict(x: number, y: number): BinomialPrediction {
    return mapBinomialPrediction(this.requireInner().predict(x, y));
  }

  /** Batch prevalence prediction at multiple `(x, y)` pairs. */
  predictBatch(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): BinomialPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(xs),
      toFloat64Array(ys)
    );
    return mapBinomialPredictionArray(out);
  }

  /** Batch prevalence prediction returning typed arrays. Prefer for large grids. */
  predictBatchArrays(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(xs),
      toFloat64Array(ys)
    );
    return mapBinomialBatchArrayOutput(out);
  }

  /**
   * Predict prevalence on a rectangular `(x, y)` grid in projected units. Result arrays have
   * shape `[yCells][xCells]` (row = low to high `y`, column = low to high `x`).
   */
  predictGrid(options: PredictProjectedGridOptions): BinomialGridOutput {
    const inner = this.requireInner();
    const nRows = Math.max(1, Math.floor(options.yCells));
    const nCols = Math.max(1, Math.floor(options.xCells));
    const out = inner.predictGridArrays(
      options.xMin,
      options.xMax,
      options.yMin,
      options.yMax,
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
