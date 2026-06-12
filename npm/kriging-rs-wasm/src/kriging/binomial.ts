/**
 * Binomial kriging: prevalence / proportion surfaces from count data (successes/trials).
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
  BinomialKrigingOptionsWasm,
  BinomialKrigingWithPriorOptionsWasm,
  WasmBinomialInstance,
} from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialDiagnostics,
  BinomialFromPrecomputedLogitsOptions,
  BinomialFromPrecomputedLogitsWithVariancesOptions,
  BinomialGridOutput,
  BinomialKrigingFromFittedVariogramOptions,
  BinomialKrigingFromFittedVariogramWithPriorOptions,
  BinomialKrigingOptions,
  BinomialKrigingWithPriorOptions,
  BinomialPrediction,
  IntegerArrayInput,
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
  private inner: WasmBinomialInstance | null;

  /**
   * Build a binomial kriging model from locations, success/trial counts, and variogram parameters.
   *
   * @param options - Sample coordinates, successes, trials, and variogram (nugget, sill, range, type, optional shape).
   * @throws {KrigingError} When the WASM module is not loaded or when inputs are invalid.
   */
  constructor(options: BinomialKrigingOptions) {
    const mod = requireLoadedModule();
    try {
      this.inner = mod.WasmBinomialKriging.fromArrays(
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

  private requireInner(): WasmBinomialInstance {
    if (this.inner === null) {
      throw new KrigingError(BINOMIAL_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Build a binomial kriging model from locations and **pre-computed logits**. Bypasses
   * the empirical-Bayes shrinkage step used by the default constructor; use when you
   * already have finite logit estimates (e.g. from an externally fitted mean-field
   * model).
   *
   * @throws {KrigingError} When logits are non-finite or the WASM module is not loaded.
   */
  static fromPrecomputedLogits(
    options: BinomialFromPrecomputedLogitsOptions
  ): BinomialKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialKriging;
    const instance = Object.create(
      BinomialKriging.prototype
    ) as BinomialKriging;
    try {
      (instance as unknown as { inner: WasmBinomialInstance | null }).inner =
        ctor.fromPrecomputedLogits(
          toFloat64Array(options.lats),
          toFloat64Array(options.lons),
          toFloat64Array(options.logits),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape
        );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Like {@link BinomialKriging.fromPrecomputedLogits}, with a per-site **logit observation
   * variance** vector (diagonal). Optional `prior` sets the Beta prior recorded in build notes.
   */
  static fromPrecomputedLogitsWithVariances(
    options: BinomialFromPrecomputedLogitsWithVariancesOptions
  ): BinomialKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialKriging;
    const instance = Object.create(
      BinomialKriging.prototype
    ) as BinomialKriging;
    const priorAlpha = options.prior?.alpha;
    const priorBeta = options.prior?.beta;
    try {
      (instance as unknown as { inner: WasmBinomialInstance | null }).inner =
        ctor.fromPrecomputedLogitsWithVariances(
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
        );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Create a binomial kriging model with a Beta(alpha, beta) prior on prevalence.
   * Useful when counts are small or some locations have zero trials.
   *
   * @param options - Sample coordinates, successes, trials, variogram, and prior { alpha, beta }.
   * @returns A new BinomialKriging instance.
   * @throws {KrigingError} When the WASM module is not loaded or when inputs are invalid.
   */
  static newWithPrior(
    options: BinomialKrigingWithPriorOptions
  ): BinomialKriging {
    const mod = requireLoadedModule();
    const instance = Object.create(
      BinomialKriging.prototype
    ) as BinomialKriging;
    try {
      (instance as unknown as { inner: WasmBinomialInstance | null }).inner =
        mod.WasmBinomialKriging.newWithPrior(
          toBinomialWithPriorOptionsWasm(options)
        );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Build a binomial kriging model from count data and a fitted variogram. Prefer
   * {@link fitBinomialVariogram} on the same `successes` / `trials` (and matching
   * `prior` when using {@link BinomialKriging.fromFittedVariogramWithPrior}) so the
   * fit uses the calibrated binomial empirical variogram aligned with this model.
   *
   * @param options - Sample lats, lons, successes, trials, and a {@link FittedVariogram}.
   * @returns A new BinomialKriging instance.
   * @throws {KrigingError} When the WASM module is not loaded or when inputs are invalid.
   */
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

  /**
   * Build a binomial kriging model with a Beta prior from count data and a fitted variogram.
   * Prefer {@link fitBinomialVariogram} with the same `prior` as here so variogram
   * fitting matches the kriger’s shrinkage and observation variances.
   *
   * @param options - Sample lats, lons, successes, trials, a {@link FittedVariogram}, and prior { alpha, beta }.
   * @returns A new BinomialKriging instance.
   * @throws {KrigingError} When the WASM module is not loaded or when inputs are invalid.
   */
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

  /**
   * Release WASM-held resources. Safe to call multiple times; subsequent calls are no-ops.
   * You can call free() in a finally block after creating a model.
   */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /**
   * Build-time diagnostics (prior, dropped zero-trial rows, logit inflation, warnings, …).
   * Backed by WASM `getBuildNotes` JSON (camelCase).
   */
  get buildNotes(): BinomialBuildNotes {
    try {
      return mapBinomialBuildNotes(this.requireInner().getBuildNotes());
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Variogram parameters, {@link BinomialBuildNotes}, and optional leave-one-out logit MSDR.
   *
   * Pass the same `lats` / `lons` / `successes` / `trials` used to construct the model (after
   * any zero-trial drops) to populate `logitLooMsdr`; omit to skip LOO.
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
          "BinomialKriging.diagnostics requires WASM getDiagnostics (geographic binomial)",
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

  /**
   * Single-point prevalence prediction at (lat, lon) in degrees.
   *
   * @param lat - Latitude in degrees.
   * @param lon - Longitude in degrees.
   * @returns Prevalence in [0, 1], logit value, and kriging variance.
   */
  predict(lat: number, lon: number): BinomialPrediction {
    return mapBinomialPrediction(this.requireInner().predict(lat, lon));
  }

  /**
   * Batch prevalence prediction at multiple (lat, lon) pairs. For large grids, prefer
   * {@link BinomialKriging.predictBatchArrays}.
   *
   * @param lats - Latitudes in degrees (same length as lons).
   * @param lons - Longitudes in degrees (same length as lats).
   * @returns Array of prevalence predictions for each location.
   */
  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapBinomialPredictionArray(out);
  }

  /**
   * Batch prevalence prediction returning typed arrays. Prefer over
   * {@link BinomialKriging.predictBatch} for large grids.
   *
   * @param lats - Latitudes in degrees (same length as lons).
   * @param lons - Longitudes in degrees (same length as lats).
   * @returns Object with `prevalenceMedians`, `prevalenceMeans`, `logitValues`, `logitVariances`,
   *   and `prevalenceVariances` Float64Arrays.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapBinomialBatchArrayOutput(out);
  }

  /**
   * Predict prevalence on a rectangular grid. Same as {@link OrdinaryKriging.predictGrid}
   * but returns prevalences and logit values. Grid shape [yCells][xCells].
   *
   * @param options - west, south, east, north (degrees), xCells, yCells
   * @returns Object with prevalence median/mean grids, logits, logit variances, and prevalence
   *   variance grids as number[][] (shape [yCells][xCells]).
   */
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

  /**
   * Batch prevalence prediction using WebGPU. Requires build with GPU feature.
   * Throws {@link KrigingError} with code `backend_unavailable` if the GPU path fails for any
   * reason. Use {@link BinomialKriging.predictBatchGpuOrCpu} when silent CPU fallback is
   * acceptable.
   *
   * @param lats - Latitudes in degrees (same length as lons).
   * @param lons - Longitudes in degrees (same length as lats).
   * @returns Promise resolving to an array of prevalence predictions for each location.
   * @throws {KrigingError} code `backend_unavailable` if the GPU path fails, or if the package
   *   was not built with the `gpu` feature.
   */
  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<BinomialPrediction[]> {
    const inner = this.requireInner();
    if (typeof inner.predictBatchGpu !== "function") {
      throw new KrigingError(
        'predictBatchGpu not available; rebuild WASM package with feature "gpu"',
        { code: "backend_unavailable" }
      );
    }
    try {
      const out = await inner.predictBatchGpu(
        toFloat64Array(lats),
        toFloat64Array(lons)
      );
      return mapBinomialPredictionArray(out);
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * GPU-first batch prevalence prediction with automatic CPU fallback on any GPU error.
   */
  async predictBatchGpuOrCpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<BinomialPrediction[]> {
    const inner = this.requireInner();
    const latArr = toFloat64Array(lats);
    const lonArr = toFloat64Array(lons);
    if (typeof inner.predictBatchGpuOrCpu === "function") {
      try {
        const out = await inner.predictBatchGpuOrCpu(latArr, lonArr);
        return mapBinomialPredictionArray(out);
      } catch (e) {
        throw wrapThrown(e);
      }
    }
    return this.predictBatch(latArr, lonArr);
  }
}
