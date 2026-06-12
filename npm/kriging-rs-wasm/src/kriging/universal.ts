/**
 * Universal kriging: ordinary kriging with a polynomial drift basis in lat/lon.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import {
  mapOrdinaryBatchArrayOutput,
  mapOrdinaryPrediction,
  mapOrdinaryPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import { modelKFold, modelLeaveOneOut } from "../internal/model-cv.js";
import type { WasmKrigingModelHandle } from "../internal/wasm-shapes.js";
import type {
  CvResult,
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  UniversalKrigingOptions,
} from "../types.js";

const UNIVERSAL_FREED = "UniversalKriging model has been freed";

/**
 * Universal kriging model with a polynomial drift basis. Choose `"constant"` (equivalent
 * to ordinary kriging), `"linear"` (linear drift in lat/lon), or `"quadratic"`.
 */
export class UniversalKriging {
  private inner: WasmKrigingModelHandle | null;

  /**
   * Build a universal kriging model from sample data, variogram parameters, and a
   * polynomial trend basis.
   *
   * @param options - Sample coordinates, values, variogram, and `trend` (`"constant"`,
   *   `"linear"`, or `"quadratic"`).
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are invalid
   *   (mismatched array lengths, singular covariance, unknown trend), or when the WASM
   *   package was built without this class (code `backend_unavailable`).
   */
  constructor(options: UniversalKrigingOptions) {
    const mod = requireLoadedModule();
    const factory = mod.WasmKrigingModel?.universalGeoFromArrays;
    if (!factory) {
      throw new KrigingError(
        "UniversalKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = factory.call(
        mod.WasmKrigingModel,
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.values),
        options.trend,
        options.variogram.variogramType,
        options.variogram.nugget,
        options.variogram.sill,
        options.variogram.range,
        options.variogram.shape
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmKrigingModelHandle {
    if (this.inner === null) {
      throw new KrigingError(UNIVERSAL_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Release WASM-held resources. Safe to call multiple times; subsequent calls are no-ops.
   */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /**
   * Single-point prediction at `(lat, lon)` in degrees.
   *
   * @returns Interpolated value and kriging variance (includes trend uncertainty).
   */
  predict(lat: number, lon: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(lat, lon));
  }

  /**
   * Batch prediction at multiple `(lat, lon)` pairs. For large grids prefer
   * {@link UniversalKriging.predictBatchArrays} to avoid per-point object allocation.
   */
  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): OrdinaryPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapOrdinaryPredictionArray(out);
  }

  /**
   * Batch prediction returning typed arrays `{ values, variances }`. Prefer over
   * {@link UniversalKriging.predictBatch} for large grids.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons)
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
