/**
 * Space-time ordinary kriging over geographic coordinates with a scalar time axis.
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
import { packSpaceTimeVariogram } from "../internal/spacetime.js";
import type { WasmSpaceTimeInstance } from "../internal/wasm-shapes.js";
import type {
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  SpaceTimeOrdinaryKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeOrdinaryKriging model has been freed";

/**
 * Space-time ordinary kriging model. Spatial coordinates are interpreted as
 * geographic `(lat, lon)` in degrees (Haversine distances in km); `time` is an
 * arbitrary scalar (days, seconds, etc.).
 */
export class SpaceTimeOrdinaryKriging {
  private inner: WasmSpaceTimeInstance | null;

  constructor(options: SpaceTimeOrdinaryKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeOrdinaryKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeOrdinaryKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const packed = packSpaceTimeVariogram(options.variogram);
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.times),
        toFloat64Array(options.values),
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

  private requireInner(): WasmSpaceTimeInstance {
    if (this.inner === null) {
      throw new KrigingError(FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /** Release WASM-held resources. Safe to call multiple times. */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Single-point prediction at `(lat, lon, time)`. */
  predict(lat: number, lon: number, time: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(lat, lon, time));
  }

  /**
   * Batch prediction returning typed arrays `{ values, variances }`.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }
}
