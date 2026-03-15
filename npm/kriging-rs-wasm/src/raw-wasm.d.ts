declare module "../pkg/kriging_rs.js" {
  const init: (input?: unknown) => Promise<unknown>;
  export default init;

  export const initSync: (module: unknown) => unknown;
  export const WasmOrdinaryKriging: new (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    variogram_type: string,
    nugget: number,
    sill: number,
    range: number
  ) => unknown;
  export const WasmBinomialKriging: {
    new (
      lats: Float64Array,
      lons: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogram_type: string,
      nugget: number,
      sill: number,
      range: number
    ): unknown;
    newWithPrior(
      lats: Float64Array,
      lons: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogram_type: string,
      nugget: number,
      sill: number,
      range: number,
      alpha: number,
      beta: number
    ): unknown;
  };
  export const WasmVariogramType: {
    readonly Spherical: number;
    readonly Exponential: number;
    readonly Gaussian: number;
    readonly Cubic: number;
    readonly Stable: number;
    readonly Matern: number;
  };
  export const fitOrdinaryVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number
  ) => unknown;
  export const webgpuAvailable: (...args: unknown[]) => Promise<unknown>;
}
