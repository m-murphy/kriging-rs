declare module "../pkg/kriging_rs.js" {
  const init: (input?: unknown) => Promise<unknown>;
  export default init;

  export const initSync: (module: unknown) => unknown;
  export const WasmOrdinaryKriging: new (options: unknown) => unknown;
  export const WasmBinomialKriging: {
    new (options: unknown): unknown;
    newWithPrior(options: unknown): unknown;
  };
  export const WasmVariogramType: {
    readonly Spherical: number;
    readonly Exponential: number;
    readonly Gaussian: number;
    readonly Cubic: number;
    readonly Stable: number;
    readonly Matern: number;
  };
  export const fitVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number
  ) => unknown;
  export const webgpuAvailable: (...args: unknown[]) => Promise<unknown>;
}
