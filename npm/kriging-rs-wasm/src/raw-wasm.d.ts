declare module "../pkg/kriging_rs.js" {
  const init: (input?: unknown) => Promise<unknown>;
  export default init;

  export const initSync: (module: unknown) => unknown;
  /** Use `fromArrays` for the normal fast path; see internal/wasm-shapes.ts. */
  export const WasmOrdinaryKriging: {
    new (options: unknown): unknown;
    fromArrays(...args: unknown[]): unknown;
  };
  /** Factory methods; instance shape includes getBuildNotes and predictGridArrays. */
  export const WasmBinomialKriging: {
    new (options: unknown): unknown;
    newWithPrior(options: unknown): unknown;
    fromArrays(...args: unknown[]): unknown;
    fromPrecomputedLogits(...args: unknown[]): unknown;
    fromPrecomputedLogitsWithVariances(...args: unknown[]): unknown;
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
    variogramType: number,
    estimator?: string
  ) => unknown;
  export const fitBinomialVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number,
    estimator: string | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    relWeightEps: number | undefined
  ) => unknown;
  export const estimateBinomialPrior: (
    successes: Uint32Array,
    trials: Uint32Array
  ) => unknown;
  export const webgpuAvailable: (...args: unknown[]) => Promise<unknown>;
}
