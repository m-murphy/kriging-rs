declare module "../pkg/kriging_rs.js" {
  const init: (input?: unknown) => Promise<unknown>;
  export default init;

  export const initSync: (module: unknown) => unknown;

  /** Unified model handle factories (ADR-0002). See internal/wasm-shapes.ts. */
  export const WasmKrigingModel: {
    ordinaryGeoFromArrays(...args: unknown[]): unknown;
    ordinaryGeoNew(options: unknown): unknown;
    simpleGeoFromArrays(...args: unknown[]): unknown;
    universalGeoFromArrays(...args: unknown[]): unknown;
    projectedOrdinaryFromArrays(...args: unknown[]): unknown;
    binomialGeoNew(options: unknown): unknown;
    binomialGeoFromArrays(...args: unknown[]): unknown;
    binomialGeoNewWithPrior(options: unknown): unknown;
    binomialGeoFromPrecomputedLogits(...args: unknown[]): unknown;
    binomialGeoFromPrecomputedLogitsWithVariances(...args: unknown[]): unknown;
    binomialProjectedFromArrays(...args: unknown[]): unknown;
    binomialProjectedFromArraysWithPrior(...args: unknown[]): unknown;
    binomialProjectedFromPrecomputedLogits(...args: unknown[]): unknown;
    binomialProjectedFromPrecomputedLogitsWithVariances(...args: unknown[]): unknown;
    binomialTangentPlaneNew(options: unknown): unknown;
    binomialTangentPlaneNewWithPrior(options: unknown): unknown;
    binomialTangentPlaneFromArrays(...args: unknown[]): unknown;
    spacetimeOrdinaryGeoFromArrays(...args: unknown[]): unknown;
    spacetimeSimpleGeoFromArrays(...args: unknown[]): unknown;
    spacetimeUniversalGeoFromArrays(...args: unknown[]): unknown;
    spacetimeBinomialGeoFromArrays(...args: unknown[]): unknown;
    spacetimeBinomialGeoFromArraysWithPrior(...args: unknown[]): unknown;
    spacetimeBinomialGeoFromPrecomputedLogits(...args: unknown[]): unknown;
    spacetimeBinomialGeoFromPrecomputedLogitsWithVariances(...args: unknown[]): unknown;
    spacetimeOrdinaryProjectedFromArrays(...args: unknown[]): unknown;
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
