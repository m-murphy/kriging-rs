/**
 * Internal: TypeScript shapes for the WASM-exposed classes, functions, and options.
 * Purely type-level; not part of the public API.
 *
 * @module
 */

/** Instance CV methods shared by fitted WASM model handles. */
export interface WasmModelCvMethods {
  leaveOneOut(): unknown;
  kFold(k: number): unknown;
}

/** Unified WASM kriging model handle (ADR-0002). */
export interface WasmKrigingModelHandle extends WasmModelCvMethods {
  readonly geometry: string;
  readonly family: string;
  predict(a: number, b: number): unknown;
  predictSpaceTime(a: number, b: number, time: number): unknown;
  predictBatch(a: Float64Array, b: Float64Array): unknown;
  predictBatchArrays(a: Float64Array, b: Float64Array): unknown;
  predictBatchSpaceTime(
    a: Float64Array,
    b: Float64Array,
    times: Float64Array
  ): unknown;
  predictBatchArraysSpaceTime(
    a: Float64Array,
    b: Float64Array,
    times: Float64Array
  ): unknown;
  predictGridArrays(
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    xCells: number,
    yCells: number
  ): unknown;
  setNeighborhood(
    maxNeighbors: number | undefined,
    maxRadius: number | undefined
  ): void;
  neighborhood(): unknown;
  mean(): number;
  getBuildNotes(): unknown;
  getDiagnostics?(options?: unknown): unknown;
  free?: () => void;
  predictBatchGpu?(lats: Float64Array, lons: Float64Array): Promise<unknown>;
  predictBatchGpuOrCpu?(
    lats: Float64Array,
    lons: Float64Array
  ): Promise<unknown>;
}

/** Shape passed to WASM (plain arrays for serde deserialization). */
export interface OrdinaryKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  values: number[];
  variogram: {
    variogramType: string;
    nugget: number;
    sill: number;
    range: number;
    shape?: number;
  };
}

export interface BinomialKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  successes: number[];
  trials: number[];
  variogram: {
    variogramType: string;
    nugget: number;
    sill: number;
    range: number;
    shape?: number;
  };
  /** Maps to Rust `BinomialStability` preset (`default` | `strict` | `permissive`). */
  stability?: string;
  /** When true, enables one-step Laplace logit observation variance (calibration version 3). */
  oneStepLaplaceObservationVariance?: boolean;
}

export interface BinomialKrigingWithPriorOptionsWasm extends BinomialKrigingOptionsWasm {
  prior: { alpha: number; beta: number };
}

export interface BinomialTangentPlaneKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  successes: number[];
  trials: number[];
  variogram: {
    variogramType: string;
    nugget: number;
    sill: number;
    range: number;
    shape?: number;
  };
  majorAngleDeg: number;
  rangeRatio: number;
  tangentPlaneRefLat?: number;
  tangentPlaneRefLon?: number;
  stability?: string;
  oneStepLaplaceObservationVariance?: boolean;
}

export interface BinomialTangentPlaneKrigingWithPriorOptionsWasm
  extends BinomialTangentPlaneKrigingOptionsWasm {
  prior: { alpha: number; beta: number };
}

/** Shape of the raw `pkg/kriging_rs.js` glue module, typed for TS consumers. */
export type RawModule = {
  default: (input?: unknown) => Promise<unknown>;
  WasmKrigingModel: {
    ordinaryGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      values: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmKrigingModelHandle;
    ordinaryGeoNew(options: OrdinaryKrigingOptionsWasm): WasmKrigingModelHandle;
    simpleGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      values: Float64Array,
      mean: number,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmKrigingModelHandle;
    universalGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      values: Float64Array,
      trend: string,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmKrigingModelHandle;
    projectedOrdinaryFromArrays(
      xs: Float64Array,
      ys: Float64Array,
      values: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number
    ): WasmKrigingModelHandle;
    binomialGeoNew(options: BinomialKrigingOptionsWasm): WasmKrigingModelHandle;
    binomialGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    binomialGeoNewWithPrior(
      options: BinomialKrigingWithPriorOptionsWasm
    ): WasmKrigingModelHandle;
    binomialGeoFromPrecomputedLogits(
      lats: Float64Array,
      lons: Float64Array,
      logits: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmKrigingModelHandle;
    binomialGeoFromPrecomputedLogitsWithVariances(
      lats: Float64Array,
      lons: Float64Array,
      logits: Float64Array,
      logitObservationVariance: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      priorAlpha: number | undefined,
      priorBeta: number | undefined,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    binomialProjectedFromArrays(
      xs: Float64Array,
      ys: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    binomialProjectedFromArraysWithPrior(
      xs: Float64Array,
      ys: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number,
      priorAlpha: number,
      priorBeta: number,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    binomialProjectedFromPrecomputedLogits(
      xs: Float64Array,
      ys: Float64Array,
      logits: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number
    ): WasmKrigingModelHandle;
    binomialProjectedFromPrecomputedLogitsWithVariances(
      xs: Float64Array,
      ys: Float64Array,
      logits: Float64Array,
      logitObservationVariance: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number,
      priorAlpha: number | undefined,
      priorBeta: number | undefined,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    binomialTangentPlaneNew(
      options: BinomialTangentPlaneKrigingOptionsWasm
    ): WasmKrigingModelHandle;
    binomialTangentPlaneNewWithPrior(
      options: BinomialTangentPlaneKrigingWithPriorOptionsWasm
    ): WasmKrigingModelHandle;
    binomialTangentPlaneFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number,
      tangentPlaneRefLat: number | undefined,
      tangentPlaneRefLon: number | undefined,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    spacetimeOrdinaryGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      values: Float64Array,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmKrigingModelHandle;
    spacetimeSimpleGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      values: Float64Array,
      mean: number,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmKrigingModelHandle;
    spacetimeUniversalGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      values: Float64Array,
      trend: string,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmKrigingModelHandle;
    spacetimeBinomialGeoFromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    spacetimeBinomialGeoFromArraysWithPrior(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      priorAlpha: number,
      priorBeta: number,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    spacetimeBinomialGeoFromPrecomputedLogits(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      logits: Float64Array,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmKrigingModelHandle;
    spacetimeBinomialGeoFromPrecomputedLogitsWithVariances(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      logits: Float64Array,
      logitObservationVariance: Float64Array,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined,
      priorAlpha: number | undefined,
      priorBeta: number | undefined,
      stability?: string,
      oneStepLaplaceObservationVariance?: boolean
    ): WasmKrigingModelHandle;
    spacetimeOrdinaryProjectedFromArrays(
      xs: Float64Array,
      ys: Float64Array,
      times: Float64Array,
      values: Float64Array,
      majorAngleDeg: number,
      rangeRatio: number,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmKrigingModelHandle;
  };
  WasmVariogramType: {
    readonly Spherical: number;
    readonly Exponential: number;
    readonly Gaussian: number;
    readonly Cubic: number;
    readonly Stable: number;
    readonly Matern: number;
    readonly Power: number;
    readonly HoleEffect: number;
  };
  fitVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number,
    estimator?: string
  ) => unknown;
  fitBinomialVariogram: (
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
  estimateBinomialPrior: (
    successes: Uint32Array,
    trials: Uint32Array
  ) => unknown;
  computeEmpiricalVariogram: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    estimator?: string
  ) => unknown;
  computeDirectionalEmpiricalVariogram: (
    xs: Float64Array,
    ys: Float64Array,
    values: Float64Array,
    maxDistance: number,
    nBins: number,
    azimuthDeg: number,
    toleranceDeg: number
  ) => unknown;
  cv: (options: Record<string, unknown>) => unknown;
  simulate: (options: Record<string, unknown>) => unknown;
  webgpuAvailable?: () => boolean;
  evaluateNestedVariogram: (
    components: unknown,
    distances: Float64Array
  ) => unknown;
  aggregatePolygonsOverEnsemble: (
    samples: Float64Array,
    nRealizations: number,
    nTargets: number,
    polygonIndices: Uint32Array,
    polygonWeights: Float64Array,
    polygonOffsets: Uint32Array,
    quantiles: Float64Array
  ) => unknown;
  wasmComputeEmpiricalSpaceTimeVariogram: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    maxSpatialDistance: number | undefined,
    maxTemporalLag: number | undefined,
    nSpatialBins: number,
    nTemporalBins: number,
    estimator: string
  ) => unknown;
  wasmFitSpaceTimeVariogram: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    maxSpatialDistance: number | undefined,
    maxTemporalLag: number | undefined,
    nSpatialBins: number,
    nTemporalBins: number,
    estimator: string,
    family: string,
    spatialModel: string,
    temporalModel: string
  ) => unknown;
};
