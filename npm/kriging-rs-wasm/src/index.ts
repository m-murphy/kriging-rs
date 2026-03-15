/**
 * Supported variogram model type names (string form, e.g. in {@link FittedVariogram}).
 */
export type VariogramTypeName =
  | "spherical"
  | "exponential"
  | "gaussian"
  | "cubic"
  | "stable"
  | "matern";

/**
 * Input type for numeric coordinate or value arrays; accepts plain arrays or typed arrays.
 */
export type NumericArrayInput = number[] | ArrayLike<number>;

/**
 * Input type for integer counts (e.g. successes, trials); accepts plain arrays or typed arrays.
 */
export type IntegerArrayInput = number[] | ArrayLike<number>;

/**
 * Result of a single ordinary kriging prediction.
 * @property value - Interpolated value at the location
 * @property variance - Kriging variance (prediction uncertainty)
 */
export interface OrdinaryPrediction {
  value: number;
  variance: number;
}

/**
 * Result of a single binomial kriging prediction (prevalence surface).
 * @property prevalence - Estimated prevalence in [0, 1]
 * @property logitValue - Logit-scale value
 * @property variance - Kriging variance (prediction uncertainty)
 */
export interface BinomialPrediction {
  prevalence: number;
  logitValue: number;
  variance: number;
}

/**
 * Batch ordinary kriging output as typed arrays (avoids per-point object allocation).
 * Use for large prediction grids.
 */
export interface OrdinaryBatchArrayOutput {
  values: Float64Array;
  variances: Float64Array;
}

/**
 * Batch binomial kriging output as typed arrays (avoids per-point object allocation).
 * Use for large prediction grids.
 */
export interface BinomialBatchArrayOutput {
  prevalences: Float64Array;
  logitValues: Float64Array;
  variances: Float64Array;
}

/**
 * Fitted variogram parameters from {@link fitOrdinaryVariogram}.
 * Use these to construct an {@link OrdinaryKriging} model.
 */
export interface FittedVariogram {
  variogramType: VariogramTypeName;
  nugget: number;
  sill: number;
  range: number;
  /** Shape parameter (alpha for stable, nu for matern); present only for stable/matern. */
  shape?: number;
  residuals: number;
}

/**
 * Variogram parameters for model construction (nugget, sill, range, optional shape).
 */
export interface VariogramParams {
  variogramType: VariogramTypeName;
  nugget: number;
  sill: number;
  range: number;
  /** Shape parameter for stable/matern; omit for other types. */
  shape?: number;
}

/**
 * Options for constructing an ordinary kriging model. Pass a single object to
 * {@link OrdinaryKriging} constructor.
 */
export interface OrdinaryKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
}

/**
 * Options for constructing a binomial kriging model. Pass a single object to
 * {@link BinomialKriging} constructor.
 */
export interface BinomialKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: VariogramParams;
}

/**
 * Beta(alpha, beta) prior parameters for binomial kriging.
 */
export interface BinomialPriorParams {
  alpha: number;
  beta: number;
}

/**
 * Options for constructing a binomial kriging model with a prior. Pass a single
 * object to {@link BinomialKriging.newWithPrior}.
 */
export interface BinomialKrigingWithPriorOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: VariogramParams;
  prior: BinomialPriorParams;
}

/**
 * Error thrown by the library when WASM operations fail (invalid inputs, model build failure, etc.).
 * The underlying cause is attached as `cause` when available.
 */
export class KrigingError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = "KrigingError";
    if (options?.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

/** Internal: WASM ordinary kriging instance shape */
interface WasmOrdinaryInstance {
  predict(lat: number, lon: number): unknown;
  predictBatch(lats: Float64Array, lons: Float64Array): unknown;
  predictBatchArrays(lats: Float64Array, lons: Float64Array): unknown;
  free?: () => void;
  predictBatchGpu?(lats: Float64Array, lons: Float64Array): Promise<unknown>;
}

/** Internal: WASM binomial kriging instance shape */
interface WasmBinomialInstance {
  predict(lat: number, lon: number): unknown;
  predictBatch(lats: Float64Array, lons: Float64Array): unknown;
  predictBatchArrays(lats: Float64Array, lons: Float64Array): unknown;
  free?: () => void;
  predictBatchGpu?(lats: Float64Array, lons: Float64Array): Promise<unknown>;
}

/** Shape passed to WASM (plain arrays for serde deserialization). */
interface OrdinaryKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  values: number[];
  variogram: { variogramType: string; nugget: number; sill: number; range: number; shape?: number };
}

interface BinomialKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  successes: number[];
  trials: number[];
  variogram: { variogramType: string; nugget: number; sill: number; range: number; shape?: number };
}

interface BinomialKrigingWithPriorOptionsWasm extends BinomialKrigingOptionsWasm {
  prior: { alpha: number; beta: number };
}

type RawModule = {
  default: (input?: unknown) => Promise<unknown>;
  WasmOrdinaryKriging: new (options: OrdinaryKrigingOptionsWasm) => WasmOrdinaryInstance;
  WasmBinomialKriging: {
    new (options: BinomialKrigingOptionsWasm): WasmBinomialInstance;
    newWithPrior(options: BinomialKrigingWithPriorOptionsWasm): WasmBinomialInstance;
  };
  WasmVariogramType: {
    readonly Spherical: number;
    readonly Exponential: number;
    readonly Gaussian: number;
    readonly Cubic: number;
    readonly Stable: number;
    readonly Matern: number;
  };
  fitOrdinaryVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number
  ) => unknown;
  webgpuAvailable?: () => Promise<unknown>;
};

let rawModulePromise: Promise<RawModule> | null = null;
let rawModuleLoaded: RawModule | null = null;
const wasmModuleSpecifier: string = "../pkg/kriging_rs.js";

function loadRawModule(): Promise<RawModule> {
  if (!rawModulePromise) {
    rawModulePromise = import(wasmModuleSpecifier) as Promise<RawModule>;
    void rawModulePromise.then((mod) => {
      rawModuleLoaded = mod;
    });
  }
  return rawModulePromise;
}

function requireLoadedModule(): RawModule {
  if (!rawModuleLoaded) {
    throw new Error(
      "WASM module is not loaded; call and await init() before using APIs"
    );
  }
  return rawModuleLoaded;
}

/** Variogram type enum (use e.g. VariogramType.Exponential). Available after {@link init}. */
export const VariogramType: RawModule["WasmVariogramType"] = new Proxy(
  {} as RawModule["WasmVariogramType"],
  {
    get(_, prop) {
      return (requireLoadedModule().WasmVariogramType as Record<string, number>)[prop as string];
    },
  }
);

/**
 * Initialize the WebAssembly module. Call and await this once before using any other API.
 * Can be called with no arguments (loads WASM from the default location) or with pre-fetched
 * WASM bytes (e.g. from a custom URL or bundler) for offline or custom loading.
 *
 * @param input - Optional: `ArrayBuffer` or `Response` of WASM bytes; omit to use default loader
 * @returns Promise that resolves when initialization is complete
 */
export async function init(input?: unknown): Promise<unknown> {
  const mod = await loadRawModule();
  rawModuleLoaded = mod;
  return mod.default(input);
}

export default init;

function toOrdinaryOptionsWasm(opts: OrdinaryKrigingOptions): OrdinaryKrigingOptionsWasm {
  return {
    lats: Array.from(toFloat64Array(opts.lats)),
    lons: Array.from(toFloat64Array(opts.lons)),
    values: Array.from(toFloat64Array(opts.values)),
    variogram: {
      variogramType: opts.variogram.variogramType,
      nugget: opts.variogram.nugget,
      sill: opts.variogram.sill,
      range: opts.variogram.range,
      shape: opts.variogram.shape,
    },
  };
}

function toBinomialOptionsWasm(opts: BinomialKrigingOptions): BinomialKrigingOptionsWasm {
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
    }),
    prior: { alpha: opts.prior.alpha, beta: opts.prior.beta },
  };
}

/**
 * Ordinary kriging model for spatial interpolation of continuous values.
 * Coordinates are in degrees (latitude, longitude); distances use Haversine (great-circle).
 * Pass a single options object with data and variogram parameters.
 */
export class OrdinaryKriging {
  private readonly inner: WasmOrdinaryInstance;

  constructor(options: OrdinaryKrigingOptions) {
    const mod = requireLoadedModule();
    try {
      this.inner = new mod.WasmOrdinaryKriging(toOrdinaryOptionsWasm(options));
    } catch (e) {
      throw new KrigingError(e instanceof Error ? e.message : String(e), {
        cause: e,
      });
    }
  }

  /** Release WASM-held resources. Call when the model is no longer needed. */
  free(): void {
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
  }

  /** Single-point prediction at (lat, lon) in degrees. */
  predict(lat: number, lon: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.inner.predict(lat, lon));
  }

  /** Batch prediction at multiple (lat, lon) pairs; returns an array of {@link OrdinaryPrediction}. */
  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): OrdinaryPrediction[] {
    const out = this.inner.predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapOrdinaryPredictionArray(out);
  }

  /**
   * Batch prediction returning typed arrays (values, variances). Prefer over predictBatch
   * for large grids to avoid per-point object allocation.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.inner.predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }

  /**
   * Batch prediction using WebGPU when available. Requires building with `npm run build:wasm:gpu`.
   * Throws if the GPU feature was not included in the build.
   */
  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<OrdinaryPrediction[]> {
    if (typeof this.inner.predictBatchGpu !== "function") {
      throw new Error(
        'predictBatchGpu not available; rebuild WASM package with feature "gpu"'
      );
    }
    const out = await this.inner.predictBatchGpu(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapOrdinaryPredictionArray(out);
  }
}

/**
 * Binomial kriging model for prevalence (proportion) surfaces from count data (successes/trials).
 * Coordinates are in degrees; distances use Haversine. Use {@link BinomialKriging.newWithPrior}
 * to supply a Beta(alpha, beta) prior for stabilization.
 */
export class BinomialKriging {
  private inner: WasmBinomialInstance;

  constructor(options: BinomialKrigingOptions) {
    const mod = requireLoadedModule();
    try {
      this.inner = new mod.WasmBinomialKriging(toBinomialOptionsWasm(options));
    } catch (e) {
      throw new KrigingError(e instanceof Error ? e.message : String(e), {
        cause: e,
      });
    }
  }

  /**
   * Create a binomial kriging model with a Beta(alpha, beta) prior on prevalence.
   * Useful when counts are small or some locations have zero trials.
   */
  static newWithPrior(options: BinomialKrigingWithPriorOptions): BinomialKriging {
    const mod = requireLoadedModule();
    const instance = Object.create(
      BinomialKriging.prototype
    ) as BinomialKriging;
    try {
      instance.inner = mod.WasmBinomialKriging.newWithPrior(
        toBinomialWithPriorOptionsWasm(options)
      );
    } catch (e) {
      throw new KrigingError(e instanceof Error ? e.message : String(e), {
        cause: e,
      });
    }
    return instance;
  }

  /** Release WASM-held resources. Call when the model is no longer needed. */
  free(): void {
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
  }

  /** Single-point prevalence prediction at (lat, lon) in degrees. */
  predict(lat: number, lon: number): BinomialPrediction {
    return mapBinomialPrediction(this.inner.predict(lat, lon));
  }

  /** Batch prevalence prediction; returns an array of {@link BinomialPrediction}. */
  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialPrediction[] {
    const out = this.inner.predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapBinomialPredictionArray(out);
  }

  /**
   * Batch prevalence prediction returning typed arrays. Prefer for large grids.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this.inner.predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapBinomialBatchArrayOutput(out);
  }

  /**
   * Batch prevalence prediction using WebGPU when available. Requires build with GPU feature.
   */
  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<BinomialPrediction[]> {
    if (typeof this.inner.predictBatchGpu !== "function") {
      throw new Error(
        'predictBatchGpu not available; rebuild WASM package with feature "gpu"'
      );
    }
    const out = await this.inner.predictBatchGpu(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapBinomialPredictionArray(out);
  }
}

/**
 * Fit a variogram model to sample data by computing an empirical variogram and fitting
 * the specified model type. Use the returned {@link FittedVariogram} to construct an
 * {@link OrdinaryKriging} model.
 *
 * @param sampleLats - Sample latitudes (degrees)
 * @param sampleLons - Sample longitudes (degrees)
 * @param values - Sample values (same length as lats/lons)
 * @param maxDistance - Optional maximum distance for binning (same units as range); omit for auto
 * @param nBins - Number of distance bins for the empirical variogram
 * @param variogramType - Variogram model type (e.g. {@link VariogramType}.Exponential)
 * @returns Fitted variogram parameters
 */
export function fitOrdinaryVariogram(
  sampleLats: NumericArrayInput,
  sampleLons: NumericArrayInput,
  values: NumericArrayInput,
  maxDistance: number | undefined,
  nBins: number,
  variogramType: number
): FittedVariogram {
  const mod = requireLoadedModule();
  let out: unknown;
  try {
    out = mod.fitOrdinaryVariogram(
      toFloat64Array(sampleLats),
      toFloat64Array(sampleLons),
      toFloat64Array(values),
      maxDistance,
      nBins,
      variogramType
    );
  } catch (e) {
    throw new KrigingError(e instanceof Error ? e.message : String(e), {
      cause: e,
    });
  }
  const result = asRecord(out);
  const fitted: FittedVariogram = {
    variogramType: requireVariogramType(result.variogramType),
    nugget: requireNumber(result.nugget),
    sill: requireNumber(result.sill),
    range: requireNumber(result.range),
    residuals: requireNumber(result.residuals),
  };
  if (result.shape !== undefined && typeof result.shape === "number" && Number.isFinite(result.shape)) {
    fitted.shape = result.shape;
  }
  return fitted;
}

/**
 * Check whether WebGPU-backed batch prediction is available. Returns false if the package
 * was built without the `gpu` feature or if the environment does not support WebGPU.
 */
export async function webgpuAvailable(): Promise<boolean> {
  if (!rawModuleLoaded) {
    await loadRawModule();
  }
  const mod = requireLoadedModule();
  if (typeof mod.webgpuAvailable !== "function") {
    return false;
  }
  return Boolean(await mod.webgpuAvailable());
}

function toFloat64Array(input: NumericArrayInput): Float64Array {
  return input instanceof Float64Array ? input : Float64Array.from(input);
}

function toUint32Array(input: IntegerArrayInput): Uint32Array {
  return input instanceof Uint32Array ? input : Uint32Array.from(input);
}

function mapOrdinaryPrediction(value: unknown): OrdinaryPrediction {
  const item = asRecord(value);
  return {
    value: requireNumber(item.value),
    variance: requireNumber(item.variance),
  };
}

function mapOrdinaryPredictionArray(value: unknown): OrdinaryPrediction[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected ordinary prediction array output");
  }
  return value.map((item) => mapOrdinaryPrediction(item));
}

function mapBinomialPrediction(value: unknown): BinomialPrediction {
  const item = asRecord(value);
  const maybeSnake = item.logit_value;
  const maybeCamel = item.logitValue;
  const logit = maybeCamel ?? maybeSnake;
  return {
    prevalence: requireNumber(item.prevalence),
    logitValue: requireNumber(logit),
    variance: requireNumber(item.variance),
  };
}

function mapBinomialPredictionArray(value: unknown): BinomialPrediction[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected binomial prediction array output");
  }
  return value.map((item) => mapBinomialPrediction(item));
}

function mapOrdinaryBatchArrayOutput(value: unknown): OrdinaryBatchArrayOutput {
  const out = asRecord(value);
  return {
    values: requireFloat64Array(out.values),
    variances: requireFloat64Array(out.variances),
  };
}

function mapBinomialBatchArrayOutput(value: unknown): BinomialBatchArrayOutput {
  const out = asRecord(value);
  return {
    prevalences: requireFloat64Array(out.prevalences),
    logitValues: requireFloat64Array(out.logitValues),
    variances: requireFloat64Array(out.variances),
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null) {
    throw new Error("Expected object output from WASM");
  }
  return value as Record<string, unknown>;
}

function requireFloat64Array(value: unknown): Float64Array {
  if (!(value instanceof Float64Array)) {
    throw new Error("Expected Float64Array output from WASM");
  }
  return value;
}

function requireNumber(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("Expected finite numeric output from WASM");
  }
  return value;
}

function requireVariogramType(value: unknown): VariogramTypeName {
  if (
    value === "spherical" ||
    value === "exponential" ||
    value === "gaussian" ||
    value === "cubic" ||
    value === "stable" ||
    value === "matern"
  ) {
    return value;
  }
  throw new Error("Expected known variogram type from WASM");
}
