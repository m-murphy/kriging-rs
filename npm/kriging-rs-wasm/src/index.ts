export type VariogramType = "spherical" | "exponential" | "gaussian";
export type NumericArrayInput = number[] | ArrayLike<number>;
export type IntegerArrayInput = number[] | ArrayLike<number>;

export interface OrdinaryPrediction {
  value: number;
  variance: number;
}

export interface BinomialPrediction {
  prevalence: number;
  logitValue: number;
  variance: number;
}

export interface OrdinaryBatchArrayOutput {
  values: Float64Array;
  variances: Float64Array;
}

export interface BinomialBatchArrayOutput {
  prevalences: Float64Array;
  logitValues: Float64Array;
  variances: Float64Array;
}

export interface FittedVariogram {
  variogramType: VariogramType;
  nugget: number;
  sill: number;
  range: number;
  residuals: number;
}

type RawModule = {
  default: (input?: unknown) => Promise<unknown>;
  WasmOrdinaryKriging: new (...args: unknown[]) => any;
  WasmBinomialKriging: {
    new (...args: unknown[]): any;
    newWithPrior: (...args: unknown[]) => any;
  };
  fitOrdinaryVariogram: (...args: unknown[]) => unknown;
  webgpuAvailable?: (...args: unknown[]) => Promise<unknown>;
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
    throw new Error('WASM module is not loaded; call and await init() before using APIs');
  }
  return rawModuleLoaded;
}

export async function init(input?: unknown): Promise<unknown> {
  const mod = await loadRawModule();
  rawModuleLoaded = mod;
  return mod.default(input);
}

export default init;

export class OrdinaryKriging {
  private readonly inner: any;

  constructor(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    values: NumericArrayInput,
    variogramType: VariogramType,
    nugget: number,
    sill: number,
    range: number
  ) {
    const mod = requireLoadedModule();
    this.inner = new mod.WasmOrdinaryKriging(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(values),
      variogramType,
      nugget,
      sill,
      range
    );
  }

  free(): void {
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
  }

  predict(lat: number, lon: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.inner.predict(lat, lon));
  }

  predictBatch(lats: NumericArrayInput, lons: NumericArrayInput): OrdinaryPrediction[] {
    const out = this.inner.predictBatch(toFloat64Array(lats), toFloat64Array(lons));
    return mapOrdinaryPredictionArray(out);
  }

  predictBatchArrays(lats: NumericArrayInput, lons: NumericArrayInput): OrdinaryBatchArrayOutput {
    const out = this.inner.predictBatchArrays(toFloat64Array(lats), toFloat64Array(lons));
    return mapOrdinaryBatchArrayOutput(out);
  }

  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<OrdinaryPrediction[]> {
    if (typeof this.inner.predictBatchGpu !== "function") {
      throw new Error('predictBatchGpu not available; rebuild WASM package with feature "gpu"');
    }
    const out = await this.inner.predictBatchGpu(toFloat64Array(lats), toFloat64Array(lons));
    return mapOrdinaryPredictionArray(out);
  }
}

export class BinomialKriging {
  private inner: any;

  constructor(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    successes: IntegerArrayInput,
    trials: IntegerArrayInput,
    variogramType: VariogramType,
    nugget: number,
    sill: number,
    range: number
  ) {
    const mod = requireLoadedModule();
    this.inner = new mod.WasmBinomialKriging(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toUint32Array(successes),
      toUint32Array(trials),
      variogramType,
      nugget,
      sill,
      range
    );
  }

  static newWithPrior(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    successes: IntegerArrayInput,
    trials: IntegerArrayInput,
    variogramType: VariogramType,
    nugget: number,
    sill: number,
    range: number,
    alpha: number,
    beta: number
  ): BinomialKriging {
    const mod = requireLoadedModule();
    const instance = Object.create(BinomialKriging.prototype) as BinomialKriging;
    instance.inner = mod.WasmBinomialKriging.newWithPrior(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toUint32Array(successes),
      toUint32Array(trials),
      variogramType,
      nugget,
      sill,
      range,
      alpha,
      beta
    );
    return instance;
  }

  free(): void {
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
  }

  predict(lat: number, lon: number): BinomialPrediction {
    return mapBinomialPrediction(this.inner.predict(lat, lon));
  }

  predictBatch(lats: NumericArrayInput, lons: NumericArrayInput): BinomialPrediction[] {
    const out = this.inner.predictBatch(toFloat64Array(lats), toFloat64Array(lons));
    return mapBinomialPredictionArray(out);
  }

  predictBatchArrays(lats: NumericArrayInput, lons: NumericArrayInput): BinomialBatchArrayOutput {
    const out = this.inner.predictBatchArrays(toFloat64Array(lats), toFloat64Array(lons));
    return mapBinomialBatchArrayOutput(out);
  }

  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<BinomialPrediction[]> {
    if (typeof this.inner.predictBatchGpu !== "function") {
      throw new Error('predictBatchGpu not available; rebuild WASM package with feature "gpu"');
    }
    const out = await this.inner.predictBatchGpu(toFloat64Array(lats), toFloat64Array(lons));
    return mapBinomialPredictionArray(out);
  }
}

export function fitOrdinaryVariogram(
  sampleLats: NumericArrayInput,
  sampleLons: NumericArrayInput,
  values: NumericArrayInput,
  maxDistance: number | undefined,
  nBins: number,
  variogramTypes: readonly VariogramType[]
): FittedVariogram {
  const mod = requireLoadedModule();
  const out = mod.fitOrdinaryVariogram(
    toFloat64Array(sampleLats),
    toFloat64Array(sampleLons),
    toFloat64Array(values),
    maxDistance,
    nBins,
    [...variogramTypes]
  );
  const result = asRecord(out);
  return {
    variogramType: requireVariogramType(result.variogramType),
    nugget: requireNumber(result.nugget),
    sill: requireNumber(result.sill),
    range: requireNumber(result.range),
    residuals: requireNumber(result.residuals)
  };
}

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
    variance: requireNumber(item.variance)
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
    variance: requireNumber(item.variance)
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
    variances: requireFloat64Array(out.variances)
  };
}

function mapBinomialBatchArrayOutput(value: unknown): BinomialBatchArrayOutput {
  const out = asRecord(value);
  return {
    prevalences: requireFloat64Array(out.prevalences),
    logitValues: requireFloat64Array(out.logitValues),
    variances: requireFloat64Array(out.variances)
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

function requireVariogramType(value: unknown): VariogramType {
  if (value === "spherical" || value === "exponential" || value === "gaussian") {
    return value;
  }
  throw new Error("Expected known variogram type from WASM");
}
