/**
 * Shared lifecycle, diagnostics, CV, and batch prediction for 2-D binomial WASM handles.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array, toUint32Array } from "./convert.js";
import { reshapeFlatToGrid } from "./grid.js";
import {
  mapBinomialBatchArrayOutput,
  mapBinomialBuildNotes,
  mapBinomialDiagnostics,
  mapBinomialPrediction,
  mapBinomialPredictionArray,
} from "./mappers.js";
import { modelKFold, modelLeaveOneOut } from "./model-cv.js";
import type { WasmKrigingModelHandle } from "./wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResult,
  BinomialDiagnostics,
  BinomialGridOutput,
  BinomialPrediction,
  IntegerArrayInput,
  NumericArrayInput,
  PredictGridOptions,
  PredictProjectedGridOptions,
} from "../types.js";

export type GeoBinomialDiagnosticsCounts = {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
};

export type ProjectedBinomialDiagnosticsCounts = {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
};

export function attachBinomialHandle<T>(
  proto: object,
  inner: WasmKrigingModelHandle
): T {
  const instance = Object.create(proto) as T & {
    inner: WasmKrigingModelHandle | null;
  };
  instance.inner = inner;
  return instance as T;
}

export function requireBinomialHandle(
  inner: WasmKrigingModelHandle | null,
  freedMessage: string
): WasmKrigingModelHandle {
  if (inner === null) {
    throw new KrigingError(freedMessage, { code: "model_freed" });
  }
  return inner;
}

export function freeBinomialHandle(
  inner: WasmKrigingModelHandle | null
): null {
  if (inner === null) return null;
  if (typeof inner.free === "function") {
    inner.free();
  }
  return null;
}

export function getBinomialBuildNotes(
  inner: WasmKrigingModelHandle,
  freedMessage: string
): BinomialBuildNotes {
  try {
    return mapBinomialBuildNotes(
      requireBinomialHandle(inner, freedMessage).getBuildNotes()
    );
  } catch (e) {
    throw wrapThrown(e);
  }
}

export function getBinomialDiagnostics2d<
  T extends GeoBinomialDiagnosticsCounts | ProjectedBinomialDiagnosticsCounts,
>(
  inner: WasmKrigingModelHandle,
  freedMessage: string,
  counts: T | undefined,
  packOpts: (counts: T) => unknown
): BinomialDiagnostics {
  try {
    const handle = requireBinomialHandle(inner, freedMessage);
    const getDiagnostics = handle.getDiagnostics;
    if (typeof getDiagnostics !== "function") {
      throw new KrigingError(
        "binomial diagnostics requires WASM getDiagnostics",
        { code: "internal_error" }
      );
    }
    const opts: unknown =
      counts === undefined ? undefined : packOpts(counts);
    return mapBinomialDiagnostics(
      opts === undefined
        ? getDiagnostics.call(handle)
        : getDiagnostics.call(handle, opts)
    );
  } catch (e) {
    throw wrapThrown(e);
  }
}

export function packGeoDiagnosticsOpts(
  counts: GeoBinomialDiagnosticsCounts
): unknown {
  return {
    lats: toFloat64Array(counts.lats),
    lons: toFloat64Array(counts.lons),
    successes: toUint32Array(counts.successes),
    trials: toUint32Array(counts.trials),
  };
}

export function packProjectedDiagnosticsOpts(
  counts: ProjectedBinomialDiagnosticsCounts
): unknown {
  return {
    xs: toFloat64Array(counts.xs),
    ys: toFloat64Array(counts.ys),
    successes: toUint32Array(counts.successes),
    trials: toUint32Array(counts.trials),
  };
}

export function binomialLeaveOneOut(
  inner: WasmKrigingModelHandle,
  freedMessage: string
): BinomialCvResult {
  return modelLeaveOneOut(
    requireBinomialHandle(inner, freedMessage),
    "binomial"
  ) as BinomialCvResult;
}

export function binomialKFold(
  inner: WasmKrigingModelHandle,
  freedMessage: string,
  k: number
): BinomialCvResult {
  return modelKFold(
    requireBinomialHandle(inner, freedMessage),
    k,
    "binomial"
  ) as BinomialCvResult;
}

export function predictBinomialGeo(
  inner: WasmKrigingModelHandle,
  lat: number,
  lon: number
): BinomialPrediction {
  return mapBinomialPrediction(inner.predict(lat, lon));
}

export function predictBatchBinomialGeo(
  inner: WasmKrigingModelHandle,
  lats: NumericArrayInput,
  lons: NumericArrayInput
): BinomialPrediction[] {
  const out = inner.predictBatch(toFloat64Array(lats), toFloat64Array(lons));
  return mapBinomialPredictionArray(out);
}

export function predictBatchArraysBinomialGeo(
  inner: WasmKrigingModelHandle,
  lats: NumericArrayInput,
  lons: NumericArrayInput
): BinomialBatchArrayOutput {
  const out = inner.predictBatchArrays(
    toFloat64Array(lats),
    toFloat64Array(lons)
  );
  return mapBinomialBatchArrayOutput(out);
}

export function predictGridBinomialGeo(
  inner: WasmKrigingModelHandle,
  options: PredictGridOptions
): BinomialGridOutput {
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

export async function predictBatchGpuBinomialGeo(
  inner: WasmKrigingModelHandle,
  lats: NumericArrayInput,
  lons: NumericArrayInput
): Promise<BinomialPrediction[]> {
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

export async function predictBatchGpuOrCpuBinomialGeo(
  inner: WasmKrigingModelHandle,
  lats: NumericArrayInput,
  lons: NumericArrayInput,
  cpuFallback: () => BinomialPrediction[]
): Promise<BinomialPrediction[]> {
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
  return cpuFallback();
}

export function predictBinomialProjected(
  inner: WasmKrigingModelHandle,
  x: number,
  y: number
): BinomialPrediction {
  return mapBinomialPrediction(inner.predict(x, y));
}

export function predictBatchBinomialProjected(
  inner: WasmKrigingModelHandle,
  xs: NumericArrayInput,
  ys: NumericArrayInput
): BinomialPrediction[] {
  const out = inner.predictBatch(toFloat64Array(xs), toFloat64Array(ys));
  return mapBinomialPredictionArray(out);
}

export function predictBatchArraysBinomialProjected(
  inner: WasmKrigingModelHandle,
  xs: NumericArrayInput,
  ys: NumericArrayInput
): BinomialBatchArrayOutput {
  const out = inner.predictBatchArrays(toFloat64Array(xs), toFloat64Array(ys));
  return mapBinomialBatchArrayOutput(out);
}

export function predictGridBinomialProjected(
  inner: WasmKrigingModelHandle,
  options: PredictProjectedGridOptions
): BinomialGridOutput {
  const nRows = Math.max(1, Math.floor(options.yCells));
  const nCols = Math.max(1, Math.floor(options.xCells));
  const out = inner.predictGridArrays(
    options.xMin,
    options.xMax,
    options.yMin,
    options.yMax,
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
