/**
 * Internal: value converters that turn raw WASM outputs into the strongly-typed shapes
 * exposed by the public API.
 *
 * @module
 */

import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResidual,
  BinomialCvResult,
  BinomialCvSummary,
  BinomialDiagnostics,
  BinomialPrediction,
  BinomialPriorParams,
  CvResidual,
  CvResult,
  CvSummary,
  EmpiricalSpaceTimeVariogramResult,
  EmpiricalVariogramResult,
  FitSpaceTimeVariogramResult,
  FittedSpaceTimeVariogram,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  PrevalenceCalibrationBin,
  SpaceTimeBinomialDiagnostics,
  SpaceTimeVariogramParams,
  VariogramParams,
  VariogramTypeName,
} from "../types.js";
import {
  asRecord,
  requireFiniteOrNaN,
  requireFloat64Array,
  requireNumber,
  requireUint32Array,
  requireVariogramType,
} from "./convert.js";

export function mapOrdinaryPrediction(value: unknown): OrdinaryPrediction {
  const item = asRecord(value);
  return {
    value: requireNumber(item.value),
    variance: requireNumber(item.variance),
  };
}

export function mapOrdinaryPredictionArray(
  value: unknown
): OrdinaryPrediction[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected ordinary prediction array output");
  }
  return value.map((item) => mapOrdinaryPrediction(item));
}

export function mapBinomialPrediction(value: unknown): BinomialPrediction {
  const item = asRecord(value);
  const prevalenceMedian = requireNumber(item.prevalenceMedian);
  const prevalenceMean = requireNumber(item.prevalenceMean);
  const logit = requireNumber(item.logit);
  const logitVariance = requireNumber(item.logitVariance);
  const explicitPv = item.prevalenceVariance;
  const prevalenceVariance =
    typeof explicitPv === "number" && Number.isFinite(explicitPv)
      ? explicitPv
      : deltaPrevalenceVariance(prevalenceMedian, logitVariance);
  return {
    prevalenceMedian,
    prevalenceMean,
    logit,
    logitVariance,
    prevalenceVariance,
  };
}

export function mapBinomialPredictionArray(
  value: unknown
): BinomialPrediction[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected binomial prediction array output");
  }
  return value.map((item) => mapBinomialPrediction(item));
}

function mapBinomialPriorFromNotes(value: unknown): BinomialPriorParams {
  const rec = asRecord(value);
  return {
    alpha: requireNumber(rec.alpha),
    beta: requireNumber(rec.beta),
  };
}

function mapZeroTrialDroppedIndices(value: unknown): number[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) {
    throw new Error("Expected zeroTrialDroppedIndices array from WASM");
  }
  return value.map((x) => requireNumber(x));
}

/** Map raw `getBuildNotes()` JSON from WASM to {@link BinomialBuildNotes}. */
export function mapBinomialBuildNotes(value: unknown): BinomialBuildNotes {
  const rec = asRecord(value);
  const priorRaw = rec.prior;
  const calVer = rec.calibrationVersion;
  const logitInfl = rec.logitInflation;
  const nAttempts = rec.nBuildAttempts;
  const dropped = rec.zeroTrialDroppedIndices;
  const fromLogitsOnly = rec.fromPrecomputedLogitsOnly;
  if (typeof fromLogitsOnly !== "boolean") {
    throw new Error("Expected boolean fromPrecomputedLogitsOnly from WASM");
  }
  const warningsRaw = rec.warnings;
  const warnings =
    warningsRaw === undefined || warningsRaw === null
      ? []
      : Array.isArray(warningsRaw)
        ? warningsRaw.map((x) => String(x))
        : [];
  return {
    calibrationVersion: requireNumber(calVer),
    logitInflation: requireNumber(logitInfl),
    nBuildAttempts: requireNumber(nAttempts),
    prior: mapBinomialPriorFromNotes(priorRaw),
    zeroTrialDroppedIndices: mapZeroTrialDroppedIndices(dropped),
    fromPrecomputedLogitsOnly: fromLogitsOnly,
    warnings,
    conditionNumber:
      rec.conditionNumber === undefined || rec.conditionNumber === null
        ? undefined
        : requireNumber(rec.conditionNumber),
    effectiveDof:
      rec.effectiveDof === undefined || rec.effectiveDof === null
        ? undefined
        : requireNumber(rec.effectiveDof),
    lastMsdr:
      rec.lastMsdr === undefined || rec.lastMsdr === null
        ? undefined
        : requireNumber(rec.lastMsdr),
  };
}

/** Map raw `getDiagnostics()` JSON from WASM to {@link BinomialDiagnostics}. */
export function mapBinomialDiagnostics(value: unknown): BinomialDiagnostics {
  const rec = asRecord(value);
  const msdr = rec.logitLooMsdr;
  return {
    variogram: mapVariogramParams(rec.variogram),
    buildNotes: mapBinomialBuildNotes(rec.buildNotes),
    logitLooMsdr:
      msdr === undefined || msdr === null ? undefined : requireNumber(msdr),
  };
}

function mapSpaceTimeVariogramParamsFromDiagnostics(
  value: unknown
): SpaceTimeVariogramParams {
  const rec = asRecord(value);
  const fam = rec.family;
  if (fam !== "separable" && fam !== "productSum") {
    throw new Error("Expected variogram.family separable|productSum from WASM diagnostics");
  }
  const spatial = mapVariogramParams(rec.spatial);
  const temporal = mapVariogramParams(rec.temporal);
  if (fam === "separable") {
    return { family: "separable", spatial, temporal };
  }
  return {
    family: "productSum",
    spatial,
    temporal,
    k1: requireNumber(rec.k1),
    k2: requireNumber(rec.k2),
    k3: requireNumber(rec.k3),
  };
}

/** Map raw `getDiagnostics()` JSON from WASM space–time binomial. */
export function mapSpaceTimeBinomialDiagnostics(
  value: unknown
): SpaceTimeBinomialDiagnostics {
  const rec = asRecord(value);
  const msdr = rec.logitLooMsdr;
  return {
    variogram: mapSpaceTimeVariogramParamsFromDiagnostics(rec.variogram),
    buildNotes: mapBinomialBuildNotes(rec.buildNotes),
    logitLooMsdr:
      msdr === undefined || msdr === null ? undefined : requireNumber(msdr),
  };
}

export function mapOrdinaryBatchArrayOutput(
  value: unknown
): OrdinaryBatchArrayOutput {
  const out = asRecord(value);
  return {
    values: requireFloat64Array(out.values),
    variances: requireFloat64Array(out.variances),
  };
}

export function mapBinomialBatchArrayOutput(
  value: unknown
): BinomialBatchArrayOutput {
  const out = asRecord(value);
  const prevalenceMedians = requireFloat64Array(out.prevalenceMedians);
  const prevalenceMeans = requireFloat64Array(out.prevalenceMeans);
  const logitValues = requireFloat64Array(out.logitValues);
  const logitVariances = requireFloat64Array(out.logitVariances);
  const explicit = out.prevalenceVariances;
  const prevalenceVariances =
    explicit instanceof Float64Array
      ? explicit
      : computeDeltaPrevalenceVariances(prevalenceMedians, logitVariances);
  return {
    prevalenceMedians,
    prevalenceMeans,
    logitValues,
    logitVariances,
    prevalenceVariances,
  };
}

export function deltaPrevalenceVariance(
  prevalence: number,
  logitVariance: number
): number {
  const factor = prevalence * (1 - prevalence);
  return factor * factor * Math.max(0, logitVariance);
}

function computeDeltaPrevalenceVariances(
  prevalenceMedians: Float64Array,
  logitVariances: Float64Array
): Float64Array {
  const n = prevalenceMedians.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = deltaPrevalenceVariance(prevalenceMedians[i], logitVariances[i]);
  }
  return out;
}

export function mapEmpiricalVariogram(
  value: unknown
): EmpiricalVariogramResult {
  const rec = asRecord(value);
  return {
    distances: requireFloat64Array(rec.distances),
    semivariances: requireFloat64Array(rec.semivariances),
    counts: requireUint32Array(rec.counts),
  };
}

export function mapCvOutput(value: unknown): CvResult {
  const rec = asRecord(value);
  const indices = requireUint32Array(rec.indices);
  const observed = requireFloat64Array(rec.observed);
  const predicted = requireFloat64Array(rec.predicted);
  const variances = requireFloat64Array(rec.variances);
  const summaryRec = asRecord(rec.summary);
  const summary: CvSummary = {
    n: requireNumber(summaryRec.n),
    meanError: requireFiniteOrNaN(summaryRec.meanError),
    rmse: requireFiniteOrNaN(summaryRec.rmse),
    msdr: requireFiniteOrNaN(summaryRec.msdr),
  };
  const residuals: CvResidual[] = [];
  for (let i = 0; i < indices.length; i++) {
    residuals.push({
      index: indices[i],
      observed: observed[i],
      predicted: predicted[i],
      variance: variances[i],
      error: observed[i] - predicted[i],
    });
  }
  return {
    residuals,
    summary,
    arrays: { indices, observed, predicted, variances },
  };
}

function mapCvSummary(value: unknown): CvSummary {
  const rec = asRecord(value);
  return {
    n: requireNumber(rec.n),
    meanError: requireFiniteOrNaN(rec.meanError),
    rmse: requireFiniteOrNaN(rec.rmse),
    msdr: requireFiniteOrNaN(rec.msdr),
  };
}

function mapPrevalenceCalibrationBin(value: unknown): PrevalenceCalibrationBin {
  const rec = asRecord(value);
  return {
    binIndex: requireNumber(rec.binIndex),
    predictedLo: requireFiniteOrNaN(rec.predictedLo),
    predictedHi: requireFiniteOrNaN(rec.predictedHi),
    nStations: requireNumber(rec.nStations),
    sumTrials: requireNumber(rec.sumTrials),
    sumSuccesses: requireNumber(rec.sumSuccesses),
    meanPredicted: requireFiniteOrNaN(rec.meanPredicted),
    pooledObservedPrevalence: requireFiniteOrNaN(rec.pooledObservedPrevalence),
  };
}

function mapPrevalenceCalibrationBins(value: unknown): PrevalenceCalibrationBin[] {
  if (!Array.isArray(value)) {
    throw new Error("calibrationBins must be an array");
  }
  return value.map(mapPrevalenceCalibrationBin);
}

export function mapBinomialCvOutput(value: unknown): BinomialCvResult {
  const rec = asRecord(value);
  const indices = requireUint32Array(rec.indices);
  const successes = requireUint32Array(rec.successes);
  const trials = requireUint32Array(rec.trials);
  const observedLogit = requireFloat64Array(rec.observedLogit);
  const predictedLogit = requireFloat64Array(rec.predictedLogit);
  const logitVariance = requireFloat64Array(rec.logitVariance);
  const observedPrevalence = requireFloat64Array(rec.observedPrevalence);
  const predictedPrevalence = requireFloat64Array(rec.predictedPrevalence);
  const prevalenceVariance = requireFloat64Array(rec.prevalenceVariance);
  const summaryRec = asRecord(rec.summary);
  const summary: BinomialCvSummary = {
    n: requireNumber(summaryRec.n),
    nEvaluated: requireNumber(summaryRec.nEvaluated),
    logit: mapCvSummary(summaryRec.logit),
    prevalence: mapCvSummary(summaryRec.prevalence),
    brier: requireFiniteOrNaN(summaryRec.brier),
    logScorePerTrial: requireFiniteOrNaN(summaryRec.logScorePerTrial),
    calibrationBins: mapPrevalenceCalibrationBins(summaryRec.calibrationBins),
  };
  const residuals: BinomialCvResidual[] = [];
  for (let i = 0; i < indices.length; i++) {
    residuals.push({
      index: indices[i],
      successes: successes[i],
      trials: trials[i],
      observedLogit: observedLogit[i],
      predictedLogit: predictedLogit[i],
      logitVariance: logitVariance[i],
      observedPrevalence: observedPrevalence[i],
      predictedPrevalence: predictedPrevalence[i],
      prevalenceVariance: prevalenceVariance[i],
      logitError: observedLogit[i] - predictedLogit[i],
      prevalenceError: observedPrevalence[i] - predictedPrevalence[i],
    });
  }
  return {
    residuals,
    summary,
    arrays: {
      indices,
      successes,
      trials,
      observedLogit,
      predictedLogit,
      logitVariance,
      observedPrevalence,
      predictedPrevalence,
      prevalenceVariance,
    },
  };
}

// ---------- Space-time mappers ----------

export function mapVariogramParams(value: unknown): VariogramParams {
  const rec = asRecord(value);
  const vt = requireVariogramType(rec.variogramType);
  const out: VariogramParams = {
    variogramType: vt as VariogramTypeName,
    nugget: requireNumber(rec.nugget),
    sill: requireNumber(rec.sill),
    range: requireNumber(rec.range),
  };
  if (typeof rec.shape === "number" && Number.isFinite(rec.shape)) {
    out.shape = rec.shape;
  }
  return out;
}

export function mapFittedSpaceTimeVariogram(
  value: unknown
): FittedSpaceTimeVariogram {
  const rec = asRecord(value);
  const family = rec.family;
  if (family !== "separable" && family !== "productSum") {
    throw new Error("Unknown space-time variogram family from WASM");
  }
  const spatial = mapVariogramParams(rec.spatial);
  const temporal = mapVariogramParams(rec.temporal);
  const residuals = requireFiniteOrNaN(rec.residuals);
  if (family === "productSum") {
    return {
      family: "productSum",
      spatial,
      temporal,
      k1: requireNumber(rec.k1),
      k2: requireNumber(rec.k2),
      k3: requireNumber(rec.k3),
      residuals,
    };
  }
  return {
    family: "separable",
    spatial,
    temporal,
    residuals,
  };
}

export function mapEmpiricalSpaceTimeVariogram(
  value: unknown
): EmpiricalSpaceTimeVariogramResult {
  const rec = asRecord(value);
  return {
    nSpatialBins: requireNumber(rec.nSpatialBins),
    nTemporalBins: requireNumber(rec.nTemporalBins),
    spatialLags: requireFloat64Array(rec.spatialLags),
    temporalLags: requireFloat64Array(rec.temporalLags),
    semivariances: requireFloat64Array(rec.semivariances),
    nPairs: requireFloat64Array(rec.nPairs),
  };
}

export function mapFitSpaceTimeVariogramResult(
  value: unknown
): FitSpaceTimeVariogramResult {
  const rec = asRecord(value);
  return {
    empirical: mapEmpiricalSpaceTimeVariogram(rec.empirical),
    fit: mapFittedSpaceTimeVariogram(rec.fit),
  };
}
