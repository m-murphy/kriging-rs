import {
  BinomialKriging,
  BinomialProjectedKriging,
  binomialPreprocess,
  OrdinaryKriging,
  SpaceTimeBinomialKriging,
  SpaceTimeOrdinaryKriging,
  SpaceTimeProjectedOrdinaryKriging,
  SpaceTimeSimpleKriging,
  SpaceTimeUniversalKriging,
  computeEmpiricalSpaceTimeVariogram,
  cv,
  conditionalSimulate,
  fitSpaceTimeVariogram,
  fitBinomialVariogram,
  fitVariogram,
  estimateBinomialPrior,
  init,
  interpolateOrdinaryToGrid,
  interpolateBinomialToGrid,
  leaveOneOut,
  kFold,
  VariogramType,
  type BinomialBatchArrayOutput,
  type BinomialBuildNotes,
  type BinomialCvResult,
  type BinomialCvSummary,
  type BinomialPrediction,
  type BinomialPreprocessResult,
  type BinomialGridOutput,
  type BinomialPriorInput,
  type BinomialPriorParams,
  type InterpolateBinomialToGridResult,
  type BinomialSimulationResult,
  type CvResult,
  type EmpiricalSpaceTimeVariogramResult,
  type FitSpaceTimeVariogramResult,
  type FittedVariogram,
  type OrdinaryBatchArrayOutput,
  type OrdinaryPrediction,
  type OrdinaryGridOutput,
  type PredictGridOptions,
  type PredictProjectedGridOptions,
  type PrevalenceCalibrationBin,
  type SpaceTimeVariogramParams,
  type VariogramTypeName,
} from "../src/index.js";

// Contract: init() must return Promise<void>
type _InitReturn = ReturnType<typeof init>;
const _initReturnsVoid: _InitReturn extends Promise<void> ? true : false = true;

type IsAny<T> = 0 extends 1 & T ? true : false;
type AssertNotAny<T> = IsAny<T> extends true ? never : true;

const variogram: VariogramTypeName = "gaussian";
const lats = new Float64Array([0, 1, 2]);
const lons = new Float64Array([0, 1, 2]);
const values = new Float64Array([3, 4, 5]);

const ordinary = new OrdinaryKriging({
  lats,
  lons,
  values,
  variogram: { variogramType: variogram, nugget: 0.01, sill: 1.0, range: 100 },
});
const pred = ordinary.predict(0.5, 0.5);
const batch = ordinary.predictBatch(lats, lons);
const batchArrays = ordinary.predictBatchArrays(lats, lons);

const _predNotAny: AssertNotAny<typeof pred> = true;
const _batchNotAny: AssertNotAny<typeof batch> = true;
const _batchArraysNotAny: AssertNotAny<typeof batchArrays> = true;
const _predType: OrdinaryPrediction = pred;
const _batchItemType: OrdinaryPrediction = batch[0];
const _batchArraysType: OrdinaryBatchArrayOutput = batchArrays;

const fit = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: VariogramType.Gaussian,
  nBins: 12,
});
const fitWithString = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: "exponential",
});
const _fitWithStringType: VariogramTypeName = fitWithString.variogramType;
const _fitVariogramType: VariogramTypeName = fit.variogramType;
const fittedOrdinary = new OrdinaryKriging({
  lats,
  lons,
  values,
  variogram: {
    variogramType: fit.variogramType,
    nugget: fit.nugget,
    sill: fit.sill,
    range: fit.range,
    shape: fit.shape,
  },
});
const fromFittedOrdinary = OrdinaryKriging.fromFitted({
  lats,
  lons,
  values,
  fittedVariogram: fit,
  nuggetOverride: 0.05,
});
const _fromFittedPred: OrdinaryPrediction = fromFittedOrdinary.predict(
  0.5,
  0.5
);
const gridOpts: PredictGridOptions = {
  west: 0,
  south: 0,
  east: 1,
  north: 1,
  xCells: 5,
  yCells: 4,
};
const ordinaryGrid: OrdinaryGridOutput =
  fromFittedOrdinary.predictGrid(gridOpts);
const _ordinaryGridType: OrdinaryGridOutput = ordinaryGrid;
const fittedBatch = fittedOrdinary.predictBatch(lats, lons);
const _fittedBatchItemType: OrdinaryPrediction = fittedBatch[0];
const fittedBatchArrays = fittedOrdinary.predictBatchArrays(lats, lons);
const _fittedBatchArrayType: OrdinaryBatchArrayOutput = fittedBatchArrays;

const successes = new Uint32Array([2, 4, 6]);
const trials = new Uint32Array([10, 10, 10]);
const binomial = new BinomialKriging({
  lats,
  lons,
  successes,
  trials,
  variogram: {
    variogramType: "exponential",
    nugget: 0.01,
    sill: 1.0,
    range: 100,
  },
  stability: "strict",
});
const bPred = binomial.predict(0.4, 0.4);
const _bPredType: BinomialPrediction = bPred;
const _bPredNotAny: AssertNotAny<typeof bPred> = true;
const bArrayOut = binomial.predictBatchArrays(lats, lons);
const _bArrayType: BinomialBatchArrayOutput = bArrayOut;

const fitBinomial = fitBinomialVariogram({
  sampleLats: lats,
  sampleLons: lons,
  successes,
  trials,
  variogramType: VariogramType.Exponential,
  nBins: 12,
});
const _autoPrior: BinomialPriorInput = "auto";
const _priorFromCounts: BinomialPriorParams = estimateBinomialPrior({
  successes,
  trials,
});
const fitBinomialAutoPrior = fitBinomialVariogram({
  sampleLats: lats,
  sampleLons: lons,
  successes,
  trials,
  variogramType: VariogramType.Exponential,
  nBins: 12,
  prior: "auto",
});
const _fitBinomialAutoPrior: FittedVariogram = fitBinomialAutoPrior;
const _fitBinomialFitted: FittedVariogram = fitBinomial;

const binomialFromFitted = BinomialKriging.fromFittedVariogram({
  lats,
  lons,
  successes,
  trials,
  fittedVariogram: fit,
});
const _binomialFromFittedPred: BinomialPrediction = binomialFromFitted.predict(
  0.4,
  0.4
);

const binomialFromFittedPrior = BinomialKriging.fromFittedVariogramWithPrior({
  lats,
  lons,
  successes,
  trials,
  fittedVariogram: fit,
  prior: { alpha: 1, beta: 1 },
});
const _binomialFromFittedPriorPred: BinomialPrediction =
  binomialFromFittedPrior.predict(0.4, 0.4);
const binomialFromFcVar = BinomialKriging.fromPrecomputedLogitsWithVariances({
  lats,
  lons,
  logits: new Float64Array([0, 0.1, -0.1]),
  logitObservationVariance: new Float64Array([0.05, 0.05, 0.05]),
  variogram: {
    variogramType: "exponential",
    nugget: 0.05,
    sill: 1.0,
    range: 100,
  },
  prior: { alpha: 1, beta: 2 },
});
const _binomialFromFcVarPred: BinomialPrediction = binomialFromFcVar.predict(
  0.5,
  0.5
);
const binomialGrid: BinomialGridOutput =
  binomialFromFittedPrior.predictGrid(gridOpts);
const _binomialGridType: BinomialGridOutput = binomialGrid;

const _oneShotOrdinary: OrdinaryGridOutput = interpolateOrdinaryToGrid({
  lats: Array.from(lats),
  lons: Array.from(lons),
  values: Array.from(values),
  west: 0,
  south: 0,
  east: 2,
  north: 2,
  xCells: 3,
  yCells: 3,
  variogramType: "exponential",
  nBins: 12,
});

const fittedProjectedBinomial: FittedVariogram = {
  variogramType: "exponential",
  nugget: 0.05,
  sill: 1.0,
  range: 200,
  residuals: 0.02,
};
const bpFromFitted = BinomialProjectedKriging.fromFittedVariogram({
  xs: lats,
  ys: lons,
  successes,
  trials,
  fittedVariogram: fittedProjectedBinomial,
  majorAngleDeg: 0,
  rangeRatio: 1,
});
const _projGridOpts: PredictProjectedGridOptions = {
  xMin: 0,
  xMax: 2,
  yMin: 0,
  yMax: 2,
  xCells: 3,
  yCells: 3,
};
const _bpProjGrid: BinomialGridOutput = bpFromFitted.predictGrid(_projGridOpts);

const bpFromFcVar = BinomialProjectedKriging.fromPrecomputedLogitsWithVariances({
  xs: lats,
  ys: lons,
  logits: new Float64Array([0, 0.1, -0.05]),
  logitObservationVariance: new Float64Array([0.02, 0.02, 0.02]),
  variogram: {
    variogramType: "gaussian",
    nugget: 0.01,
    sill: 1.0,
    range: 50,
  },
  majorAngleDeg: 0,
  rangeRatio: 1,
});
const _bpFromFcVarPred: BinomialPrediction = bpFromFcVar.predict(0.5, 0.5);

// ---------- Space-time contracts ----------

const times = new Float64Array([0, 1, 2]);

const stVariogram = {
  family: "separable" as const,
  spatial: {
    variogramType: "exponential" as const,
    nugget: 0.01,
    sill: 1.0,
    range: 100,
  },
  temporal: {
    variogramType: "exponential" as const,
    nugget: 0.01,
    sill: 1.0,
    range: 5,
  },
};

const stOrdinary = new SpaceTimeOrdinaryKriging({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
});
const _stPred: OrdinaryPrediction = stOrdinary.predict(0.5, 0.5, 1.0);
const _stBatchArrays: OrdinaryBatchArrayOutput = stOrdinary.predictBatchArrays(
  lats,
  lons,
  times
);

const stSimple = new SpaceTimeSimpleKriging({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  mean: 4.0,
});
const _stSimplePred: OrdinaryPrediction = stSimple.predict(0.5, 0.5, 1.0);

const stUniversal = new SpaceTimeUniversalKriging({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  trend: "linearInSpaceAndTime",
});
const _stUniversalPred: OrdinaryPrediction = stUniversal.predict(0.5, 0.5, 1.0);

const stBinomial = new SpaceTimeBinomialKriging({
  lats,
  lons,
  times,
  successes,
  trials,
  variogram: stVariogram,
});
const _stBinomialPred: BinomialPrediction = stBinomial.predict(0.5, 0.5, 1.0);
const _stBinNotes: BinomialBuildNotes = stBinomial.buildNotes;
const _pre: BinomialPreprocessResult = binomialPreprocess({
  successes: [2, 3],
  trials: [10, 10],
});
const _stBinomialBatchArrays: BinomialBatchArrayOutput =
  stBinomial.predictBatchArrays(lats, lons, times);

const stBinomialPrior = SpaceTimeBinomialKriging.newWithPrior({
  lats,
  lons,
  times,
  successes,
  trials,
  variogram: stVariogram,
  prior: { alpha: 2, beta: 2 },
});
const _stBinomialPriorPred: BinomialPrediction = stBinomialPrior.predict(
  0.5,
  0.5,
  1.0
);

const stBinomialFc = SpaceTimeBinomialKriging.fromPrecomputedLogits({
  lats,
  lons,
  times,
  logits: [0.05, 0.12, -0.03],
  variogram: stVariogram,
});
const _stBinomialFcPred: BinomialPrediction = stBinomialFc.predict(0.5, 0.5, 1.0);

const stBinomialFcVar = SpaceTimeBinomialKriging.fromPrecomputedLogitsWithVariances({
  lats,
  lons,
  times,
  logits: [0.01, 0.02, -0.01],
  logitObservationVariance: [0.05, 0.05, 0.05],
  variogram: stVariogram,
});
const _stBinomialFcVarPred: BinomialPrediction = stBinomialFcVar.predict(
  0.5,
  0.5,
  1.0
);

const stProjected = new SpaceTimeProjectedOrdinaryKriging({
  xs: lats,
  ys: lons,
  times,
  values,
  variogram: stVariogram,
  majorAngleDeg: 0,
  rangeRatio: 1.0,
});
const _stProjectedPred: OrdinaryPrediction = stProjected.predict(0.5, 0.5, 1.0);

const _stEmpirical: EmpiricalSpaceTimeVariogramResult =
  computeEmpiricalSpaceTimeVariogram({
    lats,
    lons,
    times,
    values,
    nSpatialBins: 4,
    nTemporalBins: 3,
  });

const _stFit: FitSpaceTimeVariogramResult = fitSpaceTimeVariogram({
  lats,
  lons,
  times,
  values,
  nSpatialBins: 4,
  nTemporalBins: 3,
  family: "separable",
  spatialModel: "exponential",
  temporalModel: "exponential",
});

// Space-time CV / SGS contracts
const _stCv: CvResult = leaveOneOut({ geometry: "spacetime", family: "ordinary", 
  lats,
  lons,
  times,
  values,
  spaceTimeVariogram: stVariogram,
}) as CvResult;
const _stCvKf: CvResult = kFold({ geometry: "spacetime", family: "ordinary", 
  lats,
  lons,
  times,
  values,
  spaceTimeVariogram: stVariogram,
  k: 3,
}) as CvResult;
const _stCvSimple: CvResult = leaveOneOut({ geometry: "spacetime", family: "simple", 
  lats,
  lons,
  times,
  values,
  spaceTimeVariogram: stVariogram,
  mean: 4,
}) as CvResult;
const _stCvSimpleKf: CvResult = kFold({ geometry: "spacetime", family: "simple", 
  lats,
  lons,
  times,
  values,
  spaceTimeVariogram: stVariogram,
  mean: 4,
  k: 3,
}) as CvResult;
const _stCvUniv: CvResult = leaveOneOut({ geometry: "spacetime", family: "universal", 
  lats,
  lons,
  times,
  values,
  spaceTimeVariogram: stVariogram,
  trend: "linearInSpaceAndTime",
}) as CvResult;
const _stCvUnivKf: CvResult = kFold({ geometry: "spacetime", family: "universal", 
  lats,
  lons,
  times,
  values,
  spaceTimeVariogram: stVariogram,
  trend: "linearInSpaceAndTime",
  k: 3,
}) as CvResult;
const _stCvBin: BinomialCvResult = cv({ geometry: "spacetime", family: "binomial", 
  lats,
  lons,
  times,
  successes,
  trials,
  spaceTimeVariogram: stVariogram,
}) as BinomialCvResult;
const _stCvBinKf: BinomialCvResult = cv({ geometry: "spacetime", family: "binomial",
  k: 3,
  lats,
  lons,
  times,
  successes,
  trials,
  spaceTimeVariogram: stVariogram,
}) as BinomialCvResult;

const _stSgs: Float64Array = conditionalSimulate({ geometry: "spacetime", family: "ordinary", 
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats: new Float64Array([0.5]),
  targetLons: new Float64Array([0.5]),
  targetTimes: new Float64Array([0.5]),
  spaceTimeVariogram: stVariogram,
  seed: 1n,
}) as Float64Array;
const _stSgsSimple: Float64Array = conditionalSimulate({ geometry: "spacetime", family: "simple", 
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats: new Float64Array([0.5]),
  targetLons: new Float64Array([0.5]),
  targetTimes: new Float64Array([0.5]),
  spaceTimeVariogram: stVariogram,
  mean: 4,
  seed: 1n,
}) as Float64Array;
const _stSgsUniv: Float64Array = conditionalSimulate({ geometry: "spacetime", family: "universal", 
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats: new Float64Array([0.5]),
  targetLons: new Float64Array([0.5]),
  targetTimes: new Float64Array([0.5]),
  spaceTimeVariogram: stVariogram,
  trend: "constant",
  seed: 1n,
}) as Float64Array;
const _stSgsBin: BinomialSimulationResult =
  conditionalSimulate({ geometry: "spacetime", family: "binomial", 
    conditioningLats: lats,
    conditioningLons: lons,
    conditioningTimes: times,
    conditioningSuccesses: successes,
    conditioningTrials: trials,
    targetLats: new Float64Array([0.5]),
    targetLons: new Float64Array([0.5]),
    targetTimes: new Float64Array([0.5]),
    spaceTimeVariogram: stVariogram,
    seed: 1n,
  }) as BinomialSimulationResult;

const _oneShotBinomial: InterpolateBinomialToGridResult = interpolateBinomialToGrid({
  lats: Array.from(lats),
  lons: Array.from(lons),
  successes: Array.from(successes),
  trials: Array.from(trials),
  west: 0,
  south: 0,
  east: 2,
  north: 2,
  xCells: 3,
  yCells: 3,
  variogramType: "exponential",
  prior: { alpha: 1, beta: 1 },
});

const _calBinShape: PrevalenceCalibrationBin = {
  binIndex: 0,
  predictedLo: 0,
  predictedHi: 0.1,
  nStations: 0,
  sumTrials: 0,
  sumSuccesses: 0,
  meanPredicted: NaN,
  pooledObservedPrevalence: NaN,
};
const _binomialCvSummaryShape: BinomialCvSummary = {
  n: 0,
  nEvaluated: 0,
  logit: { n: 0, meanError: 0, rmse: 0, msdr: 0 },
  prevalence: { n: 0, meanError: 0, rmse: 0, msdr: 0 },
  brier: NaN,
  logScorePerTrial: NaN,
  calibrationBins: [_calBinShape],
};

// ---------- SpaceTimeVariogramParams discriminated-union contracts ----------

const _separableOk: SpaceTimeVariogramParams = {
  family: "separable",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
};

const _productSumOk: SpaceTimeVariogramParams = {
  family: "productSum",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
  k1: 1,
  k2: 0,
  k3: 0,
};

// Separable MUST NOT accept k1/k2/k3.
const _separableRejectsK: SpaceTimeVariogramParams = {
  family: "separable",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
  // @ts-expect-error separable has no k coefficients
  k1: 1,
};

// productSum MUST require k1/k2/k3.
// @ts-expect-error productSum requires k1, k2, k3
const _productSumRequiresK: SpaceTimeVariogramParams = {
  family: "productSum",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
};

// snake_case "product_sum" must be rejected at the TS boundary.
const _snakeCaseRejected: SpaceTimeVariogramParams = {
  // @ts-expect-error snake_case family name is not part of the public API
  family: "product_sum",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
  k1: 1,
  k2: 0,
  k3: 0,
};
