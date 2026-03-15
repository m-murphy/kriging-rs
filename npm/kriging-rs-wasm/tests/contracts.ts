import {
  BinomialKriging,
  OrdinaryKriging,
  fitOrdinaryVariogram,
  VariogramType,
  type BinomialBatchArrayOutput,
  type BinomialPrediction,
  type OrdinaryBatchArrayOutput,
  type OrdinaryPrediction,
  type VariogramTypeName,
} from "../src/index.js";

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

const fit = fitOrdinaryVariogram(
  lats,
  lons,
  values,
  undefined,
  12,
  VariogramType.Gaussian
);
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
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 100 },
});
const bPred = binomial.predict(0.4, 0.4);
const _bPredType: BinomialPrediction = bPred;
const _bPredNotAny: AssertNotAny<typeof bPred> = true;
const bArrayOut = binomial.predictBatchArrays(lats, lons);
const _bArrayType: BinomialBatchArrayOutput = bArrayOut;
