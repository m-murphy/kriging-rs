/**
 * Entry point for `kriging-rs-wasm`. This module only re-exports the public API;
 * implementation lives in sibling modules (see `types.ts`, `errors.ts`, `kriging/*`,
 * `variogram.ts`, `cv.ts`, `simulation.ts`, `interpolate.ts`).
 *
 * Typical usage:
 *
 * ```ts
 * import init, { OrdinaryKriging, fitVariogram } from "kriging-rs-wasm";
 *
 * await init();
 * const fitted = fitVariogram({ sampleLats, sampleLons, values, variogramType: "exponential" });
 * const model = OrdinaryKriging.fromFitted({ lats, lons, values, fittedVariogram: fitted });
 * const { value, variance } = model.predict(37.7, -122.4);
 * model.free();
 * ```
 *
 * @module
 */

export { init, VariogramType, webgpuAvailable } from "./internal/module.js";
export { KrigingError } from "./errors.js";
export { OrdinaryKriging } from "./kriging/ordinary.js";
export { SimpleKriging } from "./kriging/simple.js";
export { UniversalKriging } from "./kriging/universal.js";
export { ProjectedKriging } from "./kriging/projected.js";
export { BinomialKriging } from "./kriging/binomial.js";
export {
  computeDirectionalEmpiricalVariogram,
  computeEmpiricalVariogram,
  evaluateNestedVariogram,
  fitVariogram,
} from "./variogram.js";
export {
  kFold,
  kFoldBinomial,
  kFoldProjected,
  kFoldSimple,
  kFoldUniversal,
  leaveOneOut,
  leaveOneOutBinomial,
  leaveOneOutProjected,
  leaveOneOutSimple,
  leaveOneOutUniversal,
} from "./cv.js";
export {
  conditionalSimulate,
  conditionalSimulateBinomial,
  conditionalSimulateProjected,
  conditionalSimulateSimple,
  conditionalSimulateUniversal,
} from "./simulation.js";
export {
  interpolateBinomialToGrid,
  interpolateOrdinaryToGrid,
} from "./interpolate.js";

export type {
  BinomialBatchArrayOutput,
  BinomialCvResidual,
  BinomialCvResult,
  BinomialCvSummary,
  BinomialFromPrecomputedLogitsOptions,
  BinomialGridOutput,
  BinomialKrigingFromFittedVariogramOptions,
  BinomialKrigingFromFittedVariogramWithPriorOptions,
  BinomialKrigingOptions,
  BinomialKrigingWithPriorOptions,
  BinomialPrediction,
  BinomialPriorParams,
  BinomialSimulationResult,
  ComputeDirectionalEmpiricalVariogramOptions,
  ComputeEmpiricalVariogramOptions,
  ConditionalSimulateBinomialOptions,
  ConditionalSimulateOptions,
  ConditionalSimulateProjectedOptions,
  ConditionalSimulateSimpleOptions,
  ConditionalSimulateUniversalOptions,
  CvResidual,
  CvResult,
  CvSummary,
  EmpiricalEstimator,
  EmpiricalVariogramResult,
  FitVariogramOptions,
  FittedVariogram,
  IntegerArrayInput,
  InterpolateBinomialToGridOptions,
  InterpolateOrdinaryToGridOptions,
  KFoldBinomialOptions,
  KFoldOptions,
  KFoldProjectedOptions,
  KFoldSimpleOptions,
  KFoldUniversalOptions,
  KrigingErrorCode,
  LeaveOneOutBinomialOptions,
  LeaveOneOutOptions,
  LeaveOneOutProjectedOptions,
  LeaveOneOutSimpleOptions,
  LeaveOneOutUniversalOptions,
  NeighborhoodOptions,
  NestedVariogramComponent,
  NestedVariogramEvaluation,
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryGridOutput,
  OrdinaryKrigingFromFittedOptions,
  OrdinaryKrigingOptions,
  OrdinaryPrediction,
  PredictGridOptions,
  ProjectedKrigingOptions,
  SimpleKrigingOptions,
  UniversalKrigingOptions,
  UniversalTrend,
  VariogramParams,
  VariogramTypeName,
} from "./types.js";

import { init } from "./internal/module.js";

export default init;
