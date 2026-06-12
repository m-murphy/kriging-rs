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
export { BinomialProjectedKriging } from "./kriging/binomial-projected.js";
export { BinomialTangentPlaneKriging } from "./kriging/binomial-tangent-plane.js";
export {
  computeDirectionalEmpiricalVariogram,
  computeEmpiricalVariogram,
  evaluateNestedVariogram,
  fitBinomialVariogram,
  fitVariogram,
} from "./variogram.js";
export { estimateBinomialPrior } from "./estimate-binomial-prior.js";
export {
  kFold,
  kFoldBinomial,
  kFoldBinomialProjected,
  kFoldProjected,
  kFoldSimple,
  kFoldSpaceTime,
  kFoldSpaceTimeBinomial,
  kFoldSpaceTimeSimple,
  kFoldSpaceTimeUniversal,
  kFoldUniversal,
  leaveOneOut,
  leaveOneOutBinomial,
  leaveOneOutBinomialProjected,
  leaveOneOutProjected,
  leaveOneOutSimple,
  leaveOneOutSpaceTime,
  leaveOneOutSpaceTimeBinomial,
  leaveOneOutSpaceTimeSimple,
  leaveOneOutSpaceTimeUniversal,
  leaveOneOutUniversal,
} from "./cv.js";
export {
  conditionalSimulate,
  conditionalSimulateBinomial,
  conditionalSimulateBinomialProjected,
  conditionalSimulateMany,
  conditionalSimulateManyBinomial,
  conditionalSimulateManyBinomialProjected,
  conditionalSimulateManySpaceTime,
  conditionalSimulateManySpaceTimeBinomial,
  conditionalSimulateProjected,
  conditionalSimulateSimple,
  conditionalSimulateSpaceTime,
  conditionalSimulateSpaceTimeBinomial,
  conditionalSimulateSpaceTimeSimple,
  conditionalSimulateSpaceTimeUniversal,
  conditionalSimulateUniversal,
} from "./simulation.js";
export {
  ensembleExceedanceProbability,
  ensembleMean,
  ensembleQuantiles,
  ensembleVariance,
} from "./aggregate.js";
export {
  gridCellCenters,
  reshapeGridRow,
  simulateBinomialGrid,
  simulateBinomialGridEnsemble,
  simulateBinomialGridSummary,
} from "./simulate-grid.js";
export {
  simulateBinomialSpaceTimeGrid,
  simulateBinomialSpaceTimeGridAtDate,
  simulateBinomialSpaceTimeGridEnsemble,
  simulateBinomialSpaceTimeGridEnsembleAtDate,
  simulateBinomialSpaceTimeGridSummary,
  simulateBinomialSpaceTimeGridSummaryAtDate,
} from "./simulate-grid-spacetime.js";
export {
  aggregatePrevalenceByPolygon,
  polygonCellsFromMask,
} from "./polygons.js";
export {
  interpolateBinomialToGrid,
  interpolateOrdinaryToGrid,
} from "./interpolate.js";
export {
  SpaceTimeBinomialKriging,
  SpaceTimeOrdinaryKriging,
  SpaceTimeProjectedOrdinaryKriging,
  SpaceTimeSimpleKriging,
  SpaceTimeUniversalKriging,
  computeEmpiricalSpaceTimeVariogram,
  fitSpaceTimeVariogram,
} from "./spacetime/index.js";
export { datesFromTimes, timesFromDates } from "./time.js";

export type {
  AggregatePrevalenceByPolygonOptions,
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResidual,
  BinomialCvResult,
  BinomialCvSummary,
  BinomialDiagnostics,
  BinomialFromPrecomputedLogitsOptions,
  PrevalenceCalibrationBin,
  BinomialFromPrecomputedLogitsWithVariancesOptions,
  BinomialGridEnsemble,
  BinomialGridOutput,
  BinomialGridSimulation,
  BinomialGridSummary,
  BinomialKrigingFromFittedVariogramOptions,
  BinomialKrigingFromFittedVariogramWithPriorOptions,
  BinomialKrigingOptions,
  BinomialKrigingWithPriorOptions,
  BinomialTangentPlaneKrigingOptions,
  BinomialTangentPlaneKrigingWithPriorOptions,
  BinomialPrediction,
  BinomialPriorInput,
  BinomialPriorParams,
  BinomialStabilityPreset,
  BinomialProjectedFromPrecomputedLogitsOptions,
  BinomialProjectedFromPrecomputedLogitsWithVariancesOptions,
  BinomialProjectedKrigingFromFittedVariogramOptions,
  BinomialProjectedKrigingFromFittedVariogramWithPriorOptions,
  BinomialProjectedKrigingOptions,
  BinomialProjectedKrigingWithPriorOptions,
  BinomialSimulationManyResult,
  BinomialSimulationResult,
  ComputeDirectionalEmpiricalVariogramOptions,
  ComputeEmpiricalVariogramOptions,
  ConditionalSimulateBinomialOptions,
  ConditionalSimulateBinomialProjectedOptions,
  ConditionalSimulateManyBinomialOptions,
  ConditionalSimulateManyBinomialProjectedOptions,
  ConditionalSimulateManyOptions,
  ConditionalSimulateManySpaceTimeBinomialOptions,
  ConditionalSimulateManySpaceTimeOptions,
  ConditionalSimulateOptions,
  ConditionalSimulateProjectedOptions,
  ConditionalSimulateSimpleOptions,
  ConditionalSimulateSpaceTimeBinomialOptions,
  ConditionalSimulateSpaceTimeOptions,
  ConditionalSimulateSpaceTimeSimpleOptions,
  ConditionalSimulateSpaceTimeUniversalOptions,
  ConditionalSimulateUniversalOptions,
  CvResidual,
  DateAxisOptions,
  CvResult,
  CvSummary,
  EmpiricalEstimator,
  EmpiricalVariogramResult,
  FitBinomialVariogramOptions,
  FitVariogramOptions,
  FittedVariogram,
  GeoGridBounds,
  IntegerArrayInput,
  InterpolateBinomialToGridOptions,
  InterpolateBinomialToGridResult,
  InterpolateOrdinaryToGridOptions,
  KFoldBinomialOptions,
  KFoldBinomialProjectedOptions,
  KFoldOptions,
  KFoldProjectedOptions,
  KFoldSimpleOptions,
  KFoldSpaceTimeBinomialOptions,
  KFoldSpaceTimeOptions,
  KFoldSpaceTimeSimpleOptions,
  KFoldSpaceTimeUniversalOptions,
  KFoldUniversalOptions,
  KrigingErrorCode,
  LeaveOneOutBinomialOptions,
  LeaveOneOutBinomialProjectedOptions,
  LeaveOneOutOptions,
  LeaveOneOutProjectedOptions,
  LeaveOneOutSimpleOptions,
  LeaveOneOutSpaceTimeBinomialOptions,
  LeaveOneOutSpaceTimeOptions,
  LeaveOneOutSpaceTimeSimpleOptions,
  LeaveOneOutSpaceTimeUniversalOptions,
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
  PolygonAggregateResult,
  PolygonCells,
  PolygonCellsFromMaskOptions,
  PredictGridAtDateOptions,
  PredictGridAtTimeOptions,
  PredictGridOptions,
  PredictProjectedGridOptions,
  ProjectedKrigingOptions,
  SimpleKrigingOptions,
  SimulateBinomialGridEnsembleOptions,
  SimulateBinomialGridOptions,
  SimulateBinomialGridSummaryOptions,
  SimulateBinomialSpaceTimeGridAtDateOptions,
  SimulateBinomialSpaceTimeGridEnsembleAtDateOptions,
  SimulateBinomialSpaceTimeGridEnsembleOptions,
  SimulateBinomialSpaceTimeGridOptions,
  SimulateBinomialSpaceTimeGridSummaryAtDateOptions,
  SimulateBinomialSpaceTimeGridSummaryOptions,
  UniversalKrigingOptions,
  UniversalTrend,
  VariogramParams,
  VariogramTypeName,
  ComputeEmpiricalSpaceTimeVariogramOptions,
  EmpiricalSpaceTimeVariogramResult,
  FitSpaceTimeVariogramOptions,
  FitSpaceTimeVariogramResult,
  FittedSpaceTimeVariogram,
  SpaceTimeBinomialDiagnostics,
  SpaceTimeBinomialKrigingFromFittedOptions,
  SpaceTimeBinomialFromPrecomputedLogitsOptions,
  SpaceTimeBinomialFromPrecomputedLogitsWithVariancesOptions,
  SpaceTimeBinomialKrigingOptions,
  SpaceTimeBinomialKrigingWithPriorOptions,
  SpaceTimeOrdinaryKrigingFromFittedOptions,
  SpaceTimeOrdinaryKrigingOptions,
  SpaceTimeProjectedOrdinaryKrigingFromFittedOptions,
  SpaceTimeProjectedOrdinaryKrigingOptions,
  SpaceTimeSimpleKrigingFromFittedOptions,
  SpaceTimeSimpleKrigingOptions,
  SpaceTimeUniversalKrigingFromFittedOptions,
  SpaceTimeUniversalKrigingOptions,
  SpaceTimeUniversalTrend,
  SpaceTimeVariogramFamily,
  SpaceTimeVariogramParams,
} from "./types.js";
export type { TimeUnit } from "./time.js";

export { binomialPreprocess } from "./binomial-preprocess.js";
export type {
  BinomialPreprocessOptions,
  BinomialPreprocessResult,
} from "./binomial-preprocess.js";
export {
  DEFAULT_BINOMIAL_PRIOR,
  resolveBinomialPriorOrDefault,
  smoothedLogit,
  smoothedLogits,
} from "./internal/prior.js";

import { init } from "./internal/module.js";

export default init;
