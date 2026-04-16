/**
 * Public types and interfaces for kriging-rs-wasm. Purely type-level; no runtime
 * behavior lives in this module.
 *
 * @module
 */

/**
 * Supported variogram model type names (string form, e.g. in {@link FittedVariogram}).
 */
export type VariogramTypeName =
  | "spherical"
  | "exponential"
  | "gaussian"
  | "cubic"
  | "stable"
  | "matern"
  | "power"
  | "holeeffect";

/**
 * Empirical variogram estimator choice.
 *
 * - `"classical"` — Matheron's estimator (mean-squared differences; default).
 * - `"cressie-hawkins"` — robust fourth-root estimator, resistant to outliers.
 */
export type EmpiricalEstimator = "classical" | "cressie-hawkins";

/**
 * Universal kriging drift / trend basis.
 *
 * - `"constant"` — equivalent to ordinary kriging.
 * - `"linear"` — `[1, lat, lon]`.
 * - `"quadratic"` — `[1, lat, lon, lat², lat·lon, lon²]`.
 */
export type UniversalTrend = "constant" | "linear" | "quadratic";

/**
 * Input type for numeric coordinate or value arrays; accepts plain arrays or typed arrays.
 */
export type NumericArrayInput = number[] | ArrayLike<number>;

/**
 * Input type for integer counts (e.g. successes, trials); accepts plain arrays or typed arrays.
 * Values represent counts and should be non-negative integers (semantic only; no runtime check).
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
 * @property variance - Kriging variance of the **logit**, not of prevalence. See
 *   {@link BinomialPrediction.prevalenceVariance} for a probability-scale approximation.
 * @property prevalenceVariance - Delta-method approximation of Var(prevalence),
 *   i.e. `[p(1-p)]^2 * variance`. Use this for approximate CIs on the probability scale.
 */
export interface BinomialPrediction {
  prevalence: number;
  logitValue: number;
  variance: number;
  prevalenceVariance: number;
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
 * Use for large prediction grids. `variances` is the kriging variance of the logit;
 * `prevalenceVariances` is the delta-method probability-scale approximation.
 */
export interface BinomialBatchArrayOutput {
  prevalences: Float64Array;
  logitValues: Float64Array;
  variances: Float64Array;
  prevalenceVariances: Float64Array;
}

/**
 * Options for grid prediction: rectangular bounds in degrees and number of cells.
 * Cell centers are computed in row-major order (first row = south, last row = north;
 * within a row, west to east). Result grids have shape [yCells][xCells] (row index = latitude).
 */
export interface PredictGridOptions {
  /** Western longitude in degrees. */
  west: number;
  /** Southern latitude in degrees. */
  south: number;
  /** Eastern longitude in degrees. */
  east: number;
  /** Northern latitude in degrees. */
  north: number;
  /** Number of cells in the x (longitude) direction. */
  xCells: number;
  /** Number of cells in the y (latitude) direction. */
  yCells: number;
}

/**
 * Ordinary kriging grid output: 2D arrays with shape [yCells][xCells].
 * values[j][i] and variances[j][i] correspond to row j (latitude), column i (longitude).
 */
export interface OrdinaryGridOutput {
  values: number[][];
  variances: number[][];
}

/**
 * Binomial kriging grid output: 2D arrays with shape [yCells][xCells]. `variances` is the
 * kriging variance on the logit scale; `prevalenceVariances` is the delta-method
 * probability-scale approximation `[p(1-p)]^2 * variance`.
 */
export interface BinomialGridOutput {
  prevalences: number[][];
  logitValues: number[][];
  variances: number[][];
  prevalenceVariances: number[][];
}

/**
 * Options for one-shot ordinary kriging: fit variogram from sample data, build model, predict on grid, then free.
 */
export interface InterpolateOrdinaryToGridOptions {
  /** Sample latitudes in degrees. */
  lats: NumericArrayInput;
  /** Sample longitudes in degrees. */
  lons: NumericArrayInput;
  /** Sample values (same length as lats/lons). */
  values: NumericArrayInput;
  /** Grid bounds and cell counts. */
  west: number;
  south: number;
  east: number;
  north: number;
  xCells: number;
  yCells: number;
  /** Variogram model type (e.g. "exponential"). */
  variogramType: VariogramTypeName | number;
  /** Optional number of bins for empirical variogram (default 12). */
  nBins?: number;
  /** Optional max distance for binning. */
  maxDistance?: number;
  /** Optional nugget override when building model from fitted variogram. */
  nuggetOverride?: number;
}

/**
 * Options for one-shot binomial kriging: fit variogram, build model, predict on grid, then free.
 */
export interface InterpolateBinomialToGridOptions {
  /** Sample latitudes in degrees. */
  lats: NumericArrayInput;
  /** Sample longitudes in degrees. */
  lons: NumericArrayInput;
  /** Success counts (same length as lats/lons). */
  successes: IntegerArrayInput;
  /** Trial counts (same length as lats/lons). */
  trials: IntegerArrayInput;
  /** Grid bounds and cell counts. */
  west: number;
  south: number;
  east: number;
  north: number;
  xCells: number;
  yCells: number;
  /** Variogram model type (e.g. "exponential"). */
  variogramType: VariogramTypeName | number;
  /** Optional number of bins for empirical variogram (default 12). */
  nBins?: number;
  /** Optional nugget override when building model from fitted variogram. */
  nuggetOverride?: number;
  /** Optional Beta(alpha, beta) prior for binomial model. */
  prior?: BinomialPriorParams;
}

/**
 * Fitted variogram parameters from {@link fitVariogram}.
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
 * Options for {@link fitVariogram}. Pass a single object with sample data and
 * variogram model type; optional settings control binning for the empirical variogram.
 */
export interface FitVariogramOptions {
  /** Sample latitudes in degrees. */
  sampleLats: NumericArrayInput;
  /** Sample longitudes in degrees. */
  sampleLons: NumericArrayInput;
  /** Sample values (same length as sampleLats/sampleLons). */
  values: NumericArrayInput;
  /** Variogram model type: string (e.g. `"exponential"`) or {@link VariogramType} enum value. */
  variogramType: VariogramTypeName | number;
  /** Optional maximum distance for binning; omit for automatic choice. */
  maxDistance?: number;
  /** Number of distance bins for the empirical variogram (default 12). */
  nBins?: number;
  /** Empirical estimator: `"classical"` (default) or `"cressie-hawkins"` (robust). */
  estimator?: EmpiricalEstimator;
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
 * Search-neighborhood restriction for a kriging model. Omit both fields (or pass `{}`)
 * to clear the neighborhood. When both are given, the intersection applies (k-nearest
 * within radius).
 */
export interface NeighborhoodOptions {
  /** Keep only the `k` closest stations at each prediction location. */
  maxNeighbors?: number;
  /** Keep only stations within this great-circle distance (kilometers). */
  maxRadius?: number;
}

/**
 * Options for constructing a simple kriging model (with known mean).
 */
export interface SimpleKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Known mean used for the residual kriging system. */
  mean: number;
}

/**
 * Options for constructing a universal kriging model with a polynomial drift basis.
 */
export interface UniversalKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Drift basis: `"constant"`, `"linear"` (in lat/lon), or `"quadratic"`. */
  trend: UniversalTrend;
}

/**
 * Projected (planar) kriging options with 2D anisotropy.
 *
 * Coordinates are arbitrary planar `(x, y)` values (for example projected meters);
 * distances are Euclidean. The anisotropy model rotates the correlation ellipse by
 * `majorAngleDeg` (counter-clockwise from +x) and scales the minor axis by
 * `rangeRatio ∈ (0, 1]`.
 */
export interface ProjectedKrigingOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Azimuth of the major axis, in degrees counter-clockwise from +x. */
  majorAngleDeg: number;
  /** Ratio of minor to major range, in `(0, 1]`. */
  rangeRatio: number;
}

/**
 * Options for binomial kriging from pre-computed logits (bypasses the default
 * empirical-Bayes shrinkage from success/trial counts).
 */
export interface BinomialFromPrecomputedLogitsOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  logits: NumericArrayInput;
  variogram: VariogramParams;
}

/**
 * Result of {@link computeEmpiricalVariogram} / {@link computeDirectionalEmpiricalVariogram}.
 * One entry per distance bin in increasing-distance order.
 */
export interface EmpiricalVariogramResult {
  distances: Float64Array;
  semivariances: Float64Array;
  /** Pair counts per bin. */
  counts: Uint32Array;
}

/**
 * Options for {@link computeEmpiricalVariogram}.
 */
export interface ComputeEmpiricalVariogramOptions {
  sampleLats: NumericArrayInput;
  sampleLons: NumericArrayInput;
  values: NumericArrayInput;
  /** Optional maximum distance for binning; omit for automatic choice. */
  maxDistance?: number;
  /** Number of distance bins for the empirical variogram (default 12). */
  nBins?: number;
  /** Estimator choice; defaults to `"classical"`. */
  estimator?: EmpiricalEstimator;
}

/**
 * Options for {@link computeDirectionalEmpiricalVariogram}.
 * Uses planar `(x, y)` coordinates with Euclidean distances.
 */
export interface ComputeDirectionalEmpiricalVariogramOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  values: NumericArrayInput;
  /** Maximum distance for binning (same units as `xs`/`ys`). */
  maxDistance: number;
  /** Number of distance bins. */
  nBins: number;
  /** Azimuth in degrees measured counter-clockwise from +x. */
  azimuthDeg: number;
  /** Half-angle tolerance in degrees (0, 90]. */
  toleranceDeg: number;
}

/** Per-station residual from cross-validation. */
export interface CvResidual {
  /** Index of the held-out station in the original input arrays. */
  index: number;
  /** Observed value at the held-out station. */
  observed: number;
  /** Kriging prediction from the training fold. */
  predicted: number;
  /** Kriging variance. */
  variance: number;
  /** Signed residual `observed − predicted`. */
  error: number;
}

/** Summary statistics over cross-validation residuals. */
export interface CvSummary {
  n: number;
  /** Mean signed error (bias). */
  meanError: number;
  /** Root mean squared error. */
  rmse: number;
  /** Mean squared deviation ratio; ≈ 1 when variogram is well calibrated. */
  msdr: number;
}

/** Result of {@link leaveOneOut} / {@link kFold}. */
export interface CvResult {
  /** Per-station residuals in input order. */
  residuals: CvResidual[];
  /** Aggregate summary statistics. */
  summary: CvSummary;
  /** Typed-array view of per-station predictions (convenient for plotting). */
  arrays: {
    indices: Uint32Array;
    observed: Float64Array;
    predicted: Float64Array;
    variances: Float64Array;
  };
}

/**
 * Options for {@link leaveOneOut}. Uses ordinary kriging with the supplied variogram.
 */
export interface LeaveOneOutOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
}

/**
 * Options for {@link kFold}. Folds are deterministic round-robin (station `i` → fold `i % k`).
 * Caller should shuffle inputs for randomized validation.
 */
export interface KFoldOptions extends LeaveOneOutOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutSimple}. Simple kriging treats the supplied `mean` as
 * known for every fold (no in-fold refit), matching practice for an externally estimated
 * mean.
 */
export interface LeaveOneOutSimpleOptions extends LeaveOneOutOptions {
  /** Known constant mean used by simple kriging inside each fold. */
  mean: number;
}

/** Options for {@link kFoldSimple}. */
export interface KFoldSimpleOptions extends LeaveOneOutSimpleOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutUniversal}. Trend coefficients are re-estimated inside
 * each fold from the training stations, so the trend contributes no in-sample leakage.
 */
export interface LeaveOneOutUniversalOptions extends LeaveOneOutOptions {
  /** Polynomial drift basis. `"constant"` is equivalent to ordinary kriging. */
  trend: UniversalTrend;
}

/** Options for {@link kFoldUniversal}. */
export interface KFoldUniversalOptions extends LeaveOneOutUniversalOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutProjected}. Uses planar `(x, y)` coordinates and the
 * kriging variogram's `range` must be expressed in the same linear units. When
 * `rangeRatio === 1` the model is isotropic and `majorAngleDeg` is ignored.
 */
export interface LeaveOneOutProjectedOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Angle of the major (longer-range) axis in degrees (0 = +x, counter-clockwise). */
  majorAngleDeg: number;
  /** Ratio of minor range to major range, in (0, 1]. `1` = isotropic. */
  rangeRatio: number;
}

/** Options for {@link kFoldProjected}. */
export interface KFoldProjectedOptions extends LeaveOneOutProjectedOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutBinomial}. A station whose `trials[i] === 0` is treated as
 * unobservable: it participates in no training fold, and its residual carries `NaN` for
 * observed fields (prediction is still populated). The `summary.logit` and
 * `summary.prevalence` aggregates skip those stations automatically.
 */
export interface LeaveOneOutBinomialOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: ArrayLike<number> | Uint32Array;
  trials: ArrayLike<number> | Uint32Array;
  variogram: VariogramParams;
  /** Optional Beta prior hyperparameters; both or neither must be given. Default Beta(½, ½). */
  priorAlpha?: number;
  priorBeta?: number;
}

/** Options for {@link kFoldBinomial}. */
export interface KFoldBinomialOptions extends LeaveOneOutBinomialOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Per-station residual from binomial cross-validation. Reports the held-out observation
 * and the model's prediction on **both** the logit scale (directly comparable to
 * continuous kriging and MSDR-calibratable) and the prevalence scale (intuitive; delta-
 * method variance).
 *
 * When `trials === 0`, observed fields are `NaN`; see {@link LeaveOneOutBinomialOptions}.
 */
export interface BinomialCvResidual {
  /** Index of the held-out station in the original input arrays. */
  index: number;
  /** Held-out success count. */
  successes: number;
  /** Held-out trial count. `0` means the observation is undefined (`NaN` observed fields). */
  trials: number;
  /** Observed logit (log-odds of prevalence). `NaN` when `trials === 0`. */
  observedLogit: number;
  /** Model prediction on the logit scale. */
  predictedLogit: number;
  /** Kriging variance on the logit scale. */
  logitVariance: number;
  /** Observed prevalence `successes / trials`. `NaN` when `trials === 0`. */
  observedPrevalence: number;
  /** Model prediction on the prevalence scale (logistic of `predictedLogit`). */
  predictedPrevalence: number;
  /** Delta-method approximation of the variance of `predictedPrevalence`. */
  prevalenceVariance: number;
  /** Signed logit-scale error `observedLogit − predictedLogit` (`NaN` when `trials === 0`). */
  logitError: number;
  /** Signed prevalence-scale error `observedPrevalence − predictedPrevalence` (`NaN` when `trials === 0`). */
  prevalenceError: number;
}

/**
 * Aggregate binomial-CV summary reported on **both** scales. `nEvaluated` excludes
 * stations with `trials === 0` (which contribute `NaN` observations).
 */
export interface BinomialCvSummary {
  /** Total residuals, including any with `trials === 0`. */
  n: number;
  /** Number of residuals with `trials > 0`, i.e. those contributing to `logit`/`prevalence`. */
  nEvaluated: number;
  /** Summary statistics on the logit scale. */
  logit: CvSummary;
  /** Summary statistics on the prevalence scale. */
  prevalence: CvSummary;
}

/** Result of {@link leaveOneOutBinomial} / {@link kFoldBinomial}. */
export interface BinomialCvResult {
  /** Per-station residuals in input order. */
  residuals: BinomialCvResidual[];
  /** Aggregate summary on both scales. */
  summary: BinomialCvSummary;
  /** Typed-array view of per-station fields (convenient for plotting). */
  arrays: {
    indices: Uint32Array;
    successes: Uint32Array;
    trials: Uint32Array;
    observedLogit: Float64Array;
    predictedLogit: Float64Array;
    logitVariance: Float64Array;
    observedPrevalence: Float64Array;
    predictedPrevalence: Float64Array;
    prevalenceVariance: Float64Array;
  };
}

/**
 * Options for {@link conditionalSimulate}.
 *
 * Returns one sample per target in input order. When called repeatedly with the same
 * arguments (including `seed`), the output is deterministic.
 */
export interface ConditionalSimulateOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  variogram: VariogramParams;
  /** RNG seed for reproducibility (defaults to `0n`). Accepts number or bigint. */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Options for {@link conditionalSimulateSimple}.
 *
 * Simulation uses simple kriging with the supplied known `mean` at every step.
 */
export interface ConditionalSimulateSimpleOptions extends ConditionalSimulateOptions {
  /** Known constant mean used by simple kriging inside the simulation loop. */
  mean: number;
}

/**
 * Options for {@link conditionalSimulateUniversal}.
 *
 * Trend coefficients are re-estimated at each simulation step. Requires at least
 * `p + 1` conditioning stations, where `p = 1` (constant), `3` (linear), or `6` (quadratic).
 */
export interface ConditionalSimulateUniversalOptions extends ConditionalSimulateOptions {
  /** Polynomial drift basis. `"constant"` is equivalent to ordinary kriging. */
  trend: UniversalTrend;
}

/**
 * Options for {@link conditionalSimulateProjected}.
 *
 * Uses planar `(x, y)` coordinates and optional 2-D geometric anisotropy. Pass
 * `rangeRatio = 1` for isotropic simulation (angle is then ignored).
 */
export interface ConditionalSimulateProjectedOptions {
  conditioningXs: NumericArrayInput;
  conditioningYs: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetXs: NumericArrayInput;
  targetYs: NumericArrayInput;
  variogram: VariogramParams;
  /** Angle of the major (longer-range) axis in degrees (0 = +x, counter-clockwise). */
  majorAngleDeg: number;
  /** Ratio of minor range to major range, in (0, 1]. `1` = isotropic. */
  rangeRatio: number;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Options for {@link conditionalSimulateBinomial}.
 *
 * Simulation happens on the **logit** scale (where the Gaussian assumption is natural) and
 * results are reported on both the logit and prevalence scales via
 * {@link BinomialSimulationResult}. Stations with `trials === 0` are dropped from the
 * initial conditioning pool.
 */
export interface ConditionalSimulateBinomialOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  variogram: VariogramParams;
  /** Optional Beta prior hyperparameters; both or neither must be given. Default Beta(½, ½). */
  priorAlpha?: number;
  priorBeta?: number;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Result of {@link conditionalSimulateBinomial}. Contains samples on both the logit scale
 * (unbounded) and the prevalence scale (in `(0, 1)`), in the original target input order.
 *
 * By construction, `prevalenceSamples[i] === logistic(logitSamples[i])`.
 */
export interface BinomialSimulationResult {
  /** Simulated logit values (unbounded). */
  logitSamples: Float64Array;
  /** Simulated prevalence values in `(0, 1)`. */
  prevalenceSamples: Float64Array;
}

/** A single component of a nested (additive) variogram model. */
export interface NestedVariogramComponent {
  variogramType: VariogramTypeName;
  nugget: number;
  sill: number;
  range: number;
  shape?: number;
}

/** Result of {@link evaluateNestedVariogram}. */
export interface NestedVariogramEvaluation {
  distances: Float64Array;
  semivariances: Float64Array;
  covariances: Float64Array;
}

/**
 * Options for {@link OrdinaryKriging.fromFitted}: sample data plus a fitted variogram
 * (e.g. from {@link fitVariogram}) to build the model without manually spreading variogram fields.
 */
export interface OrdinaryKrigingFromFittedOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  fittedVariogram: FittedVariogram;
  /** If set, overrides the fitted variogram nugget when building the model (e.g. for UI-tuned sigma²). */
  nuggetOverride?: number;
}

/**
 * Options for {@link BinomialKriging.fromFittedVariogram}: count data plus a fitted variogram
 * (e.g. from fitting on logits or reusing ordinary-fit params) to build the model.
 */
export interface BinomialKrigingFromFittedVariogramOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  fittedVariogram: FittedVariogram;
  /** If set, overrides the fitted variogram nugget when building the model. */
  nuggetOverride?: number;
}

/**
 * Options for {@link BinomialKriging.fromFittedVariogramWithPrior}: count data, fitted variogram,
 * and Beta prior to build a binomial kriging model with a prior.
 */
export interface BinomialKrigingFromFittedVariogramWithPriorOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  fittedVariogram: FittedVariogram;
  prior: BinomialPriorParams;
  /** If set, overrides the fitted variogram nugget when building the model. */
  nuggetOverride?: number;
}

/**
 * Stable error codes for UI-friendly handling. When present, `KrigingError.code` is one of these.
 * Not every error has a code; the library may add new codes in minor releases.
 */
export type KrigingErrorCode =
  | "not_loaded"
  | "model_freed"
  | "mismatched_arrays"
  | "invalid_variogram"
  | "invalid_bins"
  | "singular_covariance"
  | "too_few_points"
  | "unknown_variogram"
  | "invalid_input"
  | "backend_unavailable"
  | "internal_error"
  | "unknown_family"
  | "unknown_trend"
  | "unknown_estimator";

// ---------------------------------------------------------------------------
// Spatio-temporal kriging types
// ---------------------------------------------------------------------------

/**
 * Family of space-time variogram models.
 *
 * - `"separable"` — `C(h_s, h_t) = C_s(h_s) · C_t(h_t) / C_s(0)` (product of normalized marginals).
 * - `"product_sum"` — `C(h_s, h_t) = k1·C_s(h_s)·C_t(h_t) + k2·C_s(h_s) + k3·C_t(h_t)` with
 *   `k1 ≥ 0`, `k2 ≥ 0`, `k3 ≥ 0` and `k1 + k2 + k3 > 0`.
 */
export type SpaceTimeVariogramFamily = "separable" | "product_sum";

/**
 * Fully-specified space-time variogram. Spatial and temporal marginals are ordinary
 * 2-D variograms (e.g. `{ variogramType: "exponential", nugget, sill, range }`); `k1/k2/k3`
 * are only used for `"product_sum"` and default to `1/0/0` for `"separable"`.
 */
export interface SpaceTimeVariogramParams {
  family: SpaceTimeVariogramFamily;
  spatial: VariogramParams;
  temporal: VariogramParams;
  k1?: number;
  k2?: number;
  k3?: number;
}

/**
 * Drift basis for space-time universal kriging.
 *
 * Spatial components are scalar projections `(a, b)` of the coordinate (`(lat, lon)` for
 * geographic, `(x, y)` for projected), time is a scalar `t`.
 *
 * - `"constant"` — `[1]`; equivalent to ordinary space-time kriging.
 * - `"linearInTime"` — `[1, t]`.
 * - `"quadraticInTime"` — `[1, t, t²]`.
 * - `"linearInSpace"` — `[1, a, b]`.
 * - `"linearInSpaceAndTime"` — `[1, a, b, t]`.
 * - `"quadraticInSpaceAndTime"` — `[1, a, b, t, a², a·b, b², t², a·t, b·t]`.
 */
export type SpaceTimeUniversalTrend =
  | "constant"
  | "linearInTime"
  | "quadraticInTime"
  | "linearInSpace"
  | "linearInSpaceAndTime"
  | "quadraticInSpaceAndTime";

/** Common base for geographic space-time kriging model options. */
export interface SpaceTimeOrdinaryKrigingOptions {
  /** Latitudes in degrees. */
  lats: NumericArrayInput;
  /** Longitudes in degrees. */
  lons: NumericArrayInput;
  /** Sample times (scalar; any monotone unit — e.g. days, seconds). */
  times: NumericArrayInput;
  /** Sample values (same length as lats/lons/times). */
  values: NumericArrayInput;
  /** Space-time variogram parameters. */
  variogram: SpaceTimeVariogramParams;
}

/** Options for building a space-time simple kriging model with known mean. */
export interface SpaceTimeSimpleKrigingOptions extends SpaceTimeOrdinaryKrigingOptions {
  mean: number;
}

/** Options for building a space-time universal kriging model. */
export interface SpaceTimeUniversalKrigingOptions extends SpaceTimeOrdinaryKrigingOptions {
  trend: SpaceTimeUniversalTrend;
}

/** Options for building a space-time binomial kriging model (count data). */
export interface SpaceTimeBinomialKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: SpaceTimeVariogramParams;
}

/** Options for building a projected (planar) space-time ordinary kriging model. */
export interface SpaceTimeProjectedOrdinaryKrigingOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** Azimuth of the major axis, in degrees counter-clockwise from +x. */
  majorAngleDeg: number;
  /** Ratio of minor to major range, in `(0, 1]`. */
  rangeRatio: number;
}

/**
 * Options for {@link computeEmpiricalSpaceTimeVariogram}. Produces a 2-D empirical
 * variogram binned simultaneously in space and time.
 */
export interface ComputeEmpiricalSpaceTimeVariogramOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  /** Maximum spatial distance (km) for binning; defaults to half the largest pair distance. */
  maxSpatialDistance?: number;
  /** Maximum temporal lag (time units) for binning; defaults to half the largest pair lag. */
  maxTemporalLag?: number;
  /** Number of spatial bins (required, ≥ 1). */
  nSpatialBins: number;
  /** Number of temporal bins (required, ≥ 1). */
  nTemporalBins: number;
  /** Empirical estimator: `"classical"` (Matheron, default) or `"cressie-hawkins"` (robust). */
  estimator?: EmpiricalEstimator;
}

/**
 * Empirical space-time variogram: row-major flat arrays indexed by
 * `i = spatialBin * nTemporalBins + temporalBin`.
 */
export interface EmpiricalSpaceTimeVariogramResult {
  nSpatialBins: number;
  nTemporalBins: number;
  /** Mean spatial lag per bin (length = `nSpatialBins * nTemporalBins`). */
  spatialLags: Float64Array;
  /** Mean temporal lag per bin. */
  temporalLags: Float64Array;
  /** Mean semivariance per bin. */
  semivariances: Float64Array;
  /** Pair count per bin. */
  nPairs: Float64Array;
}

/** Options for {@link fitSpaceTimeVariogram}. */
export interface FitSpaceTimeVariogramOptions extends ComputeEmpiricalSpaceTimeVariogramOptions {
  /** Space-time family to fit: `"separable"` or `"product_sum"`. */
  family: SpaceTimeVariogramFamily;
  /** Parametric model for the spatial marginal (e.g. `"exponential"`). */
  spatialModel: VariogramTypeName;
  /** Parametric model for the temporal marginal (e.g. `"exponential"`). */
  temporalModel: VariogramTypeName;
}

/** Result of fitting a space-time variogram. */
export interface FittedSpaceTimeVariogram {
  family: SpaceTimeVariogramFamily;
  spatial: VariogramParams;
  temporal: VariogramParams;
  /** Product coefficient (0 for separable-only; default 1). */
  k1: number;
  /** Spatial-marginal coefficient (0 unless `product_sum`). */
  k2: number;
  /** Temporal-marginal coefficient (0 unless `product_sum`). */
  k3: number;
  /** Sum-of-squared-errors of the final fit against the empirical variogram. */
  residuals: number;
}

/**
 * Combined empirical + parametric fit result returned by {@link fitSpaceTimeVariogram}.
 */
export interface FitSpaceTimeVariogramResult {
  empirical: EmpiricalSpaceTimeVariogramResult;
  fit: FittedSpaceTimeVariogram;
}

/**
 * Options for {@link leaveOneOutSpaceTime}. Space-time ordinary kriging CV over geographic
 * coordinates with a scalar time axis.
 */
export interface LeaveOneOutSpaceTimeOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
}

/** Options for {@link kFoldSpaceTime}. */
export interface KFoldSpaceTimeOptions extends LeaveOneOutSpaceTimeOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/** Options for {@link leaveOneOutSpaceTimeSimple}. */
export interface LeaveOneOutSpaceTimeSimpleOptions extends LeaveOneOutSpaceTimeOptions {
  /** Known constant mean used by simple ST kriging inside each fold. */
  mean: number;
}

/** Options for {@link kFoldSpaceTimeSimple}. */
export interface KFoldSpaceTimeSimpleOptions extends LeaveOneOutSpaceTimeSimpleOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/** Options for {@link leaveOneOutSpaceTimeUniversal}. */
export interface LeaveOneOutSpaceTimeUniversalOptions extends LeaveOneOutSpaceTimeOptions {
  /** Polynomial drift basis for universal ST kriging. */
  trend: SpaceTimeUniversalTrend;
}

/** Options for {@link kFoldSpaceTimeUniversal}. */
export interface KFoldSpaceTimeUniversalOptions extends LeaveOneOutSpaceTimeUniversalOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutSpaceTimeBinomial}. Stations with `trials[i] === 0` are
 * treated as unobservable and carry `NaN` observed fields; summaries skip them.
 */
export interface LeaveOneOutSpaceTimeBinomialOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** Optional Beta prior; both or neither must be given. Default Beta(½, ½). */
  priorAlpha?: number;
  priorBeta?: number;
}

/** Options for {@link kFoldSpaceTimeBinomial}. */
export interface KFoldSpaceTimeBinomialOptions extends LeaveOneOutSpaceTimeBinomialOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link conditionalSimulateSpaceTime}. Space-time SGS returns one sample per
 * target in input order; deterministic for a given `seed`.
 */
export interface ConditionalSimulateSpaceTimeOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningTimes: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  targetTimes: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/** Options for {@link conditionalSimulateSpaceTimeSimple}. */
export interface ConditionalSimulateSpaceTimeSimpleOptions extends ConditionalSimulateSpaceTimeOptions {
  /** Known constant mean used by simple ST kriging inside the simulation loop. */
  mean: number;
}

/** Options for {@link conditionalSimulateSpaceTimeUniversal}. */
export interface ConditionalSimulateSpaceTimeUniversalOptions extends ConditionalSimulateSpaceTimeOptions {
  /** Polynomial drift basis for universal ST kriging. */
  trend: SpaceTimeUniversalTrend;
}

/**
 * Options for {@link conditionalSimulateSpaceTimeBinomial}. Simulation happens on the logit
 * scale; results are returned on both logit and prevalence scales.
 */
export interface ConditionalSimulateSpaceTimeBinomialOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningTimes: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  targetTimes: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** Optional Beta prior; both or neither must be given. Default Beta(½, ½). */
  priorAlpha?: number;
  priorBeta?: number;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}
