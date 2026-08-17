# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`BinomialCounts`** — geometry-free `(successes, trials)` retained on count-based
  calibrated binomial fits for instance CV and diagnostics without re-packing tensors.
- **`binomial_logit_loo_msdr`** — shared Rust helper for logit-scale LOO MSDR on any
  binomial [`KrigingPredictor`](src/predictor/cv.rs).
- **`SpaceTimeBinomialDiagnostics`** export from the `spacetime` module.
- **Tangent-plane binomial instance CV** — `leaveOneOut` / `kFold` on
  `BinomialTangentPlaneKriging` (npm) and `WasmKrigingModel` handles (WASM).
- **`BinomialAdapterRef`** — WASM dispatch for binomial build notes, diagnostics, and
  instance CV across geo, projected, tangent-plane, and spacetime adapters.
- **`KrigingConditioner<Site, Scale>`** — opaque live fitted state for ordinary, simple,
  universal, and binomial sequential Gaussian simulation, with compile-time continuous/logit
  scale separation and atomic condition append.

### Changed

- **Universal kriging engine** — [`UniversalKrigingEngine<K, T>`](src/kriging/universal_engine.rs)
  is parameterized by pairwise covariance and [trend basis](docs/adr/0004-trend-basis-universal-engine.md);
  [`SpaceTimeUniversalKrigingEngine<M>`](src/spacetime/kriging/universal_engine.rs) is a type alias.
- **Binomial `diagnostics()`** (Rust and npm) — count-based builds compute LOO logit MSDR
  from retained training counts and model coordinates; explicit count tensors are only
  required for precomputed-logit builds.
- **WASM binomial adapters** hold full calibrated fits instead of split model/notes/CV
  buffers; diagnostics and instance CV read stored training counts.
- **npm binomial quartet** — shared `binomial-model-shared` helpers for lifecycle,
  diagnostics, CV, and batch/grid prediction across geo, projected, tangent-plane, and
  spacetime classes.
- **SGS construction** — fitted models now provide `.into_conditioner()`; the four
  `sequential_*` functions consume conditioners while retaining their RNG, target-order, and
  result semantics.
- **Binomial SGS calibration** — simulation now uses the fitted model's canonical
  Laplace/Fisher logit observation variance instead of a separate empirical-Bayes variance path.

### Removed

- **Breaking Rust API:** `KrigingSimulator`, `BinomialKrigingSimulator`, the ten
  model/domain-specific `*Simulator` structs, and the duplicate `predictor::simulation` route.
  Use a fitted model's `.into_conditioner()` and the functions in `simulation` instead.

## [0.4.0] - 2026-06-12

Large release after 0.3.0: calibrated binomial kriging, dual-SPD engines, generic
predictor/simulator harnesses, unified WASM CV/simulation and model handle, and faster LOO CV.

### Added

- **Calibrated binomial kriging (default):** per-site logit *observation* variance
  (Empirical-Bayes `Beta(α,β)` + delta method) on the ordinary-kriging diagonal, with
  automatic inflation retries and always-returned [`BinomialBuildNotes`]
  (`calibrationVersion`, `logitInflation`, `nBuildAttempts`, prior, dropped zero-trial
  indices). The geographic fit type is `BinomialFit` (`BinomialCalibratedResult<BinomialKrigingModel>`);
  use [`BinomialKrigingModel::new_with_config`] for full control. `new` / `new_with_prior`
  are heteroskedastic by default with `HeteroskedasticBinomialConfig::default()`.
- **Empirical variogram (binomial):** `compute_empirical_variogram_binomial_calibrated`
  (pair-noise aware classical estimator) for the calibrated default.
- **TypeScript:** `interpolateBinomialToGrid` includes `buildNotes` on
  `InterpolateBinomialToGridResult`. `getBuildNotes()` is available on
  `BinomialKriging`, `BinomialProjectedKriging`, and `SpaceTimeBinomialKriging`.
- **Dual SPD ordinary kriging engine** — [`OrdinaryKrigingEngine<K: PairwiseCovariance>`](src/kriging/engine.rs)
  and [`SpaceTimeOrdinaryKrigingEngine<M>`](src/spacetime/kriging/engine.rs) (type alias): Cholesky on the
  covariance block `C` plus precomputed `β = C⁻¹·1`, with incremental conditioning via
  [`cholesky_extend_spd_lower`](src/cholesky_update.rs). See [ADR-0001](docs/adr/0001-dual-spd-ordinary-kriging.md)
  and [ADR-0003](docs/adr/0003-pairwise-covariance-ordinary-engine.md).
- **Simple kriging engines** for incremental SGS and prediction wrappers:
  [`SimpleKrigingEngine<K: PairwiseCovariance>`](src/kriging/simple_engine.rs) and
  [`SpaceTimeSimpleKrigingEngine<M>`](src/spacetime/kriging/simple_engine.rs) (type alias).
- **Universal kriging engines** (dual SPD with multi-constraint Schur complement):
  [`UniversalKrigingEngine<K, T>`](src/kriging/universal_engine.rs) parameterized by
  pairwise covariance and [trend basis](docs/adr/0004-trend-basis-universal-engine.md), and
  [`SpaceTimeUniversalKrigingEngine<M>`](src/spacetime/kriging/universal_engine.rs) (type alias).
  Constant trend delegates to the ordinary engine.
- **Calibrated logit ordinary build** — shared [`build_calibrated_logit_ordinary`](src/kriging/binomial.rs)
  for geographic, projected, and space–time binomial models.
- **Generic predictor / simulator harnesses** (the simulator backend surface was later
  replaced by the Unreleased conditioner API):
  - [`KrigingPredictor`](src/predictor/cv.rs) + [`leave_one_out_cv`](src/predictor/cv.rs) /
    [`k_fold_cv`](src/predictor/cv.rs)
  - Legacy `KrigingSimulator` +
    [`sequential_gaussian_simulate`](src/simulation.rs) /
    [`sequential_binomial_simulate`](src/simulation.rs)
  - Per-geometry backend structs (e.g. [`OrdinaryGeoPredictor`](src/predictor/cv.rs),
    legacy `BinomialProjectedSimulator`)
- **Domain vocabulary** — [`CONTEXT.md`](CONTEXT.md) and architecture review artifacts.
- **`cv()`** — single WASM entry point dispatching on `(geometry, family)` for
  geo, projected, and spacetime ordinary / simple / universal / binomial CV.
- **`simulate()`** — single WASM entry point for all 2-D and spacetime SGS
  variants; set `nRealizations > 1` for ensemble output.
- TypeScript types **`CvOptions`**, **`SimulateOptions`**, **`KrigingGeometry`**,
  **`KrigingFamily`**.
- **Instance CV on fitted models** — all continuous and binomial model classes
  (2-D geo/projected and spacetime) expose `.leaveOneOut()` and `.kFold(k)` using
  the model's training data and variogram (binomial requires building from
  `successes`/`trials`, not precomputed logits).
- **`WasmKrigingModel` tagged handle** (ADR-0002) — one WASM type wraps every fitted
  variant with shared `geometry` / `family` tags, predict dispatch, instance CV,
  and static factories (`ordinaryGeoFromArrays`, `spacetimeBinomialGeoFromArrays`,
  etc.). TypeScript model classes (`OrdinaryKriging`, `SpaceTimeOrdinaryKriging`, …)
  are thin adapters over this handle.
- `OrdinaryKrigingModel::new_with_extra_diagonal` for arbitrary (homogeneous or) **site-specific**
  non-spatial noise on the covariance diagonal.
- Browser-representative workload and **prediction-only** benchmark: `benches/BROWSER_BENCHMARKS.md`
  and Criterion `bench_binomial_browser_representative` (`400` stations, `200×200` target grid;
  optional `gpu-blocking` sub-bench for WebGPU class RHS).
- **Fast LOO CV (ordinary / simple / ST ordinary)** — `leave_one_out_predictions` on the dual-SPD
  engines deletes one station from the fitted Cholesky factor per hold-out via O(n²) rank-one
  downdate (Krause & Igel 2015) instead of refitting each fold from scratch. Constant-trend
  universal CV delegates to the ordinary fast path.

### Changed (breaking / migration from 0.3.x)

- **Default binomial prior is `Beta(1, 1)`** in Rust and when the TS/JS `prior` is
  omitted; use explicit `0.5/0.5` for Jeffreys.
- **Rust return types:** `BinomialKrigingModel::new` / `new_with_prior` /
  `new_with_config` (and the analogous **projected** and **space–time** binomial
  builders, including some `from_precomputed_logits*`) return
  `BinomialCalibratedResult<…>` (alias `BinomialFit` in geographic 2D). Use
  `.model`, `.into_model()`, or `Deref` to `&InnerModel` for `predict` / `predict_batch`.
- **Numerical drift vs 0.3.x:** ordinary (and binomial-via-ordinary) predictions may differ at
  the last few ULPs in `f32` due to the dual-SPD formulation (mathematically equivalent in exact
  arithmetic). One CV tolerance was relaxed accordingly.
- **CV and simulation Rust API:** the 20 named `leave_one_out_*` / `k_fold_*` and 16
  `conditional_simulate_*` entry points added in 0.3.0 are **removed**. Use predictor/simulator
  backend structs with the generic harnesses instead (re-exported from [`cv`](src/cv.rs) and
  [`simulation`](src/simulation.rs)).
- **Unified WASM/TypeScript CV and simulation seam** — the per-variant suffixed exports from
  0.3.0 (`leaveOneOutSimple`, `leaveOneOutBinomial`, `conditionalSimulateSpaceTimeUniversal`,
  `conditionalSimulateManyBinomial`, …) are **removed**. Dispatch through `cv()` /
  `simulate()` instead. The base convenience wrappers **`leaveOneOut`**, **`kFold`**,
  **`conditionalSimulate`**, and **`conditionalSimulateMany`** remain, but now take unified
  `CvOptions` / `SimulateOptions` with optional `geometry` and `family` (default `"geo"` +
  `"ordinary"`). In 0.3.x those four names were geo-ordinary-only with variant-specific option
  types.
- **Spacetime binomial CV** no longer requires a dummy `values` array; only
  `successes` / `trials` are validated for the binomial family.
- **Per-family WASM model exports removed** — `WasmOrdinaryKriging`,
  `WasmBinomialKriging`, `WasmSpaceTimeOrdinaryKriging`, and the other eleven
  legacy `Wasm*Kriging` types are no longer exported from the WASM binary. Use
  `WasmKrigingModel` static factories or the public TypeScript model classes.
- Count inputs with `trials==0` are dropped before binomial fit (at least two retained sites
  required). Rebuild the published `pkg` after upgrading (`npm run build:wasm`).

### Changed

- **Incremental SGS:** all continuous and binomial simulator backends extend the kriging
  conditioner per target (ordinary, simple, universal with any supported drift, projected,
  space–time). No refit-per-target paths remain in production SGS.
- **SGS harness:** skips appending a sampled target when kriging variance is below
  `1e-10` (target already in the conditioning set), avoiding duplicate-site Cholesky failures.
- **Binomial SGS** uses heteroskedastic (logit observation variance) conditioning in the
  per-target sequential fits for closer alignment with calibrated binomial prediction.
- **Size-independent diagonal jitter** — `kriging_diagonal_jitter` / `spacetime_diagonal_jitter` no longer
  scale with `√n`, so incremental Cholesky extend/downdate and LOO CV stay consistent when the
  conditioning set grows or shrinks by one station.

### Removed

- Separate **hetero-only** binomial constructor (superseded by calibrated default on `new` /
  `new_with_prior` / `new_with_config`).
- Dead helpers duplicated from the former `predictor/simulation` module in
  [`simulation.rs`](src/simulation.rs) (`Rng`, `resolve_target_order`, `validate_continuous_inputs`).
- Per-variant WASM/TypeScript CV and SGS exports from 0.3.0 (see migration table below).

### Fixed

- **GPU batch prediction** on [`OrdinaryKrigingModel`](src/kriging/ordinary.rs): uses
  `engine.coords()` and `engine.variogram()` after the engine refactor.
- **Ordinary engine condition append** now appends to `coords` as well as `prepared`/`values`.

### Migration (Rust)

```rust
// CV — was leave_one_out_ordinary(&coords, &values, variogram)
leave_one_out_cv(&OrdinaryGeoPredictor { coords, values, variogram })?;

// SGS — was conditional_simulate_ordinary(...)
sequential_gaussian_simulate(
    OrdinaryKrigingModel::new(dataset, variogram)?.into_conditioner()?,
    &targets,
    SimulationOptions::new(seed),
)?;
```

### Migration (npm / WASM)

**Retained wrappers (signature change):** `leaveOneOut`, `kFold`, `conditionalSimulate`, and
`conditionalSimulateMany` are still exported. Pass unified options; omit `geometry` / `family`
for geo + ordinary (same default as 0.3.x geo-ordinary calls).

| Removed export (0.3.x suffixed variants) | Replacement |
|------------------------------------------|-------------|
| `leaveOneOutSimple` / `kFoldSimple` | `cv({ geometry: "geo", family: "simple", mean, … })` or `leaveOneOut({ … })` |
| `leaveOneOutUniversal` / `kFoldUniversal` | `cv({ geometry: "geo", family: "universal", trend, … })` |
| `leaveOneOutProjected` / `kFoldProjected` | `cv({ geometry: "projected", family: "ordinary", xs, ys, majorAngleDeg, rangeRatio, … })` |
| `leaveOneOutBinomial` / `kFoldBinomial` | `cv({ geometry: "geo", family: "binomial", successes, trials, … })` |
| `leaveOneOutBinomialProjected` / `kFoldBinomialProjected` | `cv({ geometry: "projected", family: "binomial", xs, ys, successes, trials, … })` |
| `leaveOneOutSpaceTime*` / `kFoldSpaceTime*` | `cv({ geometry: "spacetime", family: "ordinary" \| "simple" \| "universal" \| "binomial", times, spaceTimeVariogram, … })` |
| `conditionalSimulateSimple` / `Universal` / `Projected` / `Binomial` | `simulate({ geometry, family, … })` or `conditionalSimulate({ … })` |
| `conditionalSimulateSpaceTime*` | `simulate({ geometry: "spacetime", family, spaceTimeVariogram, … })` |
| `conditionalSimulateManySimple` / `*Binomial` / `*SpaceTime*` (suffixed) | `conditionalSimulateMany({ geometry, family, nRealizations, baseSeed, … })` |
| `WasmOrdinaryKriging`, `WasmSimpleKriging`, … | `WasmKrigingModel.ordinaryGeoFromArrays`, `WasmKrigingModel.simpleGeoFromArrays`, … |
| `WasmSpaceTimeOrdinaryKriging`, … | `WasmKrigingModel.spacetimeOrdinaryGeoFromArrays`, … |

Spacetime calls use **`spaceTimeVariogram`** (not `variogram`). Binomial simulation
uses **`conditioningSuccesses`** / **`conditioningTrials`**; binomial CV uses
**`successes`** / **`trials`**.

```ts
import { cv, leaveOneOut, kFold, simulate, conditionalSimulate } from "kriging-rs-wasm";

// LOO ordinary CV (geometry/family optional on convenience wrappers)
const loo = leaveOneOut({ lats, lons, values, variogram });

// K-fold projected binomial CV
const kf = cv({
  geometry: "projected",
  family: "binomial",
  xs, ys, successes, trials, variogram,
  majorAngleDeg: 0, rangeRatio: 1, k: 5,
});

// Spacetime SGS
const samples = simulate({
  geometry: "spacetime",
  family: "ordinary",
  conditioningLats, conditioningLons, conditioningTimes, conditioningValues,
  targetLats, targetLons, targetTimes,
  spaceTimeVariogram,
  seed: 42n,
});
```

Public TypeScript classes construct via `WasmKrigingModel.*` factories internally.
If you call WASM glue directly, use `WasmKrigingModel` static factories.

## [0.3.0] - 2026-04-16

0.3.0 is a large feature release. Highlights:

- A new **spatio-temporal kriging** module that extends the 2-D surface with a time axis.
- **Cross-validation** (leave-one-out and K-fold) and **conditional simulation** (Sequential Gaussian Simulation) — both new to the project — shipped for every kriging variant in both 2-D and space-time.
- A focused **disease prevalence mapping cookbook**: tightened one-shot binomial interpolation, fast multi-realization SGS, generic ensemble aggregators, grid-shaped binomial simulation, polygon roll-ups, projected binomial kriging, and date-aware space-time helpers.
- The remaining `kriging-rs-wasm` feature parity gaps from 0.2.x are filled (`SimpleKriging`, `UniversalKriging`, `ProjectedKriging`, neighborhoods, more variogram families).
- A round of TypeScript ergonomics: discriminated-union variograms, `fromFitted`, `Symbol.dispose`, date helpers, fixed-time grids, and multi-realization SGS.

### Added

#### Disease prevalence mapping cookbook (TypeScript + Rust)

A focused workflow for the project's primary use case: building, validating,
sampling, and reporting prevalence surfaces from binomial count data.

- **Tightened one-shot binomial interpolation.** `interpolateBinomialToGrid`
  now fits the variogram on the same EB-smoothed logits used by
  `BinomialKriging`, accepts a `prior` and an `estimator` (`"classical" |
  "cressie-hawkins"`), exposes the resulting `fittedVariogram` on the result,
  and can run leave-one-out / k-fold CV in the same call via
  `withCv: true | "loo" | { k }`. The new `BinomialCvSummary` (both logit and
  prevalence scales) is reported alongside the grid.
- **Multi-realization SGS — fast path.** New Rust entry points
  `conditional_simulate_many{,_spacetime}{,_binomial}` amortize the
  conditioning factorization across realizations and ship behind WASM bridges
  (`conditionalSimulateMany{,Binomial}` and the ST variants now do a single
  JS↔WASM crossing per ensemble).
- **Generic ensemble aggregators.** Pure-TS `ensembleMean`, `ensembleVariance`,
  `ensembleQuantiles`, and `ensembleExceedanceProbability` operate over the
  flat row-major buffers produced by the `*Many*` simulators.
- **Grid-shaped binomial simulation.** `simulateBinomialGrid`,
  `simulateBinomialGridEnsemble`, and `simulateBinomialGridSummary` build
  conditioning + target geometry from a `GeoGridBounds` and return either
  nested `[yCells][xCells]` arrays (single realization), a flat ensemble
  buffer (many realizations), or per-cell mean / variance / quantile /
  exceedance maps (one-shot summary).
- **Polygon aggregation over ensembles.** New Rust `aggregate` module
  (`polygon_weighted_summary`, `polygon_weighted_summaries_batch`) reduces an
  ensemble buffer to per-polygon mean / variance / quantiles using
  `(indices, weights)` cell lists; bridged through WASM as
  `aggregatePolygonsOverEnsemble`. The TS `aggregatePrevalenceByPolygon` and
  `polygonCellsFromMask` helpers wrap this for population-weighted
  district-level reporting.
- **Projected binomial kriging.** New Rust `BinomialProjectedKrigingModel`
  with `ProjectedBinomialObservation` (Beta priors and pre-computed logits
  supported), CV (`leave_one_out_binomial_projected`,
  `k_fold_binomial_projected`), SGS
  (`conditional_simulate_binomial_projected`,
  `conditional_simulate_many_binomial_projected`), and a WASM bridge
  (`WasmBinomialProjectedKriging`). The TS layer ships
  `BinomialProjectedKriging` plus `leaveOneOutBinomialProjected`,
  `kFoldBinomialProjected`, `conditionalSimulateBinomialProjected`, and
  `conditionalSimulateManyBinomialProjected` for projected (planar, optionally
  anisotropic) prevalence mapping.
- **Date-aware space-time helpers.** `SpaceTimeBinomialKriging` gains
  `predictAtDate`, `predictBatchAtDates`, `predictBatchArraysAtDates`, and
  `predictGridAtDate` (date-axis configured per call via
  `DateAxisOptions = { timeUnit?, epoch? }`). New free functions
  `simulateBinomialSpaceTimeGrid{,Ensemble,Summary}` and their `*AtDate`
  counterparts mirror the spatial grid family for ST binomial conditioning.

#### Spatio-temporal kriging (new `spacetime` module)

- **Coordinates & metrics.** `SpaceTimeCoord<C>` and `SpaceTimeDataset<C>`, generic over any spatial-coordinate type that implements the new `SpatialMetric` trait. Two implementations ship out of the box: `GeoMetric` (Haversine on `GeoCoord`) and `ProjectedMetric` (Euclidean on `ProjectedCoord` with optional `Anisotropy2D`). All re-exported from `spacetime`.
- **Variogram families.** `SpaceTimeVariogram::new_separable(spatial, temporal)` (normalized product) and `SpaceTimeVariogram::new_product_sum(spatial, temporal, k1, k2, k3)` (`k₁·C_s·C_t + k₂·C_s + k₃·C_t`), with admissibility checks (nonnegative `k_i`, `k₁+k₂+k₃ > 0`, power-law marginals rejected).
- **Empirical & fitted variograms.** `compute_empirical_spacetime_variogram` computes a 2-D spatio-temporal variogram (Matheron and Cressie–Hawkins estimators); `spatial_marginal` / `temporal_marginal` project it to 1-D slices. `fit_spacetime_variogram` fits the two marginals independently and, for product-sum, solves a non-negative least-squares problem for `(k1, k2, k3)`. Supporting types: `EmpiricalSpaceTimeVariogram`, `SpaceTimeVariogramConfig`, `SpaceTimeFitConfig`, `SpaceTimeFitResult`.
- **Kriging models** (all generic over `SpatialMetric`): `SpaceTimeOrdinaryKrigingModel<M>`, `SpaceTimeSimpleKrigingModel<M>`, `SpaceTimeUniversalKrigingModel<M>` (trends: `Constant`, `LinearInTime`, `QuadraticInTime`, `LinearInSpace`, `LinearInSpaceAndTime`, `QuadraticInSpaceAndTime`), and `SpaceTimeBinomialKrigingModel<M>` (Beta priors via `SpaceTimeBinomialObservation`; pre-computed logits supported).
- **WASM / TypeScript bindings.** `WasmSpaceTimeOrdinaryKriging`, `WasmSpaceTimeSimpleKriging`, `WasmSpaceTimeUniversalKriging`, `WasmSpaceTimeBinomialKriging`, `WasmSpaceTimeOrdinaryProjectedKriging`, plus `wasmComputeEmpiricalSpaceTimeVariogram` / `wasmFitSpaceTimeVariogram`. TypeScript wrappers under `src/spacetime/`: `SpaceTime{Ordinary,Simple,Universal,Binomial,ProjectedOrdinary}Kriging`, `computeEmpiricalSpaceTimeVariogram`, `fitSpaceTimeVariogram`. New `KrigingErrorCode` values: `"unknown_family"`, `"unknown_trend"`, `"unknown_estimator"`.

#### Cross-validation (new in 0.3.0, every variant)

Cross-validation is brand new in this release: leave-one-out and K-fold helpers
ship for every kriging variant, both 2-D and space-time, from day one.

- **Rust (2-D):** `leave_one_out` / `k_fold` (ordinary) plus `_simple`, `_universal`, `_projected`, `_binomial`. Fold iteration is factored into shared `for_each_loo_fold` / `for_each_k_fold` helpers, so adding new variants is trivial.
- **Rust (space-time):** `leave_one_out_spacetime` / `k_fold_spacetime` (ordinary), plus `_simple`, `_universal`, `_binomial` — generic over `SpatialMetric` (ordinary / simple / binomial) or `SpatialBasis` (universal).
- **TypeScript:** `leaveOneOut` / `kFold` (ordinary) plus `Simple`, `Universal`, `Projected`, `Binomial`, `SpaceTime`, `SpaceTimeSimple`, `SpaceTimeUniversal`, `SpaceTimeBinomial` suffixes for both verbs.
- **Binomial CV reports both scales.** New `BinomialCvResidual` / `BinomialCvSummary` / `BinomialCvResult` types carry per-station observed/predicted values and variances on **both** the logit scale (directly comparable to continuous kriging; calibratable via MSDR) and the prevalence scale (delta-method variance). Stations with `trials == 0` retain their index with `NaN` observed fields and are automatically skipped in summary aggregation.

#### Conditional simulation (Sequential Gaussian Simulation, new in 0.3.0, every variant)

Also brand new: SGS helpers ship for every kriging variant, both 2-D and
space-time.

- **Rust (2-D):** `conditional_simulate` (ordinary) plus `_simple`, `_universal`, `_projected`, `_binomial`. Target-order validation is factored into a shared `resolve_target_order` helper.
- **Rust (space-time):** `conditional_simulate_spacetime` (ordinary), plus `_simple`, `_universal`, `_binomial`.
- **TypeScript:** `conditionalSimulate` (ordinary) plus `Simple`, `Universal`, `Projected`, `Binomial`, `SpaceTime`, `SpaceTimeSimple`, `SpaceTimeUniversal`, `SpaceTimeBinomial` suffixes. All are deterministic for a given `seed` and accept an optional `targetOrder` override.
- **Binomial simulation reports both scales.** New `BinomialSimulationResult` (Rust and TS) carries `logit_samples` / `logitSamples` (unbounded) and `prevalence_samples` / `prevalenceSamples` (in `(0, 1)`, by construction `logistic(logit_samples)`). Simulation runs on the logit scale; stations with `trials == 0` are dropped from the initial conditioning pool. Beta prior supported via `prior_alpha` / `prior_beta` (Rust) or `prior?: BinomialPriorParams` (TypeScript).

#### WASM / npm feature parity

`kriging-rs-wasm` now mirrors the full Rust surface — these bullets close the
2-D gaps left over from 0.2.x (ordinary + binomial only). The new CV and SGS
helpers are listed in their own sections above.

- New classes: `SimpleKriging` (known mean), `UniversalKriging` (trends `"constant" | "linear" | "quadratic"`), and `ProjectedKriging` (planar `(x, y)` with 2-D anisotropy via `majorAngleDeg`, `rangeRatio`).
- `OrdinaryKriging.setNeighborhood({ maxNeighbors?, maxRadius? })` and `neighborhood()` for search-neighborhood configuration.
- `BinomialKriging.fromPrecomputedLogits(...)` factory that bypasses empirical-Bayes shrinkage.
- Top-level helpers: `computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`, `evaluateNestedVariogram`.
- `fitVariogram` and `computeEmpiricalVariogram` accept `estimator: "classical" | "cressie-hawkins"`.
- `VariogramTypeName` gains `"power"` and `"holeeffect"`.

#### TypeScript DX polish (`kriging-rs-wasm`)

Higher-level ergonomics layered on top of the WASM bindings.

- **Discriminated union variograms.** `SpaceTimeVariogramParams` and `FittedSpaceTimeVariogram` are now discriminated on `family`: `"separable"` (no product-sum coefficients) or `"productSum"` (requires `k1`, `k2`, `k3`). TypeScript enforces the coefficient shape; passing snake_case is a type error.
- **`fromFitted(...)`** static on every space-time kriging class — pass a `fitSpaceTimeVariogram` result directly as `fittedVariogram`, no destructuring required.
- **`[Symbol.dispose]()`** on every 2-D and space-time kriging class, enabling `using model = new OrdinaryKriging(...)` (ES2023 explicit resource management) to release WASM memory automatically.
- **Unified binomial prior.** CV and SGS options take a single `prior?: BinomialPriorParams` object (both `alpha` and `beta` must be set together). Validation is centralized in the new `resolveBinomialPrior` helper.
- **Date helpers.** `timesFromDates(dates, unit?, epoch?)` and `datesFromTimes(times, unit?, epoch?)` with `TimeUnit = "ms" | "s" | "minutes" | "hours" | "days"` for converting between JS `Date` and the scalar time axis used by ST models.
- **`predictGridAtTime(options)`** on `SpaceTimeOrdinaryKriging` and `SpaceTimeBinomialKriging` — predicts a rectangular lat/lon grid at a fixed time slice and returns 2-D arrays shaped `[yCells][xCells]`.
- **Multi-realization SGS.** `conditionalSimulateMany({ ..., nRealizations, baseSeed? })` and `conditionalSimulateManySpaceTime(...)` draw N independent realizations with seeds `baseSeed + k` and return a flat row-major `Float64Array(nRealizations * nTargets)`.

#### Rust ergonomics

- `OrdinaryKrigingModel::set_neighborhood` — in-place counterpart of `with_neighborhood`, for FFI-friendly updates that don't consume `self`.

### Changed

- **No more silent fallbacks for missing WASM symbols.** Every TypeScript wrapper for an unconditional WASM export (`leaveOneOut*`, `kFold*`, `conditionalSimulate*`, `conditionalSimulateMany*`, `computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`, `evaluateNestedVariogram`, `aggregatePolygonsOverEnsemble`) now calls the WASM export directly without a `typeof === "function"` guard or pure-JS fallback. The `RawModule` shape in `internal/wasm-shapes.ts` declares these as required (only the `gpu`-feature-gated `webgpuAvailable` and instance-level `predictBatchGpu` / `predictBatchGpuOrCpu` remain optional, since they really may be absent from a GPU-less build). This eliminates a class of dead code and ensures any genuine packaging mistake fails loudly at first call rather than silently falling back to a slower path.
- **CV summary statistics now propagate `NaN`.** The internal `requireFiniteOrZero` helper has been replaced with `requireFiniteOrNaN` (in `internal/convert.ts`), and is now used for `meanError`, `rmse`, and `msdr` on `CvSummary` / `BinomialCvSummary`, plus `residuals` on `FittedSpaceTimeVariogram`. Previously these were silently coerced to `0` when the underlying Rust statistic was `NaN` (for example when zero residuals were evaluated), which masked "no data to summarize" as "perfectly calibrated". The honest `NaN` is now returned; non-numeric payloads still throw.
- **`unpackBinomialManyPayload` no longer carries fallback `nRealizations` / `nTargets`.** The Rust `binomial_many_to_js` serializer always populates those fields, so the JS-side `?? fallback` defaults were defensive against our own code. They have been removed; both fields are now resolved with `requireNumber`, surfacing any inconsistency in the WASM payload immediately.

### Documentation

- **Disease prevalence mapping cookbook** in `npm/kriging-rs-wasm/README.md`
  walking through fit → validate → simulate → aggregate end-to-end.
- **`examples/prevalence_mapping.rs`** — Rust runnable example showing the
  same workflow against the native crate.

## [0.2.3] - 2026-04-06

### Fixed

- **npm package (kriging-rs-wasm)** – Load the wasm-pack glue with a static `import` so bundlers (e.g. Vite) include `pkg/kriging_rs.js` and the `.wasm` file in the app output. Previously a dynamic `import("../pkg/kriging_rs.js")` was not analyzed at build time, so production requests to `/pkg/kriging_rs.js` failed unless consumers copied `pkg/` into `dist/`.
- **npm package (kriging-rs-wasm)** – `typecheck:contracts` and `verify` run `build:wasm` before TypeScript so `pkg/kriging_rs.js` exists on fresh checkouts (fixes CI TS2307 after the static import change).

## [0.2.2] - 2025-03-15

### Fixed

- **npm package (kriging-rs-wasm)** – The published tarball again correctly includes the `pkg/` directory (`kriging_rs.js`, `kriging_rs_bg.wasm`, etc.). Root `.gitignore` was narrowed to `/pkg/` and `/dist/` so npm pack no longer excludes the package’s build output (monorepo ignore behavior). The build script now removes wasm-pack’s generated `pkg/.gitignore` (which ignored all files) so npm pack includes the WASM artifacts.

## [0.2.1] - 2025-03-15

### Fixed

- **npm package (kriging-rs-wasm)** – The published tarball now includes the `pkg/` directory (wasm-pack output: `kriging_rs.js`, `kriging_rs_bg.wasm`, etc.). Previously `build:wasm` used `--out-dir npm/kriging-rs-wasm/pkg`, which from the package directory resolved to a nested path, so the top-level `pkg/` listed in `files` was never created. Build scripts now use `--out-dir pkg`. `prepublishOnly` also verifies that `pkg/` exists and contains the expected files before publishing.

## [0.2.0] - 2025-03-15

### Added

- **npm package (kriging-rs-wasm)**
  - `predictGrid(options)` on `OrdinaryKriging` and `BinomialKriging`: rectangular grid prediction from bounds and cell counts; returns 2D value/variance (and prevalence/logit) grids. Types: `PredictGridOptions`, `OrdinaryGridOutput`, `BinomialGridOutput`.
  - One-shot helpers: `interpolateOrdinaryToGrid(options)` and `interpolateBinomialToGrid(options)` (fit → build model → predict grid → free). Types: `InterpolateOrdinaryToGridOptions`, `InterpolateBinomialToGridOptions`.
  - `KrigingError.code` and `KrigingErrorCode` for UI-friendly error handling.

### Changed

- **npm package (kriging-rs-wasm)**
  - `init()` now returns `Promise<void>` (was `Promise<unknown>`) so callers can use it directly without a wrapper.
  - Optional `nuggetOverride` on `OrdinaryKrigingFromFittedOptions` to override the fitted variogram nugget when building the model. Binomial count data uses `fitBinomialVariogram` (calibrated path); use its returned `FittedVariogram` with `BinomialKriging.fromFittedVariogram*` instead of overriding the nugget separately.
  - `model.free()` is idempotent: safe to call multiple times; subsequent calls are no-ops. The TypeScript wrapper clears its reference after the first call so use-after-free throws a clear error. Documented in README and JSDoc.

## [0.1.0] - 2025-03-15

### Added

- **Rust crate (kriging-rs)**
  - Ordinary kriging: `OrdinaryKrigingModel`, `Prediction`, single-point and batch prediction.
  - Binomial kriging: `BinomialKrigingModel`, `BinomialObservation`, `BinomialPrediction`, `BinomialPrior`; optional prior; single-point and batch prediction.
  - Empirical variogram: `compute_empirical_variogram`, `EmpiricalVariogram`, `VariogramConfig`, `PositiveReal`.
  - Variogram fitting: `fit_variogram`, `FitResult`.
  - Parametric variogram models: `VariogramModel`, `VariogramType` — Spherical, Exponential, Gaussian, Cubic, Stable, Matérn (Stable and Matérn with optional shape parameter).
  - `GeoCoord` (lat/lon with validation), Haversine distance and distance matrix; `GeoDataset` for coordinates and values.
  - `Real` type alias (`f32`), `KrigingError`; utilities: `Probability`, `clamp_probability`, `logistic`, `logit`, `logit_clamped`.
  - Optional `wasm` feature for browser WASM bindings (see npm package).
  - Optional `gpu` and `gpu-blocking` features for WebGPU-based batch prediction and RHS covariance building via `wgpu`; `GpuBackend`, `GpuSupport`, `detect_gpu_support`, `build_rhs_covariances_gpu`, and related APIs.
- **npm package (kriging-rs-wasm)**
  - TypeScript-first WASM bindings; `init()` required before use.
  - `OrdinaryKriging` and `BinomialKriging` (constructors, `predict`, `predict_batch`, `predict_batch_arrays`).
  - `fitVariogram` with configurable bins and variogram type (string or `VariogramType` enum).
  - Variogram types: spherical, exponential, gaussian, cubic, stable, matern (optional `shape` for stable and matern).
  - `KrigingError` (JS class with `cause`); `webgpuAvailable` when built with GPU support.
  - Batch and typed-array prediction APIs.

[0.4.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.4.0
[0.3.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.3.0
[0.2.3]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.3
[0.2.2]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.2
[0.2.1]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.1
[0.2.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.0
[0.1.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.1.0
