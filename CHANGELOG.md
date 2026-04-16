# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-16

### Added

- **Spatio-temporal kriging.** New `spacetime` module with full feature parity against the 2-D kriging surface:
  - Coordinates and datasets: `SpaceTimeCoord<C>`, `SpaceTimeDataset<C>`, generic over a spatial-coordinate type that implements the new `SpatialMetric` trait.
  - Metrics: `GeoMetric` (Haversine on `GeoCoord`) and `ProjectedMetric` (Euclidean on `ProjectedCoord` with optional `Anisotropy2D`), both re-exported from `spacetime`.
  - Variogram families: `SpaceTimeVariogram::new_separable(spatial, temporal)` (normalized product) and `SpaceTimeVariogram::new_product_sum(spatial, temporal, k1, k2, k3)` with admissibility checks (k's non-negative, k1+k2+k3 > 0, power-law marginals rejected).
  - Empirical 2-D space-time variogram (`EmpiricalSpaceTimeVariogram`, `SpaceTimeVariogramConfig`, `compute_empirical_spacetime_variogram`) with Matheron and Cressie–Hawkins estimators; `spatial_marginal` / `temporal_marginal` helpers project to 1-D slices.
  - Parametric fitting (`fit_spacetime_variogram`, `SpaceTimeFitConfig`, `SpaceTimeFitResult`) that fits marginals independently and, for product-sum, solves a non-negative least-squares problem for `(k1, k2, k3)`.
  - Kriging models (all generic over `SpatialMetric`): `SpaceTimeOrdinaryKrigingModel<M>`, `SpaceTimeSimpleKrigingModel<M>`, `SpaceTimeUniversalKrigingModel<M>` with time-extended trends (`Constant`, `LinearInTime`, `QuadraticInTime`, `LinearInSpace`, `LinearInSpaceAndTime`, `QuadraticInSpaceAndTime`), and `SpaceTimeBinomialKrigingModel<M>` (with `SpaceTimeBinomialObservation`, supports Beta priors and pre-computed logits).
  - WASM bindings: `WasmSpaceTimeOrdinaryKriging`, `WasmSpaceTimeSimpleKriging`, `WasmSpaceTimeUniversalKriging`, `WasmSpaceTimeBinomialKriging`, `WasmSpaceTimeOrdinaryProjectedKriging`, plus `wasmComputeEmpiricalSpaceTimeVariogram` and `wasmFitSpaceTimeVariogram`.
  - TypeScript wrappers under `npm/kriging-rs-wasm/src/spacetime/`: `SpaceTimeOrdinaryKriging`, `SpaceTimeSimpleKriging`, `SpaceTimeUniversalKriging`, `SpaceTimeBinomialKriging`, `SpaceTimeProjectedOrdinaryKriging`, and top-level `computeEmpiricalSpaceTimeVariogram` / `fitSpaceTimeVariogram`. New `KrigingErrorCode`s: `"unknown_family"`, `"unknown_trend"`, `"unknown_estimator"`.
  - Space-time cross-validation: leave-one-out and K-fold for every ST variant.
    - Rust: `leave_one_out_spacetime` / `k_fold_spacetime` (ordinary), `_simple`, `_universal`, `_binomial` — all generic over `SpatialMetric` (ordinary/simple/binomial) or `SpatialBasis` (universal).
    - WASM / TypeScript: `leaveOneOutSpaceTime`, `kFoldSpaceTime`, `leaveOneOutSpaceTimeSimple`, `kFoldSpaceTimeSimple`, `leaveOneOutSpaceTimeUniversal`, `kFoldSpaceTimeUniversal`, `leaveOneOutSpaceTimeBinomial`, `kFoldSpaceTimeBinomial`. Binomial variants return dual-scale residuals (logit + prevalence).
  - Space-time conditional simulation (Sequential Gaussian Simulation): one helper per ST variant.
    - Rust: `conditional_simulate_spacetime`, `_simple`, `_universal`, `_binomial`.
    - WASM / TypeScript: `conditionalSimulateSpaceTime`, `conditionalSimulateSpaceTimeSimple`, `conditionalSimulateSpaceTimeUniversal`, `conditionalSimulateSpaceTimeBinomial`. Deterministic for a given `seed`; optional `targetOrder` override; binomial variant returns `logitSamples` and `prevalenceSamples`.
- **Cross-validation for every kriging variant.** Previously ordinary-only; now each variant has its own leave-one-out and K-fold helper:
  - Rust: `leave_one_out_simple` / `k_fold_simple`, `leave_one_out_universal` / `k_fold_universal`, `leave_one_out_projected` / `k_fold_projected`, `leave_one_out_binomial` / `k_fold_binomial`. Internal fold iteration is now factored into shared helpers (`for_each_loo_fold`, `for_each_k_fold`) so adding new variants is trivial.
  - WASM / TypeScript: `leaveOneOutSimple`, `kFoldSimple`, `leaveOneOutUniversal`, `kFoldUniversal`, `leaveOneOutProjected`, `kFoldProjected`, `leaveOneOutBinomial`, `kFoldBinomial`.
  - Existing `leave_one_out` / `k_fold` (and JS `leaveOneOut` / `kFold`) are unchanged — no breaking change.
- **Binomial CV reports both scales.** New `BinomialCvResidual` / `BinomialCvSummary` (Rust) and `BinomialCvResidual` / `BinomialCvSummary` / `BinomialCvResult` (TS) carry per-station observed/predicted values and variances on **both** the logit scale (directly comparable to continuous kriging; calibratable via MSDR) and the prevalence scale (delta-method variance). Stations with `trials == 0` retain their index with `NaN` observed fields; the summary aggregates on each scale skip them automatically.
- **Conditional simulation for every kriging variant.** Previously ordinary-only; now each variant has its own sequential Gaussian simulation helper:
  - Rust: `conditional_simulate_simple`, `conditional_simulate_universal`, `conditional_simulate_projected`, `conditional_simulate_binomial`. Target-order validation is now factored into a shared `resolve_target_order` helper.
  - WASM / TypeScript: `conditionalSimulateSimple`, `conditionalSimulateUniversal`, `conditionalSimulateProjected`, `conditionalSimulateBinomial`.
  - Existing `conditional_simulate` (and JS `conditionalSimulate`) are unchanged — no breaking change.
- **Binomial simulation reports both scales.** New `BinomialSimulationResult` (Rust and TS) carries `logit_samples` / `logitSamples` (unbounded) and `prevalence_samples` / `prevalenceSamples` (in `(0, 1)`, by construction equal to `logistic(logit_samples)`). Simulation happens on the logit scale; stations with `trials == 0` are dropped from the initial conditioning pool. Accepts optional `priorAlpha` / `priorBeta` matching the binomial kriging model.
- **WASM / npm feature parity** – The `kriging-rs-wasm` wrapper now mirrors the full Rust feature surface:
  - `OrdinaryKriging.setNeighborhood({ maxNeighbors?, maxRadius? })` and `neighborhood()` for search-neighborhood configuration.
  - New `SimpleKriging` (known-mean) and `UniversalKriging` (with `"constant"`, `"linear"`, `"quadratic"` trend) classes.
  - New `ProjectedKriging` class for planar `(x, y)` kriging with 2D anisotropy (`majorAngleDeg`, `rangeRatio`).
  - `BinomialKriging.fromPrecomputedLogits(...)` factory that bypasses empirical-Bayes shrinkage.
  - Top-level functions `computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`, `leaveOneOut`, `kFold`, `conditionalSimulate`, and `evaluateNestedVariogram`.
  - `fitVariogram` and `computeEmpiricalVariogram` now accept `estimator: "classical" | "cressie-hawkins"`.
  - Extended `VariogramTypeName` to include `"power"` and `"holeeffect"`.
- **Rust** – `OrdinaryKrigingModel::set_neighborhood` (in-place variant of `with_neighborhood`) for FFI-friendly updates without consuming `self`.

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
  - Optional `nuggetOverride` on `OrdinaryKrigingFromFittedOptions`, `BinomialKrigingFromFittedVariogramOptions`, and `BinomialKrigingFromFittedVariogramWithPriorOptions` to override the fitted variogram nugget when building the model.
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

[0.2.3]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.3
[0.2.2]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.2
[0.2.1]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.1
[0.2.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.0
[0.1.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.1.0
