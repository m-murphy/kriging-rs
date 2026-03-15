# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.1.0
