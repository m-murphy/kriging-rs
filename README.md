# kriging-rs

Geostatistical kriging library with WASM support.

[Documentation](https://docs.rs/kriging-rs)

## Installation

```toml
[dependencies]
kriging-rs = "0.1"
```

Or `cargo add kriging-rs`.

## Usage

```rust
use kriging_rs::{GeoCoord, GeoDataset, OrdinaryKrigingModel, VariogramModel, VariogramType};

let coords = vec![
    GeoCoord::try_new(0.0, 0.0)?,
    GeoCoord::try_new(0.0, 1.0)?,
    GeoCoord::try_new(1.0, 0.0)?,
];
let values = vec![1.0, 2.0, 1.5];
let dataset = GeoDataset::new(coords, values)?;
let variogram = VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential)?;
let model = OrdinaryKrigingModel::new(dataset, variogram)?;
let prediction = model.predict(GeoCoord::try_new(0.3, 0.3)?)?;
println!("{:?}", prediction.value);
```

## Features

- Ordinary, simple, universal, and binomial kriging for 2-D spatial interpolation
- **Spatio-temporal kriging** (ordinary / simple / universal / binomial) over a 2-D spatial
  axis and a scalar time axis, with separable and product-sum space-time variograms
- Empirical and parametric 2-D and space-time variogram fitting
- Leave-one-out and K-fold cross-validation for every 2-D and space-time variant (continuous
  residuals plus dual-scale logit+prevalence for binomial)
- Sequential Gaussian simulation (conditional simulation) for every 2-D and space-time variant,
  deterministic for a given RNG seed
- Variogram models: spherical, exponential, Gaussian, cubic, stable, Matérn, power, hole-effect
  (stable and Matérn accept an optional shape parameter)
- Geographic coordinates with Haversine distances and planar `(x, y)` coordinates with 2-D
  anisotropy — both usable in the spatial and spatio-temporal paths
- Optional WASM bindings for browser applications
- `Real` abstraction defaults to `f32` for compute paths
- Optional cross-platform GPU capability path via `wgpu`
- **Binomial (prevalence) default:** empirical-Bayes `Beta(1,1)` (or user prior) → logit
  working values → **ordinary kriging with per-site logit observation variance** (calibrated
  binomial) on the covariance diagonal, then `logistic` to prevalence; *not* a full
  binomial-likelihood field model. Build diagnostics are always returned on the Rust side
  ([`BinomialBuildNotes`]) and exposed in WASM as `getBuildNotes()`. See
  [benches/BROWSER_BENCHMARKS.md](benches/BROWSER_BENCHMARKS.md) for a large
  browser-representative prediction workload

Build with `--features wasm` for browser; see below for GPU.

## Repository layout

Root is the Rust crate. `npm/kriging-rs-wasm/` is the TypeScript/WASM npm package. `www/` is a browser demo (see [www/README.md](www/README.md)).

## WASM and npm package

Build WASM:

```bash
wasm-pack build --target web -- --features wasm
```

The TypeScript/npm facade lives in `npm/kriging-rs-wasm`. See that package’s README for install, verify, and batch/typed-array APIs.

Browser demo: [www/README.md](www/README.md).

## Development

Install [pre-commit](https://pre-commit.com/) and run `pre-commit install` so fmt and clippy run before each commit and match CI.

```bash
cargo test
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

## GPU

Features: `gpu` (async WebGPU via `wgpu` on native + web, including GPU-assisted RHS covariance for batch prediction) and `gpu-blocking` (native blocking helpers via `pollster`).

```bash
cargo run --example gpu_probe --features "gpu,gpu-blocking"
```

GPU batch prediction APIs are on `OrdinaryKrigingModel` / `BinomialKrigingModel` (Rust) and the WASM types (with `wasm,gpu`). See examples and the npm README for details.

## Performance

Run `cargo bench` for current numbers; see [bench-results/README.md](bench-results/README.md) for logging and comparison. A **browser-oriented** (large grid, mixed trial counts) binomial *prediction* benchmark and workload description is in [benches/BROWSER_BENCHMARKS.md](benches/BROWSER_BENCHMARKS.md) (`bench_binomial_browser_representative`).

## License

Licensed under MIT.
