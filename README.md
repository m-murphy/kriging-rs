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

- Ordinary kriging for spatial interpolation
- Binomial kriging for prevalence surface estimation
- Variogram models: spherical, exponential, Gaussian, cubic, stable, Matérn (stable and Matérn accept an optional shape parameter)
- Geographic coordinate support with Haversine distances
- Optional WASM bindings for browser applications
- `Real` abstraction defaults to `f32` for compute paths
- Optional cross-platform GPU capability path via `wgpu`

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

Run `cargo bench` for current numbers; see [bench-results/README.md](bench-results/README.md) for logging and comparison.

## License

Licensed under MIT.
