# kriging-rs

Geostatistical kriging library for Rust with WebAssembly support.

## Features

- Ordinary kriging for spatial interpolation
- Binomial kriging for prevalence surface estimation
- Variogram models: spherical, exponential, Gaussian, cubic, stable, Matérn (stable and Matérn accept an optional shape parameter)
- Geographic coordinate support with Haversine distances
- Optional WASM bindings for browser applications
- `Real` abstraction defaults to `f32` for compute paths
- Optional cross-platform GPU capability path via `wgpu`

## Repository layout

- **Root:** Rust crate (`Cargo.toml`, `src/`, `examples/`, `tests/`, `benches/`), shared docs, and CI.
- **`npm/kriging-rs-wasm/`:** TypeScript/WASM npm package; builds from this crate and is publish-ready.
- **`www/`:** Optional browser demo that depends on the npm package. See `www/README.md`.

## Rust usage

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

## WASM usage

```bash
wasm-pack build --target web -- --features wasm
```

## TypeScript package (publish-ready)

A TypeScript-first npm package facade is available at `npm/kriging-rs-wasm`.

It wraps generated `wasm-bindgen` exports with stable domain types and a stronger
TypeScript API surface (`VariogramType` unions, typed prediction outputs, and
typed-array batch result shapes).

From `npm/kriging-rs-wasm`:

```bash
npm install
npm run verify
```

`npm run verify` performs:

- TS contract typechecks (`tsc --noEmit`)
- WASM package generation (`wasm-pack`)
- TS facade compilation
- runtime smoke test (`import + init + fitVariogram + predictBatch`)

## Simple web app demo

See `www/README.md` for a minimal browser demo using the generated WASM package.

## Development

```bash
cargo test
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

## GPU feature flags

- `gpu`: enables async WebGPU support (`wgpu`) on native + web targets, including
  GPU-assisted RHS covariance assembly for batch prediction
- `gpu-blocking`: adds native blocking helper wrappers (via `pollster`)

Native probe example:

```bash
cargo run --example gpu_probe --features "gpu,gpu-blocking"
```

GPU-assisted batch prediction APIs:

- Rust async: `OrdinaryKrigingModel::predict_batch_gpu(...)`,
  `BinomialKrigingModel::predict_batch_gpu(...)`
- Rust native blocking: `predict_batch_gpu_blocking(...)` (with `gpu-blocking`)
- WASM async: `WasmOrdinaryKriging.predictBatchGpu(...)`,
  `WasmBinomialKriging.predictBatchGpu(...)` (with `wasm,gpu`)
- Explicit model-first path: fit (`compute_empirical_variogram` +
  `fit_variogram`) and then predict using `OrdinaryKrigingModel` or
  `BinomialKrigingModel`

## Performance

Run `cargo bench` for current numbers; see `bench-results/README.md` for logging and comparison.

## WASM batch and typed-array APIs

For large prediction batches and typed-array usage (`predictBatchArrays`, `fitVariogram`, `VariogramType`), see the [npm package README](npm/kriging-rs-wasm/README.md).

## License

Licensed under MIT.
