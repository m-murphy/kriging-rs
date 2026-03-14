# kriging-rs

Geostatistical kriging library for Rust with WebAssembly support.

## Features

- Ordinary kriging for spatial interpolation
- Binomial kriging for prevalence surface estimation
- Geographic coordinate support with Haversine distances
- Optional WASM bindings for browser applications
- `Real` abstraction defaults to `f32` for compute paths
- Optional cross-platform GPU capability path via `wgpu`

## Rust usage

```rust
use kriging_rs::{GeoCoord, OrdinaryKrigingModel, VariogramModel};

let coords = vec![
    GeoCoord { lat: 0.0, lon: 0.0 },
    GeoCoord { lat: 0.0, lon: 1.0 },
    GeoCoord { lat: 1.0, lon: 0.0 },
];
let values = vec![1.0, 2.0, 1.5];
let model = OrdinaryKrigingModel::new(
    coords,
    values,
    VariogramModel::Exponential {
        nugget: 0.01,
        sill: 2.0,
        range: 300.0,
    },
)?;
let prediction = model.predict(GeoCoord { lat: 0.3, lon: 0.3 })?;
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
- runtime smoke test (`import + init + fitOrdinaryVariogram + predictBatch`)

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

## Performance notes (1000-point workloads)

Recent benchmark and profiling work focused on `~1000` sample points and `~1000` prediction points.
The latest pass uses `f32` default compute plus native parallel batch prediction (WASM keeps sequential batch execution).

- `ordinary_predict_batch_1000_with_1000pts`: `~263.6ms -> ~12.7ms`
- `pipeline_ordinary_end_to_end_1000`: `~387.0ms -> ~83.6ms`
- `pipeline_binomial_end_to_end_1000`: `~389.4ms -> ~84.4ms`
- `pipeline_ordinary_model_build_1000`: `~109.3ms -> ~101.0ms` (about `7.5%` faster)
- `pipeline_binomial_model_build_1000`: `~105.2ms -> ~101.5ms` (about `3.5%` faster)

All regression and simulation tests remain green after optimization, and existing APIs remain backward compatible.

## WASM high-throughput output APIs

For large prediction batches, typed-array APIs avoid per-record JS object creation:

- `WasmOrdinaryKriging.predictBatchArrays(...)` returns `{ values: Float64Array, variances: Float64Array }`
- `WasmBinomialKriging.predictBatchArrays(...)` returns `{ prevalences: Float64Array, logitValues: Float64Array, variances: Float64Array }`
- Use `fitOrdinaryVariogram(...)` + model constructors and call `predictBatchArrays(...)` for typed-array throughput

## License

Licensed under either Apache-2.0 or MIT at your option.
