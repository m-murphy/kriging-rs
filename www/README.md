# Web demo

This folder contains a simple no-bundler browser demo for `kriging-rs`.

For external TypeScript consumers, use the publish-ready package facade in
`npm/kriging-rs-wasm` (typed wrapper API + package verification workflow).

## 1) Build WASM package into this folder

From repository root:

```bash
wasm-pack build --target web --out-dir pkg -- --features wasm
```

To include WebGPU capability checks in the WASM package:

```bash
wasm-pack build --target web --out-dir pkg -- --features "wasm,gpu"
```

## 2) Serve `www/` over HTTP

From repository root:

```bash
python3 -m http.server 8000 -d www
```

Then open `http://localhost:8000`.

## What it demonstrates

- `WasmOrdinaryKriging` constructor and `predict`
- `WasmOrdinaryKriging.predictBatch` for gridded surface prediction
- `WasmOrdinaryKriging.predictBatchArrays` for typed-array output
- `WasmOrdinaryKriging.predictBatchGpu` (async WebGPU path)
- `WasmBinomialKriging` constructor and `predict`
- `WasmBinomialKriging.predictBatch` for gridded prevalence prediction
- `WasmBinomialKriging.predictBatchArrays` for typed-array output
- `WasmBinomialKriging.predictBatchGpu` (async WebGPU path)
- `fitOrdinaryVariogram` for variogram fitting before model construction
- `webgpuAvailable` async helper when built with `gpu` feature
- Canvas-based 2D heatmap rendering of kriged values

Current demo wiring uses model-based prediction for both ordinary and binomial surface modes, with optional WebGPU acceleration selected from the UI backend control.

## 2D surface demo

Click **Run 2D surface** to:

1. Generate a fresh synthetic random sample dataset with 324 points on each run.
2. Build a rectangular prediction grid from sample bounds.
3. Call `predictBatch` using ordinary or binomial kriging.
4. Render predicted values/prevalence as a heatmap on the canvas.
5. Overlay sample locations colored by observed value/prevalence.
6. Render a residual plot (observed - predicted at sample locations).
7. Render an empirical variogram to inspect semivariance by distance.

Controls:

- **Kriging mode**: ordinary value surface or binomial prevalence surface.
- **Grid resolution**: lower is faster, higher is smoother.
- **Variogram model**: choose spherical/exponential/gaussian.
- **Surface layer**: prediction values or kriging variance (uncertainty).
- **Residual plot**: scatter, histogram, or QQ-style diagnostic.
- **Execution backend**:
  - `Auto` uses WebGPU when available (requires `--features "wasm,gpu"`).
  - `CPU` forces the existing CPU path.
  - `WebGPU required` fails fast if browser WebGPU is unavailable.
- **Run performance harness**: executes a 350-point ordinary-kriging
  workload multiple times and prints timing stats.

The legend below the plot shows the min/max predicted values for the current run.
Residual mean and RMSE are shown below the residual canvas.
The empirical variogram helps interpret certainty: lower semivariance at short
distance implies stronger local similarity.
An overlay curve for the selected variogram model is drawn on the empirical
variogram for visual comparison.

## Browser performance harness

Use **Run performance harness** to benchmark browser-side analysis performance with:

- 350 random sample points.
- A 36x36 prediction grid (1296 predictions).
- Warmup runs followed by measured runs.
- Timing breakdown for:
  - data preparation
  - variogram fitting
  - model build
  - batch prediction
  - JS mapping/aggregation

The output panel reports per-phase `mean/p50/p95` and phase share percentages,
plus total runtime summary stats for measured runs.
