# kriging-rs Web Demo

A Vite + React browser demo that uses the **kriging-rs-wasm** npm package for ordinary and binomial kriging in the browser.

## Prerequisites

- Node.js 18+
- Build the WASM package first (from repo root or `npm/kriging-rs-wasm`)

## Build the package

From the repository root:

```bash
cd npm/kriging-rs-wasm
npm run build
```

For WebGPU support (optional):

```bash
npm run build:wasm:gpu
npm run build:ts
```

## Install and run the demo

From the repository root:

```bash
cd www
npm install
npm run dev
```

Then open **http://localhost:5173**.

## Build for production

```bash
cd www
npm run build
npm run preview
```

`npm run build` outputs to `www/dist/`. `npm run preview` serves that build locally.

## What the demo uses

The demo imports only the **kriging-rs-wasm** package (no raw WASM in `www`). It uses:

- `init()` and `webgpuAvailable()` for one-time setup
- `OrdinaryKriging` and `BinomialKriging` for models
- `fitVariogram` for variogram fitting
- `VariogramType` for the variogram enum

## Features

- **Quick demos** — Run a single ordinary or binomial prediction, or the full fitted pipeline (fit variogram then predict).
- **2D Surface** — Predict an ordinary or binomial kriging surface from synthetic or uploaded data; view heatmap, residuals, and empirical variogram. Optional WebGPU backend. Export the surface as PNG or the prediction grid as CSV.
- **Data upload** — Upload a CSV with columns `lat`, `lon`, and either `value` (ordinary) or `successes` and `trials` (binomial). The 2D Surface uses this data when provided.
- **Compare variogram models** — Run ordinary kriging on the same synthetic dataset with two different variogram models (e.g. Exponential vs Spherical) and view the surfaces side-by-side.
- **Performance harness** — Benchmark a 350-point ordinary-kriging workload and view timing breakdown (data prep, variogram fit, model build, predict, mapping).

## Controls (2D Surface)

- **Kriging mode** — Ordinary (value) or binomial (prevalence).
- **Grid resolution** — 24×24 (fast), 36×36 (default), or 48×48 (detailed).
- **Variogram model** — Spherical, Exponential, Gaussian, Cubic, Stable, Matérn.
- **Surface layer** — Prediction values or kriging variance (uncertainty).
- **Residual plot** — Scatter, histogram, or QQ-style.
- **Backend** — Auto (prefer WebGPU), CPU only, or WebGPU required.
