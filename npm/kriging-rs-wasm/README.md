# kriging-rs-wasm

TypeScript-first WebAssembly package for [kriging-rs](https://github.com/m-murphy/kriging-rs): ordinary and binomial kriging with optional WebGPU acceleration.

## Quick start

1. Install: `npm install kriging-rs-wasm`
2. Initialize once: `await init()`
3. Build a model and predict:

```ts
import init, { OrdinaryKriging } from "kriging-rs-wasm";
await init();

const model = new OrdinaryKriging({
  lats: [37.7, 37.71, 37.72],
  lons: [-122.45, -122.44, -122.43],
  values: [10, 12, 11],
  variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1.5, range: 5.0 },
});
const pred = model.predict(37.705, -122.435);
```

## Requirements

- Browser or Node.js; ES2022. For custom WASM loading you can pass pre-fetched bytes to `init(wasmArrayBuffer)`.

## API overview

| Export | Purpose |
|--------|---------|
| `init` | Initialize the WASM module (call and await once before any other API). |
| `OrdinaryKriging` | Spatial interpolation of **continuous** values (e.g. temperature, elevation). |
| `BinomialKriging` | **Prevalence/proportion** surfaces from count data (successes out of trials). |
| `fitVariogram` | Fit a variogram model to sample data; use the result to build an `OrdinaryKriging` model. |
| `interpolateOrdinaryToGrid` | One-shot: fit + build + predict on grid + free; returns value and variance grids. |
| `interpolateBinomialToGrid` | One-shot: fit + build + predict on grid + free; returns prevalence and variance grids. |
| `VariogramType` | Enum for variogram model type (optional; you can pass string names like `"exponential"` instead). |
| `KrigingError` | Error class thrown on invalid inputs or model build failure; `cause` holds the underlying error. |
| `webgpuAvailable` | Check if WebGPU-backed batch prediction is available (requires GPU build). |

**When to use which:** Use **ordinary kriging** when you have continuous measurements at locations (e.g. sensor values, elevations). Use **binomial kriging** when you have counts (successes and trials) and want to estimate a proportion or prevalence surface.

## Build

From this directory:

```bash
npm install
npm run build
```

The build performs:

- `wasm-pack` generation into `pkg/`
- TypeScript facade compilation into `dist/`

For WebGPU-backed batch prediction, use:

```bash
npm run build:wasm:gpu
npm run build:ts
```

## Verify

```bash
npm run verify
```

This checks:

- Type contracts (`tsc --noEmit`)
- WASM + TypeScript build
- Runtime smoke test
- Test suite (`npm test`)

## Usage

Call and await `init()` once before using any other API. **VariogramType** (e.g. `VariogramType.Exponential`) is only safe to use after `init()` has been awaited; accessing it before init will throw when the value is used. You can pass string names like `"exponential"` instead.

Supported variogram types: `"spherical"`, `"exponential"`, `"gaussian"`, `"cubic"`, `"stable"`, `"matern"`. You can pass the model type as a string (e.g. `"exponential"`) or as `VariogramType.Exponential`. For `stable` and `matern`, pass an optional `shape` when constructing a model; `fitVariogram` returns a `shape` field when the fitted model is stable or Matérn.

### Ordinary kriging

```ts
import init, {
  OrdinaryKriging,
  fitVariogram,
} from "kriging-rs-wasm";

await init();

const model = new OrdinaryKriging({
  lats: [37.7, 37.71, 37.72],
  lons: [-122.45, -122.44, -122.43],
  values: [10, 12, 11],
  variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1.5, range: 5.0 },
});

const prediction = model.predict(37.705, -122.435);

// Fit variogram from data (options object; variogramType can be string or VariogramType enum)
const lats = [37.7, 37.71, 37.72];
const lons = [-122.45, -122.44, -122.43];
const values = [10, 12, 11];
const fitted = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: "exponential",
  nBins: 12,  // optional; default 12
});
const fittedModel = new OrdinaryKriging({
  lats,
  lons,
  values,
  variogram: {
    variogramType: fitted.variogramType,
    nugget: fitted.nugget,
    sill: fitted.sill,
    range: fitted.range,
    shape: fitted.shape,  // optional; used for stable/matern
  },
});
const batch = fittedModel.predictBatch(lats, lons);
```

**Convenience factories (fit → model):** To avoid manually spreading `fitted` fields, use the static factories:

- **Ordinary:** `OrdinaryKriging.fromFitted({ lats, lons, values, fittedVariogram: fitted })`
- **Binomial:** `BinomialKriging.fromFittedVariogram({ lats, lons, successes, trials, fittedVariogram })`
- **Binomial with prior:** `BinomialKriging.fromFittedVariogramWithPrior({ lats, lons, successes, trials, fittedVariogram, prior: { alpha, beta } })`

Example:

```ts
const fitted = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: "exponential",
  nBins: 12,
});
const model = OrdinaryKriging.fromFitted({ lats, lons, values, fittedVariogram: fitted });
const pred = model.predict(37.705, -122.435);
```

### Binomial kriging (prevalence surfaces)

For count data (successes out of trials) at locations:

```ts
import init, { BinomialKriging } from "kriging-rs-wasm";

await init();

const lats = [37.7, 37.71, 37.72];
const lons = [-122.45, -122.44, -122.43];
const successes = [2, 5, 3];  // counts
const trials = [10, 10, 10];

const model = new BinomialKriging({
  lats,
  lons,
  successes,
  trials,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 100 },
});
const pred = model.predict(37.705, -122.435);
// pred.prevalence in [0, 1], pred.variance, pred.logitValue
```

With a Beta prior (e.g. when counts are small):

```ts
const model = BinomialKriging.newWithPrior({
  lats,
  lons,
  successes,
  trials,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 100 },
  prior: { alpha: 1, beta: 1 },
});
```

### Batch prediction and typed arrays

For large prediction grids, use `predictBatchArrays` to get `Float64Array` outputs and avoid per-point object allocation:

```ts
const { values, variances } = model.predictBatchArrays(gridLats, gridLons);
// values.length === gridLats.length; same for variances
```

For ordinary kriging the result is `{ values, variances }`; for binomial it is `{ prevalences, logitValues, variances }`.

### Grid prediction (bounds + resolution)

For a rectangular grid defined by bounds and cell counts, use `predictGrid` so the library handles building cell-center coordinates and reshaping results:

```ts
const { values, variances } = model.predictGrid({
  west: -122.5,
  south: 37.6,
  east: -122.3,
  north: 37.8,
  xCells: 50,
  yCells: 40,
});
```

**Grid layout:** Results are 2D arrays (not flat). Row index = latitude index, column index = longitude index. First row (`j = 0`) = south, last row = north; first column (`i = 0`) = west, last column = east. So `values[j][i]` is the prediction at the cell with latitude row `j` and longitude column `i`. Internally the library uses row-major order (south to north, then west to east within each row). Ordinary kriging returns `{ values, variances }`; binomial returns `{ prevalences, logitValues, variances }`, all with shape `[yCells][xCells]`.

### One-shot interpolate to grid

For the common flow "fit variogram → build model → predict on grid → free", you can use a single call:

```ts
// Ordinary: sample data + grid spec + variogram type
const { values, variances } = interpolateOrdinaryToGrid({
  lats, lons, values: sampleValues,
  west: -122.5, south: 37.6, east: -122.3, north: 37.8,
  xCells: 50, yCells: 40,
  variogramType: "exponential",
  nBins: 12,
  nuggetOverride: 0.01,  // optional
});

// Binomial: count data + grid spec + variogram type; optional prior
const { prevalences, variances } = interpolateBinomialToGrid({
  lats, lons, successes, trials,
  west: -122.5, south: 37.6, east: -122.3, north: 37.8,
  xCells: 50, yCells: 40,
  variogramType: "exponential",
  prior: { alpha: 1, beta: 1 },  // optional
});
```

The model is created internally and freed before returning, so you do not need to call `free()`.

### WebGPU (optional)

If the package is built with the GPU feature (`npm run build:wasm:gpu`), you can use async batch prediction and check availability:

```ts
import init, { OrdinaryKriging, webgpuAvailable } from "kriging-rs-wasm";
await init();

if (await webgpuAvailable()) {
  const predictions = await model.predictBatchGpu(gridLats, gridLons);
}
```

If `predictBatchGpu` is called without a GPU build, it throws.

## Error handling

Constructors (`OrdinaryKriging`, `BinomialKriging`, `BinomialKriging.newWithPrior`) and `fitVariogram` throw on invalid inputs or model build failure (e.g. singular covariance). Errors are rethrown as `KrigingError` with the underlying cause attached as `cause`. Typical causes:

- Mismatched array lengths (lats, lons, values or successes/trials)
- Invalid coordinates or variogram parameters
- Using the API before calling and awaiting `init()`

```ts
import { KrigingError } from "kriging-rs-wasm";
try {
  const model = new OrdinaryKriging({
    lats,
    lons,
    values,
    variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1, range: 100 },
  });
} catch (e) {
  if (e instanceof KrigingError) {
    console.error(e.message, e.cause);
  }
  throw e;
}
```

## Publishing

From `npm/kriging-rs-wasm`, run `npm run verify`, then `npm publish` (dry-run: `npm publish --dry-run`). The package publishes the `dist/`, `pkg/`, and `README.md` listed in `files`. Follow semver for releases.

## Notes

- Call and await `init(...)` once before invoking model constructors or variogram-fitting APIs. You can pass pre-fetched WASM bytes: `await init(wasmArrayBuffer)`.
- Coordinates are in degrees (latitude, longitude); distances use Haversine (great-circle).
- For GPU-enabled exports, build with `npm run build:wasm:gpu`.
- **Resource management:** When a model is no longer needed, call `model.free()` to release WASM-held memory. This is optional but recommended in long-lived applications.
