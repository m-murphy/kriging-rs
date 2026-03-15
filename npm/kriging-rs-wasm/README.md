# kriging-rs-wasm

TypeScript-first WebAssembly package for [kriging-rs](https://github.com/m-murphy/kriging-rs): ordinary and binomial kriging with optional WebGPU acceleration.

Supported variogram types: `"spherical"`, `"exponential"`, `"gaussian"`, `"cubic"`, `"stable"`, `"matern"`. Pass the model type to `fitOrdinaryVariogram` as the **enum** (e.g. `VariogramType.Exponential`). For `stable` and `matern`, pass an optional `shape` parameter when constructing a model; `fitOrdinaryVariogram` returns a `shape` field when the fitted model is stable or Matérn.

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

Call and await `init()` once before using any other API.

### Ordinary kriging

```ts
import init, {
  OrdinaryKriging,
  fitOrdinaryVariogram,
  VariogramType,
} from "kriging-rs-wasm";

await init();

const model = new OrdinaryKriging(
  [37.7, 37.71, 37.72],
  [-122.45, -122.44, -122.43],
  [10, 12, 11],
  "gaussian",
  0.01,
  1.5,
  5.0
);

const prediction = model.predict(37.705, -122.435);

// Fit variogram from data (specify model type as enum), then build model
const fitted = fitOrdinaryVariogram(
  [37.7, 37.71, 37.72],
  [-122.45, -122.44, -122.43],
  [10, 12, 11],
  undefined,
  12,
  VariogramType.Exponential
);
const fittedModel = new OrdinaryKriging(
  [37.7, 37.71, 37.72],
  [-122.45, -122.44, -122.43],
  [10, 12, 11],
  fitted.variogramType,
  fitted.nugget,
  fitted.sill,
  fitted.range,
  fitted.shape  // optional; used for stable/matern
);
const batch = fittedModel.predictBatch(lats, lons);
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

const model = new BinomialKriging(
  lats,
  lons,
  successes,
  trials,
  "exponential",
  0.01,
  1.0,
  100
);
const pred = model.predict(37.705, -122.435);
// pred.prevalence in [0, 1], pred.variance, pred.logitValue
```

With a Beta prior (e.g. when counts are small):

```ts
const model = BinomialKriging.newWithPrior(
  lats,
  lons,
  successes,
  trials,
  "exponential",
  0.01,
  1.0,
  100,
  1, 1  // alpha, beta
);
```

### Batch prediction and typed arrays

For large prediction grids, use `predictBatchArrays` to get `Float64Array` outputs and avoid per-point object allocation:

```ts
const { values, variances } = model.predictBatchArrays(gridLats, gridLons);
// values.length === gridLats.length; same for variances
```

For ordinary kriging the result is `{ values, variances }`; for binomial it is `{ prevalences, logitValues, variances }`.

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

Constructors (`OrdinaryKriging`, `BinomialKriging`, `BinomialKriging.newWithPrior`) and `fitOrdinaryVariogram` throw on invalid inputs or model build failure (e.g. singular covariance). Errors are rethrown as `KrigingError` with the underlying cause attached as `cause`. Typical causes:

- Mismatched array lengths (lats, lons, values or successes/trials)
- Invalid coordinates or variogram parameters
- Using the API before calling and awaiting `init()`

```ts
import { KrigingError } from "kriging-rs-wasm";
try {
  const model = new OrdinaryKriging(lats, lons, values, "gaussian", 0.01, 1, 100);
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
