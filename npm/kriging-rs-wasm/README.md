# kriging-rs-wasm

TypeScript-first WebAssembly package for `kriging-rs`.

## Build

From this directory:

```bash
npm install
npm run build
```

The build performs:

- `wasm-pack` generation into `pkg/`
- TypeScript facade compilation into `dist/`

## Verify

```bash
npm run verify
```

This checks:

- Type contracts (`tsc --noEmit`)
- WASM + TypeScript build
- Runtime smoke test (`import + init + fitOrdinaryVariogram + predictBatch`)

## Usage

```ts
import init, { OrdinaryKriging, fitOrdinaryVariogram } from "kriging-rs-wasm";

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
const fitted = fitOrdinaryVariogram(
  [37.7, 37.71, 37.72],
  [-122.45, -122.44, -122.43],
  [10, 12, 11],
  undefined,
  12,
  ["gaussian", "exponential"]
);
const fittedModel = new OrdinaryKriging(
  [37.7, 37.71, 37.72],
  [-122.45, -122.44, -122.43],
  [10, 12, 11],
  fitted.variogramType,
  fitted.nugget,
  fitted.sill,
  fitted.range
);
const fittedPrediction = fittedModel.predict(37.705, -122.435);
```

## Notes

- Call and await `init(...)` once before invoking model constructors or variogram-fitting APIs.
- For GPU-enabled exports, build with `npm run build:wasm:gpu`.
