import { readFile } from "node:fs/promises";
import * as kriging from "../dist/index.js";

const wasmBytes = await readFile(new URL("../pkg/kriging_rs_bg.wasm", import.meta.url));
await kriging.init(wasmBytes);

const sampleLats = [0.0, 0.0, 1.0];
const sampleLons = [0.0, 1.0, 0.0];
const sampleValues = [1.0, 2.0, 1.5];
const fit = kriging.fitOrdinaryVariogram(
  sampleLats,
  sampleLons,
  sampleValues,
  undefined,
  12,
  ["spherical", "exponential", "gaussian"]
);
const model = new kriging.OrdinaryKriging(
  sampleLats,
  sampleLons,
  sampleValues,
  fit.variogramType,
  fit.nugget,
  fit.sill,
  fit.range
);
const out = model.predictBatch([0.25, 0.5], [0.25, 0.5]);

if (!Array.isArray(out) || out.length !== 2) {
  throw new Error("Smoke test failed: predictBatch did not return prediction array");
}

for (const item of out) {
  if (typeof item.value !== "number" || typeof item.variance !== "number") {
    throw new Error("Smoke test failed: prediction item shape is invalid");
  }
}

console.log("Smoke test passed.");
