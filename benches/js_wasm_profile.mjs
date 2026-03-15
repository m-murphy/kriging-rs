import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const pkgPath = path.join(__dirname, "..", "target", "wasm-node-pkg", "kriging_rs.js");

const wasm = await import(pkgPath);

const {
  WasmOrdinaryKriging,
  WasmVariogramType,
  fitOrdinaryVariogram,
} = wasm;

function mulberry32(seed) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function summarize(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const p50 = sorted[Math.floor((n - 1) * 0.5)];
  const p95 = sorted[Math.floor((n - 1) * 0.95)];
  return {
    meanMs: sum / n,
    p50Ms: p50,
    p95Ms: p95,
    minMs: sorted[0],
    maxMs: sorted[n - 1],
  };
}

function buildSurfaceSamples(sampleCount = 350, seed = 12345) {
  const rng = mulberry32(seed);
  const lats = new Float64Array(sampleCount);
  const lons = new Float64Array(sampleCount);
  const values = new Float64Array(sampleCount);

  const latMin = 37.7;
  const latMax = 37.82;
  const lonMin = -122.48;
  const lonMax = -122.35;

  for (let i = 0; i < sampleCount; i += 1) {
    const lat = latMin + rng() * (latMax - latMin);
    const lon = lonMin + rng() * (lonMax - lonMin);
    const smoothField = Math.sin((lat - latMin) * 65) * 0.8 + Math.cos((lon - lonMin) * 72) * 0.75;
    const trend = ((lat - latMin) / (latMax - latMin) - 0.5) * 0.8;
    const valueNoise = (rng() - 0.5) * 0.25;
    const value = 15.0 + smoothField * 2.3 + trend + valueNoise;
    lats[i] = lat;
    lons[i] = lon;
    values[i] = value;
  }

  return { lats, lons, values };
}

function buildPredictionGrid(lats, lons, resolution) {
  const latMinBase = Math.min(...lats);
  const latMaxBase = Math.max(...lats);
  const lonMinBase = Math.min(...lons);
  const lonMaxBase = Math.max(...lons);
  const latSpan = Math.max(0.01, latMaxBase - latMinBase);
  const lonSpan = Math.max(0.01, lonMaxBase - lonMinBase);
  const latPad = latSpan * 0.15;
  const lonPad = lonSpan * 0.15;
  const latMin = latMinBase - latPad;
  const latMax = latMaxBase + latPad;
  const lonMin = lonMinBase - lonPad;
  const lonMax = lonMaxBase + lonPad;

  const predLats = new Float64Array(resolution * resolution);
  const predLons = new Float64Array(resolution * resolution);
  const latStep = (latMax - latMin) / (resolution - 1);
  const lonStep = (lonMax - lonMin) / (resolution - 1);

  let index = 0;
  for (let row = 0; row < resolution; row += 1) {
    const lat = latMax - row * latStep;
    for (let col = 0; col < resolution; col += 1) {
      const lon = lonMin + col * lonStep;
      predLats[index] = lat;
      predLons[index] = lon;
      index += 1;
    }
  }
  return { predLats, predLons };
}

function profileWasm(rounds = 10, warmup = 3) {
  const sample = buildSurfaceSamples(350, 20260313);
  const grid = buildPredictionGrid(sample.lats, sample.lons, 36);
  const variogramType = WasmVariogramType.Exponential;

  const phases = {
    fitMs: [],
    modelBuildMs: [],
    predictBatchMs: [],
    jsMapFromObjectsMs: [],
    pipelineEndToEndMs: [],
  };
  let checksum = 0;

  for (let run = 0; run < warmup + rounds; run += 1) {
    const measured = run >= warmup;

    let t0 = performance.now();
    const fitted = fitOrdinaryVariogram(
      sample.lats,
      sample.lons,
      sample.values,
      undefined,
      12,
      variogramType,
    );
    let t1 = performance.now();
    const fitMs = t1 - t0;

    t0 = performance.now();
    const model = new WasmOrdinaryKriging(
      sample.lats,
      sample.lons,
      sample.values,
      fitted.variogramType ?? fitted.variogram_type,
      fitted.nugget,
      fitted.sill,
      fitted.range,
    );
    t1 = performance.now();
    const modelBuildMs = t1 - t0;

    t0 = performance.now();
    const batch = model.predictBatch(grid.predLats, grid.predLons);
    t1 = performance.now();
    const predictBatchMs = t1 - t0;

    t0 = performance.now();
    checksum += batch.reduce((acc, p) => acc + p.value + p.variance, 0);
    t1 = performance.now();
    const jsMapFromObjectsMs = t1 - t0;
    model.free();

    t0 = performance.now();
    const pipelineFit = fitOrdinaryVariogram(
      sample.lats,
      sample.lons,
      sample.values,
      undefined,
      12,
      variogramType,
    );
    const pipelineModel = new WasmOrdinaryKriging(
      sample.lats,
      sample.lons,
      sample.values,
      pipelineFit.variogramType ?? pipelineFit.variogram_type,
      pipelineFit.nugget,
      pipelineFit.sill,
      pipelineFit.range,
    );
    const pipelineOut = pipelineModel.predictBatch(grid.predLats, grid.predLons);
    pipelineModel.free();
    t1 = performance.now();
    const pipelineEndToEndMs = t1 - t0;
    checksum += pipelineOut[0].value + pipelineOut[0].variance;

    if (measured) {
      phases.fitMs.push(fitMs);
      phases.modelBuildMs.push(modelBuildMs);
      phases.predictBatchMs.push(predictBatchMs);
      phases.jsMapFromObjectsMs.push(jsMapFromObjectsMs);
      phases.pipelineEndToEndMs.push(pipelineEndToEndMs);
    }
  }

  return {
    runtime: {
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
    },
    samplePoints: sample.lats.length,
    predictionPoints: grid.predLats.length,
    rounds,
    warmup,
    phases: Object.fromEntries(
      Object.entries(phases).map(([name, values]) => [name, summarize(values)]),
    ),
    checksum,
  };
}

const report = profileWasm();
console.log(JSON.stringify(report, null, 2));
