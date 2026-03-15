import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { beforeAll, describe, test, expect } from "vitest";
import {
  init,
  KrigingError,
  OrdinaryKriging,
  BinomialKriging,
  fitVariogram,
  VariogramType,
  type OrdinaryPrediction,
  type BinomialPrediction,
  type VariogramTypeName,
} from "../dist/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const pkgDir = resolve(__dirname, "..");
const wasmPath = resolve(pkgDir, "pkg/kriging_rs_bg.wasm");

async function loadWasm(): Promise<Buffer> {
  return readFile(wasmPath);
}

beforeAll(async () => {
  const wasmBytes = await loadWasm();
  await init(wasmBytes);
});

describe("Ordinary kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const values = [10, 12, 11, 13];
  const variogramType: VariogramTypeName = "gaussian";
  const nugget = 0.01;
  const sill = 1.5;
  const range = 5.0;

  test("fitVariogram returns valid FittedVariogram (enum)", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Exponential,
      nBins: 12,
    });
    expect(fit).toMatchObject({
      variogramType: "exponential",
      nugget: expect.any(Number),
      sill: expect.any(Number),
      range: expect.any(Number),
      residuals: expect.any(Number),
    });
    expect(fit.nugget).toBeGreaterThanOrEqual(0);
    expect(fit.sill).toBeGreaterThanOrEqual(0);
    expect(fit.range).toBeGreaterThanOrEqual(0);
  });

  test("fitVariogram accepts string variogramType", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: "gaussian",
    });
    expect(fit.variogramType).toBe("gaussian");
    expect(fit.nugget).toBeGreaterThanOrEqual(0);
    expect(fit.sill).toBeGreaterThanOrEqual(0);
    expect(fit.range).toBeGreaterThanOrEqual(0);
  });

  test("OrdinaryKriging predict returns value and variance", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const pred = model.predict(0.5, 0.5);
    expect(pred).toMatchObject({
      value: expect.any(Number),
      variance: expect.any(Number),
    });
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("OrdinaryKriging predictBatch returns array of predictions", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const batchLats = [0.25, 0.5, 0.75];
    const batchLons = [0.25, 0.5, 0.75];
    const out = model.predictBatch(batchLats, batchLons);
    expect(Array.isArray(out)).toBe(true);
    expect(out.length).toBe(3);
    out.forEach((p: OrdinaryPrediction) => {
      expect(p).toMatchObject({
        value: expect.any(Number),
        variance: expect.any(Number),
      });
      expect(p.variance).toBeGreaterThanOrEqual(0);
    });
    model.free();
  });

  test("OrdinaryKriging predictBatchArrays returns Float64Arrays", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const batchLats = [0.25, 0.5];
    const batchLons = [0.25, 0.5];
    const out = model.predictBatchArrays(batchLats, batchLons);
    expect(out.values).toBeInstanceOf(Float64Array);
    expect(out.variances).toBeInstanceOf(Float64Array);
    expect(out.values.length).toBe(2);
    expect(out.variances.length).toBe(2);
    out.variances.forEach((v) => expect(v).toBeGreaterThanOrEqual(0));
    model.free();
  });

  test("model from fitVariogram produces consistent predictions", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Gaussian,
      nBins: 12,
    });
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: {
        variogramType: fit.variogramType,
        nugget: fit.nugget,
        sill: fit.sill,
        range: fit.range,
        shape: fit.shape,
      },
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.value)).toBe(true);
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("OrdinaryKriging.fromFitted produces same predictions as constructor with fitted variogram", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Gaussian,
      nBins: 12,
    });
    const model = OrdinaryKriging.fromFitted({
      lats,
      lons,
      values,
      fittedVariogram: fit,
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.value)).toBe(true);
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });
});

describe("Binomial kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const successes = [2, 4, 3, 5];
  const trials = [10, 10, 10, 10];
  const variogramType: VariogramTypeName = "exponential";
  const nugget = 0.01;
  const sill = 1.0;
  const range = 100;

  test("BinomialKriging predict returns prevalence, logitValue, variance", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const pred = model.predict(0.5, 0.5);
    expect(pred).toMatchObject({
      prevalence: expect.any(Number),
      logitValue: expect.any(Number),
      variance: expect.any(Number),
    });
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("BinomialKriging predictBatch returns array of predictions", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const out = model.predictBatch([0.25, 0.5], [0.25, 0.5]);
    expect(Array.isArray(out)).toBe(true);
    expect(out.length).toBe(2);
    out.forEach((p: BinomialPrediction) => {
      expect(p.prevalence).toBeGreaterThanOrEqual(0);
      expect(p.prevalence).toBeLessThanOrEqual(1);
      expect(p.variance).toBeGreaterThanOrEqual(0);
    });
    model.free();
  });

  test("BinomialKriging predictBatchArrays returns Float64Arrays", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const out = model.predictBatchArrays([0.25, 0.5], [0.25, 0.5]);
    expect(out.prevalences).toBeInstanceOf(Float64Array);
    expect(out.logitValues).toBeInstanceOf(Float64Array);
    expect(out.variances).toBeInstanceOf(Float64Array);
    expect(out.prevalences.length).toBe(2);
    model.free();
  });

  test("BinomialKriging.newWithPrior produces valid predictions", () => {
    const alpha = 1;
    const beta = 1;
    const model = BinomialKriging.newWithPrior({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
      prior: { alpha, beta },
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.prevalence)).toBe(true);
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    model.free();
  });

  test("BinomialKriging.fromFittedVariogram produces valid predictions", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values: lats.map((_, i) => successes[i] / trials[i]),
      variogramType: "exponential",
      nBins: 12,
    });
    const model = BinomialKriging.fromFittedVariogram({
      lats,
      lons,
      successes,
      trials,
      fittedVariogram: fit,
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.prevalence)).toBe(true);
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    model.free();
  });

  test("BinomialKriging.fromFittedVariogramWithPrior produces valid predictions", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values: lats.map((_, i) => successes[i] / trials[i]),
      variogramType: "exponential",
      nBins: 12,
    });
    const model = BinomialKriging.fromFittedVariogramWithPrior({
      lats,
      lons,
      successes,
      trials,
      fittedVariogram: fit,
      prior: { alpha: 1, beta: 1 },
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.prevalence)).toBe(true);
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    model.free();
  });
});

describe("Error handling", () => {
  test("fitVariogram with mismatched array lengths throws KrigingError", () => {
    expect(() =>
      fitVariogram({
        sampleLats: [0, 1],
        sampleLons: [0, 1, 2],
        values: [1, 2, 3],
        variogramType: VariogramType.Gaussian,
      })
    ).toThrow(KrigingError);
  });

  test("OrdinaryKriging with mismatched lats/lons throws KrigingError", () => {
    expect(
      () =>
        new OrdinaryKriging({
          lats: [0, 1],
          lons: [0, 1, 2],
          values: [1, 2, 3],
          variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1, range: 100 },
        })
    ).toThrow(KrigingError);
  });

  test("BinomialKriging with mismatched arrays throws KrigingError", () => {
    expect(
      () =>
        new BinomialKriging({
          lats: [0, 1],
          lons: [0, 1],
          successes: [1, 2, 3],
          trials: [10, 10, 10],
          variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1, range: 100 },
        })
    ).toThrow(KrigingError);
  });
});

describe("Uninitialized API", () => {
  test("using API before init throws (run in subprocess)", () => {
    const scriptPath = resolve(
      pkgDir,
      "scripts/assert-uninitialized-throws.mjs"
    );
    const result = spawnSync(process.execPath, [scriptPath], {
      cwd: pkgDir,
      encoding: "utf8",
    });
    expect(result.status).toBe(0);
  });
});
