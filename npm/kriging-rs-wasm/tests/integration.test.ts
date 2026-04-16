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
  SimpleKriging,
  UniversalKriging,
  ProjectedKriging,
  SpaceTimeOrdinaryKriging,
  SpaceTimeSimpleKriging,
  SpaceTimeUniversalKriging,
  SpaceTimeBinomialKriging,
  SpaceTimeProjectedOrdinaryKriging,
  computeEmpiricalSpaceTimeVariogram,
  fitSpaceTimeVariogram,
  fitVariogram,
  interpolateOrdinaryToGrid,
  interpolateBinomialToGrid,
  computeEmpiricalVariogram,
  computeDirectionalEmpiricalVariogram,
  leaveOneOut,
  kFold,
  leaveOneOutSimple,
  kFoldSimple,
  leaveOneOutUniversal,
  kFoldUniversal,
  leaveOneOutProjected,
  kFoldProjected,
  leaveOneOutBinomial,
  kFoldBinomial,
  conditionalSimulate,
  conditionalSimulateSimple,
  conditionalSimulateUniversal,
  conditionalSimulateProjected,
  conditionalSimulateBinomial,
  leaveOneOutSpaceTime,
  kFoldSpaceTime,
  leaveOneOutSpaceTimeSimple,
  kFoldSpaceTimeSimple,
  leaveOneOutSpaceTimeUniversal,
  kFoldSpaceTimeUniversal,
  leaveOneOutSpaceTimeBinomial,
  kFoldSpaceTimeBinomial,
  conditionalSimulateSpaceTime,
  conditionalSimulateSpaceTimeSimple,
  conditionalSimulateSpaceTimeUniversal,
  conditionalSimulateSpaceTimeBinomial,
  evaluateNestedVariogram,
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

  test("OrdinaryKriging.fromFitted with nuggetOverride uses override", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Gaussian,
      nBins: 12,
    });
    const modelNoOverride = OrdinaryKriging.fromFitted({
      lats,
      lons,
      values,
      fittedVariogram: fit,
    });
    const predNoOverride = modelNoOverride.predict(0.5, 0.5);
    modelNoOverride.free();
    const modelOverride = OrdinaryKriging.fromFitted({
      lats,
      lons,
      values,
      fittedVariogram: fit,
      nuggetOverride: 0,
    });
    const predOverride = modelOverride.predict(0.5, 0.5);
    modelOverride.free();
    expect(Number.isFinite(predOverride.value)).toBe(true);
    expect(Number.isFinite(predOverride.variance)).toBe(true);
    expect(predOverride.variance).not.toBe(predNoOverride.variance);
  });

  test("OrdinaryKriging free() is idempotent", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    model.free();
    model.free();
    expect(() => model.predict(0.5, 0.5)).toThrow(KrigingError);
  });

  test("KrigingError has code model_freed after free()", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    model.free();
    try {
      model.predict(0.5, 0.5);
    } catch (e) {
      expect(e).toBeInstanceOf(KrigingError);
      expect((e as KrigingError).code).toBe("model_freed");
      return;
    }
    expect.fail("should have thrown");
  });

  test("OrdinaryKriging predictGrid returns 2D arrays matching predictBatchArrays", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const gridOpts = {
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 3,
    };
    const out = model.predictGrid(gridOpts);
    expect(out.values.length).toBe(3);
    expect(out.values[0].length).toBe(2);
    expect(out.variances.length).toBe(3);
    expect(out.variances[0].length).toBe(2);
    const { lats: flatLats, lons: flatLons } = (() => {
      const nRows = 3;
      const nCols = 2;
      const latsArr = new Float64Array(nRows * nCols);
      const lonsArr = new Float64Array(nRows * nCols);
      let k = 0;
      for (let j = 0; j < nRows; j++) {
        const lat = 0 + (j + 0.5) * (1 / nRows);
        for (let i = 0; i < nCols; i++) {
          latsArr[k] = lat;
          lonsArr[k] = 0 + (i + 0.5) * (1 / nCols);
          k++;
        }
      }
      return { lats: latsArr, lons: lonsArr };
    })();
    const batch = model.predictBatchArrays(flatLats, flatLons);
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 2; i++) {
        expect(out.values[j][i]).toBe(batch.values[j * 2 + i]);
        expect(out.variances[j][i]).toBe(batch.variances[j * 2 + i]);
      }
    }
    model.free();
  });

  test("interpolateOrdinaryToGrid returns grid and frees model", () => {
    const out = interpolateOrdinaryToGrid({
      lats,
      lons,
      values,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
    });
    expect(out.values.length).toBe(2);
    expect(out.values[0].length).toBe(2);
    expect(out.variances.length).toBe(2);
    expect(Number.isFinite(out.values[0][0])).toBe(true);
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

  test("BinomialKriging free() is idempotent", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    model.free();
    model.free();
    expect(() => model.predict(0.5, 0.5)).toThrow(KrigingError);
  });

  test("BinomialKriging predictGrid returns 2D arrays", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const out = model.predictGrid({
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
    });
    expect(out.prevalences.length).toBe(2);
    expect(out.prevalences[0].length).toBe(2);
    expect(out.variances.length).toBe(2);
    model.free();
  });

  test("interpolateBinomialToGrid returns grid and frees model", () => {
    const out = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
    });
    expect(out.prevalences.length).toBe(2);
    expect(out.prevalences[0].length).toBe(2);
    expect(out.variances.length).toBe(2);
    expect(Number.isFinite(out.prevalences[0][0])).toBe(true);
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
  test("fitVariogram with mismatched array lengths throws KrigingError with code", () => {
    try {
      fitVariogram({
        sampleLats: [0, 1],
        sampleLons: [0, 1, 2],
        values: [1, 2, 3],
        variogramType: VariogramType.Gaussian,
      });
    } catch (e) {
      expect(e).toBeInstanceOf(KrigingError);
      expect((e as KrigingError).code).toBe("mismatched_arrays");
      return;
    }
    expect.fail("should have thrown");
  });

  test("OrdinaryKriging with mismatched lats/lons throws KrigingError", () => {
    expect(
      () =>
        new OrdinaryKriging({
          lats: [0, 1],
          lons: [0, 1, 2],
          values: [1, 2, 3],
          variogram: {
            variogramType: "gaussian",
            nugget: 0.01,
            sill: 1,
            range: 100,
          },
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
          variogram: {
            variogramType: "gaussian",
            nugget: 0.01,
            sill: 1,
            range: 100,
          },
        })
    ).toThrow(KrigingError);
  });
});

describe("Search neighborhood (ordinary kriging)", () => {
  const lats = [0, 0, 1, 1, 2, 2];
  const lons = [0, 1, 0, 1, 0, 1];
  const values = [10, 12, 11, 13, 14, 15];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.01,
    sill: 1.0,
    range: 500.0,
  };

  test("setNeighborhood limits to k nearest and is reversible", () => {
    const model = new OrdinaryKriging({ lats, lons, values, variogram });
    try {
      expect(model.neighborhood()).toBeNull();
      model.setNeighborhood({ maxNeighbors: 3 });
      expect(model.neighborhood()).toEqual({ maxNeighbors: 3 });
      const pred = model.predict(0.5, 0.5);
      expect(Number.isFinite(pred.value)).toBe(true);
      model.setNeighborhood();
      expect(model.neighborhood()).toBeNull();
    } finally {
      model.free();
    }
  });

  test("setNeighborhood with invalid radius throws", () => {
    const model = new OrdinaryKriging({ lats, lons, values, variogram });
    try {
      expect(() => model.setNeighborhood({ maxRadius: -1 })).toThrow(
        KrigingError
      );
    } finally {
      model.free();
    }
  });
});

describe("Simple kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const values = [10, 12, 11, 13];

  test("SimpleKriging predicts with known mean", () => {
    const model = new SimpleKriging({
      lats,
      lons,
      values,
      mean: 11.5,
      variogram: {
        variogramType: "exponential",
        nugget: 0.01,
        sill: 1,
        range: 500,
      },
    });
    try {
      expect(model.mean).toBeCloseTo(11.5, 6);
      const pred = model.predict(0.5, 0.5);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
      const batch = model.predictBatchArrays([0.5, 0.25], [0.5, 0.25]);
      expect(batch.values.length).toBe(2);
      expect(batch.variances.length).toBe(2);
    } finally {
      model.free();
    }
  });
});

describe("Universal kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const values = [10, 12, 11, 13];

  test("UniversalKriging with linear trend predicts finite values", () => {
    const model = new UniversalKriging({
      lats,
      lons,
      values,
      trend: "linear",
      variogram: {
        variogramType: "exponential",
        nugget: 0.01,
        sill: 1,
        range: 500,
      },
    });
    try {
      const pred = model.predict(0.5, 0.5);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("UniversalKriging rejects unknown trend", () => {
    expect(
      () =>
        new UniversalKriging({
          lats,
          lons,
          values,
          trend: "cubic" as unknown as "linear",
          variogram: {
            variogramType: "exponential",
            nugget: 0.01,
            sill: 1,
            range: 500,
          },
        })
    ).toThrow(KrigingError);
  });
});

describe("Projected kriging", () => {
  test("ProjectedKriging with 2D anisotropy predicts finite values", () => {
    const xs = [0, 1000, 0, 1000];
    const ys = [0, 0, 1000, 1000];
    const values = [10, 11, 12, 13];
    const model = new ProjectedKriging({
      xs,
      ys,
      values,
      variogram: {
        variogramType: "spherical",
        nugget: 0.1,
        sill: 1.0,
        range: 5000,
      },
      majorAngleDeg: 0,
      rangeRatio: 0.5,
    });
    try {
      const pred = model.predict(500, 500);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("ProjectedKriging rejects range ratio outside (0, 1]", () => {
    expect(
      () =>
        new ProjectedKriging({
          xs: [0, 1, 0, 1],
          ys: [0, 0, 1, 1],
          values: [10, 11, 12, 13],
          variogram: {
            variogramType: "spherical",
            nugget: 0.1,
            sill: 1,
            range: 5,
          },
          majorAngleDeg: 0,
          rangeRatio: 2.0,
        })
    ).toThrow(KrigingError);
  });
});

describe("Empirical variogram (direct)", () => {
  const lats = [0, 0, 1, 1, 2, 2];
  const lons = [0, 1, 0, 1, 0, 1];
  const values = [10, 12, 11, 13, 14, 15];

  test("computeEmpiricalVariogram returns non-empty bins", () => {
    const out = computeEmpiricalVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      nBins: 5,
    });
    expect(out.distances.length).toBe(out.semivariances.length);
    expect(out.counts.length).toBe(out.distances.length);
    expect(out.distances.length).toBeGreaterThan(0);
  });

  test("Cressie-Hawkins estimator produces non-negative semivariances", () => {
    const out = computeEmpiricalVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      nBins: 5,
      estimator: "cressie-hawkins",
    });
    for (const s of out.semivariances) {
      if (Number.isFinite(s)) expect(s).toBeGreaterThanOrEqual(0);
    }
  });

  test("computeDirectionalEmpiricalVariogram respects tolerance", () => {
    const xs = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    const ys = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    const values = xs.map((_, i) => i * 0.5 + ys[i] * 2);
    const out = computeDirectionalEmpiricalVariogram({
      xs,
      ys,
      values,
      maxDistance: 5,
      nBins: 4,
      azimuthDeg: 0,
      toleranceDeg: 15,
    });
    expect(out.distances.length).toBe(4);
  });
});

describe("Cross-validation", () => {
  const lats = [0, 0, 1, 1, 2, 2];
  const lons = [0, 1, 0, 1, 0, 1];
  const values = [10, 12, 11, 13, 14, 15];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("leaveOneOut returns one residual per station", () => {
    const out = leaveOneOut({ lats, lons, values, variogram });
    expect(out.residuals.length).toBe(values.length);
    expect(out.summary.n).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
      expect(r.error).toBeCloseTo(r.observed - r.predicted, 6);
    }
    expect(Number.isFinite(out.summary.rmse)).toBe(true);
  });

  test("kFold returns residuals for all stations", () => {
    const out = kFold({ lats, lons, values, variogram, k: 3 });
    expect(out.residuals.length).toBe(values.length);
    expect(out.summary.n).toBe(values.length);
  });

  test("kFold rejects k < 2", () => {
    expect(() => kFold({ lats, lons, values, variogram, k: 1 })).toThrow(
      KrigingError
    );
  });
});

describe("Cross-validation (per-variant)", () => {
  const lats = [0, 0, 1, 1, 2, 2, 3, 3];
  const lons = [0, 1, 0, 1, 0, 1, 0, 1];
  const values = lats.map((lat, i) => 2 * lat + 3 * lons[i] + 1);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("leaveOneOutSimple returns one residual per station with known mean", () => {
    const out = leaveOneOutSimple({ lats, lons, values, variogram, mean });
    expect(out.residuals.length).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
      expect(Number.isFinite(r.variance)).toBe(true);
    }
  });

  test("kFoldSimple covers every station exactly once", () => {
    const out = kFoldSimple({ lats, lons, values, variogram, mean, k: 4 });
    expect(out.residuals.length).toBe(values.length);
    const seen = new Set<number>();
    for (const r of out.residuals) {
      expect(seen.has(r.index)).toBe(false);
      seen.add(r.index);
    }
    expect(seen.size).toBe(values.length);
  });

  test("leaveOneOutUniversal with constant trend matches leaveOneOut (ordinary)", () => {
    const ok = leaveOneOut({ lats, lons, values, variogram });
    const uk = leaveOneOutUniversal({
      lats,
      lons,
      values,
      variogram,
      trend: "constant",
    });
    expect(uk.residuals.length).toBe(ok.residuals.length);
    for (let i = 0; i < ok.residuals.length; i++) {
      expect(uk.residuals[i].predicted).toBeCloseTo(
        ok.residuals[i].predicted,
        6
      );
    }
  });

  test("kFoldUniversal with linear trend returns finite residuals", () => {
    const out = kFoldUniversal({
      lats,
      lons,
      values,
      variogram,
      trend: "linear",
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
    }
  });

  test("leaveOneOutUniversal rejects unknown trend", () => {
    expect(() =>
      leaveOneOutUniversal({
        lats,
        lons,
        values,
        variogram,
        // @ts-expect-error — deliberately invalid
        trend: "cubic",
      })
    ).toThrow(KrigingError);
  });

  test("leaveOneOutProjected (isotropic) returns residuals in input order", () => {
    // Planar grid; reuse lats/lons as xs/ys for structure.
    const xs = lats;
    const ys = lons;
    const projectedVariogram = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 5.0,
    };
    const out = leaveOneOutProjected({
      xs,
      ys,
      values,
      variogram: projectedVariogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
    });
    expect(out.residuals.length).toBe(values.length);
    for (let i = 0; i < out.residuals.length; i++) {
      expect(out.residuals[i].index).toBe(i);
      expect(Number.isFinite(out.residuals[i].predicted)).toBe(true);
    }
  });

  test("kFoldProjected rejects invalid rangeRatio", () => {
    const xs = lats;
    const ys = lons;
    const projectedVariogram = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 5.0,
    };
    expect(() =>
      kFoldProjected({
        xs,
        ys,
        values,
        variogram: projectedVariogram,
        majorAngleDeg: 0,
        rangeRatio: 2,
        k: 4,
      })
    ).toThrow(KrigingError);
  });
});

describe("Cross-validation (binomial, both scales)", () => {
  // Smooth logit gradient; prevalences ~ 0.1..0.9 across a 4x4 grid.
  const coords: { lat: number; lon: number }[] = [];
  const successes: number[] = [];
  const trials: number[] = [];
  const logistic = (x: number) => 1 / (1 + Math.exp(-x));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      coords.push({ lat: i, lon: j });
      const p = logistic(-2 + 0.5 * i + 0.5 * j);
      const n = 40;
      successes.push(Math.round(p * n));
      trials.push(n);
    }
  }
  const lats = coords.map((c) => c.lat);
  const lons = coords.map((c) => c.lon);
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 2.0,
    range: 5.0,
  };

  test("leaveOneOutBinomial reports both logit and prevalence scales in input order", () => {
    const out = leaveOneOutBinomial({
      lats,
      lons,
      successes,
      trials,
      variogram,
    });
    expect(out.residuals.length).toBe(lats.length);
    for (let i = 0; i < out.residuals.length; i++) {
      const r = out.residuals[i];
      expect(r.index).toBe(i);
      expect(r.trials).toBe(trials[i]);
      expect(r.successes).toBe(successes[i]);
      expect(Number.isFinite(r.observedLogit)).toBe(true);
      expect(Number.isFinite(r.predictedLogit)).toBe(true);
      expect(Number.isFinite(r.logitVariance)).toBe(true);
      expect(Number.isFinite(r.observedPrevalence)).toBe(true);
      expect(Number.isFinite(r.predictedPrevalence)).toBe(true);
      expect(Number.isFinite(r.prevalenceVariance)).toBe(true);
      expect(r.predictedPrevalence).toBeGreaterThanOrEqual(0);
      expect(r.predictedPrevalence).toBeLessThanOrEqual(1);
      expect(r.logitError).toBeCloseTo(r.observedLogit - r.predictedLogit, 9);
      expect(r.prevalenceError).toBeCloseTo(
        r.observedPrevalence - r.predictedPrevalence,
        9
      );
    }
    expect(out.summary.n).toBe(lats.length);
    expect(out.summary.nEvaluated).toBe(lats.length);
    expect(Number.isFinite(out.summary.logit.rmse)).toBe(true);
    expect(Number.isFinite(out.summary.prevalence.rmse)).toBe(true);
  });

  test("leaveOneOutBinomial handles trials==0 with NaN observations and excluded summary", () => {
    const successesZ = successes.slice();
    const trialsZ = trials.slice();
    successesZ[0] = 0;
    trialsZ[0] = 0;
    const out = leaveOneOutBinomial({
      lats,
      lons,
      successes: successesZ,
      trials: trialsZ,
      variogram,
    });
    const r0 = out.residuals[0];
    expect(r0.trials).toBe(0);
    expect(Number.isNaN(r0.observedLogit)).toBe(true);
    expect(Number.isNaN(r0.observedPrevalence)).toBe(true);
    expect(Number.isNaN(r0.logitError)).toBe(true);
    expect(Number.isNaN(r0.prevalenceError)).toBe(true);
    // Prediction still populated.
    expect(Number.isFinite(r0.predictedLogit)).toBe(true);
    expect(Number.isFinite(r0.predictedPrevalence)).toBe(true);
    // Summary excludes the zero-trials station.
    expect(out.summary.n).toBe(lats.length);
    expect(out.summary.nEvaluated).toBe(lats.length - 1);
    expect(out.summary.logit.n).toBe(lats.length - 1);
    expect(out.summary.prevalence.n).toBe(lats.length - 1);
  });

  test("kFoldBinomial covers every station exactly once", () => {
    const out = kFoldBinomial({
      lats,
      lons,
      successes,
      trials,
      variogram,
      k: 4,
    });
    expect(out.residuals.length).toBe(lats.length);
    const seen = new Set<number>();
    for (const r of out.residuals) {
      expect(seen.has(r.index)).toBe(false);
      seen.add(r.index);
    }
    expect(seen.size).toBe(lats.length);
  });

  test("leaveOneOutBinomial accepts custom Beta(alpha, beta) prior", () => {
    const out = leaveOneOutBinomial({
      lats,
      lons,
      successes,
      trials,
      variogram,
      priorAlpha: 1,
      priorBeta: 1,
    });
    expect(out.residuals.length).toBe(lats.length);
    expect(Number.isFinite(out.summary.prevalence.rmse)).toBe(true);
  });

  test("leaveOneOutBinomial rejects partial prior specification", () => {
    expect(() =>
      leaveOneOutBinomial({
        lats,
        lons,
        successes,
        trials,
        variogram,
        priorAlpha: 1,
      })
    ).toThrow(KrigingError);
  });
});

describe("Conditional simulation", () => {
  const condLats = [0, 0, 1, 1];
  const condLons = [0, 1, 0, 1];
  const condValues = [10, 12, 11, 13];
  const targetLats = [0.5, 0.25, 0.75];
  const targetLons = [0.5, 0.25, 0.75];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("conditionalSimulate is deterministic under fixed seed", () => {
    const a = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 42,
    });
    const b = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 42,
    });
    expect(a.length).toBe(targetLats.length);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  test("different seeds produce different realizations", () => {
    const a = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 1,
    });
    const b = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 2,
    });
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });
});

describe("Conditional simulation (per-variant)", () => {
  const condLats = [0, 0, 1, 1];
  const condLons = [0, 1, 0, 1];
  const condValues = [10, 12, 11, 13];
  const targetLats = [0.5, 0.25, 0.75];
  const targetLons = [0.5, 0.25, 0.75];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("conditionalSimulateSimple is deterministic and returns finite samples", () => {
    const a = conditionalSimulateSimple({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      mean: 11.5,
      seed: 101,
    });
    const b = conditionalSimulateSimple({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      mean: 11.5,
      seed: 101,
    });
    expect(a.length).toBe(targetLats.length);
    expect(Array.from(a)).toEqual(Array.from(b));
    for (const v of a) expect(Number.isFinite(v)).toBe(true);
  });

  test("conditionalSimulateUniversal with constant trend ≈ conditionalSimulate", () => {
    // Constant universal trend is mathematically equivalent to ordinary kriging.
    // Both use the same seeded RNG path, so realizations should match exactly.
    const ord = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 7,
    });
    const uni = conditionalSimulateUniversal({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      trend: "constant",
      seed: 7,
    });
    expect(uni.length).toBe(targetLats.length);
    for (let i = 0; i < ord.length; i += 1) {
      expect(Math.abs(uni[i] - ord[i])).toBeLessThan(1e-6);
    }
  });

  test("conditionalSimulateUniversal rejects too-few conditioning points for linear trend", () => {
    expect(() =>
      conditionalSimulateUniversal({
        conditioningLats: [0, 0, 1],
        conditioningLons: [0, 1, 0],
        conditioningValues: [10, 12, 11],
        targetLats: [0.5],
        targetLons: [0.5],
        variogram,
        trend: "linear",
        seed: 0,
      })
    ).toThrow(KrigingError);
  });

  test("conditionalSimulateProjected is deterministic on planar coordinates", () => {
    const xs = [0, 0, 1, 1];
    const ys = [0, 1, 0, 1];
    const values = [10, 12, 11, 13];
    const targetXs = [0.5, 0.25];
    const targetYs = [0.5, 0.75];
    const vg = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 2.0,
    };
    const a = conditionalSimulateProjected({
      conditioningXs: xs,
      conditioningYs: ys,
      conditioningValues: values,
      targetXs,
      targetYs,
      variogram: vg,
      majorAngleDeg: 0,
      rangeRatio: 1,
      seed: 13,
    });
    const b = conditionalSimulateProjected({
      conditioningXs: xs,
      conditioningYs: ys,
      conditioningValues: values,
      targetXs,
      targetYs,
      variogram: vg,
      majorAngleDeg: 0,
      rangeRatio: 1,
      seed: 13,
    });
    expect(a.length).toBe(targetXs.length);
    expect(Array.from(a)).toEqual(Array.from(b));
    for (const v of a) expect(Number.isFinite(v)).toBe(true);
  });
});

describe("Conditional simulation (binomial, both scales)", () => {
  const condLats = [0, 0, 1, 1];
  const condLons = [0, 1, 0, 1];
  const successes = [3, 7, 4, 9];
  const trials = [10, 12, 9, 15];
  const targetLats = [0.5, 0.25];
  const targetLons = [0.5, 0.75];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("conditionalSimulateBinomial reports both scales consistently", () => {
    const result = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 42,
    });
    expect(result.logitSamples.length).toBe(targetLats.length);
    expect(result.prevalenceSamples.length).toBe(targetLats.length);
    for (let i = 0; i < result.logitSamples.length; i += 1) {
      const logit = result.logitSamples[i];
      const prev = result.prevalenceSamples[i];
      expect(Number.isFinite(logit)).toBe(true);
      expect(prev).toBeGreaterThan(0);
      expect(prev).toBeLessThan(1);
      const expected = 1 / (1 + Math.exp(-logit));
      expect(Math.abs(prev - expected)).toBeLessThan(1e-6);
    }
  });

  test("conditionalSimulateBinomial is deterministic for same seed", () => {
    const a = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 99,
    });
    const b = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 99,
    });
    expect(Array.from(a.logitSamples)).toEqual(Array.from(b.logitSamples));
    expect(Array.from(a.prevalenceSamples)).toEqual(
      Array.from(b.prevalenceSamples)
    );
  });

  test("conditionalSimulateBinomial drops trials === 0 stations from conditioning", () => {
    const withZero = conditionalSimulateBinomial({
      conditioningLats: [...condLats, 0.5],
      conditioningLons: [...condLons, 0.5],
      successes: [...successes, 0],
      trials: [...trials, 0],
      targetLats,
      targetLons,
      variogram,
      seed: 4,
    });
    const without = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 4,
    });
    expect(Array.from(withZero.logitSamples)).toEqual(
      Array.from(without.logitSamples)
    );
    expect(Array.from(withZero.prevalenceSamples)).toEqual(
      Array.from(without.prevalenceSamples)
    );
  });

  test("conditionalSimulateBinomial rejects lopsided priors", () => {
    expect(() =>
      conditionalSimulateBinomial({
        conditioningLats: condLats,
        conditioningLons: condLons,
        successes,
        trials,
        targetLats,
        targetLons,
        variogram,
        priorAlpha: 2,
        seed: 0,
      })
    ).toThrow(KrigingError);
  });

  test("conditionalSimulateBinomial with custom prior differs from default", () => {
    const defaultPrior = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 55,
    });
    const customPrior = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      priorAlpha: 2,
      priorBeta: 5,
      seed: 55,
    });
    expect(
      Math.abs(defaultPrior.logitSamples[0] - customPrior.logitSamples[0])
    ).toBeGreaterThan(1e-9);
  });
});

describe("Binomial kriging: fromPrecomputedLogits", () => {
  test("builds from externally supplied logits", () => {
    const lats = [0, 0, 1, 1];
    const lons = [0, 1, 0, 1];
    const logits = [-1.5, 0.0, 0.5, 1.5];
    const model = BinomialKriging.fromPrecomputedLogits({
      lats,
      lons,
      logits,
      variogram: {
        variogramType: "exponential",
        nugget: 0.05,
        sill: 1,
        range: 500,
      },
    });
    try {
      const pred = model.predict(0.5, 0.5);
      expect(pred.prevalence).toBeGreaterThanOrEqual(0);
      expect(pred.prevalence).toBeLessThanOrEqual(1);
    } finally {
      model.free();
    }
  });

  test("rejects non-finite logits", () => {
    expect(() =>
      BinomialKriging.fromPrecomputedLogits({
        lats: [0, 0, 1],
        lons: [0, 1, 0],
        logits: [0, Number.POSITIVE_INFINITY, 1],
        variogram: {
          variogramType: "exponential",
          nugget: 0.05,
          sill: 1,
          range: 500,
        },
      })
    ).toThrow(KrigingError);
  });
});

describe("Nested variograms", () => {
  test("evaluateNestedVariogram sums components on the semivariance scale", () => {
    const distances = [0, 100, 500, 1000];
    const out = evaluateNestedVariogram(
      [
        { variogramType: "exponential", nugget: 0.1, sill: 0.5, range: 200 },
        { variogramType: "spherical", nugget: 0.0, sill: 0.5, range: 800 },
      ],
      distances
    );
    expect(out.distances.length).toBe(distances.length);
    expect(out.semivariances.length).toBe(distances.length);
    expect(out.covariances.length).toBe(distances.length);
    for (let i = 0; i < out.semivariances.length; i++) {
      expect(Number.isFinite(out.semivariances[i])).toBe(true);
      expect(out.semivariances[i]).toBeGreaterThanOrEqual(0);
    }
    expect(out.semivariances[out.semivariances.length - 1]).toBeGreaterThan(
      out.semivariances[0]
    );
  });

  test("evaluateNestedVariogram rejects empty components", () => {
    expect(() => evaluateNestedVariogram([], [1, 2, 3])).toThrow(KrigingError);
  });
});

describe("fitVariogram estimator option", () => {
  test("Cressie-Hawkins estimator runs and returns finite params", () => {
    const lats = [0, 0, 1, 1, 2, 2];
    const lons = [0, 1, 0, 1, 0, 1];
    const values = [10, 12, 11, 13, 14, 15];
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: "exponential",
      estimator: "cressie-hawkins",
      nBins: 5,
    });
    expect(Number.isFinite(fit.nugget)).toBe(true);
    expect(Number.isFinite(fit.sill)).toBe(true);
    expect(Number.isFinite(fit.range)).toBe(true);
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

describe("Space-time kriging", () => {
  const lats = [0, 0, 0, 0, 1, 1, 1, 1];
  const lons = [0, 0, 1, 1, 0, 0, 1, 1];
  const times = [0, 1, 0, 1, 0, 1, 0, 1];
  const values = [1, 2, 1.5, 2.5, 1.2, 2.1, 1.8, 2.7];
  const successes = [2, 4, 3, 5, 3, 5, 4, 6];
  const trials = [10, 10, 10, 10, 10, 10, 10, 10];
  const variogram = {
    family: "separable" as const,
    spatial: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 300,
    },
    temporal: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 2,
    },
  };

  test("ordinary model predicts finite value and non-negative variance", () => {
    const model = new SpaceTimeOrdinaryKriging({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    try {
      const { value, variance } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
      expect(variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("ordinary batch-arrays returns typed arrays sized by targets", () => {
    const model = new SpaceTimeOrdinaryKriging({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    try {
      const out = model.predictBatchArrays(
        [0.25, 0.75],
        [0.25, 0.75],
        [0.25, 0.75]
      );
      expect(out.values).toBeInstanceOf(Float64Array);
      expect(out.values.length).toBe(2);
      expect(out.variances.length).toBe(2);
    } finally {
      model.free();
    }
  });

  test("simple model reverts to the supplied mean far outside the data", () => {
    const mean = 100;
    const model = new SpaceTimeSimpleKriging({
      lats,
      lons,
      times,
      values,
      variogram: {
        ...variogram,
        spatial: { ...variogram.spatial, range: 0.5 },
        temporal: { ...variogram.temporal, range: 0.5 },
      },
      mean,
    });
    try {
      const { value } = model.predict(50, 50, 500);
      expect(Math.abs(value - mean)).toBeLessThan(1);
    } finally {
      model.free();
    }
  });

  test("universal model with constant trend predicts finite value", () => {
    const model = new SpaceTimeUniversalKriging({
      lats,
      lons,
      times,
      values,
      variogram,
      trend: "constant",
    });
    try {
      const { value } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
    } finally {
      model.free();
    }
  });

  test("binomial model returns prevalence in the unit interval", () => {
    const model = new SpaceTimeBinomialKriging({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    try {
      const { prevalence, logitValue, variance, prevalenceVariance } =
        model.predict(0.5, 0.5, 0.5);
      expect(prevalence).toBeGreaterThan(0);
      expect(prevalence).toBeLessThan(1);
      expect(Number.isFinite(logitValue)).toBe(true);
      expect(variance).toBeGreaterThanOrEqual(0);
      expect(prevalenceVariance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("projected ordinary model predicts finite value", () => {
    const model = new SpaceTimeProjectedOrdinaryKriging({
      xs: lats,
      ys: lons,
      times,
      values,
      variogram: {
        ...variogram,
        spatial: { ...variogram.spatial, range: 2.0 },
      },
      majorAngleDeg: 0,
      rangeRatio: 1.0,
    });
    try {
      const { value } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
    } finally {
      model.free();
    }
  });

  test("computeEmpiricalSpaceTimeVariogram produces bin arrays of the expected size", () => {
    const emp = computeEmpiricalSpaceTimeVariogram({
      lats,
      lons,
      times,
      values,
      nSpatialBins: 3,
      nTemporalBins: 2,
    });
    expect(emp.nSpatialBins).toBe(3);
    expect(emp.nTemporalBins).toBe(2);
    const expected = 3 * 2;
    expect(emp.semivariances.length).toBe(expected);
    expect(emp.spatialLags.length).toBe(expected);
    expect(emp.temporalLags.length).toBe(expected);
    expect(emp.nPairs.length).toBe(expected);
  });

  test("fitSpaceTimeVariogram returns a separable fit with finite residuals", () => {
    const denseLats: number[] = [];
    const denseLons: number[] = [];
    const denseTimes: number[] = [];
    const denseValues: number[] = [];
    for (let i = 0; i < 6; i++) {
      for (let t = 0; t < 5; t++) {
        denseLats.push(i * 0.1);
        denseLons.push(0.05 * t);
        denseTimes.push(t);
        denseValues.push((i + t) * 0.5);
      }
    }
    const result = fitSpaceTimeVariogram({
      lats: denseLats,
      lons: denseLons,
      times: denseTimes,
      values: denseValues,
      nSpatialBins: 5,
      nTemporalBins: 5,
      family: "separable",
      spatialModel: "exponential",
      temporalModel: "exponential",
    });
    expect(result.fit.family).toBe("separable");
    expect(result.fit.spatial.variogramType).toBe("exponential");
    expect(result.fit.temporal.variogramType).toBe("exponential");
    expect(Number.isFinite(result.fit.residuals)).toBe(true);
  });

  test("mismatched arrays raise a KrigingError with a code", () => {
    expect(
      () =>
        new SpaceTimeOrdinaryKriging({
          lats: [0, 0],
          lons: [0, 1, 2],
          times: [0, 1],
          values: [1, 2],
          variogram,
        })
    ).toThrow(KrigingError);
  });
});

describe("Space-time cross-validation", () => {
  const lats = [0, 0, 0, 0, 1, 1, 1, 1];
  const lons = [0, 0, 1, 1, 0, 0, 1, 1];
  const times = [0, 1, 0, 1, 0, 1, 0, 1];
  const values = [1, 2, 1.5, 2.5, 1.2, 2.1, 1.8, 2.7];
  const successes = [2, 4, 3, 5, 3, 5, 4, 6];
  const trials = [10, 10, 10, 10, 10, 10, 10, 10];
  const variogram = {
    family: "separable" as const,
    spatial: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 300,
    },
    temporal: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 2,
    },
  };

  test("leaveOneOutSpaceTime returns one residual per station", () => {
    const out = leaveOneOutSpaceTime({ lats, lons, times, values, variogram });
    expect(out.residuals.length).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
      expect(r.variance).toBeGreaterThanOrEqual(0);
    }
    expect(out.summary.n).toBe(values.length);
  });

  test("kFoldSpaceTime returns residuals for all stations", () => {
    const out = kFoldSpaceTime({
      lats,
      lons,
      times,
      values,
      variogram,
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("kFoldSpaceTime rejects k < 2", () => {
    expect(() =>
      kFoldSpaceTime({ lats, lons, times, values, variogram, k: 1 })
    ).toThrow(KrigingError);
  });

  test("leaveOneOutSpaceTimeSimple honors known mean", () => {
    const out = leaveOneOutSpaceTimeSimple({
      lats,
      lons,
      times,
      values,
      variogram,
      mean: 2,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("kFoldSpaceTimeSimple returns residuals for all stations", () => {
    const out = kFoldSpaceTimeSimple({
      lats,
      lons,
      times,
      values,
      variogram,
      mean: 2,
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("leaveOneOutSpaceTimeUniversal with constant trend matches ordinary CV", () => {
    const ok = leaveOneOutSpaceTime({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    const uk = leaveOneOutSpaceTimeUniversal({
      lats,
      lons,
      times,
      values,
      variogram,
      trend: "constant",
    });
    expect(uk.residuals.length).toBe(ok.residuals.length);
    for (let i = 0; i < ok.residuals.length; i++) {
      expect(uk.residuals[i].predicted).toBeCloseTo(
        ok.residuals[i].predicted,
        6
      );
    }
  });

  test("kFoldSpaceTimeUniversal returns residuals for all stations", () => {
    const out = kFoldSpaceTimeUniversal({
      lats,
      lons,
      times,
      values,
      variogram,
      trend: "linearInTime",
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("leaveOneOutSpaceTimeBinomial returns dual-scale residuals", () => {
    const out = leaveOneOutSpaceTimeBinomial({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    expect(out.residuals.length).toBe(successes.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predictedLogit)).toBe(true);
      expect(r.predictedPrevalence).toBeGreaterThan(0);
      expect(r.predictedPrevalence).toBeLessThan(1);
    }
  });

  test("kFoldSpaceTimeBinomial returns residuals for all stations", () => {
    const out = kFoldSpaceTimeBinomial({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
      k: 4,
    });
    expect(out.residuals.length).toBe(successes.length);
  });

  test("binomial CV rejects prior with only alpha", () => {
    expect(() =>
      leaveOneOutSpaceTimeBinomial({
        lats,
        lons,
        times,
        successes,
        trials,
        variogram,
        priorAlpha: 1,
      })
    ).toThrow(KrigingError);
  });
});

describe("Space-time conditional simulation", () => {
  const conditioningLats = [0, 0, 0, 0, 1, 1, 1, 1];
  const conditioningLons = [0, 0, 1, 1, 0, 0, 1, 1];
  const conditioningTimes = [0, 1, 0, 1, 0, 1, 0, 1];
  const conditioningValues = [1, 2, 1.5, 2.5, 1.2, 2.1, 1.8, 2.7];
  const successes = [2, 4, 3, 5, 3, 5, 4, 6];
  const trials = [10, 10, 10, 10, 10, 10, 10, 10];
  const targetLats = [0.25, 0.75];
  const targetLons = [0.25, 0.75];
  const targetTimes = [0.5, 0.5];
  const variogram = {
    family: "separable" as const,
    spatial: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 300,
    },
    temporal: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 2,
    },
  };

  test("conditionalSimulateSpaceTime is deterministic under fixed seed", () => {
    const a = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 42n,
    });
    const b = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 42n,
    });
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  test("conditionalSimulateSpaceTime differs across seeds", () => {
    const a = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 1n,
    });
    const b = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 2n,
    });
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  test("conditionalSimulateSpaceTimeSimple honors known mean", () => {
    const out = conditionalSimulateSpaceTimeSimple({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      mean: 2,
      seed: 7n,
    });
    expect(out.length).toBe(targetLats.length);
    for (const v of out) expect(Number.isFinite(v)).toBe(true);
  });

  test("conditionalSimulateSpaceTimeUniversal with constant trend ≈ ordinary", () => {
    const ord = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 11n,
    });
    const uk = conditionalSimulateSpaceTimeUniversal({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      trend: "constant",
      seed: 11n,
    });
    expect(uk.length).toBe(ord.length);
    for (let i = 0; i < ord.length; i++) {
      expect(uk[i]).toBeCloseTo(ord[i], 5);
    }
  });

  test("conditionalSimulateSpaceTimeBinomial returns logit and prevalence samples", () => {
    const out = conditionalSimulateSpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 13n,
    });
    expect(out.logitSamples.length).toBe(targetLats.length);
    expect(out.prevalenceSamples.length).toBe(targetLats.length);
    for (let i = 0; i < targetLats.length; i++) {
      expect(Number.isFinite(out.logitSamples[i])).toBe(true);
      expect(out.prevalenceSamples[i]).toBeGreaterThan(0);
      expect(out.prevalenceSamples[i]).toBeLessThan(1);
    }
  });

  test("conditionalSimulateSpaceTimeBinomial is deterministic under fixed seed", () => {
    const a = conditionalSimulateSpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 99n,
    });
    const b = conditionalSimulateSpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 99n,
    });
    expect(Array.from(a.logitSamples)).toEqual(Array.from(b.logitSamples));
    expect(Array.from(a.prevalenceSamples)).toEqual(
      Array.from(b.prevalenceSamples)
    );
  });
});
