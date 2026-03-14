import init, * as wasm from "./pkg/kriging_rs.js";

const {
  WasmOrdinaryKriging,
  WasmBinomialKriging,
  fitOrdinaryVariogram,
  webgpuAvailable,
} = wasm;

const output = document.getElementById("output");
const surfacePlot = document.getElementById("surfacePlot");
const residualPlot = document.getElementById("residualPlot");
const variogramPlot = document.getElementById("variogramPlot");
const surfaceLabel = document.getElementById("surfaceLabel");
const surfaceMin = document.getElementById("surfaceMin");
const surfaceMax = document.getElementById("surfaceMax");
const residualMean = document.getElementById("residualMean");
const residualRmse = document.getElementById("residualRmse");
const gridResolution = document.getElementById("gridResolution");
const nBins = document.getElementById("nBins");
const maxDistanceKm = document.getElementById("maxDistanceKm");
const modelSpherical = document.getElementById("modelSpherical");
const modelExponential = document.getElementById("modelExponential");
const modelGaussian = document.getElementById("modelGaussian");
const binomialAlpha = document.getElementById("binomialAlpha");
const binomialBeta = document.getElementById("binomialBeta");
const surfaceKrigingType = document.getElementById("surfaceKrigingType");
const surfaceLayer = document.getElementById("surfaceLayer");
const residualMode = document.getElementById("residualMode");
const surfaceBackend = document.getElementById("surfaceBackend");
const webgpuStatus = document.getElementById("webgpuStatus");

let webGpuAvailableFlag = false;
let webGpuDetectionMessage = "not checked";

function selectedVariogramTypes() {
  const selected = [];
  if (modelSpherical.checked) {
    selected.push("spherical");
  }
  if (modelExponential.checked) {
    selected.push("exponential");
  }
  if (modelGaussian.checked) {
    selected.push("gaussian");
  }
  if (selected.length === 0) {
    throw new Error("Select at least one variogram candidate");
  }
  return selected;
}

function readQuickOptions() {
  const nBinsValue = Number(nBins.value);
  const maxDistanceValue = maxDistanceKm.value.trim();
  const maxDistance = maxDistanceValue === "" ? undefined : Number(maxDistanceValue);
  const variogramTypes = selectedVariogramTypes();
  const alpha = Number(binomialAlpha.value);
  const beta = Number(binomialBeta.value);
  if (!Number.isFinite(nBinsValue) || nBinsValue <= 0) {
    throw new Error("n_bins must be a positive integer");
  }
  if (maxDistance !== undefined && (!Number.isFinite(maxDistance) || maxDistance <= 0)) {
    throw new Error("max distance must be positive when set");
  }
  if (!Number.isFinite(alpha) || alpha <= 0 || !Number.isFinite(beta) || beta <= 0) {
    throw new Error("binomial alpha and beta must be > 0");
  }
  return {
    nBins: nBinsValue,
    maxDistance,
    variogramTypes,
    alpha,
    beta,
  };
}

function generateSurfaceSamples(sampleCount = 324) {
  const lats = [];
  const lons = [];
  const values = [];
  const successes = [];
  const trials = [];

  const latMin = 37.7;
  const latMax = 37.82;
  const lonMin = -122.48;
  const lonMax = -122.35;
  for (let i = 0; i < sampleCount; i += 1) {
    const lat = latMin + Math.random() * (latMax - latMin);
    const lon = lonMin + Math.random() * (lonMax - lonMin);
    const smoothField =
      Math.sin((lat - latMin) * 65) * 0.8 + Math.cos((lon - lonMin) * 72) * 0.75;
    const trend = ((lat - latMin) / (latMax - latMin) - 0.5) * 0.8;
    const valueNoise = (Math.random() - 0.5) * 0.25;
    const value = 15.0 + smoothField * 2.3 + trend + valueNoise;
    const probability = 1 / (1 + Math.exp(-(smoothField + trend)));
    const trialCount = 30 + Math.floor(Math.random() * 20);
    const successCount = Math.min(
      trialCount,
      Math.max(0, Math.round(probability * trialCount)),
    );

    lats.push(lat);
    lons.push(lon);
    values.push(value);
    trials.push(trialCount);
    successes.push(successCount);
  }

  return {
    lats,
    lons,
    values,
    successes,
    trials,
    nugget: 0.05,
    sill: 10.0,
    range: 6.0,
  };
}

let SURFACE_SAMPLE = generateSurfaceSamples();
const HARNESS_SAMPLE = generateSurfaceSamples(350);

function summarizeTimings(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const count = sorted.length;
  if (count === 0) {
    return { meanMs: 0, p50Ms: 0, p95Ms: 0, minMs: 0, maxMs: 0 };
  }
  const idx50 = Math.floor((count - 1) * 0.5);
  const idx95 = Math.floor((count - 1) * 0.95);
  const sum = sorted.reduce((acc, value) => acc + value, 0);
  return {
    meanMs: sum / count,
    p50Ms: sorted[idx50],
    p95Ms: sorted[idx95],
    minMs: sorted[0],
    maxMs: sorted[count - 1],
  };
}

function summarizePhaseShares(phaseTotals, totalValues) {
  const totalSum = totalValues.reduce((acc, value) => acc + value, 0);
  if (totalSum <= 0) {
    return Object.fromEntries(
      Object.entries(phaseTotals).map(([key]) => [key, { percent: 0, meanMs: 0 }]),
    );
  }
  return Object.fromEntries(
    Object.entries(phaseTotals).map(([key, values]) => {
      const meanMs = values.reduce((acc, value) => acc + value, 0) / values.length;
      const percent = (values.reduce((acc, value) => acc + value, 0) / totalSum) * 100;
      return [key, { meanMs, percent }];
    }),
  );
}

function resolveBackendMode(selection) {
  if (selection === "cpu") {
    return { useGpu: false, mode: "cpu" };
  }
  if (selection === "webgpu") {
    if (typeof webgpuAvailable !== "function") {
      throw new Error('webgpuAvailable export not found; rebuild with --features "wasm,gpu"');
    }
    if (!webGpuAvailableFlag) {
      throw new Error(`WebGPU required but unavailable (${webGpuDetectionMessage})`);
    }
    return { useGpu: true, mode: "webgpu" };
  }
  if (selection === "auto") {
    return { useGpu: typeof webgpuAvailable === "function" && webGpuAvailableFlag, mode: "auto" };
  }
  return { useGpu: false, mode: "cpu" };
}

function buildBinomialLogits(successes, trials, alpha, beta) {
  const logits = new Float64Array(successes.length);
  for (let i = 0; i < successes.length; i += 1) {
    const s = successes[i];
    const n = trials[i];
    const p = (s + alpha) / (n + alpha + beta);
    logits[i] = Math.log(p / (1 - p));
  }
  return logits;
}

async function runOrdinaryPerformanceHarness(options, backendSelection) {
  if (typeof fitOrdinaryVariogram !== "function") {
    throw new Error(
      "fitOrdinaryVariogram export not found; rebuild pkg with wasm-pack build --target web --out-dir pkg -- --features wasm",
    );
  }
  const warmupRuns = 3;
  const measuredRuns = 10;
  const resolution = 36;
  const phaseSeries = {
    dataPrepMs: [],
    variogramFitMs: [],
    modelBuildMs: [],
    predictBatchMs: [],
    mappingMs: [],
    totalMs: [],
  };
  const backend = resolveBackendMode(backendSelection);
  let checksum = 0;

  for (let runIndex = 0; runIndex < warmupRuns + measuredRuns; runIndex += 1) {
    const measured = runIndex >= warmupRuns;

    let t0 = performance.now();
    const sampleLats = Float64Array.from(HARNESS_SAMPLE.lats);
    const sampleLons = Float64Array.from(HARNESS_SAMPLE.lons);
    const sampleValues = Float64Array.from(HARNESS_SAMPLE.values);
    const grid = buildPredictionGrid(HARNESS_SAMPLE.lats, HARNESS_SAMPLE.lons, resolution);
    let t1 = performance.now();
    const dataPrepMs = t1 - t0;

    t0 = performance.now();
    const fitted = fitOrdinaryVariogram(
      sampleLats,
      sampleLons,
      sampleValues,
      options.maxDistance,
      options.nBins,
      options.variogramTypes,
    );
    const variogramType = fitted.variogramType ?? fitted.variogram_type;
    if (typeof variogramType !== "string") {
      throw new Error("fitOrdinaryVariogram returned invalid variogramType");
    }
    t1 = performance.now();
    const variogramFitMs = t1 - t0;

    t0 = performance.now();
    const model = new WasmOrdinaryKriging(
      sampleLats,
      sampleLons,
      sampleValues,
      variogramType,
      fitted.nugget,
      fitted.sill,
      fitted.range,
    );
    t1 = performance.now();
    const modelBuildMs = t1 - t0;

    t0 = performance.now();
    const predictions = backend.useGpu
      ? await model.predictBatchGpu(grid.predLats, grid.predLons)
      : model.predictBatch(grid.predLats, grid.predLons);
    t1 = performance.now();
    const predictBatchMs = t1 - t0;

    t0 = performance.now();
    const localChecksum = predictions.reduce((acc, pred) => acc + pred.value + pred.variance, 0);
    checksum += localChecksum;
    t1 = performance.now();
    const mappingMs = t1 - t0;

    model.free();

    if (measured) {
      const totalMs = dataPrepMs + variogramFitMs + modelBuildMs + predictBatchMs + mappingMs;
      phaseSeries.dataPrepMs.push(dataPrepMs);
      phaseSeries.variogramFitMs.push(variogramFitMs);
      phaseSeries.modelBuildMs.push(modelBuildMs);
      phaseSeries.predictBatchMs.push(predictBatchMs);
      phaseSeries.mappingMs.push(mappingMs);
      phaseSeries.totalMs.push(totalMs);
    }
  }

  const totalStats = summarizeTimings(phaseSeries.totalMs);
  return {
    scenario: "ordinary_350_samples_36x36_grid",
    warmupRuns,
    measuredRuns,
    samplePoints: HARNESS_SAMPLE.lats.length,
    predictionPoints: resolution * resolution,
    options,
    backendSelection,
    backendResolved: backend.useGpu ? "webgpu" : "cpu",
    total: totalStats,
    phases: {
      dataPrep: summarizeTimings(phaseSeries.dataPrepMs),
      variogramFit: summarizeTimings(phaseSeries.variogramFitMs),
      modelBuild: summarizeTimings(phaseSeries.modelBuildMs),
      predictBatch: summarizeTimings(phaseSeries.predictBatchMs),
      mapping: summarizeTimings(phaseSeries.mappingMs),
    },
    phaseShare: summarizePhaseShares(
      {
        dataPrep: phaseSeries.dataPrepMs,
        variogramFit: phaseSeries.variogramFitMs,
        modelBuild: phaseSeries.modelBuildMs,
        predictBatch: phaseSeries.predictBatchMs,
        mapping: phaseSeries.mappingMs,
      },
      phaseSeries.totalMs,
    ),
    checksum,
  };
}

function write(title, payload) {
  output.textContent = `${title}\n\n${JSON.stringify(payload, null, 2)}`;
}

function drawSurfacePlaceholder() {
  const ctx = surfacePlot.getContext("2d");
  ctx.clearRect(0, 0, surfacePlot.width, surfacePlot.height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, surfacePlot.width, surfacePlot.height);
  ctx.fillStyle = "#555";
  ctx.font = "16px sans-serif";
  ctx.fillText("Run 2D surface to render heatmap", 20, 30);
}

function drawResidualPlaceholder() {
  const ctx = residualPlot.getContext("2d");
  ctx.clearRect(0, 0, residualPlot.width, residualPlot.height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, residualPlot.width, residualPlot.height);
  ctx.fillStyle = "#555";
  ctx.font = "14px sans-serif";
  ctx.fillText("Residual plot appears after running the 2D surface", 16, 28);
}

function drawVariogramPlaceholder() {
  const ctx = variogramPlot.getContext("2d");
  ctx.clearRect(0, 0, variogramPlot.width, variogramPlot.height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, variogramPlot.width, variogramPlot.height);
  ctx.fillStyle = "#555";
  ctx.font = "14px sans-serif";
  ctx.fillText("Empirical variogram appears after running the 2D surface", 16, 28);
}

function normalQuantile(p) {
  const a = [
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.357751867269,
    -30.66479806614716,
    2.506628277459239,
  ];
  const b = [
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
  ];
  const c = [
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    0.007784695709041462,
    0.3224671290700398,
    2.445134137142996,
    3.754408661907416,
  ];

  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p <= 0) {
    return Number.NEGATIVE_INFINITY;
  }
  if (p >= 1) {
    return Number.POSITIVE_INFINITY;
  }

  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  const q = p - 0.5;
  const r = q * q;
  return (
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

function valueToColor(value, min, max) {
  if (max <= min) {
    return "hsl(0, 0%, 50%)";
  }
  const t = Math.min(1, Math.max(0, (value - min) / (max - min)));
  const hue = (1 - t) * 240;
  return `hsl(${hue}, 85%, 50%)`;
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

  return {
    predLats,
    predLons,
    bounds: { latMin, latMax, lonMin, lonMax },
  };
}

function renderResidualPlot(residuals, mode = "scatter") {
  const ctx = residualPlot.getContext("2d");
  const width = residualPlot.width;
  const height = residualPlot.height;
  const padding = 22;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  const minResidual = Math.min(...residuals);
  const maxResidual = Math.max(...residuals);
  const maxAbs = Math.max(Math.abs(minResidual), Math.abs(maxResidual), 1e-9);
  const mean = residuals.reduce((sum, value) => sum + value, 0) / residuals.length;
  const rmse = Math.sqrt(
    residuals.reduce((sum, value) => sum + value * value, 0) / residuals.length,
  );

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fdfdfd";
  ctx.fillRect(0, 0, width, height);

  const zeroY = padding + (plotHeight * (maxAbs - 0)) / (2 * maxAbs);
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, zeroY);
  ctx.lineTo(width - padding, zeroY);
  ctx.stroke();

  ctx.strokeStyle = "#bbb";
  ctx.strokeRect(padding, padding, plotWidth, plotHeight);

  if (mode === "qq") {
    const sd = Math.sqrt(
      residuals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / residuals.length,
    );
    const scale = Math.max(sd, 1e-9);
    const sorted = [...residuals].sort((a, b) => a - b).map((value) => (value - mean) / scale);
    const expected = sorted.map((_, i) => normalQuantile((i + 0.5) / sorted.length));
    const extent = Math.max(
      2.0,
      ...sorted.map((v) => Math.abs(v)),
      ...expected.map((v) => Math.abs(v)),
    );

    const toX = (value) => padding + ((value + extent) / (2 * extent)) * plotWidth;
    const toY = (value) => padding + ((extent - value) / (2 * extent)) * plotHeight;

    ctx.strokeStyle = "#7a7a7a";
    ctx.beginPath();
    ctx.moveTo(toX(-extent), toY(-extent));
    ctx.lineTo(toX(extent), toY(extent));
    ctx.stroke();

    for (let i = 0; i < sorted.length; i += 1) {
      const x = toX(expected[i]);
      const y = toY(sorted[i]);
      ctx.fillStyle = "#5c2d91";
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
  } else if (mode === "histogram") {
    const bins = 20;
    const min = minResidual;
    const max = maxResidual;
    const span = Math.max(1e-9, max - min);
    const counts = Array(bins).fill(0);
    for (const residual of residuals) {
      const position = (residual - min) / span;
      const bin = Math.min(bins - 1, Math.max(0, Math.floor(position * bins)));
      counts[bin] += 1;
    }
    const maxCount = Math.max(...counts, 1);
    const barWidth = plotWidth / bins;
    for (let i = 0; i < bins; i += 1) {
      const barHeight = (counts[i] / maxCount) * (plotHeight - 8);
      const x = padding + i * barWidth + 1;
      const y = padding + plotHeight - barHeight;
      ctx.fillStyle = "#4f81bd";
      ctx.fillRect(x, y, Math.max(1, barWidth - 2), barHeight);
    }
  } else {
    for (let i = 0; i < residuals.length; i += 1) {
      const residual = residuals[i];
      const x =
        padding + (i / Math.max(1, residuals.length - 1)) * plotWidth;
      const y = padding + ((maxAbs - residual) / (2 * maxAbs)) * plotHeight;
      ctx.fillStyle = residual >= 0 ? "#ba1b1b" : "#1f5aa6";
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }
  }

  ctx.fillStyle = "#444";
  ctx.font = "12px sans-serif";
  if (mode === "qq") {
    ctx.fillText("QQ residual diagnostic (standardized)", 8, padding + 10);
  } else if (mode === "histogram") {
    ctx.fillText("Residual distribution", 8, padding + 10);
  } else {
    ctx.fillText("+ residual", 8, padding + 10);
    ctx.fillText("- residual", 8, height - padding - 2);
  }

  residualMean.textContent = mean.toFixed(4);
  residualRmse.textContent = rmse.toFixed(4);

  return { mean, rmse, minResidual, maxResidual };
}

function computeEmpiricalVariogram(lats, lons, values, binCount = 18) {
  const n = values.length;
  if (n < 2) {
    return [];
  }

  let maxDistance = 0;
  const pairs = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dLat = lats[i] - lats[j];
      const dLon = lons[i] - lons[j];
      const distance = Math.sqrt(dLat * dLat + dLon * dLon);
      const semivariance = 0.5 * (values[i] - values[j]) ** 2;
      maxDistance = Math.max(maxDistance, distance);
      pairs.push({ distance, semivariance });
    }
  }

  if (maxDistance <= 0) {
    return [];
  }

  const width = maxDistance / binCount;
  const bins = Array.from({ length: binCount }, () => ({
    distanceSum: 0,
    semivarianceSum: 0,
    count: 0,
  }));

  for (const pair of pairs) {
    const index = Math.min(binCount - 1, Math.floor(pair.distance / width));
    bins[index].distanceSum += pair.distance;
    bins[index].semivarianceSum += pair.semivariance;
    bins[index].count += 1;
  }

  return bins
    .filter((bin) => bin.count > 0)
    .map((bin) => ({
      distance: bin.distanceSum / bin.count,
      semivariance: bin.semivarianceSum / bin.count,
      count: bin.count,
    }));
}

function variogramSemivariance(distance, modelType, nugget, sill, range) {
  const r = Math.max(range, 1e-9);
  const partial = Math.max(sill - nugget, 1e-9);
  const h = Math.max(distance, 0);

  if (modelType === "spherical") {
    if (h >= r) {
      return sill;
    }
    const x = h / r;
    return nugget + partial * (1.5 * x - 0.5 * x * x * x);
  }
  if (modelType === "gaussian") {
    return nugget + partial * (1 - Math.exp(-(h * h) / (r * r)));
  }
  return nugget + partial * (1 - Math.exp(-h / r));
}

function renderVariogramPlot(variogramPoints, modelType) {
  const ctx = variogramPlot.getContext("2d");
  const width = variogramPlot.width;
  const height = variogramPlot.height;
  const padding = 30;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fdfdfd";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#bbb";
  ctx.strokeRect(padding, padding, plotWidth, plotHeight);

  if (variogramPoints.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "13px sans-serif";
    ctx.fillText("Insufficient data for variogram", padding + 10, padding + 20);
    return { pointCount: 0 };
  }

  const maxDistance = Math.max(...variogramPoints.map((p) => p.distance), 1e-9);
  const maxSemivariance = Math.max(...variogramPoints.map((p) => p.semivariance), 1e-9);
  const minSemivariance = Math.min(...variogramPoints.map((p) => p.semivariance), 0);
  const overlayParams = {
    nugget: Math.max(0, minSemivariance),
    sill: maxSemivariance * 1.05,
    range: maxDistance * 0.45,
  };
  const yMax = Math.max(maxSemivariance, overlayParams.sill);

  ctx.strokeStyle = "#2c7fb8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < variogramPoints.length; i += 1) {
    const point = variogramPoints[i];
    const x = padding + (point.distance / maxDistance) * plotWidth;
    const y = padding + (1 - point.semivariance / yMax) * plotHeight;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  for (const point of variogramPoints) {
    const x = padding + (point.distance / maxDistance) * plotWidth;
    const y = padding + (1 - point.semivariance / yMax) * plotHeight;
    ctx.fillStyle = "#1f4e79";
    ctx.fillRect(x - 2, y - 2, 4, 4);
  }

  ctx.strokeStyle = "#d95f02";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const overlaySamples = 120;
  for (let i = 0; i <= overlaySamples; i += 1) {
    const distance = (i / overlaySamples) * maxDistance;
    const semivariance = variogramSemivariance(
      distance,
      modelType,
      overlayParams.nugget,
      overlayParams.sill,
      overlayParams.range,
    );
    const x = padding + (distance / maxDistance) * plotWidth;
    const y = padding + (1 - semivariance / yMax) * plotHeight;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  ctx.fillStyle = "#444";
  ctx.font = "12px sans-serif";
  ctx.fillText("Semivariance", 8, padding - 8);
  ctx.fillText("Distance", width - 85, height - 8);
  ctx.fillStyle = "#1f4e79";
  ctx.fillRect(width - 220, 12, 10, 10);
  ctx.fillStyle = "#444";
  ctx.fillText("Empirical", width - 205, 21);
  ctx.fillStyle = "#d95f02";
  ctx.fillRect(width - 130, 12, 10, 10);
  ctx.fillStyle = "#444";
  ctx.fillText("Model fit", width - 115, 21);

  return {
    pointCount: variogramPoints.length,
    maxDistance,
    maxSemivariance: yMax,
    modelType,
    overlayParams,
  };
}

function renderSurface(
  predictions,
  resolution,
  bounds,
  samplePoints,
  observedValues,
  predictionSelector,
  layerMode,
) {
  const ctx = surfacePlot.getContext("2d");
  const width = surfacePlot.width;
  const height = surfacePlot.height;
  const predictionValues = predictions.map(predictionSelector);
  const layerValues =
    layerMode === "variance" ? predictions.map((entry) => entry.variance) : predictionValues;
  const predMin = Math.min(...predictionValues);
  const predMax = Math.max(...predictionValues);
  const layerMin = Math.min(...layerValues);
  const layerMax = Math.max(...layerValues);
  const obsMin = Math.min(...observedValues);
  const obsMax = Math.max(...observedValues);
  const pointColorMin = Math.min(predMin, obsMin);
  const pointColorMax = Math.max(predMax, obsMax);
  const tileWidth = width / resolution;
  const tileHeight = height / resolution;

  ctx.clearRect(0, 0, width, height);
  for (let row = 0; row < resolution; row += 1) {
    for (let col = 0; col < resolution; col += 1) {
      const index = row * resolution + col;
      ctx.fillStyle = valueToColor(layerValues[index], layerMin, layerMax);
      ctx.fillRect(
        Math.floor(col * tileWidth),
        Math.floor(row * tileHeight),
        Math.ceil(tileWidth),
        Math.ceil(tileHeight),
      );
    }
  }

  const denseOverlay = samplePoints.lats.length > 120;
  for (let i = 0; i < samplePoints.lats.length; i += 1) {
    const lat = samplePoints.lats[i];
    const lon = samplePoints.lons[i];
    const x =
      ((lon - bounds.lonMin) / (bounds.lonMax - bounds.lonMin)) * surfacePlot.width;
    const y =
      ((bounds.latMax - lat) / (bounds.latMax - bounds.latMin)) * surfacePlot.height;

    ctx.beginPath();
    ctx.arc(x, y, denseOverlay ? 1.3 : 4, 0, Math.PI * 2);
    ctx.fillStyle = valueToColor(observedValues[i], pointColorMin, pointColorMax);
    ctx.fill();
    if (!denseOverlay) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  ctx.strokeStyle = "#111";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  surfaceLabel.textContent = layerMode === "variance" ? "Variance" : "Value";
  surfaceMin.textContent = layerMin.toFixed(3);
  surfaceMax.textContent = layerMax.toFixed(3);

  return {
    surfaceMin: layerMin,
    surfaceMax: layerMax,
    predMin,
    predMax,
    obsMin,
    obsMax,
  };
}

async function main() {
  await init();
  if (typeof webgpuAvailable === "function") {
    try {
      webGpuAvailableFlag = Boolean(await webgpuAvailable());
      webGpuDetectionMessage = webGpuAvailableFlag ? "available" : "unavailable";
    } catch (err) {
      webGpuAvailableFlag = false;
      webGpuDetectionMessage = `probe failed: ${err?.message ?? String(err)}`;
    }
  } else {
    webGpuAvailableFlag = false;
    webGpuDetectionMessage = 'WASM package missing "gpu" feature';
  }
  webgpuStatus.textContent = webGpuDetectionMessage;

  drawSurfacePlaceholder();
  drawResidualPlaceholder();
  drawVariogramPlaceholder();

  document.getElementById("runOrdinary").addEventListener("click", () => {
    const lats = [37.77, 37.78, 37.76, 37.75];
    const lons = [-122.42, -122.41, -122.4, -122.43];
    const values = [15, 18, 14, 13];

    const model = new WasmOrdinaryKriging(
      lats,
      lons,
      values,
      "exponential",
      0.1,
      6.0,
      5.0,
    );
    const pred = model.predict(37.765, -122.415);
    write("Ordinary kriging prediction", pred);
  });

  document.getElementById("runBinomial").addEventListener("click", () => {
    const lats = [40.71, 40.72, 40.7];
    const lons = [-74.0, -73.99, -74.02];
    const successes = [25, 35, 20];
    const trials = [50, 50, 50];

    const model = new WasmBinomialKriging(
      lats,
      lons,
      successes,
      trials,
      "gaussian",
      0.01,
      1.5,
      10.0,
    );
    const pred = model.predict(40.715, -74.005);
    write("Binomial kriging prediction", pred);
  });

  document.getElementById("runQuick").addEventListener("click", () => {
    try {
      const options = readQuickOptions();
      const sampleLats = [0.0, 0.5, 1.0];
      const sampleLons = [0.0, 0.5, 1.0];
      const values = [2.0, 3.0, 4.0];
      const predLats = [0.75, 0.25];
      const predLons = [0.75, 0.25];

      const ordinaryFit = fitOrdinaryVariogram(
        sampleLats,
        sampleLons,
        values,
        options.maxDistance,
        options.nBins,
        options.variogramTypes,
      );
      const ordinaryModel = new WasmOrdinaryKriging(
        sampleLats,
        sampleLons,
        values,
        ordinaryFit.variogramType ?? ordinaryFit.variogram_type,
        ordinaryFit.nugget,
        ordinaryFit.sill,
        ordinaryFit.range,
      );
      const ordinaryOut = ordinaryModel.predictBatch(predLats, predLons);
      ordinaryModel.free();

      const sampleSuccesses = [3, 6, 8];
      const sampleTrials = [10, 10, 10];
      const sampleLogits = buildBinomialLogits(
        sampleSuccesses,
        sampleTrials,
        options.alpha,
        options.beta,
      );
      const binomialFit = fitOrdinaryVariogram(
        sampleLats,
        sampleLons,
        sampleLogits,
        options.maxDistance,
        options.nBins,
        options.variogramTypes,
      );
      const binomialModel = WasmBinomialKriging.newWithPrior(
        sampleLats,
        sampleLons,
        sampleSuccesses,
        sampleTrials,
        binomialFit.variogramType ?? binomialFit.variogram_type,
        binomialFit.nugget,
        binomialFit.sill,
        binomialFit.range,
        options.alpha,
        options.beta,
      );
      const binomialOut = binomialModel.predictBatch(predLats, predLons);
      binomialModel.free();
      write("Fitted pipeline output", { ordinaryOut, binomialOut, options });
    } catch (err) {
      write("Fitted pipeline demo failed", {
        message: err?.message ?? String(err),
      });
    }
  });

  document.getElementById("runSurface").addEventListener("click", async () => {
    const resolution = Number(gridResolution.value);
    const krigingType = surfaceKrigingType.value;
    const layerMode = surfaceLayer.value;
    const residualDisplayMode = residualMode.value;

    try {
      SURFACE_SAMPLE = generateSurfaceSamples();
      const options = readQuickOptions();
      const backend = resolveBackendMode(surfaceBackend.value);
      const grid = buildPredictionGrid(
        SURFACE_SAMPLE.lats,
        SURFACE_SAMPLE.lons,
        resolution,
      );
      let predictions;
      let samplePredictions;
      let observedValues;
      let metricLabel;
      let selector;

      if (krigingType === "binomial") {
        const sampleLats = Float64Array.from(SURFACE_SAMPLE.lats);
        const sampleLons = Float64Array.from(SURFACE_SAMPLE.lons);
        const sampleSuccesses = Uint32Array.from(SURFACE_SAMPLE.successes);
        const sampleTrials = Uint32Array.from(SURFACE_SAMPLE.trials);
        const sampleLogits = buildBinomialLogits(
          SURFACE_SAMPLE.successes,
          SURFACE_SAMPLE.trials,
          options.alpha,
          options.beta,
        );
        const fitted = fitOrdinaryVariogram(
          sampleLats,
          sampleLons,
          sampleLogits,
          options.maxDistance,
          options.nBins,
          options.variogramTypes,
        );
        const variogramType = fitted.variogramType ?? fitted.variogram_type;
        if (typeof variogramType !== "string") {
          throw new Error("fitOrdinaryVariogram returned invalid variogramType");
        }
        const model = WasmBinomialKriging.newWithPrior(
          sampleLats,
          sampleLons,
          sampleSuccesses,
          sampleTrials,
          variogramType,
          fitted.nugget,
          fitted.sill,
          fitted.range,
          options.alpha,
          options.beta,
        );
        predictions = backend.useGpu
          ? await model.predictBatchGpu(grid.predLats, grid.predLons)
          : model.predictBatch(grid.predLats, grid.predLons);
        metricLabel = "prevalence";
        selector = (entry) => entry.prevalence;
        observedValues = SURFACE_SAMPLE.successes.map(
          (successes, i) => successes / SURFACE_SAMPLE.trials[i],
        );
        samplePredictions = backend.useGpu
          ? await model.predictBatchGpu(sampleLats, sampleLons)
          : model.predictBatch(sampleLats, sampleLons);
        model.free();
      } else {
        const sampleLats = Float64Array.from(SURFACE_SAMPLE.lats);
        const sampleLons = Float64Array.from(SURFACE_SAMPLE.lons);
        const sampleValues = Float64Array.from(SURFACE_SAMPLE.values);
        const fitted = fitOrdinaryVariogram(
          sampleLats,
          sampleLons,
          sampleValues,
          options.maxDistance,
          options.nBins,
          options.variogramTypes,
        );
        const variogramType = fitted.variogramType ?? fitted.variogram_type;
        if (typeof variogramType !== "string") {
          throw new Error("fitOrdinaryVariogram returned invalid variogramType");
        }
        const model = new WasmOrdinaryKriging(
          sampleLats,
          sampleLons,
          sampleValues,
          variogramType,
          fitted.nugget,
          fitted.sill,
          fitted.range,
        );
        predictions = backend.useGpu
          ? await model.predictBatchGpu(grid.predLats, grid.predLons)
          : model.predictBatch(grid.predLats, grid.predLons);
        metricLabel = "value";
        selector = (entry) => entry.value;
        observedValues = SURFACE_SAMPLE.values;
        samplePredictions = backend.useGpu
          ? await model.predictBatchGpu(sampleLats, sampleLons)
          : model.predictBatch(sampleLats, sampleLons);
        model.free();
      }

      if (!Array.isArray(predictions) || predictions.length !== resolution * resolution) {
        throw new Error("Unexpected surface output size from predictBatch");
      }
      if (
        !Array.isArray(samplePredictions) ||
        samplePredictions.length !== observedValues.length
      ) {
        throw new Error("Unexpected sample prediction output size from predictBatch");
      }
      const residuals = samplePredictions.map(
        (entry, i) => observedValues[i] - selector(entry),
      );

      const valueRange = renderSurface(
        predictions,
        resolution,
        grid.bounds,
        SURFACE_SAMPLE,
        observedValues,
        selector,
        layerMode,
      );
      const residualStats = renderResidualPlot(residuals, residualDisplayMode);
      const variogramPoints = computeEmpiricalVariogram(
        SURFACE_SAMPLE.lats,
        SURFACE_SAMPLE.lons,
        observedValues,
        options.nBins,
      );
      const variogramStats = renderVariogramPlot(variogramPoints, options.variogramTypes[0]);
      write("2D kriging surface", {
        krigingType,
        metric: metricLabel,
        layerMode,
        residualMode: residualDisplayMode,
        variogramCandidates: options.variogramTypes,
        nBins: options.nBins,
        maxDistanceKm: options.maxDistance ?? "auto",
        backendSelection: surfaceBackend.value,
        backendResolved: backend.useGpu ? "webgpu" : "cpu",
        binomialPrior: { alpha: options.alpha, beta: options.beta },
        gridResolution: `${resolution}x${resolution}`,
        predictionPoints: predictions.length,
        bounds: grid.bounds,
        valueRange,
        pointsUsed: SURFACE_SAMPLE.values.length,
        residualStats,
        variogramStats,
      });
    } catch (err) {
      drawSurfacePlaceholder();
      drawResidualPlaceholder();
      drawVariogramPlaceholder();
      surfaceMin.textContent = "-";
      surfaceMax.textContent = "-";
      residualMean.textContent = "-";
      residualRmse.textContent = "-";
      write("2D surface demo failed", {
        message: err?.message ?? String(err),
      });
    }
  });

  document.getElementById("runPerformanceHarness").addEventListener("click", async () => {
    try {
      const options = readQuickOptions();
      const report = await runOrdinaryPerformanceHarness(options, surfaceBackend.value);
      write("Browser performance harness", report);
    } catch (err) {
      write("Browser performance harness failed", {
        message: err?.message ?? String(err),
      });
    }
  });
}

main().catch((err) => {
  write("Failed to initialize WASM module", {
    message: err?.message ?? String(err),
  });
});
