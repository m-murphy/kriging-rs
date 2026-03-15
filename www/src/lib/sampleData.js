/**
 * Generate synthetic sample data for surface demo (ordinary + binomial).
 */
export function generateSurfaceSamples(sampleCount = 324) {
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

  return { lats, lons, values, successes, trials };
}

export function buildPredictionGrid(lats, lons, resolution) {
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

  return { predLats, predLons, bounds: { latMin, latMax, lonMin, lonMax } };
}

export function buildBinomialLogits(successes, trials, alpha, beta) {
  const logits = new Float64Array(successes.length);
  for (let i = 0; i < successes.length; i += 1) {
    const s = successes[i];
    const n = trials[i];
    const p = (s + alpha) / (n + alpha + beta);
    logits[i] = Math.log(p / (1 - p));
  }
  return logits;
}

export function resolveBackendMode(selection, webgpuAvailable) {
  if (selection === "cpu") return { useGpu: false, mode: "cpu" };
  if (selection === "webgpu") {
    if (!webgpuAvailable) {
      throw new Error("WebGPU required but unavailable");
    }
    return { useGpu: true, mode: "webgpu" };
  }
  if (selection === "auto") {
    return { useGpu: webgpuAvailable, mode: "auto" };
  }
  return { useGpu: false, mode: "cpu" };
}

export function summarizeTimings(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const count = sorted.length;
  if (count === 0) return { meanMs: 0, p50Ms: 0, p95Ms: 0, minMs: 0, maxMs: 0 };
  const idx50 = Math.floor((count - 1) * 0.5);
  const idx95 = Math.floor((count - 1) * 0.95);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  return {
    meanMs: sum / count,
    p50Ms: sorted[idx50],
    p95Ms: sorted[idx95],
    minMs: sorted[0],
    maxMs: sorted[count - 1],
  };
}

export function summarizePhaseShares(phaseTotals, totalValues) {
  const totalSum = totalValues.reduce((acc, v) => acc + v, 0);
  if (totalSum <= 0) {
    return Object.fromEntries(
      Object.entries(phaseTotals).map(([key]) => [key, { percent: 0, meanMs: 0 }]),
    );
  }
  return Object.fromEntries(
    Object.entries(phaseTotals).map(([key, values]) => {
      const meanMs = values.reduce((a, v) => a + v, 0) / values.length;
      const percent = (values.reduce((a, v) => a + v, 0) / totalSum) * 100;
      return [key, { meanMs, percent }];
    }),
  );
}
