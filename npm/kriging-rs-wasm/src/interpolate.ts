/**
 * One-shot "fit → build → predict on grid → free" helpers for ordinary and binomial
 * kriging. The intermediate model is created internally and freed before returning,
 * so callers do not need to manage resource lifetimes.
 *
 * @module
 */

import { toUint32Array } from "./internal/convert.js";
import { BinomialKriging } from "./kriging/binomial.js";
import { OrdinaryKriging } from "./kriging/ordinary.js";
import type {
  BinomialGridOutput,
  InterpolateBinomialToGridOptions,
  InterpolateOrdinaryToGridOptions,
  OrdinaryGridOutput,
} from "./types.js";
import { fitVariogram } from "./variogram.js";

/**
 * One-shot ordinary kriging: fit variogram from sample data, build model, predict on a
 * rectangular grid, then free the model. Returns 2D value and variance grids.
 *
 * @param options - Sample lats, lons, values; grid bounds and xCells/yCells; variogramType; optional nBins, maxDistance, nuggetOverride
 * @returns { values, variances } as number[][], shape [yCells][xCells]
 */
export function interpolateOrdinaryToGrid(
  options: InterpolateOrdinaryToGridOptions
): OrdinaryGridOutput {
  const fitted = fitVariogram({
    sampleLats: options.lats,
    sampleLons: options.lons,
    values: options.values,
    variogramType: options.variogramType,
    nBins: options.nBins,
    maxDistance: options.maxDistance,
  });
  const model = OrdinaryKriging.fromFitted({
    lats: options.lats,
    lons: options.lons,
    values: options.values,
    fittedVariogram: fitted,
    nuggetOverride: options.nuggetOverride,
  });
  try {
    return model.predictGrid({
      west: options.west,
      south: options.south,
      east: options.east,
      north: options.north,
      xCells: options.xCells,
      yCells: options.yCells,
    });
  } finally {
    model.free();
  }
}

/**
 * One-shot binomial kriging: fit variogram on logit(prevalence), build model, predict on a
 * rectangular grid, then free the model. Returns 2D prevalence and variance grids.
 *
 * @param options - Sample lats, lons, successes, trials; grid bounds and xCells/yCells; variogramType; optional nBins, nuggetOverride, prior
 * @returns { prevalences, logitValues, variances } as number[][], shape [yCells][xCells]
 */
export function interpolateBinomialToGrid(
  options: InterpolateBinomialToGridOptions
): BinomialGridOutput {
  const s = toUint32Array(options.successes);
  const t = toUint32Array(options.trials);
  const logits: number[] = [];
  for (let i = 0; i < s.length; i++) {
    const p = t[i] > 0 ? s[i] / t[i] : 0.5;
    const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p));
    logits.push(Math.log(clamped / (1 - clamped)));
  }
  const fitted = fitVariogram({
    sampleLats: options.lats,
    sampleLons: options.lons,
    values: logits,
    variogramType: options.variogramType,
    nBins: options.nBins,
  });
  const model = options.prior
    ? BinomialKriging.fromFittedVariogramWithPrior({
        lats: options.lats,
        lons: options.lons,
        successes: options.successes,
        trials: options.trials,
        fittedVariogram: fitted,
        nuggetOverride: options.nuggetOverride,
        prior: options.prior,
      })
    : BinomialKriging.fromFittedVariogram({
        lats: options.lats,
        lons: options.lons,
        successes: options.successes,
        trials: options.trials,
        fittedVariogram: fitted,
        nuggetOverride: options.nuggetOverride,
      });
  try {
    return model.predictGrid({
      west: options.west,
      south: options.south,
      east: options.east,
      north: options.north,
      xCells: options.xCells,
      yCells: options.yCells,
    });
  } finally {
    model.free();
  }
}
