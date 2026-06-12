/**
 * One-shot "fit → build → predict on grid → free" helpers for ordinary and binomial
 * kriging. The intermediate model is created internally and freed before returning,
 * so callers do not need to manage resource lifetimes.
 *
 * @module
 */

import { cv } from "./cv.js";
import { resolveBinomialPriorInput } from "./internal/prior.js";
import { BinomialKriging } from "./kriging/binomial.js";
import { OrdinaryKriging } from "./kriging/ordinary.js";
import type {
  BinomialBuildNotes,
  BinomialCvSummary,
  BinomialGridOutput,
  BinomialPriorParams,
  InterpolateBinomialToGridOptions,
  InterpolateBinomialToGridResult,
  InterpolateOrdinaryToGridOptions,
  OrdinaryGridOutput,
} from "./types.js";
import { fitBinomialVariogram, fitVariogram } from "./variogram.js";

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
 * One-shot binomial kriging: fit a variogram on the same EB-smoothed logits the
 * binomial kriger interpolates internally, build a model, predict prevalence on
 * a rectangular grid, then free the model. Returns 2D prevalence, logit, and
 * variance grids.
 *
 * The variogram is fit with {@link fitBinomialVariogram}: noise-calibrated empirical
 * variogram on EB-smoothed logits with the same `prior` (default Beta(1, 1)) as the
 * kriging model. Pass `prior: { alpha, beta }` for a fixed prior, or `prior: "auto"`
 * to estimate **α**, **β** from counts ({@link estimateBinomialPrior}).
 *
 * Pass `withCv` (boolean, `"loo"`, or `{ k }`) to additionally run binomial
 * cross-validation against the fitted variogram and include a
 * {@link BinomialCvSummary} on the returned object.
 *
 * @returns A {@link InterpolateBinomialToGridResult} carrying the prevalence /
 * logit / variance grids, the fitted variogram, and (when requested) a CV
 * summary.
 */
export function interpolateBinomialToGrid(
  options: InterpolateBinomialToGridOptions
): InterpolateBinomialToGridResult {
  const priorResolved = resolveBinomialPriorInput(options.prior, {
    successes: options.successes,
    trials: options.trials,
  });

  const fitted = fitBinomialVariogram({
    sampleLats: options.lats,
    sampleLons: options.lons,
    successes: options.successes,
    trials: options.trials,
    variogramType: options.variogramType,
    nBins: options.nBins,
    maxDistance: options.maxDistance,
    prior: priorResolved,
    relWeightEps: options.relWeightEps,
  });

  const model =
    priorResolved !== undefined
      ? BinomialKriging.fromFittedVariogramWithPrior({
          lats: options.lats,
          lons: options.lons,
          successes: options.successes,
          trials: options.trials,
          fittedVariogram: fitted,
          prior: priorResolved,
          ...(options.stability !== undefined ? { stability: options.stability } : {}),
          ...(options.oneStepLaplaceObservationVariance === true
            ? { oneStepLaplaceObservationVariance: true }
            : {}),
        })
      : BinomialKriging.fromFittedVariogram({
          lats: options.lats,
          lons: options.lons,
          successes: options.successes,
          trials: options.trials,
          fittedVariogram: fitted,
          ...(options.stability !== undefined ? { stability: options.stability } : {}),
          ...(options.oneStepLaplaceObservationVariance === true
            ? { oneStepLaplaceObservationVariance: true }
            : {}),
        });

  let grids: BinomialGridOutput;
  let buildNotes: BinomialBuildNotes;
  try {
    grids = model.predictGrid({
      west: options.west,
      south: options.south,
      east: options.east,
      north: options.north,
      xCells: options.xCells,
      yCells: options.yCells,
    });
    buildNotes = model.buildNotes;
  } finally {
    model.free();
  }

  const result: InterpolateBinomialToGridResult = {
    ...grids,
    fittedVariogram: fitted,
    buildNotes,
  };

  const cvSummary = runBinomialCvIfRequested(options, fitted, priorResolved);
  if (cvSummary !== undefined) {
    result.cv = cvSummary;
  }
  return result;
}

function runBinomialCvIfRequested(
  options: InterpolateBinomialToGridOptions,
  fittedVariogram: InterpolateBinomialToGridResult["fittedVariogram"],
  priorResolved: BinomialPriorParams | undefined
): BinomialCvSummary | undefined {
  const withCv = options.withCv;
  if (!withCv) return undefined;

  const variogram = {
    variogramType: fittedVariogram.variogramType,
    nugget: fittedVariogram.nugget,
    sill: fittedVariogram.sill,
    range: fittedVariogram.range,
    shape: fittedVariogram.shape,
  };

  if (withCv === true || withCv === "loo") {
    return cv({
      geometry: "geo",
      family: "binomial",
      lats: options.lats,
      lons: options.lons,
      successes: options.successes,
      trials: options.trials,
      variogram,
      prior: priorResolved,
    }).summary as BinomialCvSummary;
  }
  return cv({
    geometry: "geo",
    family: "binomial",
    lats: options.lats,
    lons: options.lons,
    successes: options.successes,
    trials: options.trials,
    variogram,
    k: withCv.k,
    prior: priorResolved,
  }).summary as BinomialCvSummary;
}
