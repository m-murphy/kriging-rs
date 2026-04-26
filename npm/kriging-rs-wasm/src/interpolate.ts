/**
 * One-shot "fit → build → predict on grid → free" helpers for ordinary and binomial
 * kriging. The intermediate model is created internally and freed before returning,
 * so callers do not need to manage resource lifetimes.
 *
 * @module
 */

import { kFoldBinomial, leaveOneOutBinomial } from "./cv.js";
import {
  resolveBinomialPriorOrDefault,
  smoothedLogits,
} from "./internal/prior.js";
import { BinomialKriging } from "./kriging/binomial.js";
import { OrdinaryKriging } from "./kriging/ordinary.js";
import type {
  BinomialBuildNotes,
  BinomialCvSummary,
  BinomialGridOutput,
  InterpolateBinomialToGridOptions,
  InterpolateBinomialToGridResult,
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
 * One-shot binomial kriging: fit a variogram on the same EB-smoothed logits the
 * binomial kriger interpolates internally, build a model, predict prevalence on
 * a rectangular grid, then free the model. Returns 2D prevalence, logit, and
 * variance grids.
 *
 * The variogram fit and the kriging system both see
 * `logit((sᵢ + α)/(nᵢ + α + β))` per station, with `α/β` from the supplied
 * `prior` (default Beta(1, 1)) — so the fitted nugget/sill/range are calibrated
 * for the field the model actually solves on. Pass `estimator: "cressie-hawkins"`
 * for a robust fit when counts are small / heavy-tailed.
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
  const successesArr = Array.from(options.successes as ArrayLike<number>);
  const trialsArr = Array.from(options.trials as ArrayLike<number>);
  const prior = resolveBinomialPriorOrDefault(options.prior);
  const logits = smoothedLogits(successesArr, trialsArr, prior);

  const fitted = fitVariogram({
    sampleLats: options.lats,
    sampleLons: options.lons,
    values: logits,
    variogramType: options.variogramType,
    nBins: options.nBins,
    maxDistance: options.maxDistance,
    estimator: options.estimator,
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
    buildNotes = model.getBuildNotes();
  } finally {
    model.free();
  }

  const result: InterpolateBinomialToGridResult = {
    ...grids,
    fittedVariogram: fitted,
    buildNotes,
  };

  const cvSummary = runBinomialCvIfRequested(options, fitted);
  if (cvSummary !== undefined) {
    result.cv = cvSummary;
  }
  return result;
}

function runBinomialCvIfRequested(
  options: InterpolateBinomialToGridOptions,
  fittedVariogram: InterpolateBinomialToGridResult["fittedVariogram"]
): BinomialCvSummary | undefined {
  const withCv = options.withCv;
  if (!withCv) return undefined;

  const variogram = {
    variogramType: fittedVariogram.variogramType,
    nugget: options.nuggetOverride ?? fittedVariogram.nugget,
    sill: fittedVariogram.sill,
    range: fittedVariogram.range,
    shape: fittedVariogram.shape,
  };

  if (withCv === true || withCv === "loo") {
    return leaveOneOutBinomial({
      lats: options.lats,
      lons: options.lons,
      successes: options.successes,
      trials: options.trials,
      variogram,
      prior: options.prior,
    }).summary;
  }
  return kFoldBinomial({
    lats: options.lats,
    lons: options.lons,
    successes: options.successes,
    trials: options.trials,
    variogram,
    k: withCv.k,
    prior: options.prior,
  }).summary;
}
