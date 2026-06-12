/**
 * Unified sequential Gaussian simulation (SGS) entry point for every kriging variant.
 *
 * Pass `{ geometry, family, conditioning…, target…, variogram, seed }`.
 * Set `nRealizations > 1` for ensemble output (`baseSeed` or `seed` seeds row `k`).
 *
 * @module
 */

import { KrigingError, wrapThrown } from "./errors.js";
import {
  requireFloat64Array,
  requireNumber,
} from "./internal/convert.js";
import { requireLoadedModule } from "./internal/module.js";
import { packSimulateOptions } from "./internal/unified-boundary.js";
import type {
  BinomialSimulationManyResult,
  BinomialSimulationResult,
  SimulateOptionsInput,
} from "./types.js";

function mapBinomialSimulation(raw: unknown): BinomialSimulationResult {
  const obj = raw as Record<string, unknown>;
  return {
    logitSamples: requireFloat64Array(obj.logitSamples),
    prevalenceSamples: requireFloat64Array(obj.prevalenceSamples),
  };
}

function mapBinomialSimulationMany(raw: unknown): BinomialSimulationManyResult {
  const obj = raw as Record<string, unknown>;
  return {
    nRealizations: requireNumber(obj.nRealizations),
    nTargets: requireNumber(obj.nTargets),
    logitSamples: requireFloat64Array(obj.logitSamples),
    prevalenceSamples: requireFloat64Array(obj.prevalenceSamples),
  };
}

/**
 * Sequential Gaussian simulation keyed by `geometry` and `family`.
 *
 * Returns a `Float64Array` for continuous families, or binomial logit/prevalence samples.
 * When `nRealizations > 1`, continuous output is flat row-major; binomial output includes
 * `nRealizations` and `nTargets`.
 */
export function simulate(
  options: SimulateOptionsInput
): Float64Array | BinomialSimulationResult | BinomialSimulationManyResult {
  const mod = requireLoadedModule();
  const family = options.family ?? "ordinary";
  try {
    const raw = mod.simulate(packSimulateOptions(options));
    if (family === "binomial") {
      return options.nRealizations !== undefined && options.nRealizations > 1
        ? mapBinomialSimulationMany(raw)
        : mapBinomialSimulation(raw);
    }
    return requireFloat64Array(raw);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/** Single-realization SGS — {@link simulate} with `nRealizations` unset. */
export function conditionalSimulate(
  options: SimulateOptionsInput
): Float64Array | BinomialSimulationResult {
  if (options.nRealizations !== undefined && options.nRealizations > 1) {
    throw new KrigingError(
      "conditionalSimulate expects a single realization; use simulate() or omit nRealizations",
      { code: "invalid_input" }
    );
  }
  return simulate(options) as Float64Array | BinomialSimulationResult;
}

/** Multi-realization SGS — {@link simulate} with `nRealizations >= 1`. */
export function conditionalSimulateMany(
  options: SimulateOptionsInput & { nRealizations: number }
): Float64Array | BinomialSimulationManyResult {
  if (options.nRealizations < 1) {
    throw new KrigingError("nRealizations must be >= 1", {
      code: "invalid_input",
    });
  }
  return simulate(options) as Float64Array | BinomialSimulationManyResult;
}
