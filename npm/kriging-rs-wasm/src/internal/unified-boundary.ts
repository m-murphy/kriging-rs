/**
 * Internal: pack unified CV / simulation option objects for the WASM serde boundary.
 *
 * @module
 */

import {
  toFloat64Array,
  toUint32Array,
} from "./convert.js";
import { resolveBinomialPriorInput } from "./prior.js";
import { packSpaceTimeVariogram } from "./spacetime.js";
import type {
  BinomialPriorInput,
  CvOptionsInput,
  IntegerArrayInput,
  NumericArrayInput,
  SimulateOptionsInput,
  SpaceTimeVariogramParams,
  UnifiedOptionsFlat,
  VariogramParams,
} from "../types.js";

function isSpaceTimeVariogram(
  value: unknown
): value is SpaceTimeVariogramParams {
  return (
    typeof value === "object" &&
    value !== null &&
    "family" in value &&
    "spatial" in value
  );
}

function resolveSpaceTimeVariogram(
  options: UnifiedOptionsFlat
): SpaceTimeVariogramParams | undefined {
  if (options.spaceTimeVariogram) {
    return options.spaceTimeVariogram;
  }
  if (
    options.geometry === "spacetime" &&
    isSpaceTimeVariogram(options.variogram)
  ) {
    return options.variogram;
  }
  return undefined;
}

function resolve2dVariogram(
  options: UnifiedOptionsFlat
): VariogramParams | undefined {
  if (options.variogram && !isSpaceTimeVariogram(options.variogram)) {
    return options.variogram;
  }
  return undefined;
}

function withGeometryFamilyDefaults(
  options: CvOptionsInput | SimulateOptionsInput
): UnifiedOptionsFlat {
  return {
    ...options,
    geometry: options.geometry ?? "geo",
    family: options.family ?? "ordinary",
  };
}

function packVariogram(variogram: VariogramParams) {
  return {
    variogramType: variogram.variogramType,
    nugget: variogram.nugget,
    sill: variogram.sill,
    range: variogram.range,
    shape: variogram.shape,
  };
}

function packSpaceTimeVariogramFields(
  spaceTimeVariogram: SpaceTimeVariogramParams
) {
  const packed = packSpaceTimeVariogram(spaceTimeVariogram);
  return {
    spaceTimeFamily: packed.family,
    spatialType: packed.spatialType,
    spatialNugget: packed.spatialNugget,
    spatialSill: packed.spatialSill,
    spatialRange: packed.spatialRange,
    spatialShape: packed.spatialShape,
    temporalType: packed.temporalType,
    temporalNugget: packed.temporalNugget,
    temporalSill: packed.temporalSill,
    temporalRange: packed.temporalRange,
    temporalShape: packed.temporalShape,
    k1: packed.k1,
    k2: packed.k2,
    k3: packed.k3,
  };
}

function packPrior(
  prior: BinomialPriorInput | undefined,
  context?: {
    successes?: IntegerArrayInput;
    trials?: IntegerArrayInput;
  }
) {
  if (prior === undefined) {
    return { priorAlpha: undefined, priorBeta: undefined };
  }
  if (prior === "auto") {
    if (context?.successes === undefined || context?.trials === undefined) {
      throw new Error(
        "prior 'auto' requires successes and trials in CV/simulation options"
      );
    }
    const resolved = resolveBinomialPriorInput(prior, {
      successes: context.successes,
      trials: context.trials,
    });
    return { priorAlpha: resolved?.alpha, priorBeta: resolved?.beta };
  }
  return { priorAlpha: prior.alpha, priorBeta: prior.beta };
}

function packNumericField(
  value: NumericArrayInput | undefined
): number[] | undefined {
  return value === undefined ? undefined : Array.from(toFloat64Array(value));
}

function packUintField(
  value: IntegerArrayInput | undefined
): number[] | undefined {
  return value === undefined ? undefined : Array.from(toUint32Array(value));
}

export function packCvOptions(options: CvOptionsInput): Record<string, unknown> {
  const normalized = withGeometryFamilyDefaults(options);
  const payload: Record<string, unknown> = {
    geometry: normalized.geometry,
    family: normalized.family,
  };
  if (normalized.k !== undefined) payload.k = normalized.k;

  const lats = packNumericField(normalized.lats);
  const lons = packNumericField(normalized.lons);
  const xs = packNumericField(normalized.xs);
  const ys = packNumericField(normalized.ys);
  const values = packNumericField(normalized.values);
  const times = packNumericField(normalized.times);
  const successes = packUintField(normalized.successes);
  const trials = packUintField(normalized.trials);

  if (lats) payload.lats = lats;
  if (lons) payload.lons = lons;
  if (xs) payload.xs = xs;
  if (ys) payload.ys = ys;
  if (values) payload.values = values;
  if (times) payload.times = times;
  if (successes) payload.successes = successes;
  if (trials) payload.trials = trials;

  const variogram2d = resolve2dVariogram(normalized);
  if (variogram2d) {
    payload.variogram = packVariogram(variogram2d);
  }
  const spaceTimeVariogram = resolveSpaceTimeVariogram(normalized);
  if (spaceTimeVariogram) {
    Object.assign(payload, packSpaceTimeVariogramFields(spaceTimeVariogram));
  }
  if (normalized.mean !== undefined) payload.mean = normalized.mean;
  if (normalized.trend !== undefined) payload.trend = normalized.trend;
  if (normalized.majorAngleDeg !== undefined) {
    payload.majorAngleDeg = normalized.majorAngleDeg;
  }
  if (normalized.rangeRatio !== undefined) {
    payload.rangeRatio = normalized.rangeRatio;
  }

  Object.assign(
    payload,
    packPrior(normalized.prior, {
      successes: normalized.successes,
      trials: normalized.trials,
    })
  );

  return payload;
}

function normalizeSeed(seed: number | bigint | undefined): number {
  if (typeof seed === "bigint") {
    return Number(seed & BigInt(Number.MAX_SAFE_INTEGER));
  }
  return seed ?? 0;
}

export function packSimulateOptions(
  options: SimulateOptionsInput
): Record<string, unknown> {
  const normalized = withGeometryFamilyDefaults(options);
  const legacy = normalized as UnifiedOptionsFlat & {
    successes?: IntegerArrayInput;
    trials?: IntegerArrayInput;
  };
  const payload: Record<string, unknown> = {
    geometry: normalized.geometry,
    family: normalized.family,
    seed: normalizeSeed(normalized.seed ?? normalized.baseSeed),
  };

  const assignNumeric = (
    key: string,
    value: NumericArrayInput | undefined
  ) => {
    const packed = packNumericField(value);
    if (packed) payload[key] = packed;
  };

  assignNumeric("conditioningLats", normalized.conditioningLats);
  assignNumeric("conditioningLons", normalized.conditioningLons);
  assignNumeric("conditioningXs", normalized.conditioningXs);
  assignNumeric("conditioningYs", normalized.conditioningYs);
  assignNumeric("conditioningTimes", normalized.conditioningTimes);
  assignNumeric("conditioningValues", normalized.conditioningValues);
  assignNumeric("targetLats", normalized.targetLats);
  assignNumeric("targetLons", normalized.targetLons);
  assignNumeric("targetXs", normalized.targetXs);
  assignNumeric("targetYs", normalized.targetYs);
  assignNumeric("targetTimes", normalized.targetTimes);

  const conditioningSuccesses = packUintField(
    normalized.conditioningSuccesses ?? legacy.successes
  );
  const conditioningTrials = packUintField(
    normalized.conditioningTrials ?? legacy.trials
  );
  if (conditioningSuccesses) payload.conditioningSuccesses = conditioningSuccesses;
  if (conditioningTrials) payload.conditioningTrials = conditioningTrials;

  const variogram2d = resolve2dVariogram(normalized);
  if (variogram2d) {
    payload.variogram = packVariogram(variogram2d);
  }
  const spaceTimeVariogram = resolveSpaceTimeVariogram(normalized);
  if (spaceTimeVariogram) {
    Object.assign(payload, packSpaceTimeVariogramFields(spaceTimeVariogram));
  }
  if (normalized.mean !== undefined) payload.mean = normalized.mean;
  if (normalized.trend !== undefined) payload.trend = normalized.trend;
  if (normalized.majorAngleDeg !== undefined) {
    payload.majorAngleDeg = normalized.majorAngleDeg;
  }
  if (normalized.rangeRatio !== undefined) {
    payload.rangeRatio = normalized.rangeRatio;
  }

  Object.assign(
    payload,
    packPrior(normalized.prior, {
      successes: normalized.conditioningSuccesses ?? legacy.successes,
      trials: normalized.conditioningTrials ?? legacy.trials,
    })
  );

  if (normalized.nRealizations !== undefined) {
    payload.nRealizations = normalized.nRealizations;
  }
  if (normalized.baseSeed !== undefined) {
    payload.baseSeed = normalizeSeed(normalized.baseSeed);
  }
  if (normalized.targetOrder !== undefined) {
    payload.targetOrder = Array.from(
      normalized.targetOrder instanceof Uint32Array
        ? normalized.targetOrder
        : Uint32Array.from(normalized.targetOrder as ArrayLike<number>)
    );
  }

  return payload;
}
