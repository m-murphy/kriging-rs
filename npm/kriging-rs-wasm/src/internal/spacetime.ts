/**
 * Internal: helpers for packing the space-time variogram parameters that all
 * `WasmSpaceTime*` factories accept.
 *
 * @module
 */

import type {
  SpaceTimeUniversalTrend,
  SpaceTimeVariogramFamily,
  SpaceTimeVariogramParams,
} from "../types.js";

/**
 * Positional layout of the space-time variogram arguments in the WASM `fromArrays`
 * factories. Keeping this in one place so adding a future parameter only touches the
 * wrapper glue.
 */
export interface SpaceTimeVariogramArgs {
  family: SpaceTimeVariogramFamily;
  spatialType: string;
  spatialNugget: number;
  spatialSill: number;
  spatialRange: number;
  spatialShape: number | undefined;
  temporalType: string;
  temporalNugget: number;
  temporalSill: number;
  temporalRange: number;
  temporalShape: number | undefined;
  k1: number | undefined;
  k2: number | undefined;
  k3: number | undefined;
}

export function packSpaceTimeVariogram(
  variogram: SpaceTimeVariogramParams
): SpaceTimeVariogramArgs {
  return {
    family: variogram.family,
    spatialType: variogram.spatial.variogramType,
    spatialNugget: variogram.spatial.nugget,
    spatialSill: variogram.spatial.sill,
    spatialRange: variogram.spatial.range,
    spatialShape: variogram.spatial.shape,
    temporalType: variogram.temporal.variogramType,
    temporalNugget: variogram.temporal.nugget,
    temporalSill: variogram.temporal.sill,
    temporalRange: variogram.temporal.range,
    temporalShape: variogram.temporal.shape,
    k1: variogram.k1,
    k2: variogram.k2,
    k3: variogram.k3,
  };
}

/** Validate the universal trend string. Throws `Error` when unknown. */
export function requireSpaceTimeUniversalTrend(
  trend: SpaceTimeUniversalTrend
): SpaceTimeUniversalTrend {
  switch (trend) {
    case "constant":
    case "linearInTime":
    case "quadraticInTime":
    case "linearInSpace":
    case "linearInSpaceAndTime":
    case "quadraticInSpaceAndTime":
      return trend;
    default: {
      const value: string = trend;
      throw new Error(`Unknown space-time universal trend: ${value}`);
    }
  }
}
