import { describe, expect, test } from "vitest";

import { packSimulateOptions } from "../src/internal/unified-boundary.js";
import type { SimulateOptionsInput } from "../src/types.js";

describe("simulation option packing", () => {
  test("preserves legacy binomial count aliases", () => {
    const packed = packSimulateOptions({
      geometry: "geo",
      family: "binomial",
      conditioningLats: [0, 1],
      conditioningLons: [0, 1],
      successes: new Uint32Array([1, 2]),
      trials: new Uint32Array([10, 10]),
      targetLats: [0.5],
      targetLons: [0.5],
      variogram: {
        variogramType: "exponential",
        nugget: 0.01,
        sill: 1,
        range: 100,
      },
      seed: 42,
    } as unknown as SimulateOptionsInput);

    expect(packed.conditioningSuccesses).toEqual([1, 2]);
    expect(packed.conditioningTrials).toEqual([10, 10]);
  });
});
