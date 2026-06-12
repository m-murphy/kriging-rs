/**
 * Estimate a Beta prior (α, β) from pooled binomial counts using the same heuristic
 * as the Rust crate (`estimate_binomial_prior_from_counts`).
 *
 * @module
 */

import { wrapThrown } from "./errors.js";
import { asRecord, requireNumber, toUint32Array } from "./internal/convert.js";
import { requireLoadedModule } from "./internal/module.js";
import type { BinomialPriorParams, IntegerArrayInput } from "./types.js";

/**
 * @param options - Equal-length `successes` and `trials` arrays (same semantics as
 *   {@link BinomialKriging} / {@link fitBinomialVariogram}).
 * @returns Estimated `alpha` and `beta`, both finite and strictly positive.
 */
export function estimateBinomialPrior(options: {
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
}): BinomialPriorParams {
  const mod = requireLoadedModule();
  if (typeof mod.estimateBinomialPrior !== "function") {
    throw new Error(
      "estimateBinomialPrior is not available; rebuild the WASM package"
    );
  }
  let out: unknown;
  try {
    out = mod.estimateBinomialPrior(
      toUint32Array(options.successes),
      toUint32Array(options.trials)
    );
  } catch (e) {
    throw wrapThrown(e);
  }
  const r = asRecord(out);
  return {
    alpha: requireNumber(r.alpha),
    beta: requireNumber(r.beta),
  };
}
