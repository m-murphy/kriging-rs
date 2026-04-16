/**
 * Internal: helpers for normalizing the binomial Beta prior shape used by CV and SGS.
 *
 * The public API accepts an optional {@link BinomialPriorParams} object; the WASM
 * layer historically takes two `number | undefined` scalars. This helper keeps the
 * boundary conversion in one place so each call site just unpacks `{alpha, beta}`.
 *
 * @module
 */

import type { BinomialPriorParams } from "../types.js";

/**
 * Resolve a user-supplied prior object to the positional `(alpha, beta)` form the
 * WASM functions expect. Returns `undefined` for both fields when `prior` is
 * omitted so the Rust side falls back to its default Beta(1/2, 1/2).
 */
export function resolveBinomialPrior(
  prior: BinomialPriorParams | undefined
): { alpha: number | undefined; beta: number | undefined } {
  return { alpha: prior?.alpha, beta: prior?.beta };
}
