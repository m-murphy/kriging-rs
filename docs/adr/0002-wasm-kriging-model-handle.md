# WASM kriging model handle and unified CV/simulation seam

**Status:** accepted (2026-06-12) — gates the 0.4 npm release.

## Decision

Collapse the geometry × kriging-family cartesian product at the WASM/TypeScript boundary into:

1. **`WasmKrigingModel`** — one tagged handle wrapping every fitted model variant. One `free()`, shared `predict` / `predictBatch` / `predictGrid` dispatch, and instance `leaveOneOut()` / `kFold(k)`.
2. **Unified stateless entry points** — `cv(options)` and `simulate(options)` (plus `simulateMany` for ensemble paths) keyed by `geometry` + `family` strings, for callers that skip building a model.
3. **0.4 breaking change** — remove the 20+ named `leaveOneOut*` / `kFold*` / `conditionalSimulate*` exports. TypeScript model classes (`OrdinaryKriging`, etc.) become thin adapters over `WasmKrigingModel`; they gain instance CV methods.

Rust internals already use [`KrigingPredictor`](../../src/predictor/cv.rs) and [`KrigingSimulator`](../../src/predictor/simulation.rs); this ADR completes the migration at the foreign seam that the Rust-side predictor harness refactor deliberately left unchanged.

## Why

- **Locality.** CV fold loops and SGS harness logic live in one Rust module; the WASM layer stops re-enumerating combinations.
- **Leverage.** Callers learn one CV interface and one model handle instead of a README table of 40+ exports.
- **Deletion test.** Removing a named export concentrates complexity into dispatch `match` arms — the arms belong behind one interface, not at N JS call sites.

## Considered alternatives

- **Additive deprecation (keep named exports one release).** Rejected: the cartesian product is the friction; carrying both surfaces doubles maintenance.
- **Rust-only internal refactor, unchanged npm exports.** Rejected: leaves the widest shallow seam intact; doesn't improve caller leverage.
- **Stateless `cv()` only, no model-handle methods.** Rejected: model classes are the primary ergonomic path; scripts still need the unified function.

## Consequences

- **npm 0.4.0** breaking release. CHANGELOG and migration section required.
- **`types.ts` / `wasm-shapes.ts`** shrink as dual CV/simulation option types collapse.
- **Projected geometry migration (review candidate 5)** is already complete — `ProjectedKrigingModel` uses `OrdinaryKrigingEngine<ProjectedMetric>`; no additional solver work needed for this ADR.
