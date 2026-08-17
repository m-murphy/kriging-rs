# Dual SPD formulation for ordinary kriging

**Status:** accepted (2026-06-12) — gates the 0.5 release.

## Decision

Ordinary, universal, and (by composition) binomial kriging are reformulated to solve on the symmetric **positive-definite** covariance block `C` plus a precomputed constraint vector `β = C⁻¹·1`, instead of factoring the symmetric-indefinite bordered system `[C 1; 1ᵀ 0]` with LU.

Predictions are mathematically equivalent in exact arithmetic; floating-point results may differ at the last few ULPs in `f32`.

## Why

- **One factorization path.** Cholesky on `C` replaces LU on the bordered system. The current `cholesky_update.rs` primitive (orphaned today) becomes load-bearing.
- **Incremental conditioning.** Adding a new conditioning site is a rank-1 / bordered SPD extension in O(n²) via `cholesky_extend_spd_lower`. Sequential Gaussian simulation stops rebuilding the full model per target.
- **Numerical conditioning.** The bordered system's zero corner couples to the magnitude of `C`; the SPD block alone has tighter spectral bounds.
- **Unification.** One dual-SPD engine replaces the three parallel solvers in `src/kriging/ordinary.rs`, `src/projected.rs`, and `src/spacetime/kriging/ordinary.rs`. ADR-0001 parameterized that engine by `SpatialMetric`; [ADR-0003](0003-pairwise-covariance-ordinary-engine.md) moved the seam to **pairwise covariance**.

## Considered alternatives

- **Keep LU bordered, two factorization paths in the conditioner.** Rejected: leaves duplication exactly where the refactor is aimed, and the conditioner becomes two-headed forever.
- **SPD-only conditioner for simple kriging.** Rejected: ordinary/binomial SGS keeps the O(n³) rebuild and the headline performance win stays mostly hypothetical.

## Consequences

- **Breaking release.** 0.4 → 0.5. CHANGELOG must call out small numerical drift in `f32` predictions and document the formulation switch.
- **Side-by-side validation phase recommended** in CI for one release cycle: dual and bordered run on the same inputs and the max relative drift is asserted within a documented bound.
- The LU path in `matrix.rs` was deleted; bordered LU remains only as `predict_bordered_lu` for ADR-0001 regression tests.
- **Prediction wrappers** (`SimpleKrigingModel`, `UniversalKrigingModel`, space–time analogues) now delegate to the Cholesky engines; bordered LU is removed from these paths.
