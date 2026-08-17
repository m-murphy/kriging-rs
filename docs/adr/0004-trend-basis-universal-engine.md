# Trend basis as the universal kriging engine's second parameter

**Status:** accepted (2026-08-16)

[ADR-0003](0003-pairwise-covariance-ordinary-engine.md) parameterized ordinary (and then simple) kriging by pairwise covariance. Universal kriging still cloned a second engine for space-time because the design-matrix fill (`F`) is not a covariance problem: geographic polynomial trends evaluate `(lat, lon)` while space-time trends evaluate `(s1, s2, t)` through [`SpatialBasis`](../../src/spacetime/metric.rs).

The **universal kriging engine** is parameterized by **pairwise covariance** plus **trend basis**. Two adapters justify the trend seam: geographic polynomial (`UniversalTrend`) and space-time (`SpaceTimeTrendEval` wrapping `SpaceTimeUniversalTrend`). `SpaceTimeUniversalKrigingEngine` is a type alias. Constant-trend models still delegate to the ordinary engine; that specialization is a product choice, not a leftover solver.

## Considered options

- **Parameterize only by pairwise covariance; keep two `eval_basis` fills.** Rejected: leaves the dual-SPD / Schur / `condition` clone that ADR-0003 already removed from ordinary and simple.
- **One trend enum covering space and space-time.** Rejected: the public geographic and space-time trend vocabularies are different products (`Linear` vs `LinearInTime` / `LinearInSpaceAndTime`).
- **Sugar `fit(metric, …, trend)` on the engine.** Rejected for the same reason as ADR-0003: callers construct adapters.

## Consequences

- Universal kriging consumes the same pairwise covariance adapters as ordinary and simple.
- Neighborhood search stays on 2-D models, not on the engine.
- Does not reopen the dual SPD formulation (ADR-0001).
