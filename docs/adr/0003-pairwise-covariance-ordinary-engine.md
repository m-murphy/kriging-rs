# Pairwise covariance as the ordinary kriging engine parameter

**Status:** accepted (2026-08-14)

ADR-0001 unified geographic and projected ordinary kriging on `OrdinaryKrigingEngine<M: SpatialMetric>` and claimed that would also replace the space-time solver. Distance was the wrong seam: the clone was how `C_ij` is built (`VariogramModel.covariance(d)` vs `SpaceTimeVariogram.covariance(hs, ht)`), not how spatial distance is measured. The **ordinary kriging engine** is parameterized by **pairwise covariance**. Two adapters justify the seam: 2-D (spatial metric + semivariogram model) and spatio-temporal (space-time metric + space-time variogram). `SpaceTimeOrdinaryKrigingEngine` is a type alias. Simple and universal kriging use the same adapters (universal also takes a [trend basis](0004-trend-basis-universal-engine.md)). A space-time metric trait is not introduced — only one time-distance function exists.

## Considered options

- **Parameterize the engine by a space-time metric** (`time_distance` on `SpatialMetric`). Rejected: unifies distance, leaves two engines and two covariance fills.
- **Shared helpers, two engine types.** Rejected: deleting the helpers just moves the clone.
- **Sugar `fit(metric, coords, variogram)` on the engine.** Rejected: leaks spatial domain back across the seam; callers construct the adapter.

## Consequences

- Simple and universal kriging consume the same pairwise covariance adapters without a second covariance fill. Universal's remaining clone was the design-matrix fill; see ADR-0004.
- Neighborhood search stays on `OrdinaryKrigingModel` (spatial metric), not on the engine.
- Completes the unification ADR-0001 described; does not reopen the dual SPD formulation.
