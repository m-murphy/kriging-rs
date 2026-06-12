# kriging-rs context

Vocabulary for geostatistical kriging concepts as they exist in this crate, plus the architectural seams that future readers will encounter. Use these terms exactly; aliases listed under `_Avoid_` are previous or ambient names that should not appear in new code or docs.

## Domain language

### Geostatistics

**Kriging family**:
One of the four prediction methods this crate supports: **ordinary**, **simple**, **universal**, **binomial**. Each is a kriging model under a different assumption about the mean (unknown constant, known constant, polynomial trend, logit of a probability).
_Avoid_: kriging "kind", kriging "type" (overloaded with `VariogramType`).

**Geometry**:
The spatial coordinate system a kriging model operates on. Three are supported: **geographic** (lat/lon, Haversine distance), **projected** (planar `(x, y)`, anisotropic Euclidean), and **space-time** (any spatial geometry plus a scalar time axis).
_Avoid_: backend, coordinate type, coordinate system.

**Variogram**:
The function describing how dissimilarity grows with distance. A `VariogramModel` is a parametric form (spherical, exponential, …) with nugget, sill, range, and optional shape. An *empirical* variogram is the binned-distance estimate computed from data.

**Binomial calibration**:
The default binomial kriging pipeline: empirical-Bayes Beta prior → logit working values → ordinary kriging on the logits with per-site logit observation variance on the covariance diagonal → logistic back-transform to prevalence. *Not* a full binomial-likelihood field model. Produces `BinomialBuildNotes`.
_Avoid_: "binomial kriging" (ambiguous — say "calibrated binomial" when distinguishing from a hypothetical full-likelihood path).

**Inflation retry**:
The build-time loop that doubles the per-site logit observation variance until the covariance system factorizes successfully. Owns `n_build_attempts` and `logit_inflation` in `BinomialBuildNotes`.

**Calibrated logit ordinary build**:
The shared orchestration that turns binomial counts + a variogram into an ordinary kriging model on logits: compute prior-smoothed logits, compute per-site Laplace variance, run the inflation retry loop, attach build notes. Geometry-agnostic by construction; one builder serves geographic, projected, and space-time binomial models.

**Dual SPD formulation**:
A reformulation of ordinary kriging that solves on the symmetric **positive-definite** covariance block `C` plus a precomputed constraint vector `β = C⁻¹·1`, rather than on the symmetric-indefinite bordered system `[C 1; 1ᵀ 0]`. Mathematically equivalent in exact arithmetic. Enables Cholesky factorization and incremental site addition via `cholesky_extend_spd_lower`. See ADR-0001.
_Avoid_: bordered system, Lagrangian formulation (use these only when contrasting with the dual form).

### Architectural seams

**Spatial metric**:
The trait that abstracts distance over a coordinate type. `SpatialMetric` provides `prepare(coord) -> Prepared` and `distance(a, b) -> Real`. Single source of truth for 2-D distance across geographic, projected, and the spatial part of space-time. `GeoMetric` and `ProjectedMetric` are the shipped impls; `Anisotropy2D` lives inside `ProjectedMetric` (geometric anisotropy is a property of the metric, not the variogram).
_Avoid_: distance function, distance backend, distance kernel.

**Space-time metric**:
Extends `SpatialMetric` with a scalar time distance. `SpaceTimeMetric: SpatialMetric { fn time_distance(ta, tb) -> Real; }`. Space-time variograms combine the spatial and temporal distances via separable or product-sum families.
_Avoid_: spacetime kernel.

**Ordinary kriging engine**:
The single solver behind every non-binomial 2-D and space-time ordinary kriging prediction. Parameterized by `SpatialMetric` (and `SpaceTimeMetric` for the space-time variant). Owns the dual SPD factorization, the constraint vector, batch prediction, and the jitter policy.

**Kriging conditioner**:
A live, incremental view of an ordinary kriging engine that supports `append_condition(site, value, obs_var)` in O(n²) via the SPD bordered extension (`cholesky_extend_spd_lower`). The only path that production SGS uses to add a sampled target to the conditioning pool.
_Avoid_: SGS state, simulation pool, conditioning history.

**Kriging predictor trait**:
`trait KrigingPredictor` — the cross-cutting seam consumed by `cv`. Backend structs hold borrowed conditioning data and implement `predict_fold(train, test)`; [`leave_one_out_cv`](crate::predictor::cv::leave_one_out_cv) and [`k_fold_cv`](crate::predictor::cv::k_fold_cv) drive the fold loop. Replaces the 21 hand-written CV entry points with a single generic harness.

**WASM kriging model handle**:
The single tagged WASM adapter (`WasmKrigingModel`) wrapping every fitted kriging model variant. Owns prediction dispatch, resource lifecycle (`free`), and instance cross-validation (`leaveOneOut`, `kFold`). TypeScript model classes (`OrdinaryKriging`, `BinomialKriging`, …) are thin adapters over this handle — not separate WASM types.
_Avoid_: WasmOrdinaryKriging, WasmBinomialKriging (legacy per-family WASM structs — removed in 0.6).

**Unified CV / simulation seam**:
The stateless WASM entry points `cv(options)` and `simulate(options)` keyed by `geometry` + `family`, for callers that pass raw arrays without building a model handle. Delegates to the same [`KrigingPredictor`](crate::predictor::cv::KrigingPredictor) and [`KrigingSimulator`](crate::predictor/simulation.rs) harnesses as the model-handle methods.
_Avoid_: leaveOneOutBinomialProjected, conditionalSimulateSpaceTimeUniversal (legacy named exports — removed in 0.6).

**Binomial counts**:
The geometry-free `(successes, trials)` value with `smoothed_logit(prior)` and `smoothed_probability(prior)` methods. Coordinates are paired with counts via `BinomialSite<C>` where `C: SpatialMetric::Coord`. Replaces the three observation structs.

## Relationships

- A **kriging family** runs on any **geometry**; the cartesian product was historically enumerated by hand and is now factored through the **spatial metric** seam.
- A **calibrated binomial** model is a thin wrapper over an **ordinary kriging engine** plus the **calibrated logit ordinary build**.
- **Sequential Gaussian simulation** is the **kriging conditioner** iterated over a target order.
- The **kriging predictor trait** is the read-only interface that **cv** and **simulation** see — neither knows the geometry or family directly.

## Example dialogue

> **Reviewer:** Why does spacetime binomial have its own build loop?
>
> **Author:** It doesn't — there's one *calibrated logit ordinary build* in `kriging/binomial.rs`. The space-time binomial model passes its `SpaceTimeMetric` and `SpaceTimeOrdinaryKrigingModel` into the same builder. Geometry is the only variable.
>
> **Reviewer:** And SGS — does it still rebuild the model per target?
>
> **Author:** All production SGS paths hold a *kriging conditioner* and call `append_condition` per sampled target — ordinary, simple, universal (all drift orders), and calibrated binomial across geographic, projected, and space–time geometries. The factorization extends in O(n²) via `cholesky_extend_spd_lower`. The conditioner is the only thing that touches that primitive.
>
> **Reviewer:** What about anisotropy on projected data?
>
> **Author:** Anisotropy lives inside the *spatial metric* — `ProjectedMetric { anisotropy }`. The variogram only sees pre-warped distances. There's no `Anisotropy2D` parameter on any kriging type anymore.
