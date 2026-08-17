# kriging-rs context

Vocabulary for geostatistical kriging concepts as they exist in this crate, plus the architectural seams that future readers will encounter. Use these terms exactly; aliases listed under `_Avoid_` are previous or ambient names that should not appear in new code or docs.

## Domain language

### Geostatistics

**Kriging type**:
One of the four prediction methods this crate supports: **ordinary**, **simple**, **universal**, **binomial**. Each is a kriging model under a different assumption about the mean (unknown constant, known constant, polynomial trend, logit of a probability).
_Avoid_: kriging "kind"; kriging "type" when meaning `VariogramType` (spherical, exponential, …).

**Spatial domain**:
Where the random field is defined and how distance is measured. Three are supported: **geographic** (lat/lon, Haversine / great-circle distance), **projected** (planar `(x, y)`, anisotropic Euclidean), and **spatio-temporal** (any spatial domain plus a scalar time axis).
_Avoid_: geometry (code/API tag only), backend, coordinate type.

**Semivariogram**:
The function γ(h) describing how dissimilarity grows with separation h. Related to covariance by C(h) = C(0) − γ(h). A parametric **semivariogram model** (spherical, exponential, …) has nugget, sill, range, and optional shape parameters. An **empirical semivariogram** is the binned pair-distance estimate from data.
_Avoid_: using "variogram" and "semivariogram" interchangeably without stating which is meant; `VariogramModel` is the code name for a parametric semivariogram model.

**Binomial kriging**:
Kriging of binomial proportions (successes/trials) via empirical-Bayes Beta smoothing → logit transform → ordinary kriging on the logit scale with heteroskedastic per-site observation variance → logistic back-transform to prevalence. A **logit-Gaussian random field** approach, not full binomial maximum-likelihood geostatistics. Produces `BinomialBuildNotes`.
_Avoid_: "calibrated binomial" in user-facing docs (retained in internal code names such as `BinomialCalibratedResult`).

**Variance inflation**:
The build-time loop that doubles each per-site logit observation variance until the covariance system factorizes successfully. Recorded as `n_build_attempts` and `logit_inflation` in `BinomialBuildNotes`.
_Avoid_: inflation retry.

**Logit kriging pipeline**:
The shared orchestration that turns binomial counts + a semivariogram model into an ordinary kriging model on logits: compute prior-smoothed logits, compute per-site Laplace variance, run variance inflation if needed, attach build notes. Spatial-domain-agnostic; one builder serves geographic, projected, and spatio-temporal binomial models.
_Avoid_: calibrated logit ordinary build.

**Dual SPD formulation**:
A reformulation of ordinary kriging that solves on the symmetric **positive-definite** covariance block `C` plus a precomputed constraint vector `β = C⁻¹·1`, rather than on the symmetric-indefinite bordered system `[C 1; 1ᵀ 0]`. Mathematically equivalent in exact arithmetic. Enables Cholesky factorization and incremental site addition via `cholesky_extend_spd_lower`. See ADR-0001.
_Avoid_: bordered system, Lagrangian formulation (use these only when contrasting with the dual form).

### Architectural seams

**Spatial metric**:
The trait that abstracts distance over a coordinate type. `SpatialMetric` provides `prepare(coord) -> Prepared` and `distance(a, b) -> Real`. Single source of truth for 2-D distance across geographic, projected, and the spatial part of space-time. `GeoMetric` and `ProjectedMetric` are the shipped impls; `Anisotropy2D` lives inside `ProjectedMetric` (geometric anisotropy is a property of the metric, not the variogram).
_Avoid_: distance function, distance backend, distance kernel.

**Space-time metric**:
The pair of a **spatial metric** and a scalar time distance. Space-time variograms combine those distances via separable or product-sum families. Feeds **pairwise covariance**; it does not parameterize the **ordinary kriging engine**.
_Avoid_: spacetime kernel; treating this as the ordinary engine's generic parameter.

**Pairwise covariance**:
The map from a pair of sites to an entry of C, plus C(0). C includes model jitter; per-site observation variance is not part of this seam. Spatial domain and the semivariogram model sit behind it; the ordinary, simple, and universal kriging engines do not. Two adapters exist: 2-D (spatial metric + semivariogram model) and spatio-temporal (space-time metric + space-time variogram).
_Avoid_: covariance kernel, covariance backend, distance kernel.

**Trend basis**:
The map from a site to the columns of the universal-kriging design matrix `F`. Two adapters exist: geographic polynomial (`UniversalTrend`) and space-time (`SpaceTimeTrendEval` wrapping `SpaceTimeUniversalTrend`). The **universal kriging engine** is parameterized by **pairwise covariance** plus this seam.
_Avoid_: drift functions, basis backend.

**Ordinary kriging engine**:
The single solver behind every non-binomial 2-D and space-time ordinary kriging prediction. Parameterized by **pairwise covariance**. Owns the dual SPD factorization, the constraint vector, batch prediction, and per-site observation variance on the diagonal. Neighborhood search is a 2-D model feature over the **spatial metric**, not an engine feature.

**Universal kriging engine**:
The single solver behind non-constant 2-D and space-time universal kriging. Parameterized by **pairwise covariance** and **trend basis**. Owns the dual SPD factorization of `C`, `β = C⁻¹F`, and the Schur complement on `Fᵀβ`. Constant-trend models delegate to the ordinary engine.

**Kriging conditioner**:
A live, incrementally fitted kriging state used by sequential Gaussian simulation. It predicts against the current conditioning set and can append a simulated condition without rebuilding from scratch. Ordinary, simple, and universal kriging each have a conditioner on their working scale; binomial kriging composes an ordinary conditioner on logits.
_Avoid_: SGS state, simulation pool, conditioning history.

**Kriging predictor trait**:
`trait KrigingPredictor` — the cross-cutting seam consumed by `cv`. Backend structs hold borrowed conditioning data and implement `predict_fold(train, test)`; [`leave_one_out_cv`](crate::predictor::cv::leave_one_out_cv) and [`k_fold_cv`](crate::predictor::cv::k_fold_cv) drive the fold loop. Replaces the 21 hand-written CV entry points with a single generic harness.

**WASM kriging model handle**:
The single tagged WASM adapter (`WasmKrigingModel`) wrapping every fitted kriging model variant. Owns prediction dispatch, resource lifecycle (`free`), and instance cross-validation (`leaveOneOut`, `kFold`). TypeScript model classes (`OrdinaryKriging`, `BinomialKriging`, …) are thin adapters over this handle — not separate WASM types.
_Avoid_: WasmOrdinaryKriging, WasmBinomialKriging (legacy per-family WASM structs — removed in 0.4).

**Unified CV / simulation seam**:
The stateless WASM entry points `cv(options)` and `simulate(options)` keyed by `geometry` + `family`, for callers that pass raw arrays without building a model handle. Delegates to the same kriging predictor and conditioner-backed harnesses as the model-handle methods.
_Avoid_: leaveOneOutBinomialProjected, conditionalSimulateSpaceTimeUniversal (legacy named exports — removed in 0.4).

**Binomial counts**:
The geometry-free `(successes, trials)` value with `smoothed_logit(prior)` and `smoothed_probability(prior)` methods. Coordinates are paired with counts via the domain observation structs (`BinomialObservation`, `ProjectedBinomialObservation`, `SpaceTimeBinomialObservation<C>`).
_Avoid_: `BinomialSite` (never shipped).

## Relationships

- Any **kriging type** runs on any **spatial domain**; distance is factored through the **spatial metric** seam. Ordinary, simple, and universal prediction are factored through **pairwise covariance**, not by enumerating domain × kriging type. Universal additionally factors the design matrix through **trend basis**.
- A **binomial kriging** model is a thin wrapper over an **ordinary kriging engine** plus the **logit kriging pipeline**.
- **Sequential Gaussian simulation** is the **kriging conditioner** iterated over a target order.
- The **kriging predictor trait** is the read-only interface that cross-validation sees; the **kriging conditioner** is the live fitted state that simulation sees. Neither harness knows the spatial domain or kriging type directly.

## Example dialogue

> **Reviewer:** Why does spatio-temporal binomial have its own build loop?
>
> **Author:** It doesn't — there's one *logit kriging pipeline* in `kriging/binomial.rs`. The spatio-temporal binomial model passes its `SpaceTimeMetric` and `SpaceTimeOrdinaryKrigingModel` into the same builder. Spatial domain is the only variable.
>
> **Reviewer:** And SGS — does it still rebuild the model per target?
>
> **Author:** All production SGS paths hold a *kriging conditioner* and append each sampled target — ordinary, simple, universal (all drift orders), and binomial kriging across geographic, projected, and spatio-temporal domains.
>
> **Reviewer:** What about anisotropy on projected data?
>
> **Author:** Anisotropy lives inside the *spatial metric* — `ProjectedMetric { anisotropy }`. The variogram only sees pre-warped distances. There's no `Anisotropy2D` parameter on any kriging type anymore.
