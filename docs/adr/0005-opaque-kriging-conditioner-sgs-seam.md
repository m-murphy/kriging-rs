# Opaque kriging conditioner as the SGS state seam

**Status:** accepted (2026-08-16)

Sequential Gaussian simulation previously exposed two implementation traits and ten model/domain-specific simulator structs. The dependency is entirely in-process, so that public adapter seam added surface without enabling a justified external implementation.

SGS accepts one opaque, scale-typed `KrigingConditioner<Site, Scale>` obtained by consuming a fitted model. The conditioner exposes conditional Gaussian prediction and atomic condition append; ordinary, simple, and universal engine adapters remain private, while binomial kriging composes the ordinary adapter on logits. RNG, target order, output order, and single/many realization loops remain in the simulation harness.

## Considered options

- **Keep a public conditioner trait.** Rejected: callers have no remote dependency or test substitute that justifies an implementation port.
- **Expose an enum of engine families.** Rejected: every new kriging family or spatial domain would widen a public compatibility commitment.
- **Keep model-specific conditioner structs.** Rejected: this recreates the shallow domain × kriging-type surface that the seam is intended to remove.
- **Add a direct ordinary-geographic constructor.** Rejected: fitted-model conversion keeps construction and validation with each model without making the conditioner interface asymmetric.

## Consequences

- `ContinuousScale` and `LogitScale` prevent prevalence/logit misuse at compile time.
- Models with an active search neighborhood cannot become conditioners because SGS requires every live condition to participate.
- Binomial SGS now uses the fitted model's canonical Laplace/Fisher observation-variance policy; the former simulation-only empirical-Bayes variance path is removed.
- The old simulator traits, structs, and `predictor::simulation` route are removed as a breaking Rust API change.
