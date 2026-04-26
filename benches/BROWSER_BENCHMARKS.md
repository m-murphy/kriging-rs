# Large visual browser workload (reference)

This crate does not run browsers in CI; the following numbers define what we call a **large
visual** interactive workload and are the shape targeted by
`benches/performance.rs` (`bench_binomial_browser_representative`):

| Quantity | Target |
| --- | --- |
| Conditioning (success/trial) sites | **~300–1k**; benches use **400** on a 20×20 grid |
| Prediction targets | **10k–50k**; benches use a **200×200 = 40_000** point grid in [35,39]×[−120,−116] |
| Denominators | **Mixed** — e.g. alternating 8 and 180 trials to stress heteroskedastic logit noise |

Benches are **prediction-only** after a fixed `VariogramModel` is applied (the empirical
variogram + fit is not part of the timed loop) so you can read off expected map redraw cost
in native builds. With `--features gpu-blocking` on a suitable machine, the `*_gpu` case
estimates the WebGPU RHS path. WASM timing must be profiled in-browser.

**Default** binomial path: empirical-Bayes logit + `OrdinaryKrigingModel::new_with_extra_diagonal`
per-site logit observation variance (calibrated binomial). Build cost is the same `O(n³)`
factorization class; the extra diagonal does not change prediction asymptotics (still one
solve re-used per batch in the full-data case).

See also: [`BinomialBuildNotes`](../src/kriging/binomial.rs) in the Rust API when
inflation retries run for numerical stability.
