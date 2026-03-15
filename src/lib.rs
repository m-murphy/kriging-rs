//! Geostatistical kriging library with ordinary and binomial kriging, variogram fitting,
//! and optional WASM and GPU support.
//!
//! This crate provides spatial interpolation via ordinary kriging and prevalence-surface
//! estimation via binomial kriging. It includes empirical variogram computation, parametric
//! model fitting, and Haversine-based geographic coordinates. Build with `wasm` for browser
//! bindings or `gpu` for GPU-accelerated batch prediction.
//!
//! # Quick example
//!
//! ```rust
//! use kriging_rs::{GeoCoord, GeoDataset, OrdinaryKrigingModel, VariogramModel, VariogramType};
//!
//! # fn main() -> Result<(), kriging_rs::KrigingError> {
//! let coords = vec![
//!     GeoCoord::try_new(0.0, 0.0)?,
//!     GeoCoord::try_new(0.0, 1.0)?,
//!     GeoCoord::try_new(1.0, 0.0)?,
//! ];
//! let values = vec![1.0, 2.0, 1.5];
//! let dataset = GeoDataset::new(coords, values)?;
//! let variogram = VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential)?;
//! let model = OrdinaryKrigingModel::new(dataset, variogram)?;
//! let prediction = model.predict(GeoCoord::try_new(0.3, 0.3)?)?;
//! # let _ = prediction.value;
//! # Ok(())
//! # }
//! ```
//!
//! # Module organization
//!
//! - [`kriging`] — Ordinary kriging ([`OrdinaryKrigingModel`], [`Prediction`]) and binomial
//!   kriging ([`BinomialKrigingModel`], [`BinomialObservation`], etc.) for spatial interpolation
//!   and prevalence surfaces.
//! - [`variogram`] — Empirical variogram ([`compute_empirical_variogram`]), fitting
//!   ([`fit_variogram`]), and parametric models ([`VariogramModel`], [`VariogramType`]).
//! - [`distance`] — [`GeoCoord`] and Haversine distance.
//! - [`geo_dataset`] — Coordinate–value datasets ([`GeoDataset`]).
//! - [`error`] — [`KrigingError`].
//! - `wasm` (optional, `wasm` feature) — Browser-facing WASM bindings.
//! - `gpu` (optional, `gpu` feature) — GPU backend and batch covariance helpers.
//!
//! # Features
//!
//! Default build has no WASM or GPU. Enable with:
//!
//! - `wasm` — WASM bindings and browser support.
//! - `gpu` — WebGPU-based batch prediction (native and web).
//! - `gpu-blocking` — Blocking GPU helpers on native (includes `gpu`).

/// Floating-point type used for coordinates, values, and variogram parameters; currently `f32`.
pub type Real = f32;

pub mod distance;
pub mod error;
pub mod geo_dataset;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kriging;
pub mod matrix;
pub mod utils;

pub use utils::{Probability, clamp_probability, logistic, logit, logit_clamped};
pub mod variogram;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use distance::GeoCoord;
pub use error::KrigingError;
pub use geo_dataset::GeoDataset;
#[cfg(feature = "gpu")]
pub use gpu::{GpuBackend, GpuSupport, build_rhs_covariances_gpu, detect_gpu_support, gpu_square};
pub use kriging::binomial::{
    BinomialKrigingModel, BinomialObservation, BinomialPrediction, BinomialPrior,
};
pub use kriging::ordinary::{OrdinaryKrigingModel, Prediction};
pub use variogram::fitting::{FitResult, fit_variogram};
pub use variogram::models::{VariogramModel, VariogramType};
pub use variogram::{PositiveReal, VariogramConfig, compute_empirical_variogram};
