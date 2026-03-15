//! Kriging library for Rust with optional WASM bindings.

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
