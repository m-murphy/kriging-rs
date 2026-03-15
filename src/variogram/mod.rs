//! Variogram computation, fitting, and parametric models.
//!
//! Compute an empirical variogram with [`compute_empirical_variogram`], fit a parametric model
//! with [`fit_variogram`], and build a [`VariogramModel`] (see [`VariogramType`]) for use with
//! kriging. Supported model types include spherical, exponential, Gaussian, cubic, stable, and Matérn.

pub mod empirical;
pub mod fitting;
pub mod models;

pub use empirical::{
    EmpiricalVariogram, PositiveReal, VariogramConfig, compute_empirical_variogram,
};
pub use fitting::{FitResult, fit_variogram};
pub use models::{VariogramModel, VariogramType};
