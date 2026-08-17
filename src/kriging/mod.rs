//! Kriging models for spatial interpolation and prevalence surfaces.
//!
//! - **Ordinary kriging** ([`ordinary`]): [`crate::OrdinaryKrigingModel`] and [`crate::Prediction`] for
//!   interpolating a continuous spatial field from point observations.
//! - **Binomial kriging** ([`binomial`]): [`crate::BinomialKrigingModel`], [`crate::BinomialObservation`],
//!   and related types for prevalence surfaces. The default path is empirical-Bayes
//!   logit + ordinary kriging. [`crate::HeteroBinomialFit`] documents an experimental
//!   heteroskedastic variant.

pub mod binomial;
pub mod conditioner;
pub(crate) mod engine;
pub(crate) mod numerics;
pub mod ordinary;
pub(crate) mod pairwise;
pub mod simple;
pub(crate) mod simple_engine;
pub mod universal;
pub(crate) mod universal_engine;
