//! Kriging models for spatial interpolation and prevalence surfaces.
//!
//! - **Ordinary kriging** ([`ordinary`]): [`crate::OrdinaryKrigingModel`] and [`crate::Prediction`] for
//!   interpolating a continuous spatial field from point observations.
//! - **Binomial kriging** ([`binomial`]): [`crate::BinomialKrigingModel`], [`crate::BinomialObservation`],
//!   and related types for estimating prevalence surfaces from count data (successes/trials per location).

pub mod binomial;
pub mod ordinary;
