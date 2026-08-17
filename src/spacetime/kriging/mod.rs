//! Space–time kriging model families.
//!
//! Mirrors the 2-D [`kriging`](crate::kriging) module but takes [`SpaceTimeCoord`] inputs
//! and a [`SpaceTimeVariogram`]. Parametrized over a [`SpatialMetric`] so the same
//! implementations work for geographic and projected spatial coordinates.

pub mod binomial;
pub(crate) mod engine;
pub mod ordinary;
pub mod simple;
pub(crate) mod simple_engine;
pub mod universal;
pub(crate) mod universal_engine;
