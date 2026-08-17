//! Type alias: space-time ordinary kriging is the ordinary engine over space-time pairwise covariance.

use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::kriging::pairwise::SpaceTimePairwiseCovariance;

/// Ordinary space-time engine: [`OrdinaryKrigingEngine`] with a space-time pairwise covariance adapter.
pub type SpaceTimeOrdinaryKrigingEngine<M> = OrdinaryKrigingEngine<SpaceTimePairwiseCovariance<M>>;
