//! Type alias: space-time simple kriging is the simple engine over space-time pairwise covariance.

use crate::kriging::pairwise::SpaceTimePairwiseCovariance;
use crate::kriging::simple_engine::SimpleKrigingEngine;

/// Simple space-time engine: [`SimpleKrigingEngine`] with a space-time pairwise covariance adapter.
pub type SpaceTimeSimpleKrigingEngine<M> = SimpleKrigingEngine<SpaceTimePairwiseCovariance<M>>;
