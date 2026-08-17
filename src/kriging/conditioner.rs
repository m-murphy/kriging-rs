//! Live fitted kriging state for sequential Gaussian simulation.
//!
//! A [`KrigingConditioner`] predicts against its current conditioning set and appends
//! simulated conditions incrementally. RNG, target ordering, and realization loops remain in
//! [`crate::simulation`].

use std::fmt::Debug;
use std::marker::PhantomData;

use crate::Real;
use crate::error::KrigingError;
use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::kriging::pairwise::PairwiseCovariance;
use crate::kriging::simple_engine::SimpleKrigingEngine;
use crate::kriging::universal::TrendBasis;
use crate::kriging::universal_engine::UniversalKrigingEngine;

/// Original continuous working scale.
#[derive(Debug, Clone, Copy)]
pub enum ContinuousScale {}

/// Binomial logit working scale.
#[derive(Debug, Clone, Copy)]
pub enum LogitScale {}

/// Conditional Gaussian moments on a conditioner working scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalGaussian {
    pub mean: Real,
    pub variance: Real,
}

trait ConditionerState<S>: Debug + Send + Sync {
    fn clone_box(&self) -> Box<dyn ConditionerState<S>>;

    fn predict(&self, target: S) -> Result<ConditionalGaussian, KrigingError>;

    fn append_condition(&mut self, site: S, value: Real) -> Result<(), KrigingError>;
}

impl<S> Clone for Box<dyn ConditionerState<S>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Live, incrementally fitted kriging state used by sequential Gaussian simulation.
#[derive(Debug, Clone)]
pub struct KrigingConditioner<S, Scale = ContinuousScale> {
    inner: Box<dyn ConditionerState<S>>,
    scale: PhantomData<fn() -> Scale>,
}

impl<S, Scale> KrigingConditioner<S, Scale>
where
    S: Copy,
{
    /// Predict conditional Gaussian moments at `target` on this conditioner's working scale.
    pub fn predict(&self, target: S) -> Result<ConditionalGaussian, KrigingError> {
        self.inner.predict(target)
    }

    /// Append a zero-observation-variance simulated condition.
    ///
    /// On error, the conditioner remains unchanged.
    pub fn append_condition(&mut self, site: S, value: Real) -> Result<(), KrigingError> {
        if !value.is_finite() {
            return Err(KrigingError::InvalidInput(
                "conditioned value must be finite".to_string(),
            ));
        }
        self.inner.append_condition(site, value)
    }

    pub(crate) fn from_ordinary<K>(engine: OrdinaryKrigingEngine<K>) -> Self
    where
        K: PairwiseCovariance<Site = S> + 'static,
        K::Site: 'static,
        K::Prepared: 'static,
    {
        Self {
            inner: Box::new(OrdinaryConditionerState { engine }),
            scale: PhantomData,
        }
    }

    pub(crate) fn from_simple<K>(engine: SimpleKrigingEngine<K>) -> Self
    where
        K: PairwiseCovariance<Site = S> + 'static,
        K::Site: 'static,
        K::Prepared: 'static,
    {
        Self {
            inner: Box::new(SimpleConditionerState { engine }),
            scale: PhantomData,
        }
    }

    pub(crate) fn from_universal<K, T>(engine: UniversalKrigingEngine<K, T>) -> Self
    where
        K: PairwiseCovariance<Site = S> + 'static,
        T: TrendBasis<Site = S> + 'static,
        K::Site: 'static,
        K::Prepared: 'static,
    {
        Self {
            inner: Box::new(UniversalConditionerState { engine }),
            scale: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
struct OrdinaryConditionerState<K: PairwiseCovariance> {
    engine: OrdinaryKrigingEngine<K>,
}

impl<K> ConditionerState<K::Site> for OrdinaryConditionerState<K>
where
    K: PairwiseCovariance + 'static,
    K::Site: 'static,
    K::Prepared: 'static,
{
    fn clone_box(&self) -> Box<dyn ConditionerState<K::Site>> {
        Box::new(self.clone())
    }

    fn predict(&self, target: K::Site) -> Result<ConditionalGaussian, KrigingError> {
        let prediction = self
            .engine
            .predict(&[target])?
            .pop()
            .expect("single conditioner prediction");
        Ok(ConditionalGaussian {
            mean: prediction.value,
            variance: prediction.variance,
        })
    }

    fn append_condition(&mut self, site: K::Site, value: Real) -> Result<(), KrigingError> {
        self.engine.append_condition(site, value, 0.0)
    }
}

#[derive(Debug, Clone)]
struct UniversalConditionerState<K: PairwiseCovariance, T: TrendBasis<Site = K::Site>> {
    engine: UniversalKrigingEngine<K, T>,
}

impl<K, T> ConditionerState<K::Site> for UniversalConditionerState<K, T>
where
    K: PairwiseCovariance + 'static,
    T: TrendBasis<Site = K::Site> + 'static,
    K::Site: 'static,
    K::Prepared: 'static,
{
    fn clone_box(&self) -> Box<dyn ConditionerState<K::Site>> {
        Box::new(self.clone())
    }

    fn predict(&self, target: K::Site) -> Result<ConditionalGaussian, KrigingError> {
        let prediction = self
            .engine
            .predict(&[target])?
            .pop()
            .expect("single conditioner prediction");
        Ok(ConditionalGaussian {
            mean: prediction.value,
            variance: prediction.variance,
        })
    }

    fn append_condition(&mut self, site: K::Site, value: Real) -> Result<(), KrigingError> {
        self.engine.append_condition(site, value, 0.0)
    }
}

#[derive(Debug, Clone)]
struct SimpleConditionerState<K: PairwiseCovariance> {
    engine: SimpleKrigingEngine<K>,
}

impl<K> ConditionerState<K::Site> for SimpleConditionerState<K>
where
    K: PairwiseCovariance + 'static,
    K::Site: 'static,
    K::Prepared: 'static,
{
    fn clone_box(&self) -> Box<dyn ConditionerState<K::Site>> {
        Box::new(self.clone())
    }

    fn predict(&self, target: K::Site) -> Result<ConditionalGaussian, KrigingError> {
        let prediction = self
            .engine
            .predict(&[target])?
            .pop()
            .expect("single conditioner prediction");
        Ok(ConditionalGaussian {
            mean: prediction.value,
            variance: prediction.variance,
        })
    }

    fn append_condition(&mut self, site: K::Site, value: Real) -> Result<(), KrigingError> {
        self.engine.append_condition(site, value, 0.0)
    }
}
