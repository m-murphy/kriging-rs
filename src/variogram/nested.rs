//! Nested (additive) variogram composition.
//!
//! A [`NestedVariogram`] is the sum of several component [`VariogramModel`]s. This lets you
//! model multi-scale structure (e.g. short-range exponential + long-range Gaussian) by
//! composing familiar building blocks.
//!
//! `NestedVariogram` exposes the same `semivariance` / `covariance` interface as a single
//! [`VariogramModel`] but cannot currently be passed directly to
//! [`OrdinaryKrigingModel`](crate::OrdinaryKrigingModel). It is intended for callers that
//! need multi-scale semivariances (e.g. custom kriging solvers or diagnostic plots).

use crate::Real;
use crate::error::KrigingError;
use crate::variogram::models::VariogramModel;

/// Sum of one or more component variograms. `γ(h) = Σ_k γ_k(h)`.
#[derive(Debug, Clone)]
pub struct NestedVariogram {
    components: Vec<VariogramModel>,
}

impl NestedVariogram {
    /// Construct a nested variogram from at least one component.
    pub fn new(components: Vec<VariogramModel>) -> Result<Self, KrigingError> {
        if components.is_empty() {
            return Err(KrigingError::InvalidInput(
                "nested variogram requires at least one component".to_string(),
            ));
        }
        Ok(Self { components })
    }

    /// Read-only view of the component models in insertion order.
    pub fn components(&self) -> &[VariogramModel] {
        &self.components
    }

    /// Sum of component semivariances at `distance`.
    pub fn semivariance(&self, distance: Real) -> Real {
        self.components
            .iter()
            .map(|m| m.semivariance(distance))
            .sum()
    }

    /// Sum of component covariances at `distance`. The result is interpreted as the
    /// covariance of the multi-scale Gaussian process under standard additivity assumptions.
    ///
    /// Note: components with no finite variance (e.g. the unbounded
    /// [`VariogramType::Power`](crate::VariogramType::Power) model) return `0` from
    /// [`VariogramModel::covariance`], so including them here is discouraged.
    pub fn covariance(&self, distance: Real) -> Real {
        self.components.iter().map(|m| m.covariance(distance)).sum()
    }

    /// Aggregate "variance at zero" — i.e. the effective sill `Σ_k sill_k` across bounded
    /// components.
    pub fn total_sill(&self) -> Real {
        self.components
            .iter()
            .map(|m| m.covariance(0.0) + m.semivariance(0.0))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

    #[test]
    fn new_rejects_empty_components() {
        assert!(NestedVariogram::new(vec![]).is_err());
    }

    #[test]
    fn semivariance_is_sum_of_components() {
        let a = VariogramModel::new(0.1, 1.0, 5.0, VariogramType::Exponential).unwrap();
        let b = VariogramModel::new(0.0, 2.0, 30.0, VariogramType::Gaussian).unwrap();
        let nested = NestedVariogram::new(vec![a, b]).unwrap();
        let h = 7.0;
        let sum = a.semivariance(h) + b.semivariance(h);
        assert!((nested.semivariance(h) - sum).abs() < 1e-5);
    }

    #[test]
    fn covariance_complements_semivariance() {
        let a = VariogramModel::new(0.1, 1.0, 5.0, VariogramType::Exponential).unwrap();
        let b = VariogramModel::new(0.0, 2.0, 30.0, VariogramType::Gaussian).unwrap();
        let nested = NestedVariogram::new(vec![a, b]).unwrap();
        let h = 4.0;
        let total = nested.total_sill();
        assert!((nested.covariance(h) + nested.semivariance(h) - total).abs() < 1e-4);
    }
}
