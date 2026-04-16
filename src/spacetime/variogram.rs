//! Space–time variogram models.
//!
//! Two covariance families are implemented (see the crate-level docs for background):
//!
//! - **Separable:** `C(h_s, h_t) = C_s(h_s) · C_t(h_t)`. Simple product of the spatial and
//!   temporal marginal covariances. Positive-definite whenever both marginals are valid.
//! - **Product-sum** (De Iaco / De Cesare):
//!   `C(h_s, h_t) = k1·C_s(h_s)·C_t(h_t) + k2·C_s(h_s) + k3·C_t(h_t)` with `k1, k2, k3 ≥ 0`
//!   and at least one strictly positive. Strictly more expressive than the pure product.
//!
//! Both variants expose [`covariance`](SpaceTimeVariogram::covariance),
//! [`semivariance`](SpaceTimeVariogram::semivariance), and
//! [`c_at_zero`](SpaceTimeVariogram::c_at_zero). Semivariance is computed from covariance
//! via `γ(h_s, h_t) = C(0, 0) − C(h_s, h_t)`.

use crate::Real;
use crate::error::KrigingError;
use crate::variogram::models::{VariogramModel, VariogramType};

/// A space–time covariance model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpaceTimeVariogram {
    /// Separable product model. `C(h_s, h_t) = C_s(h_s) · C_t(h_t)`.
    Separable {
        spatial: VariogramModel,
        temporal: VariogramModel,
    },
    /// Product-sum model (De Iaco/De Cesare). All three coefficients must be non-negative
    /// and at least one must be strictly positive; admissibility checks are enforced at
    /// construction.
    ProductSum {
        spatial: VariogramModel,
        temporal: VariogramModel,
        k1: Real,
        k2: Real,
        k3: Real,
    },
}

impl SpaceTimeVariogram {
    /// Build a separable model `C(h_s, h_t) = C_s(h_s) · C_t(h_t)`.
    ///
    /// Power-law marginals are rejected because their covariance is not defined at the
    /// origin — kriging systems with product-power components have no meaningful `C(0, 0)`.
    pub fn new_separable(
        spatial: VariogramModel,
        temporal: VariogramModel,
    ) -> Result<Self, KrigingError> {
        reject_power_component(spatial, "spatial")?;
        reject_power_component(temporal, "temporal")?;
        Ok(Self::Separable { spatial, temporal })
    }

    /// Build a product-sum model.
    ///
    /// Errors:
    /// - [`KrigingError::InvalidInput`] when any `k_i` is negative or not finite.
    /// - [`KrigingError::InvalidInput`] when `k1 + k2 + k3 == 0` (would give a zero kernel).
    /// - [`KrigingError::InvalidInput`] when either marginal is a power model.
    pub fn new_product_sum(
        spatial: VariogramModel,
        temporal: VariogramModel,
        k1: Real,
        k2: Real,
        k3: Real,
    ) -> Result<Self, KrigingError> {
        reject_power_component(spatial, "spatial")?;
        reject_power_component(temporal, "temporal")?;
        for (name, k) in [("k1", k1), ("k2", k2), ("k3", k3)] {
            if !k.is_finite() || k < 0.0 {
                return Err(KrigingError::InvalidInput(format!(
                    "product-sum {name} must be finite and non-negative, got {k}"
                )));
            }
        }
        if k1 + k2 + k3 <= 0.0 {
            return Err(KrigingError::InvalidInput(
                "product-sum coefficients must sum to a positive value".to_string(),
            ));
        }
        Ok(Self::ProductSum {
            spatial,
            temporal,
            k1,
            k2,
            k3,
        })
    }

    /// Spatial marginal.
    pub fn spatial(&self) -> VariogramModel {
        match *self {
            Self::Separable { spatial, .. } | Self::ProductSum { spatial, .. } => spatial,
        }
    }

    /// Temporal marginal.
    pub fn temporal(&self) -> VariogramModel {
        match *self {
            Self::Separable { temporal, .. } | Self::ProductSum { temporal, .. } => temporal,
        }
    }

    /// Covariance `C(h_s, h_t)` at the supplied spatial and temporal lags. Negative inputs
    /// are clamped to zero (matches the 1-D convention in [`VariogramModel::covariance`]).
    pub fn covariance(&self, h_spatial: Real, h_temporal: Real) -> Real {
        let hs = h_spatial.max(0.0);
        let ht = h_temporal.max(0.0);
        match *self {
            Self::Separable { spatial, temporal } => {
                spatial.covariance(hs) * temporal.covariance(ht)
            }
            Self::ProductSum {
                spatial,
                temporal,
                k1,
                k2,
                k3,
            } => {
                let cs = spatial.covariance(hs);
                let ct = temporal.covariance(ht);
                k1 * cs * ct + k2 * cs + k3 * ct
            }
        }
    }

    /// Semivariance `γ(h_s, h_t) = C(0, 0) − C(h_s, h_t)`.
    pub fn semivariance(&self, h_spatial: Real, h_temporal: Real) -> Real {
        self.c_at_zero() - self.covariance(h_spatial, h_temporal)
    }

    /// `C(0, 0)`: the process variance at zero spatial and temporal lag.
    pub fn c_at_zero(&self) -> Real {
        match *self {
            Self::Separable { spatial, temporal } => {
                spatial.covariance(0.0) * temporal.covariance(0.0)
            }
            Self::ProductSum {
                spatial,
                temporal,
                k1,
                k2,
                k3,
            } => {
                let cs0 = spatial.covariance(0.0);
                let ct0 = temporal.covariance(0.0);
                k1 * cs0 * ct0 + k2 * cs0 + k3 * ct0
            }
        }
    }

    /// Variogram types of the spatial and temporal marginals, in order. Used internally for
    /// conditioning heuristics (Gaussian/Cubic marginals need extra diagonal jitter).
    pub(crate) fn marginal_variogram_types(&self) -> [VariogramType; 2] {
        match *self {
            Self::Separable { spatial, temporal }
            | Self::ProductSum {
                spatial, temporal, ..
            } => [spatial.variogram_type(), temporal.variogram_type()],
        }
    }
}

fn reject_power_component(v: VariogramModel, label: &str) -> Result<(), KrigingError> {
    if matches!(v, VariogramModel::Power { .. }) {
        Err(KrigingError::InvalidInput(format!(
            "{label} marginal cannot be a Power variogram (unbounded covariance at origin)"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn spatial_exp() -> VariogramModel {
        VariogramModel::new(0.0, 1.0, 10.0, VariogramType::Exponential).unwrap()
    }

    fn temporal_exp() -> VariogramModel {
        VariogramModel::new(0.0, 2.0, 5.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn separable_covariance_is_product_of_marginals() {
        let stv = SpaceTimeVariogram::new_separable(spatial_exp(), temporal_exp()).unwrap();
        let hs = 3.0;
        let ht = 1.5;
        let expected = spatial_exp().covariance(hs) * temporal_exp().covariance(ht);
        assert_relative_eq!(stv.covariance(hs, ht), expected, epsilon = 1e-6);
    }

    #[test]
    fn separable_c_at_zero_is_product_of_sills() {
        let stv = SpaceTimeVariogram::new_separable(spatial_exp(), temporal_exp()).unwrap();
        let expected = spatial_exp().covariance(0.0) * temporal_exp().covariance(0.0);
        assert_relative_eq!(stv.c_at_zero(), expected, epsilon = 1e-6);
    }

    #[test]
    fn semivariance_complements_covariance() {
        let stv = SpaceTimeVariogram::new_separable(spatial_exp(), temporal_exp()).unwrap();
        let hs = 2.2;
        let ht = 0.8;
        assert_relative_eq!(
            stv.covariance(hs, ht) + stv.semivariance(hs, ht),
            stv.c_at_zero(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn separable_rejects_power_marginals() {
        let power = VariogramModel::new_power(0.0, 0.1, 1.0).unwrap();
        assert!(SpaceTimeVariogram::new_separable(power, temporal_exp()).is_err());
        assert!(SpaceTimeVariogram::new_separable(spatial_exp(), power).is_err());
    }

    #[test]
    fn product_sum_reduces_to_separable_when_k2_and_k3_are_zero() {
        let stv_sep = SpaceTimeVariogram::new_separable(spatial_exp(), temporal_exp()).unwrap();
        let stv_ps =
            SpaceTimeVariogram::new_product_sum(spatial_exp(), temporal_exp(), 1.0, 0.0, 0.0)
                .unwrap();
        assert_relative_eq!(
            stv_sep.covariance(2.0, 1.0),
            stv_ps.covariance(2.0, 1.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn product_sum_rejects_negative_coefficients() {
        assert!(
            SpaceTimeVariogram::new_product_sum(spatial_exp(), temporal_exp(), -0.1, 1.0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn product_sum_rejects_all_zero_coefficients() {
        assert!(
            SpaceTimeVariogram::new_product_sum(spatial_exp(), temporal_exp(), 0.0, 0.0, 0.0)
                .is_err()
        );
    }
}
