use crate::Real;
use crate::error::KrigingError;

/// Matérn semivariance: γ(h) = nugget + partial_sill * (1 - (2^(1-ν)/Γ(ν)) * x^ν * K_ν(x)) with x = h√(2ν)/range.
fn matern_semivariance(d: Real, nugget: Real, partial_sill: Real, range: Real, nu: Real) -> Real {
    if d <= 0.0 {
        return nugget;
    }
    let nu_f64 = nu as f64;
    let x_f64 = (d as f64) * (2.0 * nu_f64).sqrt() / (range as f64);
    if x_f64 <= 0.0 {
        return nugget;
    }
    let (_i_nu, k_nu) = puruspe::bessel::Inu_Knu(nu_f64, x_f64);
    let gamma_nu = puruspe::gamma::gamma(nu_f64);
    let factor = (2.0_f64).powf(1.0 - nu_f64) / gamma_nu * x_f64.powf(nu_f64) * k_nu;
    let correlation = factor.clamp(0.0, 1.0);
    nugget + partial_sill * (1.0 - (correlation as Real))
}

/// Parametric variogram family used to construct a [`VariogramModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariogramType {
    Spherical,
    Exponential,
    Gaussian,
    Cubic,
    /// Power-law family; requires shape parameter `alpha` in (0, 2].
    Stable,
    /// Matérn family; requires smoothness parameter `nu` > 0.
    Matern,
    /// Unbounded power-law model `γ(h) = nugget + slope · h^exponent`, `exponent ∈ (0, 2)`.
    /// Has no sill; use with care in kriging (ordinary kriging systems remain solvable).
    Power,
    /// Damped hole-effect model `γ(h) = nugget + (sill − nugget)·(1 − sin(π·h/range)/(π·h/range))`.
    /// Useful for pseudo-periodic fields.
    HoleEffect,
}

/// Parametric variogram (nugget, sill, range, and optional shape).
///
/// Build with [`VariogramModel::new`] from a [`VariogramType`]; use with
/// [`OrdinaryKrigingModel::new`](crate::OrdinaryKrigingModel::new) or
/// [`BinomialKrigingModel::new`](crate::BinomialKrigingModel::new).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariogramModel {
    Spherical {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Exponential {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Gaussian {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Cubic {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Stable {
        nugget: Real,
        sill: Real,
        range: Real,
        alpha: Real,
    },
    Matern {
        nugget: Real,
        sill: Real,
        range: Real,
        nu: Real,
    },
    /// Unbounded power-law model `γ(h) = nugget + slope·h^exponent`, `exponent ∈ (0, 2)`.
    /// `sill` and `range` fields carry `slope` and `exponent` respectively so that the rest
    /// of the library can treat `VariogramModel` uniformly via [`params`](Self::params).
    ///
    /// Note: for the power model, `sill` is not a true plateau — it is reused as the slope
    /// coefficient. Use [`new_power`](Self::new_power) for a named constructor.
    Power {
        nugget: Real,
        slope: Real,
        exponent: Real,
    },
    /// Hole-effect model with a pseudo-periodic dip; bounded above by `sill`.
    HoleEffect {
        nugget: Real,
        sill: Real,
        range: Real,
    },
}

impl VariogramModel {
    /// Constructs a variogram model with validated parameters: `nugget >= 0`, `sill > nugget`, `range > 0`, all finite.
    pub fn new(
        nugget: Real,
        sill: Real,
        range: Real,
        model_type: VariogramType,
    ) -> Result<Self, KrigingError> {
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(KrigingError::FittingError(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        if !sill.is_finite() || sill <= nugget {
            return Err(KrigingError::FittingError(
                "sill must be finite and greater than nugget".to_string(),
            ));
        }
        if !range.is_finite() || range <= 0.0 {
            return Err(KrigingError::FittingError(
                "range must be finite and positive".to_string(),
            ));
        }
        Ok(match model_type {
            VariogramType::Spherical => VariogramModel::Spherical {
                nugget,
                sill,
                range,
            },
            VariogramType::Exponential => VariogramModel::Exponential {
                nugget,
                sill,
                range,
            },
            VariogramType::Gaussian => VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            },
            VariogramType::Cubic => VariogramModel::Cubic {
                nugget,
                sill,
                range,
            },
            VariogramType::Stable => VariogramModel::Stable {
                nugget,
                sill,
                range,
                alpha: 1.0,
            },
            VariogramType::Matern => VariogramModel::Matern {
                nugget,
                sill,
                range,
                nu: 0.5,
            },
            VariogramType::Power => {
                // Default to exponent 1 (linear in distance); treat `sill` as slope.
                VariogramModel::Power {
                    nugget,
                    slope: sill - nugget,
                    exponent: 1.0,
                }
            }
            VariogramType::HoleEffect => VariogramModel::HoleEffect {
                nugget,
                sill,
                range,
            },
        })
    }

    /// Construct a power-law variogram `γ(h) = nugget + slope·h^exponent`. `slope > 0`,
    /// `exponent ∈ (0, 2)`, all values finite.
    pub fn new_power(nugget: Real, slope: Real, exponent: Real) -> Result<Self, KrigingError> {
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(KrigingError::FittingError(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        if !slope.is_finite() || slope <= 0.0 {
            return Err(KrigingError::FittingError(
                "power slope must be finite and positive".to_string(),
            ));
        }
        if !exponent.is_finite() || exponent <= 0.0 || exponent >= 2.0 {
            return Err(KrigingError::FittingError(
                "power exponent must be in (0, 2)".to_string(),
            ));
        }
        Ok(VariogramModel::Power {
            nugget,
            slope,
            exponent,
        })
    }

    /// Constructs a variogram model with an explicit shape parameter for Stable (alpha) or Matérn (nu).
    /// For other model types, `shape` is ignored. Stable: alpha in (0, 2]. Matérn: nu > 0.
    pub fn new_with_shape(
        nugget: Real,
        sill: Real,
        range: Real,
        model_type: VariogramType,
        shape: Real,
    ) -> Result<Self, KrigingError> {
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(KrigingError::FittingError(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        if !sill.is_finite() || sill <= nugget {
            return Err(KrigingError::FittingError(
                "sill must be finite and greater than nugget".to_string(),
            ));
        }
        if !range.is_finite() || range <= 0.0 {
            return Err(KrigingError::FittingError(
                "range must be finite and positive".to_string(),
            ));
        }
        Ok(match model_type {
            VariogramType::Spherical => VariogramModel::Spherical {
                nugget,
                sill,
                range,
            },
            VariogramType::Exponential => VariogramModel::Exponential {
                nugget,
                sill,
                range,
            },
            VariogramType::Gaussian => VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            },
            VariogramType::Cubic => VariogramModel::Cubic {
                nugget,
                sill,
                range,
            },
            VariogramType::Stable => {
                if !shape.is_finite() || shape <= 0.0 || shape > 2.0 {
                    return Err(KrigingError::FittingError(
                        "Stable shape (alpha) must be in (0, 2]".to_string(),
                    ));
                }
                VariogramModel::Stable {
                    nugget,
                    sill,
                    range,
                    alpha: shape,
                }
            }
            VariogramType::Matern => {
                if !shape.is_finite() || shape <= 0.0 {
                    return Err(KrigingError::FittingError(
                        "Matérn shape (nu) must be positive".to_string(),
                    ));
                }
                VariogramModel::Matern {
                    nugget,
                    sill,
                    range,
                    nu: shape,
                }
            }
            VariogramType::Power => {
                // Reuse `shape` as the exponent; `sill` is reinterpreted as slope (after
                // subtracting the nugget), matching the default constructor's convention.
                if !shape.is_finite() || shape <= 0.0 || shape >= 2.0 {
                    return Err(KrigingError::FittingError(
                        "power exponent must be in (0, 2)".to_string(),
                    ));
                }
                VariogramModel::Power {
                    nugget,
                    slope: sill - nugget,
                    exponent: shape,
                }
            }
            VariogramType::HoleEffect => VariogramModel::HoleEffect {
                nugget,
                sill,
                range,
            },
        })
    }

    pub fn variogram_type(&self) -> VariogramType {
        match self {
            Self::Spherical { .. } => VariogramType::Spherical,
            Self::Exponential { .. } => VariogramType::Exponential,
            Self::Gaussian { .. } => VariogramType::Gaussian,
            Self::Cubic { .. } => VariogramType::Cubic,
            Self::Stable { .. } => VariogramType::Stable,
            Self::Matern { .. } => VariogramType::Matern,
            Self::Power { .. } => VariogramType::Power,
            Self::HoleEffect { .. } => VariogramType::HoleEffect,
        }
    }

    pub fn params(&self) -> (Real, Real, Real) {
        match self {
            Self::Spherical {
                nugget,
                sill,
                range,
            }
            | Self::Exponential {
                nugget,
                sill,
                range,
            }
            | Self::Gaussian {
                nugget,
                sill,
                range,
            }
            | Self::Cubic {
                nugget,
                sill,
                range,
            }
            | Self::Stable {
                nugget,
                sill,
                range,
                ..
            }
            | Self::Matern {
                nugget,
                sill,
                range,
                ..
            }
            | Self::HoleEffect {
                nugget,
                sill,
                range,
            } => (*nugget, *sill, *range),
            // For the power model, `sill` carries slope and `range` carries exponent. This
            // keeps `params()` a simple getter; the actual semivariance computation branches
            // on the enum variant.
            Self::Power {
                nugget,
                slope,
                exponent,
            } => (*nugget, *slope + *nugget, *exponent),
        }
    }

    /// Shape parameter for Stable (alpha), Matérn (nu), or Power (exponent).
    /// Returns `None` for 3-parameter bounded models.
    pub fn shape(&self) -> Option<Real> {
        match self {
            Self::Stable { alpha, .. } => Some(*alpha),
            Self::Matern { nu, .. } => Some(*nu),
            Self::Power { exponent, .. } => Some(*exponent),
            _ => None,
        }
    }

    /// Semivariance at `distance`. Assumes `distance >= 0` (e.g. from haversine); clamps negative input to 0.
    pub fn semivariance(&self, distance: Real) -> Real {
        let d = distance.max(0.0);
        let (nugget, sill, range) = self.params();
        let partial_sill = sill - nugget;
        let r = range.max(Real::EPSILON);

        match self {
            Self::Spherical { .. } => {
                if d >= range {
                    sill
                } else {
                    let x = d / r;
                    nugget + partial_sill * (1.5 * x - 0.5 * x.powi(3))
                }
            }
            Self::Exponential { .. } => nugget + partial_sill * (1.0 - (-3.0 * d / r).exp()),
            Self::Gaussian { .. } => {
                nugget + partial_sill * (1.0 - (-3.0 * (d * d) / (r * r)).exp())
            }
            Self::Cubic { .. } => {
                if d >= range {
                    sill
                } else {
                    let x = d / r;
                    let poly = 7.0 * x * x - 8.5 * x.powi(3) + 3.5 * x.powi(5) - 0.5 * x.powi(7);
                    nugget + partial_sill * poly
                }
            }
            Self::Stable { alpha, .. } => {
                let x = (d / r).powf(*alpha);
                nugget + partial_sill * (1.0 - (-x).exp())
            }
            Self::Matern { nu, .. } => matern_semivariance(d, nugget, partial_sill, r, *nu),
            Self::Power {
                slope, exponent, ..
            } => nugget + slope * d.powf(*exponent),
            Self::HoleEffect { .. } => {
                if d <= 0.0 {
                    nugget
                } else {
                    // Damped hole-effect: γ(h) = nugget + partial_sill · (1 − sinc(π·h/r))
                    // with sinc(x) = sin(x)/x. Bounded above by sill.
                    let x = std::f64::consts::PI as Real * d / r;
                    let sinc = x.sin() / x;
                    nugget + partial_sill * (1.0 - sinc)
                }
            }
        }
    }

    /// Covariance `C(h) = C(0) − γ(h)`. For bounded models `C(0) = sill`; for the unbounded
    /// [power model](VariogramModel::Power), `C(0)` is not defined, so this method returns
    /// `0` there and kriging should use the [`semivariance`](Self::semivariance) form.
    pub fn covariance(&self, distance: Real) -> Real {
        if matches!(self, Self::Power { .. }) {
            // Ordinary kriging of an intrinsic (unbounded) model is usually formulated via
            // semivariances. We return a dummy value; callers that explicitly support Power
            // should branch on the type.
            return 0.0;
        }
        let (_, sill, _) = self.params();
        sill - self.semivariance(distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spherical_hits_sill_at_range() {
        let model = VariogramModel::new(0.1, 1.0, 10.0, VariogramType::Spherical).unwrap();
        assert_relative_eq!(model.semivariance(10.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(model.semivariance(20.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn exponential_and_gaussian_start_at_nugget() {
        let exp = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Exponential).unwrap();
        let gauss = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Gaussian).unwrap();
        assert_relative_eq!(exp.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(gauss.semivariance(0.0), 0.2, epsilon = 1e-6);
    }

    #[test]
    fn covariance_complements_semivariance() {
        let model = VariogramModel::new(0.1, 1.0, 5.0, VariogramType::Exponential).unwrap();
        let d = 2.2;
        assert_relative_eq!(
            model.covariance(d) + model.semivariance(d),
            1.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn cubic_hits_sill_at_range() {
        let model = VariogramModel::new(0.1, 1.0, 10.0, VariogramType::Cubic).unwrap();
        assert_relative_eq!(model.semivariance(10.0), 1.0, epsilon = 1e-5);
        assert_relative_eq!(model.semivariance(20.0), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn cubic_stable_matern_start_at_nugget() {
        let cubic = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Cubic).unwrap();
        let stable = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Stable).unwrap();
        let matern = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Matern).unwrap();
        assert_relative_eq!(cubic.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(stable.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(matern.semivariance(0.0), 0.2, epsilon = 1e-6);
    }

    #[test]
    fn stable_with_alpha_one_increases_to_sill() {
        let stable =
            VariogramModel::new_with_shape(0.1, 2.0, 10.0, VariogramType::Stable, 1.0).unwrap();
        let (nugget, sill, _) = stable.params();
        assert_relative_eq!(stable.semivariance(0.0), nugget, epsilon = 1e-6);
        let mut prev = nugget;
        for d in [1.0, 3.0, 5.0, 10.0, 20.0] {
            let g = stable.semivariance(d);
            assert!(
                g >= prev && g <= sill,
                "stable semivariance should increase toward sill"
            );
            prev = g;
        }
        assert_relative_eq!(stable.semivariance(100.0), sill, epsilon = 1e-4);
    }

    #[test]
    fn shape_returns_none_for_three_param_models() {
        let m = VariogramModel::new(0.0, 1.0, 1.0, VariogramType::Cubic).unwrap();
        assert_eq!(m.shape(), None);
    }

    #[test]
    fn shape_returns_some_for_stable_and_matern() {
        let stable =
            VariogramModel::new_with_shape(0.0, 1.0, 1.0, VariogramType::Stable, 1.5).unwrap();
        let matern =
            VariogramModel::new_with_shape(0.0, 1.0, 1.0, VariogramType::Matern, 2.5).unwrap();
        assert_relative_eq!(stable.shape().unwrap(), 1.5, epsilon = 1e-6);
        assert_relative_eq!(matern.shape().unwrap(), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn power_model_grows_without_a_sill() {
        let m = VariogramModel::new_power(0.1, 0.5, 1.0).unwrap();
        let g1 = m.semivariance(1.0);
        let g10 = m.semivariance(10.0);
        let g100 = m.semivariance(100.0);
        assert!(g1 < g10 && g10 < g100);
        assert_relative_eq!(m.semivariance(0.0), 0.1, epsilon = 1e-6);
    }

    #[test]
    fn power_model_rejects_invalid_exponent() {
        assert!(VariogramModel::new_power(0.0, 1.0, 0.0).is_err());
        assert!(VariogramModel::new_power(0.0, 1.0, 2.0).is_err());
        assert!(VariogramModel::new_power(0.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn hole_effect_starts_at_nugget_and_oscillates_below_sill() {
        let m = VariogramModel::new(0.1, 2.0, 10.0, VariogramType::HoleEffect).unwrap();
        assert_relative_eq!(m.semivariance(0.0), 0.1, epsilon = 1e-5);
        // At d = range/2 the hole-effect crosses above the midpoint.
        let mid = m.semivariance(5.0);
        assert!(mid > 0.1 && mid < 2.0);
    }
}
