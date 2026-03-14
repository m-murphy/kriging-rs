use crate::Real;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariogramType {
    Spherical,
    Exponential,
    Gaussian,
}

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
}

impl VariogramModel {
    pub fn variogram_type(&self) -> VariogramType {
        match self {
            Self::Spherical { .. } => VariogramType::Spherical,
            Self::Exponential { .. } => VariogramType::Exponential,
            Self::Gaussian { .. } => VariogramType::Gaussian,
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
            } => (*nugget, *sill, *range),
        }
    }

    pub fn semivariance(&self, distance: Real) -> Real {
        let d = distance.max(0.0);
        let (nugget, sill, range) = self.params();
        let range = range.max(Real::EPSILON);
        let partial_sill = (sill - nugget).max(0.0);

        match self {
            Self::Spherical { .. } => {
                if d >= range {
                    sill
                } else {
                    nugget + partial_sill * (1.5 * (d / range) - 0.5 * (d / range).powi(3))
                }
            }
            Self::Exponential { .. } => nugget + partial_sill * (1.0 - (-3.0 * d / range).exp()),
            Self::Gaussian { .. } => {
                nugget + partial_sill * (1.0 - (-3.0 * (d * d) / (range * range)).exp())
            }
        }
    }

    pub fn covariance(&self, distance: Real) -> Real {
        let (_, sill, _) = self.params();
        (sill - self.semivariance(distance)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spherical_hits_sill_at_range() {
        let model = VariogramModel::Spherical {
            nugget: 0.1,
            sill: 1.0,
            range: 10.0,
        };
        assert_relative_eq!(model.semivariance(10.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(model.semivariance(20.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn exponential_and_gaussian_start_at_nugget() {
        let exp = VariogramModel::Exponential {
            nugget: 0.2,
            sill: 1.2,
            range: 15.0,
        };
        let gauss = VariogramModel::Gaussian {
            nugget: 0.2,
            sill: 1.2,
            range: 15.0,
        };
        assert_relative_eq!(exp.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(gauss.semivariance(0.0), 0.2, epsilon = 1e-6);
    }

    #[test]
    fn covariance_complements_semivariance() {
        let model = VariogramModel::Exponential {
            nugget: 0.1,
            sill: 1.0,
            range: 5.0,
        };
        let d = 2.2;
        assert_relative_eq!(
            model.covariance(d) + model.semivariance(d),
            1.0,
            epsilon = 1e-5
        );
    }
}
