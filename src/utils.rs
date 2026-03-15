use crate::Real;

const PROB_EPSILON: Real = 1e-9;

/// A probability in (0, 1), enforced at construction. Use for `logit` without clamping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(Real);

impl Probability {
    /// Creates a probability. Fails if not in (0, 1) or not finite.
    pub fn try_new(p: Real) -> Result<Self, &'static str> {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err("probability must be finite and in (0, 1)");
        }
        Ok(Self(p))
    }

    /// For callers that have already ensured the value is in (0, 1) (e.g. from smoothed probability).
    #[inline]
    pub fn from_known_in_range(p: Real) -> Self {
        debug_assert!(
            p.is_finite() && p > 0.0 && p < 1.0,
            "probability must be in (0, 1)"
        );
        Self(p)
    }

    #[inline]
    pub fn get(self) -> Real {
        self.0
    }
}

pub fn clamp_probability(p: Real) -> Real {
    p.clamp(PROB_EPSILON, 1.0 - PROB_EPSILON)
}

/// Logit of a validated probability. No clamping; use `Probability::try_new` at the boundary.
#[inline]
pub fn logit(p: Probability) -> Real {
    let p = p.get();
    (p / (1.0 - p)).ln()
}

/// Logit of a raw value, with clamping. Prefer building a `Probability` and using `logit(Probability)` when the value is already in (0, 1).
pub fn logit_clamped(p: Real) -> Real {
    logit(Probability::from_known_in_range(clamp_probability(p)))
}

pub fn logistic(x: Real) -> Real {
    1.0 / (1.0 + (-x).exp())
}
