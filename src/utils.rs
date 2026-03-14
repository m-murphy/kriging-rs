use crate::Real;

pub fn clamp_probability(p: Real) -> Real {
    p.clamp(1e-9, 1.0 - 1e-9)
}

pub fn logit(p: Real) -> Real {
    let p = clamp_probability(p);
    (p / (1.0 - p)).ln()
}

pub fn logistic(x: Real) -> Real {
    1.0 / (1.0 + (-x).exp())
}
