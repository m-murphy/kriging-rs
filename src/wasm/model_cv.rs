//! Cross-validation from fitted kriging models (instance `leaveOneOut` / `kFold`).

use wasm_bindgen::prelude::*;

use crate::cv::BinomialGeoPredictor;
use crate::kriging::binomial::BinomialPrior;
use crate::kriging::universal::UniversalTrend;
use crate::predictor::cv::{
    BinomialProjectedPredictor, OrdinaryGeoPredictor, ProjectedOrdinaryPredictor,
    SimpleGeoPredictor, SpacetimeBinomialPredictor, SpacetimeOrdinaryPredictor,
    SpacetimeSimplePredictor, SpacetimeUniversalPredictor, UniversalGeoPredictor, k_fold_cv,
    leave_one_out_cv,
};
use crate::projected::Anisotropy2D;
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::kriging::universal::SpaceTimeUniversalTrend;
use crate::spacetime::metric::{GeoMetric, ProjectedMetric, SpatialMetric};
use crate::spacetime::variogram::SpaceTimeVariogram;
use crate::variogram::models::VariogramModel;
use crate::{Real, distance::GeoCoord, projected::ProjectedCoord};

use super::{binomial_cv_result_to_js, coded_err, cv_result_to_js, kriging_err_to_js};

fn run_continuous_cv<P>(predictor: &P, k: Option<usize>) -> Result<JsValue, JsValue>
where
    P: crate::predictor::cv::KrigingPredictor<Residual = crate::cv::CvResidual>,
{
    let residuals = match k {
        None => leave_one_out_cv(predictor).map_err(kriging_err_to_js)?,
        Some(k) => k_fold_cv(predictor, k).map_err(kriging_err_to_js)?,
    };
    cv_result_to_js(residuals)
}

fn run_binomial_cv<P>(predictor: &P, k: Option<usize>) -> Result<JsValue, JsValue>
where
    P: crate::predictor::cv::KrigingPredictor<Residual = crate::cv::BinomialCvResidual>,
{
    let residuals = match k {
        None => leave_one_out_cv(predictor).map_err(kriging_err_to_js)?,
        Some(k) => k_fold_cv(predictor, k).map_err(kriging_err_to_js)?,
    };
    binomial_cv_result_to_js(residuals)
}

pub(crate) fn validate_k_fold(k: usize, n: usize) -> Result<(), JsValue> {
    if k < 2 {
        return Err(coded_err("k must be >= 2 for k-fold CV", "invalid_input"));
    }
    if k > n {
        return Err(coded_err(
            &format!("k ({k}) cannot exceed number of stations ({n})"),
            "invalid_input",
        ));
    }
    Ok(())
}

pub(crate) fn ordinary_geo_cv(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = OrdinaryGeoPredictor {
        coords,
        values,
        variogram,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn simple_geo_cv(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    mean: Real,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = SimpleGeoPredictor {
        coords,
        values,
        variogram,
        mean,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn universal_geo_cv(
    coords: &[GeoCoord],
    values: &[Real],
    variogram: VariogramModel,
    trend: UniversalTrend,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = UniversalGeoPredictor {
        coords,
        values,
        variogram,
        trend,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn projected_ordinary_cv(
    coords: &[ProjectedCoord],
    values: &[Real],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = ProjectedOrdinaryPredictor {
        coords,
        values,
        variogram,
        anisotropy,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn binomial_geo_cv(
    coords: &[GeoCoord],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    prior: crate::kriging::binomial::BinomialPrior,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = BinomialGeoPredictor {
        coords,
        successes,
        trials,
        variogram,
        prior,
    };
    run_binomial_cv(&predictor, k)
}

pub(crate) fn binomial_projected_cv(
    coords: &[ProjectedCoord],
    successes: &[u32],
    trials: &[u32],
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    prior: crate::kriging::binomial::BinomialPrior,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = BinomialProjectedPredictor {
        coords,
        successes,
        trials,
        variogram,
        anisotropy,
        prior,
    };
    run_binomial_cv(&predictor, k)
}

pub(crate) fn spacetime_ordinary_cv<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = SpacetimeOrdinaryPredictor {
        metric,
        coords,
        values,
        variogram,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn spacetime_simple_cv<M: SpatialMetric>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    mean: Real,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = SpacetimeSimplePredictor {
        metric,
        coords,
        values,
        variogram,
        mean,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn spacetime_universal_cv<M: SpatialMetric + crate::spacetime::metric::SpatialBasis>(
    metric: M,
    coords: &[SpaceTimeCoord<M::Coord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    trend: SpaceTimeUniversalTrend,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = SpacetimeUniversalPredictor {
        metric,
        coords,
        values,
        variogram,
        trend,
    };
    run_continuous_cv(&predictor, k)
}

pub(crate) fn spacetime_binomial_geo_cv(
    coords: &[SpaceTimeCoord<GeoCoord>],
    successes: &[u32],
    trials: &[u32],
    variogram: SpaceTimeVariogram,
    prior: BinomialPrior,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    if let Some(k) = k {
        validate_k_fold(k, coords.len())?;
    }
    let predictor = SpacetimeBinomialPredictor {
        metric: GeoMetric,
        coords,
        successes,
        trials,
        variogram,
        prior,
    };
    run_binomial_cv(&predictor, k)
}

pub(crate) fn spacetime_ordinary_projected_cv(
    anisotropy: Anisotropy2D,
    coords: &[SpaceTimeCoord<ProjectedCoord>],
    values: &[Real],
    variogram: SpaceTimeVariogram,
    k: Option<usize>,
) -> Result<JsValue, JsValue> {
    spacetime_ordinary_cv(
        ProjectedMetric::with_anisotropy(anisotropy),
        coords,
        values,
        variogram,
        k,
    )
}
