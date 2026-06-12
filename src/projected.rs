//! Projected (planar) coordinates with Euclidean distance and 2-D anisotropy.
//!
//! The rest of the crate is geographic (`GeoCoord` + Haversine). This module adds:
//!
//! - [`ProjectedCoord`] — a Cartesian `(x, y)` pair in any linear unit (meters, km, etc.).
//! - [`euclidean_distance`] and [`euclidean_distance_squared`].
//! - [`Anisotropy2D`] — an angle + ratio that deforms distance to model direction-dependent
//!   range (e.g. stronger correlation along a 30° azimuth than perpendicular to it).
//!
//! A companion [`ProjectedKrigingModel`] performs ordinary kriging on projected coordinates,
//! optionally with an anisotropy. Geographic data can be converted to projected data with
//! [`ProjectedCoord::equirectangular`] for small-area work.

use std::num::NonZeroUsize;

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::kriging::binomial::{
    BINOMIAL_CALIBRATION_VERSION, BinomialBuildNotes, BinomialCalibratedResult,
    BinomialObservation, BinomialPrediction, BinomialPrior, HeteroskedasticBinomialConfig,
    binomial_prediction_from_ordinary, build_calibrated_logit_ordinary, finish_binomial_notes,
};

use crate::kriging::engine::OrdinaryKrigingEngine;
use crate::kriging::ordinary::Prediction;
use crate::spacetime::metric::ProjectedMetric;
use crate::utils::{Probability, logit};
use crate::variogram::empirical::{EmpiricalVariogram, PositiveReal};
use crate::variogram::models::VariogramModel;

/// A planar (projected) coordinate. Units are whatever the user supplies, as long as they
/// match the variogram's range units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedCoord {
    pub x: Real,
    pub y: Real,
}

impl ProjectedCoord {
    pub fn new(x: Real, y: Real) -> Self {
        Self { x, y }
    }

    /// Equirectangular projection of a [`GeoCoord`] around a reference point. Returns `(x, y)`
    /// in kilometers. Accurate for small areas; not suitable for global-scale data.
    ///
    /// `x = R · (lon − lon₀) · cos(lat₀)`, `y = R · (lat − lat₀)` with `R = 6371 km`.
    pub fn equirectangular(coord: GeoCoord, reference: GeoCoord) -> Self {
        const EARTH_R_KM: Real = 6_371.0;
        let cos_lat0 = reference.lat().to_radians().cos();
        let dlat = (coord.lat() - reference.lat()).to_radians();
        let dlon = (coord.lon() - reference.lon()).to_radians();
        Self {
            x: EARTH_R_KM * dlon * cos_lat0,
            y: EARTH_R_KM * dlat,
        }
    }
}

/// Squared Euclidean distance. Cheaper than [`euclidean_distance`] when only ordering matters
/// or when the caller will square-root afterwards.
#[inline]
pub fn euclidean_distance_squared(a: ProjectedCoord, b: ProjectedCoord) -> Real {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

/// Straight-line distance between two planar points.
#[inline]
pub fn euclidean_distance(a: ProjectedCoord, b: ProjectedCoord) -> Real {
    euclidean_distance_squared(a, b).sqrt()
}

/// Geometric anisotropy in 2-D: direction-dependent range.
///
/// `major_angle_deg` is the orientation (counter-clockwise from the +x axis) of the axis of
/// **maximum** continuity, in degrees. `range_ratio = minor_range / major_range ∈ (0, 1]` is
/// the ratio of minor-axis to major-axis correlation range. `range_ratio = 1` is isotropic.
///
/// The effective distance between two points in anisotropy space is
///
/// ```text
///   h' = sqrt((h_major)² + (h_minor / range_ratio)²)
/// ```
///
/// so the minor-axis separation is scaled up (shortening correlation) by `1 / range_ratio`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anisotropy2D {
    pub major_angle_deg: Real,
    pub range_ratio: Real,
}

impl Anisotropy2D {
    /// Construct a new anisotropy. Returns an error if `range_ratio` is not in `(0, 1]`
    /// or is non-finite.
    pub fn new(major_angle_deg: Real, range_ratio: Real) -> Result<Self, KrigingError> {
        if !range_ratio.is_finite() || range_ratio <= 0.0 || range_ratio > 1.0 {
            return Err(KrigingError::InvalidInput(format!(
                "range_ratio must be in (0, 1], got {range_ratio}"
            )));
        }
        if !major_angle_deg.is_finite() {
            return Err(KrigingError::InvalidInput(
                "major_angle_deg must be finite".to_string(),
            ));
        }
        Ok(Self {
            major_angle_deg,
            range_ratio,
        })
    }

    /// Isotropic anisotropy (no deformation). Equivalent to unmodified Euclidean distance.
    pub fn isotropic() -> Self {
        Self {
            major_angle_deg: 0.0,
            range_ratio: 1.0,
        }
    }

    /// Effective (anisotropy-adjusted) distance between two planar points.
    pub fn distance(self, a: ProjectedCoord, b: ProjectedCoord) -> Real {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let theta = self.major_angle_deg.to_radians();
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        // Rotate so the major axis aligns with +x.
        let h_major = dx * cos_t + dy * sin_t;
        let h_minor = -dx * sin_t + dy * cos_t;
        let inv_r = 1.0 / self.range_ratio;
        (h_major * h_major + (h_minor * inv_r) * (h_minor * inv_r)).sqrt()
    }
}

/// Directional empirical variogram configuration for planar data.
///
/// Pairs are accepted only when the vector from `i → j` lies within `tolerance_deg` of
/// `azimuth_deg` (or its 180°-mirror — the variogram is symmetric under direction reversal).
/// `azimuth_deg` is measured counter-clockwise from +x (i.e. "east"). Use
/// `azimuth_deg = 0` for east–west, `90` for north–south.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalConfig {
    pub azimuth_deg: Real,
    pub tolerance_deg: Real,
}

impl DirectionalConfig {
    pub fn new(azimuth_deg: Real, tolerance_deg: Real) -> Result<Self, KrigingError> {
        if !tolerance_deg.is_finite() || tolerance_deg <= 0.0 || tolerance_deg > 90.0 {
            return Err(KrigingError::InvalidInput(format!(
                "tolerance_deg must be in (0, 90], got {tolerance_deg}"
            )));
        }
        if !azimuth_deg.is_finite() {
            return Err(KrigingError::InvalidInput(
                "azimuth_deg must be finite".to_string(),
            ));
        }
        Ok(Self {
            azimuth_deg,
            tolerance_deg,
        })
    }
}

/// Compute a directional empirical variogram on projected coordinates. Only pairs whose
/// separation vector falls within the angular tolerance around `direction.azimuth_deg`
/// contribute. See [`compute_empirical_variogram`](crate::compute_empirical_variogram) for
/// the omnidirectional, geographic counterpart.
pub fn compute_directional_empirical_variogram(
    coords: &[ProjectedCoord],
    values: &[Real],
    max_distance: PositiveReal,
    n_bins: NonZeroUsize,
    direction: DirectionalConfig,
) -> Result<EmpiricalVariogram, KrigingError> {
    if coords.len() != values.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({}) and values ({}) must have equal length",
            coords.len(),
            values.len()
        )));
    }
    if coords.is_empty() {
        return Err(KrigingError::InsufficientData(1));
    }
    let n_bins = n_bins.get();
    let bin_width = max_distance.get() / n_bins as Real;
    let mut dist_sums = vec![0.0 as Real; n_bins];
    let mut semi_sums = vec![0.0 as Real; n_bins];
    let mut counts = vec![0usize; n_bins];

    let target = direction.azimuth_deg.to_radians();
    let tol = direction.tolerance_deg.to_radians();

    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let dx = coords[j].x - coords[i].x;
            let dy = coords[j].y - coords[i].y;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 0.0 || d > max_distance.get() {
                continue;
            }
            let angle = dy.atan2(dx);
            // Fold into [-π/2, π/2] since the variogram is symmetric under direction reversal.
            let mut diff = angle - target;
            while diff > std::f64::consts::FRAC_PI_2 as Real {
                diff -= std::f64::consts::PI as Real;
            }
            while diff < -(std::f64::consts::FRAC_PI_2 as Real) {
                diff += std::f64::consts::PI as Real;
            }
            if diff.abs() > tol {
                continue;
            }
            let g = 0.5 * (values[i] - values[j]).powi(2);
            let mut bin = (d / bin_width).floor() as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            dist_sums[bin] += d;
            semi_sums[bin] += g;
            counts[bin] += 1;
        }
    }

    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for i in 0..n_bins {
        if counts[i] == 0 {
            continue;
        }
        distances.push(dist_sums[i] / counts[i] as Real);
        semivariances.push(semi_sums[i] / counts[i] as Real);
        n_pairs.push(counts[i]);
    }
    if distances.is_empty() {
        return Err(KrigingError::FittingError(
            "no pairs in selected direction/distance range".to_string(),
        ));
    }
    Ok(EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    })
}

/// Dataset of projected coordinates and paired values. Construction validates lengths.
#[derive(Debug, Clone)]
pub struct ProjectedDataset {
    pub coords: Vec<ProjectedCoord>,
    pub values: Vec<Real>,
}

impl ProjectedDataset {
    pub fn new(coords: Vec<ProjectedCoord>, values: Vec<Real>) -> Result<Self, KrigingError> {
        if coords.len() != values.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "coords ({}) and values ({}) must have equal length",
                coords.len(),
                values.len()
            )));
        }
        if coords.is_empty() {
            return Err(KrigingError::InsufficientData(1));
        }
        Ok(Self { coords, values })
    }
}

/// Ordinary kriging on projected (planar) coordinates with optional 2-D geometric
/// anisotropy. The variogram's range must be expressed in the same linear units as the
/// coordinates (e.g. km, m).
#[derive(Debug, Clone)]
pub struct ProjectedKrigingModel {
    engine: OrdinaryKrigingEngine<ProjectedMetric>,
}

impl ProjectedKrigingModel {
    /// Build a projected ordinary kriging model. Pass [`Anisotropy2D::isotropic`] to disable
    /// anisotropy.
    pub fn new(
        dataset: ProjectedDataset,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
    ) -> Result<Self, KrigingError> {
        Self::new_with_extra_diagonal_internal(dataset, variogram, anisotropy, &[])
    }

    /// Like [`new`](Self::new) but adds `extra` (length `n`) to each main-diagonal covariance
    /// term, modeling observation-specific (non-spatial) noise in addition to jitter.
    pub fn new_with_extra_diagonal(
        dataset: ProjectedDataset,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
        extra: Vec<Real>,
    ) -> Result<Self, KrigingError> {
        if !extra.is_empty() && extra.len() != dataset.coords.len() {
            return Err(KrigingError::InvalidInput(
                "extra observation diagonal must be empty (homoscedastic) or the same length as the dataset"
                    .to_string(),
            ));
        }
        for &v in &extra {
            if !v.is_finite() || v < 0.0 {
                return Err(KrigingError::InvalidInput(
                    "observation diagonal entries must be finite and non-negative".to_string(),
                ));
            }
        }
        Self::new_with_extra_diagonal_internal(dataset, variogram, anisotropy, &extra)
    }

    fn new_with_extra_diagonal_internal(
        dataset: ProjectedDataset,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
        extra: &[Real],
    ) -> Result<Self, KrigingError> {
        let coords = dataset.coords;
        let values = dataset.values;
        let engine = OrdinaryKrigingEngine::fit_with_extra_diagonal(
            ProjectedMetric::with_anisotropy(anisotropy),
            coords,
            values,
            variogram,
            extra,
        )?;
        Ok(Self { engine })
    }

    pub fn anisotropy(&self) -> Anisotropy2D {
        self.engine.metric().anisotropy
    }

    pub fn variogram(&self) -> VariogramModel {
        self.engine.variogram()
    }

    pub fn len(&self) -> usize {
        self.engine.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn predict(&self, coord: ProjectedCoord) -> Result<Prediction, KrigingError> {
        self.engine
            .predict(&[coord])
            .map(|mut v| v.pop().expect("single prediction"))
    }

    pub fn predict_batch(
        &self,
        coords: &[ProjectedCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        self.engine.predict(coords)
    }
}

// ---------------------------------------------------------------------------
// Binomial kriging on projected coordinates
// ---------------------------------------------------------------------------

/// A single binomial observation on projected (planar) coordinates.
///
/// This is the [`crate::BinomialObservation`] equivalent for
/// [`BinomialProjectedKrigingModel`] / planar prevalence mapping.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedBinomialObservation {
    coord: ProjectedCoord,
    successes: u32,
    trials: u32,
}

impl ProjectedBinomialObservation {
    /// Validates `trials > 0` and `successes <= trials`.
    pub fn new(coord: ProjectedCoord, successes: u32, trials: u32) -> Result<Self, KrigingError> {
        if trials == 0 {
            return Err(KrigingError::InvalidBinomialData(
                "trials must be greater than 0".to_string(),
            ));
        }
        if successes > trials {
            return Err(KrigingError::InvalidBinomialData(format!(
                "successes ({}) cannot exceed trials ({})",
                successes, trials
            )));
        }
        Ok(Self {
            coord,
            successes,
            trials,
        })
    }

    #[inline]
    pub fn coord(self) -> ProjectedCoord {
        self.coord
    }

    #[inline]
    pub fn successes(self) -> u32 {
        self.successes
    }

    #[inline]
    pub fn trials(self) -> u32 {
        self.trials
    }

    pub fn smoothed_probability_with_prior(&self, prior: BinomialPrior) -> Real {
        let s = self.successes as Real;
        let n = self.trials as Real;
        (s + prior.alpha()) / (n + prior.alpha() + prior.beta())
    }

    pub fn smoothed_logit_with_prior(&self, prior: BinomialPrior) -> Real {
        let p = self.smoothed_probability_with_prior(prior);
        logit(Probability::from_known_in_range(p))
    }
}

/// Binomial kriging on projected (planar) coordinates with optional 2-D
/// geometric anisotropy.
///
/// This is the planar/anisotropic counterpart of
/// [`crate::BinomialKrigingModel`]. Kriging happens on the **logit** scale, with
/// per-site logit observation variance (calibrated binomial path) when built from
/// counts. See [`BinomialBuildNotes`].
///
/// Use [`Anisotropy2D::isotropic`] to disable anisotropy.
#[derive(Debug, Clone)]
pub struct BinomialProjectedKrigingModel {
    inner: ProjectedKrigingModel,
}

/// Planar / anisotropic binomial fit: model + [`BinomialBuildNotes`].
pub type ProjectedBinomialFit = BinomialCalibratedResult<BinomialProjectedKrigingModel>;

impl BinomialProjectedKrigingModel {
    /// Build a binomial projected kriging model from `(coord, successes, trials)` with the
    /// default `Beta(1, 1)` prior.
    pub fn new(
        observations: Vec<ProjectedBinomialObservation>,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
    ) -> Result<ProjectedBinomialFit, KrigingError> {
        Self::new_with_prior(
            observations,
            variogram,
            anisotropy,
            BinomialPrior::default(),
            HeteroskedasticBinomialConfig::default(),
        )
    }

    /// As [`new`](Self::new), with an explicit Beta prior. Uses heteroskedastic (calibrated) conditioning.
    pub fn new_with_prior(
        observations: Vec<ProjectedBinomialObservation>,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
        prior: BinomialPrior,
        hetero_config: HeteroskedasticBinomialConfig,
    ) -> Result<ProjectedBinomialFit, KrigingError> {
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let coords: Vec<ProjectedCoord> = observations.iter().map(|o| o.coord()).collect();
        let logits: Vec<Real> = observations
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect();
        let base: Vec<Real> = observations
            .iter()
            .map(|o| {
                hetero_config
                    .raw_logit_observation_variance_binomial_site(prior, o.successes(), o.trials())
                    .max(hetero_config.min_logit_observation_variance)
            })
            .collect();
        build_calibrated_logit_ordinary(
            base,
            &hetero_config,
            prior,
            &[],
            false,
            "binomial projected kriging build failed",
            |extra| {
                let dataset = ProjectedDataset::new(coords.clone(), logits.clone())?;
                ProjectedKrigingModel::new_with_extra_diagonal(
                    dataset, variogram, anisotropy, extra,
                )
                .map(|inner| Self { inner })
            },
        )
    }

    /// Pre-computed logits with no per-trial data (no observation variance on the diagonal
    /// except variogram and jitter). See [`crate::BinomialKrigingModel::from_precomputed_logits`].
    pub fn from_precomputed_logits(
        coords: Vec<ProjectedCoord>,
        logits: Vec<Real>,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
    ) -> Result<ProjectedBinomialFit, KrigingError> {
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "logits must all be finite (no NaN/inf)".to_string(),
            ));
        }
        let dataset = ProjectedDataset::new(coords, logits)?;
        let inner = ProjectedKrigingModel::new(dataset, variogram, anisotropy)?;
        Ok(BinomialCalibratedResult {
            model: Self { inner },
            notes: finish_binomial_notes(BinomialBuildNotes {
                calibration_version: BINOMIAL_CALIBRATION_VERSION,
                logit_inflation: 1.0,
                n_build_attempts: 1,
                prior: BinomialPrior::default(),
                zero_trial_dropped_indices: Vec::new(),
                from_precomputed_logits_only: true,
                warnings: Vec::new(),
                condition_number: None,
                effective_dof: None,
                last_msdr: None,
            }),
        })
    }

    /// Pre-computed logit field with per-site logit observation variances and stability policy.
    pub fn from_precomputed_logits_with_logit_observation_variances(
        coords: Vec<ProjectedCoord>,
        logits: Vec<Real>,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
        base_logit_observation_variance: Vec<Real>,
        config: HeteroskedasticBinomialConfig,
        prior_for_notes: BinomialPrior,
    ) -> Result<ProjectedBinomialFit, KrigingError> {
        if logits.len() != coords.len() {
            return Err(KrigingError::DimensionMismatch(
                "coords and logits must have equal length".to_string(),
            ));
        }
        if base_logit_observation_variance.len() != coords.len() {
            return Err(KrigingError::InvalidInput(
                "logit observation variance must match coords length".to_string(),
            ));
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "logits must all be finite (no NaN/inf)".to_string(),
            ));
        }
        build_calibrated_logit_ordinary(
            base_logit_observation_variance,
            &config,
            prior_for_notes,
            &[],
            false,
            "from_precomputed (projected) with observation variances: build failed",
            |extra| {
                let dataset = ProjectedDataset::new(coords.clone(), logits.clone())?;
                ProjectedKrigingModel::new_with_extra_diagonal(
                    dataset, variogram, anisotropy, extra,
                )
                .map(|inner| Self { inner })
            },
        )
    }

    pub fn anisotropy(&self) -> Anisotropy2D {
        self.inner.anisotropy()
    }

    pub fn variogram(&self) -> VariogramModel {
        self.inner.variogram()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn predict(&self, coord: ProjectedCoord) -> Result<BinomialPrediction, KrigingError> {
        let pred = self.inner.predict(coord)?;
        Ok(binomial_prediction_from_ordinary(pred))
    }

    pub fn predict_batch(
        &self,
        coords: &[ProjectedCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let preds = self.inner.predict_batch(coords)?;
        Ok(preds
            .into_iter()
            .map(binomial_prediction_from_ordinary)
            .collect())
    }
}

/// Simple mean latitude / longitude of sample sites (degrees), for choosing a local tangent
/// plane origin. Intended for **small** study areas where the arithmetic mean is adequate.
pub fn tangent_plane_reference_centroid(
    obs: &[BinomialObservation],
) -> Result<GeoCoord, KrigingError> {
    if obs.is_empty() {
        return Err(KrigingError::InsufficientData(2));
    }
    let mut slat = 0.0 as Real;
    let mut slon = 0.0 as Real;
    for o in obs {
        slat += o.coord().lat();
        slon += o.coord().lon();
    }
    let n = obs.len() as Real;
    GeoCoord::try_new(slat / n, slon / n)
}

/// Geographic binomial kriging on a **local tangent plane**: stations are mapped with
/// [`ProjectedCoord::equirectangular`] about a reference [`GeoCoord`], then the usual
/// [`BinomialProjectedKrigingModel`] (Euclidean distance, optional 2-D anisotropy) is applied on
/// the logit scale.
///
/// Variogram ranges are in **kilometers**, matching the equirectangular `(x, y)` units from
/// [`ProjectedCoord::equirectangular`]. This path is appropriate only when the domain is small
/// enough that planar distortion is negligible; a build note warning is always appended.
#[derive(Debug, Clone)]
pub struct BinomialTangentPlaneKrigingModel {
    reference: GeoCoord,
    inner: BinomialProjectedKrigingModel,
}

/// Tangent-plane geographic binomial fit: model + [`BinomialBuildNotes`].
pub type TangentPlaneBinomialFit = BinomialCalibratedResult<BinomialTangentPlaneKrigingModel>;

impl BinomialTangentPlaneKrigingModel {
    /// Calibrated build with default `Beta(1, 1)` prior and default heteroskedastic config.
    pub fn new(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
        reference: GeoCoord,
        anisotropy: Anisotropy2D,
    ) -> Result<TangentPlaneBinomialFit, KrigingError> {
        Self::new_with_prior(
            observations,
            variogram,
            BinomialPrior::default(),
            HeteroskedasticBinomialConfig::default(),
            &[],
            reference,
            anisotropy,
        )
    }

    /// Calibrated build with an explicit Beta prior.
    pub fn new_with_prior(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
        prior: BinomialPrior,
        config: HeteroskedasticBinomialConfig,
        extra_zero_trial_drops: &[usize],
        reference: GeoCoord,
        anisotropy: Anisotropy2D,
    ) -> Result<TangentPlaneBinomialFit, KrigingError> {
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let projected: Vec<ProjectedBinomialObservation> = observations
            .into_iter()
            .map(|o| {
                let pc = ProjectedCoord::equirectangular(o.coord(), reference);
                ProjectedBinomialObservation::new(pc, o.successes(), o.trials())
            })
            .collect::<Result<Vec<_>, KrigingError>>()?;
        let mut fit = BinomialProjectedKrigingModel::new_with_prior(
            projected, variogram, anisotropy, prior, config,
        )?;
        fit.notes
            .zero_trial_dropped_indices
            .extend_from_slice(extra_zero_trial_drops);
        fit.notes.zero_trial_dropped_indices.sort_unstable();
        if !fit
            .notes
            .warnings
            .iter()
            .any(|w| w == "tangent_plane_equirectangular_small_area")
        {
            fit.notes
                .warnings
                .push("tangent_plane_equirectangular_small_area".to_string());
        }
        Ok(BinomialCalibratedResult {
            model: Self {
                reference,
                inner: fit.model,
            },
            notes: fit.notes,
        })
    }

    #[inline]
    pub fn reference(&self) -> GeoCoord {
        self.reference
    }

    #[inline]
    pub fn anisotropy(&self) -> Anisotropy2D {
        self.inner.anisotropy()
    }

    pub fn variogram(&self) -> VariogramModel {
        self.inner.variogram()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<BinomialPrediction, KrigingError> {
        let pc = ProjectedCoord::equirectangular(coord, self.reference);
        self.inner.predict(pc)
    }

    pub fn predict_batch(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let planar: Vec<ProjectedCoord> = coords
            .iter()
            .copied()
            .map(|g| ProjectedCoord::equirectangular(g, self.reference))
            .collect();
        self.inner.predict_batch(&planar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

    #[test]
    fn euclidean_distance_matches_pythagoras() {
        let a = ProjectedCoord::new(0.0, 0.0);
        let b = ProjectedCoord::new(3.0, 4.0);
        assert!((euclidean_distance(a, b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn anisotropy_rejects_invalid_ratio() {
        assert!(Anisotropy2D::new(0.0, 0.0).is_err());
        assert!(Anisotropy2D::new(0.0, -1.0).is_err());
        assert!(Anisotropy2D::new(0.0, 1.5).is_err());
        assert!(Anisotropy2D::new(Real::NAN, 0.5).is_err());
    }

    #[test]
    fn isotropic_distance_equals_euclidean() {
        let iso = Anisotropy2D::isotropic();
        let a = ProjectedCoord::new(1.0, 2.0);
        let b = ProjectedCoord::new(4.0, 6.0);
        let want = euclidean_distance(a, b);
        assert!((iso.distance(a, b) - want).abs() < 1e-6);
    }

    #[test]
    fn anisotropy_stretches_minor_axis_distance() {
        // Major axis along +x (angle 0), ratio 0.5 => distances along +y are doubled.
        let aniso = Anisotropy2D::new(0.0, 0.5).unwrap();
        let origin = ProjectedCoord::new(0.0, 0.0);
        let along_x = ProjectedCoord::new(1.0, 0.0);
        let along_y = ProjectedCoord::new(0.0, 1.0);
        assert!((aniso.distance(origin, along_x) - 1.0).abs() < 1e-6);
        assert!((aniso.distance(origin, along_y) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn anisotropy_rotation_invariant_at_45_with_isotropic_ratio() {
        let aniso = Anisotropy2D::new(37.0, 1.0).unwrap();
        let a = ProjectedCoord::new(0.0, 0.0);
        let b = ProjectedCoord::new(3.0, 4.0);
        assert!((aniso.distance(a, b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn projected_model_recovers_training_value_at_collocated_point() {
        let coords = vec![
            ProjectedCoord::new(0.0, 0.0),
            ProjectedCoord::new(0.0, 10.0),
            ProjectedCoord::new(10.0, 0.0),
        ];
        let values = vec![10.0, 20.0, 15.0];
        let variogram = VariogramModel::new(0.01, 5.0, 30.0, VariogramType::Exponential).unwrap();
        let dataset = ProjectedDataset::new(coords.clone(), values).unwrap();
        let model = ProjectedKrigingModel::new(dataset, variogram, Anisotropy2D::isotropic())
            .expect("model");
        let pred = model.predict(coords[0]).expect("prediction");
        assert!((pred.value - 10.0).abs() < 1e-3);
        assert!(pred.variance >= 0.0);
    }

    #[test]
    fn anisotropy_makes_along_axis_prediction_closer() {
        // Two stations at (±5, 0); major axis along x with small ratio makes the x-neighbors
        // more influential than the same-distance y-direction would be.
        let coords = vec![
            ProjectedCoord::new(-5.0, 0.0),
            ProjectedCoord::new(5.0, 0.0),
            ProjectedCoord::new(0.0, 5.0),
            ProjectedCoord::new(0.0, -5.0),
        ];
        let values = vec![10.0, 10.0, 30.0, 30.0];
        let variogram = VariogramModel::new(0.01, 1.0, 20.0, VariogramType::Exponential).unwrap();
        let dataset = ProjectedDataset::new(coords.clone(), values.clone()).unwrap();
        // Strong anisotropy: along-x continuity much larger than along-y.
        let iso_model = ProjectedKrigingModel::new(
            ProjectedDataset::new(coords.clone(), values.clone()).unwrap(),
            variogram,
            Anisotropy2D::isotropic(),
        )
        .expect("iso");
        let aniso = Anisotropy2D::new(0.0, 0.2).unwrap();
        let aniso_model = ProjectedKrigingModel::new(dataset, variogram, aniso).expect("aniso");

        // At origin, iso averages all stations (value ~20) while aniso overweights x-axis
        // neighbors (value ~10).
        let target = ProjectedCoord::new(0.0, 0.0);
        let iso_pred = iso_model.predict(target).expect("iso pred");
        let aniso_pred = aniso_model.predict(target).expect("aniso pred");
        assert!(
            aniso_pred.value < iso_pred.value,
            "expected aniso.value={} < iso.value={}",
            aniso_pred.value,
            iso_pred.value
        );
    }

    #[test]
    fn equirectangular_small_area_matches_haversine_within_a_percent() {
        // Project two nearby points and compare Euclidean distance to Haversine.
        let reference = GeoCoord::try_new(10.0, 20.0).unwrap();
        let a = GeoCoord::try_new(10.1, 20.05).unwrap();
        let b = GeoCoord::try_new(10.2, 20.10).unwrap();
        let pa = ProjectedCoord::equirectangular(a, reference);
        let pb = ProjectedCoord::equirectangular(b, reference);
        let planar = euclidean_distance(pa, pb);
        let geo = crate::distance::haversine_distance(a, b);
        let rel_err = (planar - geo).abs() / geo.max(1e-6);
        assert!(rel_err < 0.01, "rel_err={rel_err}");
    }

    #[test]
    fn directional_variogram_only_counts_pairs_in_specified_direction() {
        // Three points arranged east-west and three arranged north-south, all distinct.
        // An east-west direction window should only capture the east-west pairs.
        let coords = vec![
            ProjectedCoord { x: 0.0, y: 0.0 },
            ProjectedCoord { x: 1.0, y: 0.0 },
            ProjectedCoord { x: 2.0, y: 0.0 },
            ProjectedCoord { x: 0.0, y: 10.0 },
            ProjectedCoord { x: 0.0, y: 20.0 },
        ];
        let values = vec![1.0, 2.0, 3.0, 10.0, 20.0];
        let cfg = DirectionalConfig::new(0.0, 5.0).unwrap(); // east, ±5°
        let out = compute_directional_empirical_variogram(
            &coords,
            &values,
            PositiveReal::try_new(5.0).unwrap(),
            NonZeroUsize::new(5).unwrap(),
            cfg,
        )
        .expect("directional variogram");
        // Only the three east-west pairs should be counted (0-1, 1-2, 0-2).
        let total_pairs: usize = out.n_pairs.iter().sum();
        assert_eq!(total_pairs, 3);
    }

    #[test]
    fn directional_config_rejects_invalid_tolerance() {
        assert!(DirectionalConfig::new(0.0, 0.0).is_err());
        assert!(DirectionalConfig::new(0.0, 91.0).is_err());
        assert!(DirectionalConfig::new(Real::NAN, 5.0).is_err());
    }

    #[test]
    fn binomial_projected_smoothed_logit_near_collocated_with_calibrated_diagonal() {
        let obs = vec![
            ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, 0.0), 3, 10).unwrap(),
            ProjectedBinomialObservation::new(ProjectedCoord::new(10.0, 0.0), 7, 10).unwrap(),
            ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, 10.0), 5, 10).unwrap(),
        ];
        let variogram = VariogramModel::new(0.01, 1.0, 30.0, VariogramType::Exponential).unwrap();
        let prior = BinomialPrior::default();
        let model = BinomialProjectedKrigingModel::new_with_prior(
            obs.clone(),
            variogram,
            Anisotropy2D::isotropic(),
            prior,
            HeteroskedasticBinomialConfig::default(),
        )
        .unwrap()
        .model;
        let pred = model.predict(obs[0].coord()).unwrap();
        let expected_logit = obs[0].smoothed_logit_with_prior(prior);
        // Per-site logit observation noise: collocated prediction need not match local EB logit.
        assert!((pred.logit - expected_logit).abs() < 0.35, "logit err");
        assert!(pred.prevalence_median > 0.0 && pred.prevalence_median < 1.0);
        assert!(pred.logit_variance >= 0.0);
        assert!(pred.prevalence_variance >= 0.0);
    }

    #[test]
    fn binomial_projected_anisotropy_pulls_predictions_along_major_axis() {
        // Two stations along x-axis with low prevalence, two along y-axis with high.
        let obs = vec![
            ProjectedBinomialObservation::new(ProjectedCoord::new(-5.0, 0.0), 1, 10).unwrap(),
            ProjectedBinomialObservation::new(ProjectedCoord::new(5.0, 0.0), 1, 10).unwrap(),
            ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, 5.0), 9, 10).unwrap(),
            ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, -5.0), 9, 10).unwrap(),
        ];
        let variogram = VariogramModel::new(0.01, 1.0, 20.0, VariogramType::Exponential).unwrap();
        let iso =
            BinomialProjectedKrigingModel::new(obs.clone(), variogram, Anisotropy2D::isotropic())
                .unwrap()
                .model;
        let aniso = BinomialProjectedKrigingModel::new(
            obs.clone(),
            variogram,
            Anisotropy2D::new(0.0, 0.2).unwrap(),
        )
        .unwrap()
        .model;
        let target = ProjectedCoord::new(0.0, 0.0);
        let iso_pred = iso.predict(target).unwrap();
        let aniso_pred = aniso.predict(target).unwrap();
        // Anisotropy weights x-axis (low prevalence) more heavily.
        assert!(aniso_pred.prevalence_median < iso_pred.prevalence_median);
    }

    #[test]
    fn binomial_projected_rejects_too_few_observations() {
        let one =
            vec![ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, 0.0), 1, 5).unwrap()];
        let variogram = VariogramModel::new(0.01, 1.0, 10.0, VariogramType::Exponential).unwrap();
        let r = BinomialProjectedKrigingModel::new(one, variogram, Anisotropy2D::isotropic());
        assert!(r.is_err());
    }

    #[test]
    fn tangent_plane_reference_centroid_is_mean_geo() {
        use crate::kriging::binomial::BinomialObservation;
        let obs = vec![
            BinomialObservation::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1, 5).unwrap(),
            BinomialObservation::new(GeoCoord::try_new(2.0, 4.0).unwrap(), 1, 5).unwrap(),
        ];
        let c = tangent_plane_reference_centroid(&obs).unwrap();
        assert!((c.lat() - 1.0).abs() < 1e-6);
        assert!((c.lon() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tangent_plane_binomial_matches_manual_projected_path() {
        use crate::kriging::binomial::BinomialObservation;
        let refc = GeoCoord::try_new(40.0, -80.0).unwrap();
        let obs = vec![
            BinomialObservation::new(refc, 2, 8).unwrap(),
            BinomialObservation::new(GeoCoord::try_new(40.02, -80.0).unwrap(), 5, 8).unwrap(),
            BinomialObservation::new(GeoCoord::try_new(40.0, -79.98).unwrap(), 4, 8).unwrap(),
        ];
        let v = VariogramModel::new(0.04, 1.0, 50.0, VariogramType::Exponential).unwrap();
        let prior = BinomialPrior::default();
        let iso = Anisotropy2D::isotropic();
        let cfg = HeteroskedasticBinomialConfig::default();
        let proj_obs: Vec<ProjectedBinomialObservation> = obs
            .iter()
            .map(|o| {
                let pc = ProjectedCoord::equirectangular(o.coord(), refc);
                ProjectedBinomialObservation::new(pc, o.successes(), o.trials()).unwrap()
            })
            .collect();
        let direct = BinomialProjectedKrigingModel::new_with_prior(proj_obs, v, iso, prior, cfg)
            .unwrap()
            .model;
        let tangent = BinomialTangentPlaneKrigingModel::new_with_prior(
            obs.clone(),
            v,
            prior,
            cfg,
            &[],
            refc,
            iso,
        )
        .unwrap()
        .model;

        let q = GeoCoord::try_new(40.01, -79.99).unwrap();
        let pt = ProjectedCoord::equirectangular(q, refc);
        let a = tangent.predict(q).unwrap();
        let b = direct.predict(pt).unwrap();
        assert!(
            (a.logit - b.logit).abs() < 1e-5,
            "logit a={} b={}",
            a.logit,
            b.logit
        );
        assert!(
            (a.prevalence_median - b.prevalence_median).abs() < 1e-5,
            "prev a={} b={}",
            a.prevalence_median,
            b.prevalence_median
        );
    }

    #[test]
    fn tangent_plane_build_appends_small_area_warning() {
        use crate::kriging::binomial::BinomialObservation;
        let refc = GeoCoord::try_new(40.0, -80.0).unwrap();
        let obs = vec![
            BinomialObservation::new(refc, 2, 8).unwrap(),
            BinomialObservation::new(GeoCoord::try_new(40.02, -80.0).unwrap(), 5, 8).unwrap(),
            BinomialObservation::new(GeoCoord::try_new(40.0, -79.98).unwrap(), 4, 8).unwrap(),
        ];
        let v = VariogramModel::new(0.04, 1.0, 50.0, VariogramType::Exponential).unwrap();
        let fit =
            BinomialTangentPlaneKrigingModel::new(obs, v, refc, Anisotropy2D::isotropic()).unwrap();
        assert!(
            fit.notes
                .warnings
                .iter()
                .any(|w| w == "tangent_plane_equirectangular_small_area")
        );
    }
}
