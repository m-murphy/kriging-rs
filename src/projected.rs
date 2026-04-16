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

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::num::NonZeroUsize;

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::kriging::ordinary::{Prediction, kriging_diagonal_jitter};
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
#[derive(Debug)]
pub struct ProjectedKrigingModel {
    coords: Vec<ProjectedCoord>,
    values: Vec<Real>,
    variogram: VariogramModel,
    anisotropy: Anisotropy2D,
    cov_at_zero: Real,
    system: DMatrix<Real>,
    system_lu: LU<Real, Dyn, Dyn>,
}

impl Clone for ProjectedKrigingModel {
    fn clone(&self) -> Self {
        let system = self.system.clone();
        let system_lu = system.clone().lu();
        Self {
            coords: self.coords.clone(),
            values: self.values.clone(),
            variogram: self.variogram,
            anisotropy: self.anisotropy,
            cov_at_zero: self.cov_at_zero,
            system,
            system_lu,
        }
    }
}

impl ProjectedKrigingModel {
    /// Build a projected ordinary kriging model. Pass [`Anisotropy2D::isotropic`] to disable
    /// anisotropy.
    pub fn new(
        dataset: ProjectedDataset,
        variogram: VariogramModel,
        anisotropy: Anisotropy2D,
    ) -> Result<Self, KrigingError> {
        let coords = dataset.coords;
        let values = dataset.values;
        let n = coords.len();
        let mut system = DMatrix::from_element(n + 1, n + 1, 0.0);
        let diag_eps = kriging_diagonal_jitter(n, variogram);
        for i in 0..n {
            for j in i..n {
                let d = anisotropy.distance(coords[i], coords[j]);
                let mut cov = variogram.covariance(d);
                if i == j {
                    cov += diag_eps;
                }
                system[(i, j)] = cov;
                system[(j, i)] = cov;
            }
            system[(i, n)] = 1.0;
            system[(n, i)] = 1.0;
        }
        system[(n, n)] = 0.0;

        let system_lu = system.clone().lu();
        let mut probe = DVector::from_element(n + 1, 0.0);
        probe[n] = 1.0;
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize projected kriging system".to_string(),
            ));
        }

        Ok(Self {
            coords,
            values,
            variogram,
            anisotropy,
            cov_at_zero: variogram.covariance(0.0),
            system,
            system_lu,
        })
    }

    pub fn anisotropy(&self) -> Anisotropy2D {
        self.anisotropy
    }

    pub fn predict(&self, coord: ProjectedCoord) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
        self.predict_with_rhs(coord, &mut rhs)
    }

    pub fn predict_batch(
        &self,
        coords: &[ProjectedCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let n = self.coords.len();
            coords
                .par_iter()
                .map(|c| {
                    let mut rhs = DVector::from_element(n + 1, 0.0);
                    self.predict_with_rhs(*c, &mut rhs)
                })
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(self.coords.len() + 1, 0.0);
            let mut out = Vec::with_capacity(coords.len());
            for &c in coords {
                out.push(self.predict_with_rhs(c, &mut rhs)?);
            }
            Ok(out)
        }
    }

    fn predict_with_rhs(
        &self,
        coord: ProjectedCoord,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        let n = self.coords.len();
        for i in 0..n {
            let d = self.anisotropy.distance(self.coords[i], coord);
            rhs[i] = self.variogram.covariance(d);
        }
        rhs[n] = 1.0;
        let sol = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve projected kriging system".to_string())
        })?;
        let mut value: Real = 0.0;
        let mut cov_dot: Real = 0.0;
        for i in 0..n {
            value += sol[i] * self.values[i];
            cov_dot += sol[i] * rhs[i];
        }
        let mu = sol[n];
        let variance = (self.cov_at_zero - cov_dot - mu).max(0.0);
        Ok(Prediction { value, variance })
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
}
