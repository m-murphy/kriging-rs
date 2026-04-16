//! Space–time kriging.
//!
//! Mirrors the 2-D spatial path: [`SpaceTimeCoord`] and [`SpaceTimeDataset`] pair a spatial
//! coordinate (via [`SpatialMetric`]) with a scalar time, [`SpaceTimeVariogram`] provides
//! separable and product-sum covariance families, and the `kriging` submodule hosts the
//! ordinary/simple/universal/binomial ST kriging models.
//!
//! The metric abstraction lets the same generic model serve both geographic and projected
//! data:
//!
//! - [`GeoMetric`] — Haversine distance on [`GeoCoord`](crate::GeoCoord) (kilometers).
//! - [`ProjectedMetric`] — Euclidean distance on
//!   [`ProjectedCoord`](crate::ProjectedCoord) with optional 2-D anisotropy.
//!
//! # Quick example
//!
//! ```rust
//! use kriging_rs::{
//!     GeoCoord, VariogramModel, VariogramType,
//!     spacetime::{
//!         GeoMetric, SpaceTimeCoord, SpaceTimeDataset, SpaceTimeOrdinaryKrigingModel,
//!         SpaceTimeVariogram,
//!     },
//! };
//!
//! # fn main() -> Result<(), kriging_rs::KrigingError> {
//! let coords = vec![
//!     SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0)?, 0.0)?,
//!     SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0)?, 1.0)?,
//!     SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0)?, 2.0)?,
//! ];
//! let values = vec![1.0, 2.0, 1.5];
//! let dataset = SpaceTimeDataset::new(coords, values)?;
//! let spatial = VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential)?;
//! let temporal = VariogramModel::new(0.01, 1.0, 5.0, VariogramType::Exponential)?;
//! let variogram = SpaceTimeVariogram::new_separable(spatial, temporal)?;
//! let model = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, variogram)?;
//! let pred = model.predict(SpaceTimeCoord::try_new(GeoCoord::try_new(0.3, 0.3)?, 1.0)?)?;
//! # let _ = pred.value;
//! # Ok(())
//! # }
//! ```

pub mod coord;
pub mod dataset;
pub mod empirical;
pub mod fitting;
pub mod kriging;
pub mod metric;
pub mod variogram;

pub use coord::SpaceTimeCoord;
pub use dataset::SpaceTimeDataset;
pub use empirical::{
    EmpiricalSpaceTimeVariogram, SpaceTimeVariogramConfig, compute_empirical_spacetime_variogram,
    spatial_marginal, temporal_marginal,
};
pub use fitting::{
    SpaceTimeFitConfig, SpaceTimeFitResult, SpaceTimeVariogramType, fit_spacetime_variogram,
};
pub use kriging::binomial::{SpaceTimeBinomialKrigingModel, SpaceTimeBinomialObservation};
pub use kriging::ordinary::SpaceTimeOrdinaryKrigingModel;
pub use kriging::simple::SpaceTimeSimpleKrigingModel;
pub use kriging::universal::{SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend};
pub use metric::{GeoMetric, ProjectedMetric, SpatialBasis, SpatialMetric};
pub use variogram::SpaceTimeVariogram;
