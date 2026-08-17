//! Type alias: space-time universal kriging is the universal engine over space-time
//! pairwise covariance and a space-time trend-basis adapter.

use crate::Real;
use crate::kriging::pairwise::SpaceTimePairwiseCovariance;
use crate::kriging::universal::TrendBasis;
use crate::kriging::universal_engine::UniversalKrigingEngine;
use crate::spacetime::coord::SpaceTimeCoord;
use crate::spacetime::kriging::universal::SpaceTimeUniversalTrend;
use crate::spacetime::metric::SpatialBasis;

/// Evaluates [`SpaceTimeUniversalTrend`] at a space–time site using [`SpatialBasis`].
#[derive(Debug, Clone, Copy)]
pub struct SpaceTimeTrendEval<M: SpatialBasis> {
    metric: M,
    trend: SpaceTimeUniversalTrend,
}

impl<M: SpatialBasis> SpaceTimeTrendEval<M> {
    pub fn new(metric: M, trend: SpaceTimeUniversalTrend) -> Self {
        Self { metric, trend }
    }
}

impl<M: SpatialBasis> TrendBasis for SpaceTimeTrendEval<M> {
    type Site = SpaceTimeCoord<M::Coord>;

    fn n_basis(self) -> usize {
        self.trend.n_basis()
    }

    fn eval(self, site: Self::Site, out: &mut [Real]) {
        let (s1, s2) = self.metric.spatial_components(site.spatial);
        self.trend.eval(s1, s2, site.time, out);
    }
}

/// Universal space-time engine: [`UniversalKrigingEngine`] with space-time covariance and trend adapters.
pub type SpaceTimeUniversalKrigingEngine<M> =
    UniversalKrigingEngine<SpaceTimePairwiseCovariance<M>, SpaceTimeTrendEval<M>>;

#[cfg(test)]
mod golden_tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::spacetime::coord::SpaceTimeCoord;
    use crate::spacetime::dataset::SpaceTimeDataset;
    use crate::spacetime::kriging::universal::{
        SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend,
    };
    use crate::spacetime::metric::GeoMetric;
    use crate::spacetime::variogram::SpaceTimeVariogram;
    use crate::variogram::models::{VariogramModel, VariogramType};

    #[test]
    fn spacetime_linear_in_time_matches_universal_model() {
        let coords = vec![
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 1.0).unwrap(), 3.0),
        ];
        let values = vec![1.0, 4.0, 7.0, 10.0];
        let variogram = SpaceTimeVariogram::new_separable(
            VariogramModel::new(0.05, 1.0, 300.0, VariogramType::Exponential).unwrap(),
            VariogramModel::new(0.05, 1.0, 5.0, VariogramType::Exponential).unwrap(),
        )
        .unwrap();
        let trend = SpaceTimeUniversalTrend::LinearInTime;
        let target = SpaceTimeCoord::new(GeoCoord::try_new(0.25, 0.25).unwrap(), 1.5);

        let golden = SpaceTimeUniversalKrigingModel::new(
            GeoMetric,
            SpaceTimeDataset::new(coords.clone(), values.clone()).unwrap(),
            variogram,
            trend,
        )
        .expect("golden model");
        let golden_pred = golden.predict(target).expect("golden predict");

        let engine = SpaceTimeUniversalKrigingEngine::fit(
            SpaceTimePairwiseCovariance::new(GeoMetric, variogram),
            coords,
            values,
            SpaceTimeTrendEval::new(GeoMetric, trend),
        )
        .expect("engine fit");
        let engine_pred = engine.predict(&[target]).expect("engine predict")[0];

        assert!((engine_pred.value - golden_pred.value).abs() < 1e-3);
        assert!((engine_pred.variance - golden_pred.variance).abs() < 1e-3);
    }
}
