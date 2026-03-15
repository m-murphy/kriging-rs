use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::utils::{Probability, logistic, logit};
use crate::variogram::models::VariogramModel;

#[derive(Debug, Clone, Copy)]
pub struct BinomialObservation {
    coord: GeoCoord,
    successes: u32,
    trials: u32,
}

impl BinomialObservation {
    /// Creates an observation with validated `trials > 0` and `successes <= trials`.
    pub fn new(coord: GeoCoord, successes: u32, trials: u32) -> Result<Self, KrigingError> {
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
    pub fn coord(self) -> GeoCoord {
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

    pub fn smoothed_probability(&self) -> Real {
        self.smoothed_probability_with_prior(BinomialPrior::default())
    }

    pub fn smoothed_probability_with_prior(&self, prior: BinomialPrior) -> Real {
        let s = self.successes as Real;
        let n = self.trials as Real;
        (s + prior.alpha) / (n + prior.alpha + prior.beta)
    }

    pub fn smoothed_logit(&self) -> Real {
        self.smoothed_logit_with_prior(BinomialPrior::default())
    }

    pub fn smoothed_logit_with_prior(&self, prior: BinomialPrior) -> Real {
        let p = self.smoothed_probability_with_prior(prior);
        logit(Probability::from_known_in_range(p))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialPrior {
    alpha: Real,
    beta: Real,
}

impl Default for BinomialPrior {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }
}

impl BinomialPrior {
    /// Creates a prior with validated `alpha > 0` and `beta > 0`.
    pub fn new(alpha: Real, beta: Real) -> Result<Self, KrigingError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(KrigingError::InvalidBinomialData(
                "prior alpha must be finite and positive".to_string(),
            ));
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(KrigingError::InvalidBinomialData(
                "prior beta must be finite and positive".to_string(),
            ));
        }
        Ok(Self { alpha, beta })
    }

    #[inline]
    pub fn alpha(self) -> Real {
        self.alpha
    }

    #[inline]
    pub fn beta(self) -> Real {
        self.beta
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinomialPrediction {
    pub prevalence: Real,
    pub logit_value: Real,
    pub variance: Real,
}

#[derive(Debug, Clone)]
pub struct BinomialKrigingModel {
    ordinary_model: OrdinaryKrigingModel,
}

impl BinomialKrigingModel {
    pub fn new(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        Self::new_with_prior(observations, variogram, BinomialPrior::default())
    }

    pub fn new_with_prior(
        observations: Vec<BinomialObservation>,
        variogram: VariogramModel,
        prior: BinomialPrior,
    ) -> Result<Self, KrigingError> {
        if observations.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let coords = observations.iter().map(|o| o.coord()).collect::<Vec<_>>();
        let logits = observations
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect::<Vec<_>>();
        Self::from_precomputed_logits(coords, logits, variogram)
    }

    pub(crate) fn from_precomputed_logits(
        coords: Vec<GeoCoord>,
        logits: Vec<Real>,
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        let dataset = GeoDataset::new(coords, logits)?;
        let ordinary_model = OrdinaryKrigingModel::new(dataset, variogram)?;
        Ok(Self { ordinary_model })
    }

    pub fn predict(&self, coord: GeoCoord) -> Result<BinomialPrediction, KrigingError> {
        let pred = self.ordinary_model.predict(coord)?;
        Ok(BinomialPrediction {
            prevalence: logistic(pred.value),
            logit_value: pred.value,
            variance: pred.variance,
        })
    }

    pub fn predict_batch(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch(coords)?;
        Ok(ordinary
            .into_iter()
            .map(|pred| BinomialPrediction {
                prevalence: logistic(pred.value),
                logit_value: pred.value,
                variance: pred.variance,
            })
            .collect())
    }

    #[cfg(feature = "gpu")]
    pub async fn predict_batch_gpu(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu(coords).await?;
        Ok(ordinary
            .into_iter()
            .map(|pred| BinomialPrediction {
                prevalence: logistic(pred.value),
                logit_value: pred.value,
                variance: pred.variance,
            })
            .collect())
    }

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    pub fn predict_batch_gpu_blocking(
        &self,
        coords: &[GeoCoord],
    ) -> Result<Vec<BinomialPrediction>, KrigingError> {
        let ordinary = self.ordinary_model.predict_batch_gpu_blocking(coords)?;
        Ok(ordinary
            .into_iter()
            .map(|pred| BinomialPrediction {
                prevalence: logistic(pred.value),
                logit_value: pred.value,
                variance: pred.variance,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_zero_and_all_successes_with_smoothing() {
        let o1 = BinomialObservation::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0, 10).unwrap();
        let o2 = BinomialObservation::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 10, 10).unwrap();
        let p1 = o1.smoothed_probability();
        let p2 = o2.smoothed_probability();
        assert!(p1 > 0.0 && p1 < 1.0);
        assert!(p2 > 0.0 && p2 < 1.0);
    }
}
