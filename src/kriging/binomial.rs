use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;
use crate::kriging::ordinary::OrdinaryKrigingModel;
use crate::utils::logistic;
use crate::variogram::models::VariogramModel;

#[derive(Debug, Clone, Copy)]
pub struct BinomialObservation {
    pub coord: GeoCoord,
    pub successes: u32,
    pub trials: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialPrior {
    pub alpha: Real,
    pub beta: Real,
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
    pub fn validate(self) -> Result<(), KrigingError> {
        if self.alpha <= 0.0 || self.beta <= 0.0 {
            return Err(KrigingError::InvalidBinomialData(format!(
                "prior alpha={} and beta={} must both be > 0",
                self.alpha, self.beta
            )));
        }
        Ok(())
    }
}

impl BinomialObservation {
    pub fn smoothed_probability(&self) -> Result<Real, KrigingError> {
        self.smoothed_probability_with_prior(BinomialPrior::default())
    }

    pub fn smoothed_probability_with_prior(
        &self,
        prior: BinomialPrior,
    ) -> Result<Real, KrigingError> {
        if self.trials == 0 || self.successes > self.trials {
            return Err(KrigingError::InvalidBinomialData(format!(
                "successes={} and trials={} are invalid",
                self.successes, self.trials
            )));
        }
        prior.validate()?;
        let s = self.successes as Real;
        let n = self.trials as Real;
        Ok((s + prior.alpha) / (n + prior.alpha + prior.beta))
    }

    pub fn smoothed_logit(&self) -> Result<Real, KrigingError> {
        self.smoothed_logit_with_prior(BinomialPrior::default())
    }

    pub fn smoothed_logit_with_prior(&self, prior: BinomialPrior) -> Result<Real, KrigingError> {
        let p = self.smoothed_probability_with_prior(prior)?;
        Ok((p / (1.0 - p)).ln())
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
        prior.validate()?;
        let coords = observations.iter().map(|o| o.coord).collect::<Vec<_>>();
        let logits = observations
            .iter()
            .map(|o| o.smoothed_logit_with_prior(prior))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_precomputed_logits(coords, logits, variogram)
    }

    pub(crate) fn from_precomputed_logits(
        coords: Vec<GeoCoord>,
        logits: Vec<Real>,
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let ordinary_model = OrdinaryKrigingModel::new(coords, logits, variogram)?;
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
        let o1 = BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 0.0 },
            successes: 0,
            trials: 10,
        };
        let o2 = BinomialObservation {
            coord: GeoCoord { lat: 0.0, lon: 1.0 },
            successes: 10,
            trials: 10,
        };
        let p1 = o1.smoothed_probability().expect("valid");
        let p2 = o2.smoothed_probability().expect("valid");
        assert!(p1 > 0.0 && p1 < 1.0);
        assert!(p2 > 0.0 && p2 < 1.0);
    }
}
