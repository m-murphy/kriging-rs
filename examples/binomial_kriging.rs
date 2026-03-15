use kriging_rs::{BinomialKrigingModel, BinomialObservation, GeoCoord, VariogramModel, VariogramType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observations = vec![
        BinomialObservation::new(GeoCoord::try_new(40.71, -74.00)?, 25, 50)?,
        BinomialObservation::new(GeoCoord::try_new(40.72, -73.99)?, 35, 50)?,
        BinomialObservation::new(GeoCoord::try_new(40.70, -74.02)?, 20, 50)?,
    ];

    let variogram = VariogramModel::new(0.01, 1.5, 10.0, VariogramType::Gaussian)?;
    let model = BinomialKrigingModel::new(observations, variogram)?;
    let pred = model.predict(GeoCoord::try_new(40.715, -74.005)?)?;

    println!("Predicted prevalence: {:.4}", pred.prevalence);
    println!("Predicted logit: {:.4}", pred.logit_value);
    println!("Prediction variance: {:.6}", pred.variance);
    Ok(())
}
