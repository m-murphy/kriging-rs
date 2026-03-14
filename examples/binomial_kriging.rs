use kriging_rs::{BinomialKrigingModel, BinomialObservation, GeoCoord, VariogramModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observations = vec![
        BinomialObservation {
            coord: GeoCoord {
                lat: 40.71,
                lon: -74.00,
            },
            successes: 25,
            trials: 50,
        },
        BinomialObservation {
            coord: GeoCoord {
                lat: 40.72,
                lon: -73.99,
            },
            successes: 35,
            trials: 50,
        },
        BinomialObservation {
            coord: GeoCoord {
                lat: 40.70,
                lon: -74.02,
            },
            successes: 20,
            trials: 50,
        },
    ];

    let variogram = VariogramModel::Gaussian {
        nugget: 0.01,
        sill: 1.5,
        range: 10.0,
    };
    let model = BinomialKrigingModel::new(observations, variogram)?;
    let pred = model.predict(GeoCoord {
        lat: 40.715,
        lon: -74.005,
    })?;

    println!("Predicted prevalence: {:.4}", pred.prevalence);
    println!("Predicted logit: {:.4}", pred.logit_value);
    println!("Prediction variance: {:.6}", pred.variance);
    Ok(())
}
