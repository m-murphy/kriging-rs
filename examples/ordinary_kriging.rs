use kriging_rs::{GeoCoord, OrdinaryKrigingModel, VariogramModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coords = vec![
        GeoCoord {
            lat: 37.77,
            lon: -122.42,
        },
        GeoCoord {
            lat: 37.78,
            lon: -122.41,
        },
        GeoCoord {
            lat: 37.76,
            lon: -122.40,
        },
        GeoCoord {
            lat: 37.75,
            lon: -122.43,
        },
    ];
    let values = vec![15.0, 18.0, 14.0, 13.0];
    let variogram = VariogramModel::Exponential {
        nugget: 0.1,
        sill: 6.0,
        range: 5.0,
    };

    let model = OrdinaryKrigingModel::new(coords, values, variogram)?;
    let pred = model.predict(GeoCoord {
        lat: 37.765,
        lon: -122.415,
    })?;

    println!("Predicted value: {:.3}", pred.value);
    println!("Kriging variance: {:.6}", pred.variance);
    Ok(())
}
