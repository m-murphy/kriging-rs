use kriging_rs::{GeoCoord, GeoDataset, OrdinaryKrigingModel, VariogramModel, VariogramType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coords = vec![
        GeoCoord::try_new(37.77, -122.42)?,
        GeoCoord::try_new(37.78, -122.41)?,
        GeoCoord::try_new(37.76, -122.40)?,
        GeoCoord::try_new(37.75, -122.43)?,
    ];
    let values = vec![15.0, 18.0, 14.0, 13.0];
    let variogram = VariogramModel::new(0.1, 6.0, 5.0, VariogramType::Exponential)?;
    let dataset = GeoDataset::new(coords, values)?;

    let model = OrdinaryKrigingModel::new(dataset, variogram)?;
    let pred = model.predict(GeoCoord::try_new(37.765, -122.415)?)?;

    println!("Predicted value: {:.3}", pred.value);
    println!("Kriging variance: {:.6}", pred.variance);
    Ok(())
}
