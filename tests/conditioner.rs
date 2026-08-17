use kriging_rs::simulation::{SimulationOptions, sequential_gaussian_simulate};
use kriging_rs::spacetime::{
    GeoMetric, SpaceTimeBinomialKrigingModel, SpaceTimeBinomialObservation, SpaceTimeCoord,
    SpaceTimeDataset, SpaceTimeOrdinaryKrigingModel, SpaceTimeSimpleKrigingModel,
    SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend, SpaceTimeVariogram,
};
use kriging_rs::{
    Anisotropy2D, BinomialKrigingModel, BinomialObservation, BinomialProjectedKrigingModel,
    GeoCoord, GeoDataset, HeteroskedasticBinomialConfig, KrigingConditioner, LogitScale,
    Neighborhood, OrdinaryKrigingModel, ProjectedBinomialObservation, ProjectedCoord,
    ProjectedDataset, ProjectedKrigingModel, SimpleKrigingModel, UniversalKrigingModel,
    UniversalTrend, VariogramModel, VariogramType,
};

#[test]
fn appended_condition_changes_subsequent_prediction() {
    let coords = vec![
        GeoCoord::try_new(0.0, 0.0).unwrap(),
        GeoCoord::try_new(0.0, 1.0).unwrap(),
        GeoCoord::try_new(1.0, 0.0).unwrap(),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let model =
        OrdinaryKrigingModel::new(GeoDataset::new(coords, values).unwrap(), variogram).unwrap();
    let target = GeoCoord::try_new(0.4, 0.4).unwrap();

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}

#[test]
fn projected_model_converts_to_a_conditioner() {
    let coords = vec![
        ProjectedCoord::new(0.0, 0.0),
        ProjectedCoord::new(0.0, 10.0),
        ProjectedCoord::new(10.0, 0.0),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let variogram = VariogramModel::new(0.01, 5.0, 100.0, VariogramType::Exponential).unwrap();
    let model = ProjectedKrigingModel::new(
        ProjectedDataset::new(coords, values).unwrap(),
        variogram,
        Anisotropy2D::isotropic(),
    )
    .unwrap();
    let target = ProjectedCoord::new(4.0, 4.0);

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}

#[test]
fn spacetime_ordinary_model_converts_to_a_conditioner() {
    let coords = vec![
        SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
        SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
        SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let spatial = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
    let variogram = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
    let model = SpaceTimeOrdinaryKrigingModel::new(
        GeoMetric,
        SpaceTimeDataset::new(coords, values).unwrap(),
        variogram,
    )
    .unwrap();
    let target = SpaceTimeCoord::new(GeoCoord::try_new(0.4, 0.4).unwrap(), 1.0);

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}

#[test]
fn spacetime_simple_model_converts_to_a_conditioner() {
    let coords = vec![
        SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
        SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
        SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let spatial = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
    let variogram = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
    let model = SpaceTimeSimpleKrigingModel::new(
        GeoMetric,
        SpaceTimeDataset::new(coords, values).unwrap(),
        variogram,
        15.0,
    )
    .unwrap();
    let target = SpaceTimeCoord::new(GeoCoord::try_new(0.4, 0.4).unwrap(), 1.0);

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}

#[test]
fn spacetime_universal_model_converts_to_a_conditioner() {
    let coords = vec![
        SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
        SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
        SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
        SpaceTimeCoord::new(GeoCoord::try_new(1.0, 1.0).unwrap(), 3.0),
        SpaceTimeCoord::new(GeoCoord::try_new(2.0, 0.5).unwrap(), 4.0),
    ];
    let values = coords
        .iter()
        .map(|coord| 1.0 + 2.0 * coord.spatial.lat() + 3.0 * coord.spatial.lon() + coord.time)
        .collect();
    let spatial = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
    let variogram = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
    let model = SpaceTimeUniversalKrigingModel::new(
        GeoMetric,
        SpaceTimeDataset::new(coords, values).unwrap(),
        variogram,
        SpaceTimeUniversalTrend::LinearInSpaceAndTime,
    )
    .unwrap();
    let target = SpaceTimeCoord::new(GeoCoord::try_new(0.4, 0.4).unwrap(), 1.5);

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}

#[test]
fn binomial_model_converts_to_a_logit_conditioner() {
    let observations = vec![
        BinomialObservation::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 3, 10).unwrap(),
        BinomialObservation::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 7, 12).unwrap(),
        BinomialObservation::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 4, 9).unwrap(),
    ];
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let fit = BinomialKrigingModel::new(observations, variogram).unwrap();
    let target = GeoCoord::try_new(0.4, 0.4).unwrap();
    let expected = fit.predict(target).unwrap();

    let mut conditioner: KrigingConditioner<GeoCoord, LogitScale> =
        fit.into_model().into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 2.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert_eq!(before.mean, expected.logit);
    assert_eq!(before.variance, expected.logit_variance);
    assert!(after.variance < before.variance);
    assert!((after.mean - 2.0).abs() < 0.1);
}

#[test]
fn projected_binomial_model_converts_to_a_logit_conditioner() {
    let observations = vec![
        ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, 0.0), 3, 10).unwrap(),
        ProjectedBinomialObservation::new(ProjectedCoord::new(0.0, 10.0), 7, 12).unwrap(),
        ProjectedBinomialObservation::new(ProjectedCoord::new(10.0, 0.0), 4, 9).unwrap(),
    ];
    let variogram = VariogramModel::new(0.01, 5.0, 100.0, VariogramType::Exponential).unwrap();
    let fit =
        BinomialProjectedKrigingModel::new(observations, variogram, Anisotropy2D::isotropic())
            .unwrap();
    let target = ProjectedCoord::new(4.0, 4.0);

    let mut conditioner: KrigingConditioner<ProjectedCoord, LogitScale> =
        fit.into_model().into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 2.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 2.0).abs() < 0.1);
}

#[test]
fn spacetime_binomial_model_converts_to_a_logit_conditioner() {
    let observations = vec![
        SpaceTimeBinomialObservation::new(
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0),
            3,
            10,
        )
        .unwrap(),
        SpaceTimeBinomialObservation::new(
            SpaceTimeCoord::new(GeoCoord::try_new(0.0, 1.0).unwrap(), 1.0),
            7,
            12,
        )
        .unwrap(),
        SpaceTimeBinomialObservation::new(
            SpaceTimeCoord::new(GeoCoord::try_new(1.0, 0.0).unwrap(), 2.0),
            4,
            9,
        )
        .unwrap(),
    ];
    let spatial = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let temporal = VariogramModel::new(0.01, 2.0, 5.0, VariogramType::Exponential).unwrap();
    let variogram = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
    let fit = SpaceTimeBinomialKrigingModel::new(
        GeoMetric,
        observations,
        variogram,
        HeteroskedasticBinomialConfig::default(),
    )
    .unwrap();
    let target = SpaceTimeCoord::new(GeoCoord::try_new(0.4, 0.4).unwrap(), 1.0);

    let mut conditioner: KrigingConditioner<_, LogitScale> =
        fit.into_model().into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 2.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 2.0).abs() < 0.1);
}

#[test]
fn sgs_harness_accepts_a_conditioner() {
    let coords = vec![
        GeoCoord::try_new(0.0, 0.0).unwrap(),
        GeoCoord::try_new(0.0, 1.0).unwrap(),
        GeoCoord::try_new(1.0, 0.0).unwrap(),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let conditioner =
        OrdinaryKrigingModel::new(GeoDataset::new(coords, values).unwrap(), variogram)
            .unwrap()
            .into_conditioner()
            .unwrap();
    let targets = vec![
        GeoCoord::try_new(0.4, 0.4).unwrap(),
        GeoCoord::try_new(0.6, 0.6).unwrap(),
    ];

    let samples =
        sequential_gaussian_simulate(conditioner, &targets, SimulationOptions::new(42)).unwrap();

    assert_eq!(samples.len(), targets.len());
    assert!(samples.iter().all(|sample| sample.is_finite()));
}

#[test]
fn neighborhood_model_is_rejected_as_a_conditioner() {
    let coords = vec![
        GeoCoord::try_new(0.0, 0.0).unwrap(),
        GeoCoord::try_new(0.0, 1.0).unwrap(),
        GeoCoord::try_new(1.0, 0.0).unwrap(),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let model = OrdinaryKrigingModel::new(GeoDataset::new(coords, values).unwrap(), variogram)
        .unwrap()
        .with_neighborhood(Some(Neighborhood::nearest(2)));

    let result = model.into_conditioner();

    assert!(matches!(
        result,
        Err(kriging_rs::KrigingError::InvalidInput(_))
    ));
}

#[test]
fn failed_append_leaves_conditioner_unchanged() {
    let coords = vec![
        GeoCoord::try_new(0.0, 0.0).unwrap(),
        GeoCoord::try_new(0.0, 1.0).unwrap(),
        GeoCoord::try_new(1.0, 0.0).unwrap(),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let mut conditioner =
        OrdinaryKrigingModel::new(GeoDataset::new(coords, values).unwrap(), variogram)
            .unwrap()
            .into_conditioner()
            .unwrap();
    let target = GeoCoord::try_new(0.4, 0.4).unwrap();
    let before = conditioner.predict(target).unwrap();

    let result = conditioner.append_condition(target, f64::NAN as _);
    let after = conditioner.predict(target).unwrap();

    assert!(result.is_err());
    assert_eq!(after, before);
}

#[test]
fn universal_model_converts_to_a_conditioner() {
    let coords = vec![
        GeoCoord::try_new(0.0, 0.0).unwrap(),
        GeoCoord::try_new(0.0, 1.0).unwrap(),
        GeoCoord::try_new(1.0, 0.0).unwrap(),
        GeoCoord::try_new(1.0, 1.0).unwrap(),
        GeoCoord::try_new(2.0, 0.5).unwrap(),
    ];
    let values = coords
        .iter()
        .map(|coord| 1.0 + 2.0 * coord.lat() + 3.0 * coord.lon())
        .collect();
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let model = UniversalKrigingModel::new(
        GeoDataset::new(coords, values).unwrap(),
        variogram,
        UniversalTrend::Linear,
    )
    .unwrap();
    let target = GeoCoord::try_new(0.4, 0.4).unwrap();

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}

#[test]
fn simple_model_converts_to_a_conditioner() {
    let coords = vec![
        GeoCoord::try_new(0.0, 0.0).unwrap(),
        GeoCoord::try_new(0.0, 1.0).unwrap(),
        GeoCoord::try_new(1.0, 0.0).unwrap(),
    ];
    let values = vec![10.0, 20.0, 15.0];
    let variogram = VariogramModel::new(0.01, 5.0, 300.0, VariogramType::Exponential).unwrap();
    let model =
        SimpleKrigingModel::new(GeoDataset::new(coords, values).unwrap(), variogram, 15.0).unwrap();
    let target = GeoCoord::try_new(0.4, 0.4).unwrap();

    let mut conditioner = model.into_conditioner().unwrap();
    let before = conditioner.predict(target).unwrap();
    conditioner.append_condition(target, 42.0).unwrap();
    let after = conditioner.predict(target).unwrap();

    assert!(after.variance < before.variance);
    assert!((after.mean - 42.0).abs() < 0.1);
}
