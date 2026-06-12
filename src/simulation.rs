//! Conditional simulation via sequential Gaussian simulation (SGS).
//!
//! Given a set of observed stations and a variogram model, these functions draw one realization
//! of the underlying Gaussian random field at each requested target location such that:
//!
//! - The realization honors every conditioning observation exactly (up to kriging solver
//!   tolerance).
//! - Targets are visited in a user-chosen order (default is input order). At each target the
//!   value is sampled from `N(μ̂, σ²̂)` where `μ̂` and `σ²̂` come from kriging the target against
//!   *all* already-observed-or-simulated values (conditioning + previously simulated). The
//!   sampled value is then appended to the conditioning set for subsequent targets — this is
//!   the defining "sequential" step.
//!
//! SGS reproduces the covariance structure of the variogram (asymptotically, and exactly in
//! expectation) while honoring the data. It is appropriate when users need uncertainty via
//! multiple realizations rather than a single best-guess surface.
//!
//! ## Harness
//!
//! Sequential Gaussian simulation is generic over [`KrigingSimulator`](crate::predictor::simulation::KrigingSimulator)
//! backends in [`crate::predictor::simulation`]. Construct a simulator (e.g.
//! [`OrdinaryGeoSimulator`](crate::predictor::simulation::OrdinaryGeoSimulator)) and call
//! [`sequential_gaussian_simulate`](crate::predictor::simulation::sequential_gaussian_simulate)
//! or the binomial analogue [`sequential_binomial_simulate`](crate::predictor::simulation::sequential_binomial_simulate).
//!
//! All paths accept a shared [`SimulationOptions`] (seed + optional target visit order).
//!
//! ## RNG
//!
//! The simulator uses a seedable xoshiro-style PRNG so realizations are reproducible without
//! an external `rand` runtime dependency. For production work that needs rigorous random
//! number quality, callers can post-process or wrap this module's scalar outputs.

use crate::Real;

/// Options controlling conditional simulation.
#[derive(Debug, Clone)]
pub struct SimulationOptions {
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Optional permutation of target indices specifying the order in which targets are
    /// visited. Must be a permutation of `0..targets.len()`. If `None`, targets are visited
    /// in input order.
    pub target_order: Option<Vec<usize>>,
}

impl SimulationOptions {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            target_order: None,
        }
    }
}

/// Result of a binomial conditional simulation. Contains samples on both the logit and
/// prevalence scales, in the **original** target input order.
///
/// Simulation happens on the logit scale (where the Gaussian assumption is natural);
/// `prevalence_samples[i] = logistic(logit_samples[i])`.
#[derive(Debug, Clone)]
pub struct BinomialSimulationResult {
    /// Simulated logit values (unbounded).
    pub logit_samples: Vec<Real>,
    /// Simulated prevalence values in (0, 1).
    pub prevalence_samples: Vec<Real>,
}

/// Result of a multi-realization binomial conditional simulation.
///
/// Each field is a flat row-major buffer of length `n_realizations * n_targets`. Row `k`
/// (entries `[k * n_targets .. (k + 1) * n_targets]`) holds realization `k` in the
/// **original** target input order. By construction
/// `prevalence_samples[i] = logistic(logit_samples[i])` element-wise.
#[derive(Debug, Clone)]
pub struct BinomialSimulationManyResult {
    /// Number of independent realizations stacked into the buffers.
    pub n_realizations: usize,
    /// Number of target locations per realization.
    pub n_targets: usize,
    /// Simulated logit values, row-major `[n_realizations * n_targets]`.
    pub logit_samples: Vec<Real>,
    /// Simulated prevalence values, row-major `[n_realizations * n_targets]`.
    pub prevalence_samples: Vec<Real>,
}

// Generic SGS harness — see [`crate::predictor::simulation`].
pub use crate::predictor::simulation::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::error::KrigingError;
    use crate::kriging::binomial::BinomialPrior;
    use crate::kriging::universal::UniversalTrend;
    use crate::predictor::simulation::{
        BinomialGeoSimulator, BinomialProjectedSimulator, OrdinaryGeoSimulator,
        ProjectedOrdinarySimulator, SimpleGeoSimulator, SpacetimeBinomialSimulator,
        SpacetimeOrdinarySimulator, SpacetimeSimpleSimulator, SpacetimeUniversalSimulator,
        UniversalGeoSimulator, sequential_binomial_simulate, sequential_binomial_simulate_many,
        sequential_gaussian_simulate, sequential_gaussian_simulate_many,
    };
    use crate::projected::{Anisotropy2D, ProjectedCoord};
    use crate::spacetime::coord::SpaceTimeCoord;
    use crate::spacetime::kriging::universal::SpaceTimeUniversalTrend;
    use crate::spacetime::variogram::SpaceTimeVariogram;
    use crate::utils::logistic;
    use crate::variogram::models::{VariogramModel, VariogramType};

    fn setup() -> (Vec<GeoCoord>, Vec<Real>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let variogram = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        (coords, values, variogram)
    }

    // ---- Ordinary ----------------------------------------------------------

    #[test]
    fn same_seed_gives_identical_realization() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        assert_eq!(a, b, "same seed should yield identical realization");
    }

    #[test]
    fn different_seeds_give_different_realizations() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(1),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(2),
        )
        .unwrap();
        assert!(a != b, "different seeds should differ at least somewhere");
    }

    #[test]
    fn realization_honors_conditioning_when_target_coincides() {
        let (c, v, vg) = setup();
        let targets = vec![c[2]];
        let out = sequential_gaussian_simulate(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(123),
        )
        .unwrap();
        assert!(
            (out[0] - v[2]).abs() < 0.2,
            "sampled value {} should be close to observed {}",
            out[0],
            v[2]
        );
    }

    #[test]
    fn custom_target_order_still_returns_results_in_input_order() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
            GeoCoord::try_new(0.75, 0.25).unwrap(),
        ];
        let mut opts = SimulationOptions::new(7);
        opts.target_order = Some(vec![2, 0, 1]);
        let out = sequential_gaussian_simulate(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            opts,
        )
        .unwrap();
        assert_eq!(out.len(), targets.len());
        for v in out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn invalid_permutation_is_rejected() {
        let (c, v, vg) = setup();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap(); 3];
        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 0, 1]);
        assert!(
            sequential_gaussian_simulate(
                OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
                &targets,
                opts
            )
            .is_err()
        );

        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 1, 5]);
        assert!(
            sequential_gaussian_simulate(
                OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
                &targets,
                opts
            )
            .is_err()
        );

        let mut opts = SimulationOptions::new(0);
        opts.target_order = Some(vec![0, 1]);
        assert!(
            sequential_gaussian_simulate(
                OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
                &targets,
                opts
            )
            .is_err()
        );
    }

    // ---- Simple ------------------------------------------------------------

    #[test]
    fn simple_simulation_is_deterministic_for_same_seed() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.3, 0.7).unwrap(),
        ];
        let mean = 2.5;
        let a = sequential_gaussian_simulate(
            SimpleGeoSimulator::new(&c, &v, vg, mean).unwrap(),
            &targets,
            SimulationOptions::new(9),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            SimpleGeoSimulator::new(&c, &v, vg, mean).unwrap(),
            &targets,
            SimulationOptions::new(9),
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), targets.len());
        for x in &a {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn simple_simulation_honors_conditioning_at_coincident_target() {
        let (c, v, vg) = setup();
        let targets = vec![c[1]];
        let out = sequential_gaussian_simulate(
            SimpleGeoSimulator::new(&c, &v, vg, 2.5).unwrap(),
            &targets,
            SimulationOptions::new(33),
        )
        .unwrap();
        assert!(
            (out[0] - v[1]).abs() < 0.2,
            "sampled value {} should be close to observed {}",
            out[0],
            v[1]
        );
    }

    // ---- Universal ---------------------------------------------------------

    #[test]
    fn universal_simulation_linear_trend_is_finite_and_deterministic() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.4, 0.6).unwrap(),
            GeoCoord::try_new(0.6, 0.4).unwrap(),
        ];
        let trend = UniversalTrend::Linear;
        let a = sequential_gaussian_simulate(
            UniversalGeoSimulator::new(&c, &v, vg, trend).unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            UniversalGeoSimulator::new(&c, &v, vg, trend).unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        assert_eq!(a, b);
        for x in &a {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn universal_simulation_rejects_too_few_conditioning_points() {
        // Linear trend needs n >= 3+1 = 4. Three conditioning points should fail up front.
        let c = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let v = vec![1.0, 2.0, 3.0];
        let vg = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let result = UniversalGeoSimulator::new(&c, &v, vg, UniversalTrend::Linear)
            .and_then(|sim| sequential_gaussian_simulate(sim, &targets, SimulationOptions::new(0)));
        match result {
            Err(KrigingError::InsufficientData(n)) => assert_eq!(n, 4),
            other => panic!("expected InsufficientData(4), got {other:?}"),
        }
    }

    // ---- Projected ---------------------------------------------------------

    #[test]
    fn projected_simulation_is_deterministic_for_same_seed() {
        let coords = vec![
            ProjectedCoord::new(0.0, 0.0),
            ProjectedCoord::new(0.0, 1.0),
            ProjectedCoord::new(1.0, 0.0),
            ProjectedCoord::new(1.0, 1.0),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let vg = VariogramModel::new(0.1, 5.0, 2.0, VariogramType::Exponential).unwrap();
        let targets = vec![
            ProjectedCoord::new(0.5, 0.5),
            ProjectedCoord::new(0.25, 0.75),
        ];
        let aniso = Anisotropy2D::isotropic();
        let a = sequential_gaussian_simulate(
            ProjectedOrdinarySimulator::new(&coords, &values, vg, aniso).unwrap(),
            &targets,
            SimulationOptions::new(5),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            ProjectedOrdinarySimulator::new(&coords, &values, vg, aniso).unwrap(),
            &targets,
            SimulationOptions::new(5),
        )
        .unwrap();
        assert_eq!(a, b);
        for x in &a {
            assert!(x.is_finite());
        }
    }

    // ---- Binomial projected -----------------------------------------------

    fn binomial_projected_setup() -> (Vec<ProjectedCoord>, Vec<u32>, Vec<u32>, VariogramModel) {
        let coords = vec![
            ProjectedCoord::new(0.0, 0.0),
            ProjectedCoord::new(0.0, 1.0),
            ProjectedCoord::new(1.0, 0.0),
            ProjectedCoord::new(1.0, 1.0),
        ];
        let successes = vec![3, 7, 4, 9];
        let trials = vec![10, 12, 9, 15];
        let vg = VariogramModel::new(0.1, 1.0, 1.5, VariogramType::Exponential).unwrap();
        (coords, successes, trials, vg)
    }

    #[test]
    fn binomial_projected_simulation_is_deterministic_for_same_seed() {
        let (coords, successes, trials, vg) = binomial_projected_setup();
        let targets = vec![ProjectedCoord::new(0.5, 0.5)];
        let a = sequential_binomial_simulate(
            BinomialProjectedSimulator::new(
                &coords,
                &successes,
                &trials,
                vg,
                Anisotropy2D::isotropic(),
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        let b = sequential_binomial_simulate(
            BinomialProjectedSimulator::new(
                &coords,
                &successes,
                &trials,
                vg,
                Anisotropy2D::isotropic(),
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        assert_eq!(a.logit_samples, b.logit_samples);
        assert_eq!(a.prevalence_samples, b.prevalence_samples);
        for p in &a.prevalence_samples {
            assert!(*p > 0.0 && *p < 1.0);
        }
    }

    #[test]
    fn binomial_projected_many_matches_seeded_singles() {
        let (coords, successes, trials, vg) = binomial_projected_setup();
        let targets = vec![
            ProjectedCoord::new(0.25, 0.25),
            ProjectedCoord::new(0.75, 0.5),
        ];
        let n_real = 3usize;
        let base = 100u64;
        let many = sequential_binomial_simulate_many(
            BinomialProjectedSimulator::new(
                &coords,
                &successes,
                &trials,
                vg,
                Anisotropy2D::isotropic(),
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            n_real,
            base,
            None,
        )
        .unwrap();
        assert_eq!(many.n_realizations, n_real);
        assert_eq!(many.n_targets, targets.len());
        for k in 0..n_real {
            let one = sequential_binomial_simulate(
                BinomialProjectedSimulator::new(
                    &coords,
                    &successes,
                    &trials,
                    vg,
                    Anisotropy2D::isotropic(),
                    BinomialPrior::default(),
                )
                .unwrap(),
                &targets,
                SimulationOptions::new(base + k as u64),
            )
            .unwrap();
            for j in 0..targets.len() {
                let off = k * targets.len() + j;
                assert!((many.logit_samples[off] - one.logit_samples[j]).abs() < 1e-9);
                assert!((many.prevalence_samples[off] - one.prevalence_samples[j]).abs() < 1e-9);
            }
        }
    }

    // ---- Binomial ----------------------------------------------------------

    fn binomial_setup() -> (Vec<GeoCoord>, Vec<u32>, Vec<u32>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let successes = vec![3, 7, 4, 9];
        let trials = vec![10, 12, 9, 15];
        let vg = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        (coords, successes, trials, vg)
    }

    #[test]
    fn binomial_simulation_reports_both_scales_consistently() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let prior = BinomialPrior::default();
        let result = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(17),
        )
        .unwrap();
        assert_eq!(result.logit_samples.len(), targets.len());
        assert_eq!(result.prevalence_samples.len(), targets.len());
        for (logit, prev) in result
            .logit_samples
            .iter()
            .zip(result.prevalence_samples.iter())
        {
            assert!(logit.is_finite(), "logit must be finite: got {logit}");
            assert!(*prev > 0.0 && *prev < 1.0, "prevalence must be in (0,1)");
            assert!(
                (*prev - logistic(*logit)).abs() < 1e-12,
                "prevalence must equal logistic(logit): logit={logit}, prev={prev}"
            );
        }
    }

    #[test]
    fn binomial_simulation_is_deterministic_for_same_seed() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let prior = BinomialPrior::default();
        let a = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(17),
        )
        .unwrap();
        let b = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(17),
        )
        .unwrap();
        assert_eq!(a.logit_samples, b.logit_samples);
        assert_eq!(a.prevalence_samples, b.prevalence_samples);
    }

    #[test]
    fn binomial_simulation_drops_zero_trial_stations_from_conditioning() {
        // Injecting a trials==0 station between real observations must not poison the pool
        // and must not change the realization when compared against an equivalent
        // zero-station-free call.
        let (mut c, mut s, mut t, vg) = binomial_setup();
        c.insert(2, GeoCoord::try_new(0.5, 0.5).unwrap());
        s.insert(2, 0);
        t.insert(2, 0);
        let targets = vec![GeoCoord::try_new(0.75, 0.25).unwrap()];
        let prior = BinomialPrior::default();
        let with_zero = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(4),
        )
        .unwrap();

        let (c2, s2, t2, _) = binomial_setup();
        let without = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c2, &s2, &t2, vg, prior).unwrap(),
            &targets,
            SimulationOptions::new(4),
        )
        .unwrap();
        assert_eq!(with_zero.logit_samples, without.logit_samples);
        assert_eq!(with_zero.prevalence_samples, without.prevalence_samples);
    }

    #[test]
    fn binomial_simulation_rejects_all_zero_trials() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let successes = vec![0, 0];
        let trials = vec![0, 0];
        let vg = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let result =
            BinomialGeoSimulator::new(&coords, &successes, &trials, vg, BinomialPrior::default())
                .and_then(|sim| {
                    sequential_binomial_simulate(sim, &targets, SimulationOptions::new(0))
                });
        match result {
            Err(KrigingError::InsufficientData(n)) => assert_eq!(n, 2),
            other => panic!("expected InsufficientData(2), got {other:?}"),
        }
    }

    #[test]
    fn binomial_simulation_respects_custom_prior() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let default_prior = BinomialPrior::default();
        let custom_prior = BinomialPrior::new(2.0, 5.0).unwrap();
        let a = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c, &s, &t, vg, default_prior).unwrap(),
            &targets,
            SimulationOptions::new(21),
        )
        .unwrap();
        let b = sequential_binomial_simulate(
            BinomialGeoSimulator::new(&c, &s, &t, vg, custom_prior).unwrap(),
            &targets,
            SimulationOptions::new(21),
        )
        .unwrap();
        assert!(
            (a.logit_samples[0] - b.logit_samples[0]).abs() > 1e-9,
            "different priors should produce different logit samples"
        );
    }

    // ----- Space–time SGS ---------------------------------------------------

    use crate::spacetime::GeoMetric;

    fn st_setup() -> (Vec<SpaceTimeCoord<GeoCoord>>, Vec<Real>, SpaceTimeVariogram) {
        let coords = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 1.5, 4.5];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        (coords, values, vg)
    }

    #[test]
    fn st_same_seed_gives_identical_realization() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(42),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn st_different_seeds_give_different_realizations() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let a = sequential_gaussian_simulate(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(1),
        )
        .unwrap();
        let b = sequential_gaussian_simulate(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(2),
        )
        .unwrap();
        assert!(a != b);
    }

    #[test]
    fn st_realization_honors_conditioning_when_target_coincides() {
        let (c, v, vg) = st_setup();
        let targets = vec![c[2]];
        let out = sequential_gaussian_simulate(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            SimulationOptions::new(7),
        )
        .unwrap();
        // Kriging at a known conditioning location returns the observed value with ~zero
        // variance, so the sample should equal the observation up to solver tolerance.
        assert!(
            (out[0] - v[2]).abs() < 1e-3,
            "expected ~{}, got {}",
            v[2],
            out[0]
        );
    }

    #[test]
    fn st_rejects_mismatched_arrays() {
        let (c, _, vg) = st_setup();
        let v_bad = vec![1.0, 2.0];
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap()];
        assert!(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v_bad, vg)
                .and_then(|sim| sequential_gaussian_simulate(
                    sim,
                    &targets,
                    SimulationOptions::new(1)
                ))
                .is_err()
        );
    }

    #[test]
    fn st_simple_sim_runs_with_known_mean() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let mean = v.iter().copied().sum::<Real>() / v.len() as Real;
        let out = sequential_gaussian_simulate(
            SpacetimeSimpleSimulator::new(GeoMetric, &c, &v, vg, mean).unwrap(),
            &targets,
            SimulationOptions::new(3),
        )
        .unwrap();
        assert_eq!(out.len(), targets.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn st_universal_sim_runs_with_linear_in_time_trend() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let out = sequential_gaussian_simulate(
            SpacetimeUniversalSimulator::new(
                GeoMetric,
                &c,
                &v,
                vg,
                SpaceTimeUniversalTrend::LinearInTime,
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(5),
        )
        .unwrap();
        assert_eq!(out.len(), targets.len());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn st_universal_sim_rejects_insufficient_points_for_trend() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
        ];
        let v = vec![1.0, 2.0];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.0).unwrap()];
        // LinearInSpaceAndTime requires 4 basis terms → at least 5 points.
        let r = SpacetimeUniversalSimulator::new(
            GeoMetric,
            &c,
            &v,
            vg,
            SpaceTimeUniversalTrend::LinearInSpaceAndTime,
        )
        .and_then(|sim| sequential_gaussian_simulate(sim, &targets, SimulationOptions::new(1)));
        assert!(matches!(r, Err(KrigingError::InsufficientData(_))));
    }

    #[test]
    fn st_binomial_sim_reports_both_scales_and_prevalence_in_range() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 1.0).unwrap(),
        ];
        let successes = vec![3u32, 6, 8, 15, 4, 16];
        let trials = vec![20u32, 20, 20, 20, 20, 20];
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let out = sequential_binomial_simulate(
            SpacetimeBinomialSimulator::new(
                GeoMetric,
                &c,
                &successes,
                &trials,
                vg,
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(11),
        )
        .unwrap();
        assert_eq!(out.logit_samples.len(), targets.len());
        assert_eq!(out.prevalence_samples.len(), targets.len());
        for (l, p) in out.logit_samples.iter().zip(out.prevalence_samples.iter()) {
            assert!(l.is_finite());
            assert!(p.is_finite());
            assert!(*p > 0.0 && *p < 1.0);
            // p must equal logistic(l) by construction.
            assert!((logistic(*l) - *p).abs() < 1e-6);
        }
    }

    #[test]
    fn st_binomial_sim_drops_zero_trials_from_pool() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
        ];
        // Only station 0 has trials == 0; the remaining 3 form the pool.
        let successes = vec![0u32, 6, 8, 15];
        let trials = vec![0u32, 20, 20, 20];
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap()];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let out = sequential_binomial_simulate(
            SpacetimeBinomialSimulator::new(
                GeoMetric,
                &c,
                &successes,
                &trials,
                vg,
                BinomialPrior::default(),
            )
            .unwrap(),
            &targets,
            SimulationOptions::new(2),
        )
        .unwrap();
        assert_eq!(out.logit_samples.len(), 1);
        assert!(out.logit_samples[0].is_finite());
    }

    #[test]
    fn st_binomial_sim_rejects_too_few_observable_stations() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
        ];
        let successes = vec![0u32, 6];
        let trials = vec![0u32, 20];
        let targets =
            vec![SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap()];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let r = SpacetimeBinomialSimulator::new(
            GeoMetric,
            &c,
            &successes,
            &trials,
            vg,
            BinomialPrior::default(),
        )
        .and_then(|sim| sequential_binomial_simulate(sim, &targets, SimulationOptions::new(1)));
        assert!(matches!(r, Err(KrigingError::InsufficientData(_))));
    }

    // ---- Multi-realization variants ----------------------------------------

    /// Tight contract: the k-th row of `_many` MUST equal a fresh single-realization call
    /// using `seed = base_seed + k`. This is the primary correctness guarantee for the
    /// shared-input path.
    #[test]
    fn many_ordinary_matches_per_realization_single_call() {
        let (c, v, vg) = setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
            GeoCoord::try_new(0.75, 0.25).unwrap(),
        ];
        let n_real = 4usize;
        let base_seed = 100u64;
        let many = sequential_gaussian_simulate_many(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .expect("many ordinary call");
        assert_eq!(many.len(), n_real * targets.len());
        for k in 0..n_real {
            let one = sequential_gaussian_simulate(
                OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let row = &many[k * targets.len()..(k + 1) * targets.len()];
            assert_eq!(
                row,
                one.as_slice(),
                "row {k} must match per-seed single call"
            );
        }
    }

    #[test]
    fn many_ordinary_rejects_zero_realizations() {
        let (c, v, vg) = setup();
        let targets = vec![GeoCoord::try_new(0.5, 0.5).unwrap()];
        let r = sequential_gaussian_simulate_many(
            OrdinaryGeoSimulator::new(&c, &v, vg).unwrap(),
            &targets,
            0,
            0,
            None,
        );
        assert!(matches!(r, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn many_binomial_matches_per_realization_single_call() {
        let (c, s, t, vg) = binomial_setup();
        let targets = vec![
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(0.25, 0.75).unwrap(),
        ];
        let prior = BinomialPrior::default();
        let n_real = 3usize;
        let base_seed = 555u64;
        let many = sequential_binomial_simulate_many(
            BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .expect("many binomial call");
        assert_eq!(many.n_realizations, n_real);
        assert_eq!(many.n_targets, targets.len());
        assert_eq!(many.logit_samples.len(), n_real * targets.len());
        assert_eq!(many.prevalence_samples.len(), n_real * targets.len());
        for k in 0..n_real {
            let one = sequential_binomial_simulate(
                BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let lo = k * targets.len();
            let hi = lo + targets.len();
            assert_eq!(
                &many.logit_samples[lo..hi],
                one.logit_samples.as_slice(),
                "logit row {k}"
            );
            assert_eq!(
                &many.prevalence_samples[lo..hi],
                one.prevalence_samples.as_slice(),
                "prevalence row {k}"
            );
        }
    }

    #[test]
    fn many_binomial_drops_zero_trial_stations() {
        let (mut c, mut s, mut t, vg) = binomial_setup();
        c.insert(2, GeoCoord::try_new(0.5, 0.5).unwrap());
        s.insert(2, 0);
        t.insert(2, 0);
        let targets = vec![GeoCoord::try_new(0.75, 0.25).unwrap()];
        let prior = BinomialPrior::default();
        let with_zero = sequential_binomial_simulate_many(
            BinomialGeoSimulator::new(&c, &s, &t, vg, prior).unwrap(),
            &targets,
            2,
            9,
            None,
        )
        .unwrap();
        let (c2, s2, t2, _) = binomial_setup();
        let without = sequential_binomial_simulate_many(
            BinomialGeoSimulator::new(&c2, &s2, &t2, vg, prior).unwrap(),
            &targets,
            2,
            9,
            None,
        )
        .unwrap();
        assert_eq!(with_zero.logit_samples, without.logit_samples);
        assert_eq!(with_zero.prevalence_samples, without.prevalence_samples);
    }

    #[test]
    fn many_spacetime_matches_per_realization_single_call() {
        let (c, v, vg) = st_setup();
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let n_real = 3usize;
        let base_seed = 12345u64;
        let many = sequential_gaussian_simulate_many(
            SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .expect("many st call");
        assert_eq!(many.len(), n_real * targets.len());
        for k in 0..n_real {
            let one = sequential_gaussian_simulate(
                SpacetimeOrdinarySimulator::new(GeoMetric, &c, &v, vg).unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let row = &many[k * targets.len()..(k + 1) * targets.len()];
            assert_eq!(row, one.as_slice(), "row {k}");
        }
    }

    #[test]
    fn many_spacetime_binomial_matches_per_realization_single_call() {
        let c = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 0.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 0.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.0, 0.0).unwrap(), 1.0).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(1.0, 1.0).unwrap(), 1.0).unwrap(),
        ];
        let successes = vec![3u32, 6, 8, 15, 4, 16];
        let trials = vec![20u32, 20, 20, 20, 20, 20];
        let targets = vec![
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.5, 0.5).unwrap(), 0.5).unwrap(),
            SpaceTimeCoord::try_new(GeoCoord::try_new(0.25, 0.75).unwrap(), 0.5).unwrap(),
        ];
        let spatial = VariogramModel::new(0.1, 5.0, 200.0, VariogramType::Exponential).unwrap();
        let temporal = VariogramModel::new(0.05, 1.0, 2.0, VariogramType::Exponential).unwrap();
        let vg = SpaceTimeVariogram::new_separable(spatial, temporal).unwrap();
        let prior = BinomialPrior::default();
        let n_real = 3usize;
        let base_seed = 77u64;
        let many = sequential_binomial_simulate_many(
            SpacetimeBinomialSimulator::new(GeoMetric, &c, &successes, &trials, vg, prior).unwrap(),
            &targets,
            n_real,
            base_seed,
            None,
        )
        .unwrap();
        assert_eq!(many.n_realizations, n_real);
        assert_eq!(many.n_targets, targets.len());
        for k in 0..n_real {
            let one = sequential_binomial_simulate(
                SpacetimeBinomialSimulator::new(GeoMetric, &c, &successes, &trials, vg, prior)
                    .unwrap(),
                &targets,
                SimulationOptions::new(base_seed + k as u64),
            )
            .unwrap();
            let lo = k * targets.len();
            let hi = lo + targets.len();
            assert_eq!(&many.logit_samples[lo..hi], one.logit_samples.as_slice());
            assert_eq!(
                &many.prevalence_samples[lo..hi],
                one.prevalence_samples.as_slice()
            );
        }
    }
}
