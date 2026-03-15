use std::time::Instant;

use kriging_rs::distance::haversine_distance;
use kriging_rs::{GeoCoord, Real, VariogramModel, VariogramType};

fn build_coords(count: usize, lat0: Real, lon0: Real, step: Real) -> Vec<GeoCoord> {
    (0..count)
        .map(|i| {
            GeoCoord::try_new(
                lat0 + (i as Real) * step,
                lon0 + (i as Real) * step * 0.7,
            )
            .unwrap()
        })
        .collect()
}

fn cpu_rhs_covariances(
    train_coords: &[GeoCoord],
    pred_coords: &[GeoCoord],
    variogram: VariogramModel,
) -> Vec<Real> {
    let mut out = vec![0.0; train_coords.len() * pred_coords.len()];
    for (pred_idx, pred) in pred_coords.iter().enumerate() {
        let row = pred_idx * train_coords.len();
        for (train_idx, train) in train_coords.iter().enumerate() {
            let d = haversine_distance(*train, *pred);
            out[row + train_idx] = variogram.covariance(d);
        }
    }
    out
}

fn checksum(v: &[Real]) -> Real {
    v.iter().copied().sum::<Real>()
}

fn main() {
    let n_obs = 10_000usize;
    let n_pred = 1_000usize;
    let variogram = VariogramModel::new(0.01, 4.0, 20.0, VariogramType::Exponential).unwrap();

    let train = build_coords(n_obs, 35.0, -120.0, 0.0005);
    let pred = build_coords(n_pred, 35.05, -119.95, 0.0008);

    println!("large run: observations={n_obs}, predictions={n_pred}");
    let t0 = Instant::now();
    let cpu = cpu_rhs_covariances(&train, &pred, variogram);
    let cpu_elapsed = t0.elapsed();
    println!(
        "cpu rhs covariance: {:.3}s ({} entries, checksum={:.6})",
        cpu_elapsed.as_secs_f64(),
        cpu.len(),
        checksum(&cpu)
    );

    #[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
    {
        let t1 = Instant::now();
        let gpu = kriging_rs::gpu::build_rhs_covariances_gpu_blocking(&train, &pred, variogram)
            .expect("gpu rhs covariance");
        let gpu_elapsed = t1.elapsed();
        let mut mae = 0.0;
        for i in 0..cpu.len() {
            mae += (cpu[i] - gpu[i]).abs();
        }
        mae /= cpu.len() as Real;
        println!(
            "gpu rhs covariance: {:.3}s (checksum={:.6}, mae_vs_cpu={:.8})",
            gpu_elapsed.as_secs_f64(),
            checksum(&gpu),
            mae
        );
        println!(
            "speedup gpu/cpu: {:.2}x",
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64()
        );
    }

    #[cfg(not(all(feature = "gpu-blocking", not(target_arch = "wasm32"))))]
    {
        println!("gpu comparison unavailable (enable features: \"gpu,gpu-blocking\")");
    }
}
