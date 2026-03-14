#[cfg(all(feature = "gpu", feature = "gpu-blocking", not(target_arch = "wasm32")))]
fn main() {
    let support = kriging_rs::gpu::detect_gpu_support_blocking();
    println!(
        "gpu_available={} backend={:?}",
        support.available, support.backend
    );
}

#[cfg(not(all(feature = "gpu", feature = "gpu-blocking", not(target_arch = "wasm32"))))]
fn main() {
    eprintln!("enable features \"gpu,gpu-blocking\" to run gpu_probe");
}
