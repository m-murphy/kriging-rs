//! Generic cross-validation and sequential-Gaussian-simulation harnesses.
//!
//! Model-specific logic lives in [`cv`] and [`simulation`] backend structs that implement
//! [`KrigingPredictor`](cv::KrigingPredictor) and [`KrigingSimulator`](simulation::KrigingSimulator).

pub mod cv;
pub mod simulation;
