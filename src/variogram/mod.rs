pub mod empirical;
pub mod fitting;
pub mod models;

pub use empirical::{EmpiricalVariogram, VariogramConfig, compute_empirical_variogram};
pub use fitting::{FitResult, fit_variogram};
pub use models::{VariogramModel, VariogramType};
