// Submodule declarations
mod corr;
mod feature;
mod fstat;

// Re-export public API
pub use corr::{pearson_corr, CorrelationError};
pub use feature::Feature;
pub use fstat::{f_classification, f_regression};
