#![feature(portable_simd)]

pub mod mrmr;
pub mod stats;
pub mod utils;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
/// A Python module implemented in Rust.
#[pymodule]
fn mrmrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mrmr::mrmr, m)?)?;
    m.add_class::<stats::Feature>()?;
    Ok(())
}
