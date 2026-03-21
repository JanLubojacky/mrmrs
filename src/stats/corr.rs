use multiversion::multiversion;
use polars::chunked_array::ChunkedArray;
use polars::datatypes::Float64Type;
use polars::error::PolarsError;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CorrelationError {
    #[error("Constant vector")]
    ConstantVector,

    #[error("Input vectors have different lengths: {0} vs {1}")]
    DifferentInputLengths(usize, usize),

    #[error("Insufficient valid pairs for correlation (need at least 2, got {0})")]
    InsufficientData(usize),
}

impl From<CorrelationError> for PolarsError {
    fn from(err: CorrelationError) -> Self {
        PolarsError::ComputeError(err.to_string().into())
    }
}

impl From<CorrelationError> for PyErr {
    fn from(err: CorrelationError) -> Self {
        PyValueError::new_err(format!("Correlation error: {err}"))
    }
}

/// SIMD-accelerated Pearson correlation using portable_simd.
/// Uses the computational formula
/// r = (n * sum(xy) - sum(x) * sum(y)) / sqrt((n * sum(x^2) - sum(x)^2)(n * sum(y^2) - sum(y)^2))
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
pub fn pearson_corr(
    x: &ChunkedArray<Float64Type>,
    y: &ChunkedArray<Float64Type>,
) -> Result<f64, CorrelationError> {
    if x.len() != y.len() {
        return Err(CorrelationError::DifferentInputLengths(x.len(), y.len()));
    }
    let n = x.len();
    if n < 2 {
        return Err(CorrelationError::InsufficientData(n));
    }

    const LANES: usize = 4;
    type SimdF64 = Simd<f64, LANES>;

    let x_views = x.data_views();
    let y_views = y.data_views();

    // SIMD accumulators for the 5 sums
    let mut sum_x_simd = SimdF64::splat(0.0);
    let mut sum_y_simd = SimdF64::splat(0.0);
    let mut sum_xx_simd = SimdF64::splat(0.0);
    let mut sum_yy_simd = SimdF64::splat(0.0);
    let mut sum_xy_simd = SimdF64::splat(0.0);

    for (x_chunk, y_chunk) in x_views.zip(y_views) {
        // Use as_rchunks to get remainder first, then aligned chunks
        let (x_remainder, x_simd_chunks): (&[f64], &[[f64; LANES]]) = x_chunk.as_rchunks();
        let (y_remainder, y_simd_chunks): (&[f64], &[[f64; LANES]]) = y_chunk.as_rchunks();

        // Process remainder elements (scalar)
        for (&xv, &yv) in x_remainder.iter().zip(y_remainder.iter()) {
            sum_x_simd[0] += xv;
            sum_y_simd[0] += yv;
            sum_xx_simd[0] += xv * xv;
            sum_yy_simd[0] += yv * yv;
            sum_xy_simd[0] += xv * yv;
        }

        // Process SIMD chunks
        for (x_arr, y_arr) in x_simd_chunks.iter().zip(y_simd_chunks.iter()) {
            let x_vec = SimdF64::from_array(*x_arr);
            let y_vec = SimdF64::from_array(*y_arr);

            sum_x_simd += x_vec;
            sum_y_simd += y_vec;
            sum_xx_simd = x_vec.mul_add(x_vec, sum_xx_simd);
            sum_yy_simd = y_vec.mul_add(y_vec, sum_yy_simd);
            sum_xy_simd = x_vec.mul_add(y_vec, sum_xy_simd);
        }
    }

    // Horizontal reduction - sum all lanes
    let sum_x = sum_x_simd.reduce_sum();
    let sum_y = sum_y_simd.reduce_sum();
    let sum_xx = sum_xx_simd.reduce_sum();
    let sum_yy = sum_yy_simd.reduce_sum();
    let sum_xy = sum_xy_simd.reduce_sum();

    // Compute correlation using computational formula
    let n_f = n as f64;
    let numerator = n_f * sum_xy - sum_x * sum_y;
    let denom_x = n_f * sum_xx - sum_x * sum_x;
    let denom_y = n_f * sum_yy - sum_y * sum_y;

    if denom_x <= 0.0 || denom_y <= 0.0 {
        return Err(CorrelationError::ConstantVector);
    }

    let r = numerator / (denom_x.sqrt() * denom_y.sqrt());
    Ok(r.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use polars::prelude::*;

    // Helper to create ChunkedArray from slice
    fn make_f64_array(name: &str, values: &[f64]) -> ChunkedArray<Float64Type> {
        ChunkedArray::new(name.into(), values)
    }

    #[test]
    fn test_pearson_corr_perfect_positive() {
        let x = make_f64_array("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = make_f64_array("y", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let r = pearson_corr(&x, &y).unwrap();
        assert_relative_eq!(r, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pearson_corr_perfect_negative() {
        let x = make_f64_array("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = make_f64_array("y", &[5.0, 4.0, 3.0, 2.0, 1.0]);
        let r = pearson_corr(&x, &y).unwrap();
        assert_relative_eq!(r, -1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pearson_corr_linear_relationship() {
        // y = 2x, should give r = 1.0
        let x = make_f64_array("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = make_f64_array("y", &[2.0, 4.0, 6.0, 8.0, 10.0]);
        let r = pearson_corr(&x, &y).unwrap();
        assert_relative_eq!(r, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pearson_corr_known_value() {
        // Precomputed with numpy.corrcoef([1,2,3], [2,4,5])[0,1] = 0.9819805060619657
        let x = make_f64_array("x", &[1.0, 2.0, 3.0]);
        let y = make_f64_array("y", &[2.0, 4.0, 5.0]);
        let r = pearson_corr(&x, &y).unwrap();
        assert_relative_eq!(r, 0.9819805060619657, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pearson_corr_another_known_value() {
        // Precomputed with numpy.corrcoef([1,2,3,4,5], [1,3,2,5,4])[0,1] = 0.8
        let x = make_f64_array("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = make_f64_array("y", &[1.0, 3.0, 2.0, 5.0, 4.0]);
        let r = pearson_corr(&x, &y).unwrap();
        assert_relative_eq!(r, 0.8, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pearson_corr_two_values() {
        // Minimum valid input
        let x = make_f64_array("x", &[1.0, 2.0]);
        let y = make_f64_array("y", &[1.0, 2.0]);
        let r = pearson_corr(&x, &y).unwrap();
        assert_relative_eq!(r, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pearson_corr_constant_x_error() {
        let x = make_f64_array("x", &[5.0, 5.0, 5.0, 5.0]);
        let y = make_f64_array("y", &[1.0, 2.0, 3.0, 4.0]);
        let result = pearson_corr(&x, &y);
        assert!(matches!(result, Err(CorrelationError::ConstantVector)));
    }

    #[test]
    fn test_pearson_corr_constant_y_error() {
        let x = make_f64_array("x", &[1.0, 2.0, 3.0, 4.0]);
        let y = make_f64_array("y", &[5.0, 5.0, 5.0, 5.0]);
        let result = pearson_corr(&x, &y);
        assert!(matches!(result, Err(CorrelationError::ConstantVector)));
    }

    #[test]
    fn test_pearson_corr_different_lengths_error() {
        let x = make_f64_array("x", &[1.0, 2.0, 3.0]);
        let y = make_f64_array("y", &[1.0, 2.0]);
        let result = pearson_corr(&x, &y);
        assert!(matches!(
            result,
            Err(CorrelationError::DifferentInputLengths(3, 2))
        ));
    }

    #[test]
    fn test_pearson_corr_insufficient_data_single() {
        let x = make_f64_array("x", &[1.0]);
        let y = make_f64_array("y", &[1.0]);
        let result = pearson_corr(&x, &y);
        assert!(matches!(result, Err(CorrelationError::InsufficientData(1))));
    }

    #[test]
    fn test_pearson_corr_insufficient_data_empty() {
        let x: ChunkedArray<Float64Type> = ChunkedArray::new("x".into(), &[] as &[f64]);
        let y: ChunkedArray<Float64Type> = ChunkedArray::new("y".into(), &[] as &[f64]);
        let result = pearson_corr(&x, &y);
        assert!(matches!(result, Err(CorrelationError::InsufficientData(0))));
    }
}
