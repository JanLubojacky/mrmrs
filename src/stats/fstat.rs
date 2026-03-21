use polars::datatypes::DataType;
use polars::error::{PolarsError, PolarsResult};
use polars::frame::DataFrame;
use polars::series::Series;
use rayon::prelude::*;
use std::collections::HashMap;

use super::corr::pearson_corr;
use super::feature::Feature;

/// F-statistic for classification tasks computed using a one-way ANOVA for each feature
/// <https://en.wikipedia.org/wiki/F-test>
pub fn f_classification(df: &DataFrame, y: &Series) -> PolarsResult<Vec<Feature>> {
    let num_of_samples = y.len();

    let y_i64 = y.cast(&DataType::Int64)?;
    let y_vals = y_i64.i64()?;

    let columns = df.get_columns();

    columns
        .par_iter()
        .enumerate()
        .filter_map(|(i, col)| {
            let result: PolarsResult<Option<Feature>> = (|| {
                let col_f64 = col.cast(&DataType::Float64)?;
                let vals = col_f64.f64()?;

                // Accumulate per-class statistics in a single sequential pass
                // HashMap stores (sum, sum_of_squares, count) per class
                let mut stats: HashMap<i64, (f64, f64, usize)> = HashMap::new();

                for (class_val, feature_val) in y_vals.iter().zip(vals.iter()) {
                    if let (Some(c), Some(v)) = (class_val, feature_val) {
                        let entry = stats.entry(c).or_insert((0.0, 0.0, 0));
                        entry.0 += v;
                        entry.1 += v * v;
                        entry.2 += 1;
                    }
                }

                let num_of_classes = stats.len();

                // Compute per-class mean and variance from accumulated stats
                // var = E[X²] - E[X]² (population variance)
                let group_stats: Vec<(f64, f64, f64)> = stats
                    .values()
                    .map(|(sum, sum_sq, count)| {
                        let n = *count as f64;
                        let mean = sum / n;
                        let var = (sum_sq / n) - mean * mean;
                        (n, mean, var)
                    })
                    .collect();

                // Global mean (weighted by group counts)
                let total_sum: f64 = group_stats.iter().map(|(n, mean, _)| n * mean).sum();
                let global_mean = total_sum / num_of_samples as f64;

                // Mean squared error between groups (explained variance)
                let ss_between: f64 = group_stats
                    .iter()
                    .map(|(n, mean, _)| n * (mean - global_mean).powi(2))
                    .sum();
                let mse_between = ss_between / (num_of_classes - 1) as f64;

                // Mean squared error within groups (unexplained variance)
                let ss_within: f64 = group_stats.iter().map(|(n, _, var)| n * var).sum();
                let mse_within = ss_within / (num_of_samples - num_of_classes) as f64;

                let f_stat = mse_between / mse_within;

                // filter out features with 0 relevance here
                if f_stat > 0.0 {
                    Ok(Some(Feature {
                        idx: i,
                        name: col.name().to_string(),
                        relevance: f_stat,
                        redundance: 1.0,
                        redundance_sum: 0.0,
                        score: f_stat,
                    }))
                } else {
                    Ok(None)
                }
            })();

            match result {
                Ok(Some(feature)) => Some(Ok(feature)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        })
        .collect()
}

/// F-statistic for regression tasks
/// obtained from the pearsons correlation coefficient via F = (n-2) * r^2 / (1 - r^2)
///
/// df is expected to be a dataframe with numerical-only features as columns
/// y is expected to be a regression target variable
///
pub fn f_regression(df: &DataFrame, y: &Series) -> PolarsResult<Vec<Feature>> {
    let y_f64 = y.cast(&DataType::Float64)?;
    let y_vals = y_f64.f64()?;

    let deg_of_freedom: f64 = (y_vals.len() - 2) as f64;

    let columns = df.get_columns();

    columns
        .par_iter()
        .enumerate()
        .map(|(i, col)| {
            let col_f64 = col.cast(&DataType::Float64)?;
            let x_vals = col_f64.f64()?;
            let r = pearson_corr(x_vals, y_vals).map_err(PolarsError::from)?;
            let r2 = r * r;
            let f_stat = deg_of_freedom * r2 / (1.0 - r2);

            Ok(Feature {
                idx: i,
                name: col.name().to_string(),
                relevance: f_stat,
                redundance: 1.0,
                redundance_sum: 0.0,
                score: f_stat,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use polars::prelude::*;

    #[test]
    fn test_f_classification_distinct_groups() {
        // Two groups with clearly different means should have high F-statistic
        let df = df! {
            "feature" => [1.0, 1.1, 1.2, 5.0, 5.1, 5.2]
        }
        .unwrap();
        let y = Series::new("y".into(), &[0i64, 0, 0, 1, 1, 1]);

        let features = f_classification(&df, &y).unwrap();

        assert_eq!(features.len(), 1);
        // scipy.stats.f_oneway([1.0, 1.1, 1.2], [5.0, 5.1, 5.2]) => F=2400
        assert_relative_eq!(features[0].relevance, 2400.0, epsilon = 1e-8);
    }

    #[test]
    fn test_f_classification_similar_groups() {
        // Two groups with similar means should have low F-statistic
        let df = df! {
            "feature" => [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
        }
        .unwrap();
        let y = Series::new("y".into(), &[0i64, 0, 0, 1, 1, 1]);

        let features = f_classification(&df, &y).unwrap();

        assert_eq!(features.len(), 1);
        // scipy.stats.f_oneway([1.0, 2.0, 3.0], [1.5, 2.5, 3.5]) => F=0.375
        assert_relative_eq!(features[0].relevance, 0.375, epsilon = 1e-8);
    }

    #[test]
    fn test_f_classification_multiple_features() {
        let df = df! {
            "informative" => [1.0, 2.0, 5.0, 6.0],
            "noise" => [0.5, 0.6, 0.4, 0.7]
        }
        .unwrap();
        let y = Series::new("y".into(), &[0i64, 0, 1, 1]);

        let features = f_classification(&df, &y).unwrap();

        // Find informative feature - it should have higher relevance than noise
        let informative = features.iter().find(|f| f.name == "informative").unwrap();
        // scipy.stats.f_oneway([1.0, 2.0], [5.0, 6.0]) => F=32.0
        assert_relative_eq!(informative.relevance, 32.0, epsilon = 1e-8);

        // Noise feature has F=0.0 which is filtered out (f_stat > 0.0 check)
        // scipy.stats.f_oneway([0.5, 0.6], [0.4, 0.7]) => F=0.0
        let noise = features.iter().find(|f| f.name == "noise");
        assert!(noise.is_none());
    }

    #[test]
    fn test_f_classification_multiclass() {
        // 3 classes with variance within groups
        let df = df! {
            "feature" => [1.0, 1.5, 3.0, 3.5, 5.0, 5.5]
        }
        .unwrap();
        let y = Series::new("y".into(), &[0i64, 0, 1, 1, 2, 2]);

        let features = f_classification(&df, &y).unwrap();

        assert_eq!(features.len(), 1);
        // scipy.stats.f_oneway([1.0, 1.5], [3.0, 3.5], [5.0, 5.5]) => F=64.0
        assert_relative_eq!(features[0].relevance, 64.0, epsilon = 1e-8);
    }

    // ==================== f_regression tests ====================

    #[test]
    fn test_f_regression_strong_correlation() {
        // Strong positive correlation (r ≈ 0.998)
        let df = df! {
            "feature" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();
        let y = Series::new("y".into(), &[1.1, 1.9, 3.1, 3.9, 5.1]);

        let features = f_regression(&df, &y).unwrap();

        assert_eq!(features.len(), 1);
        // scipy: F = 625.0
        assert_relative_eq!(features[0].relevance, 625.0, epsilon = 1e-8);
    }

    #[test]
    fn test_f_regression_linear_relationship() {
        // y ≈ 2x + 1 with slight noise (r ≈ 0.999)
        let df = df! {
            "feature" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();
        let y = Series::new("y".into(), &[3.1, 4.9, 7.1, 8.9, 11.1]);

        let features = f_regression(&df, &y).unwrap();

        assert_eq!(features.len(), 1);
        // scipy: F = 2500.0
        assert_relative_eq!(features[0].relevance, 2500.0, epsilon = 1e-8);
    }

    #[test]
    fn test_f_regression_anticorrelation() {
        // Strong negative correlation (r ≈ -0.998)
        let df = df! {
            "feature" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();
        let y = Series::new("y".into(), &[4.9, 4.1, 2.9, 2.1, 0.9]);

        let features = f_regression(&df, &y).unwrap();

        assert_eq!(features.len(), 1);
        // scipy: F = 625.0 (r^2 same as positive correlation)
        assert_relative_eq!(features[0].relevance, 625.0, epsilon = 1e-8);
    }

    #[test]
    fn test_f_regression_multiple_features() {
        let df = df! {
            "correlated" => [1.1, 1.9, 3.1, 3.9, 5.1],
            "anticorrelated" => [4.9, 4.1, 2.9, 2.1, 0.9],
            "noise" => [0.1, 0.9, 0.2, 0.8, 0.3]
        }
        .unwrap();
        let y = Series::new("y".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let features = f_regression(&df, &y).unwrap();

        assert_eq!(features.len(), 3);

        // Find each feature
        let correlated = features.iter().find(|f| f.name == "correlated").unwrap();
        let anticorrelated = features
            .iter()
            .find(|f| f.name == "anticorrelated")
            .unwrap();
        let noise = features.iter().find(|f| f.name == "noise").unwrap();

        // scipy: F = 625.0 for both strong correlations
        assert_relative_eq!(correlated.relevance, 625.0, epsilon = 1e-8);
        assert_relative_eq!(anticorrelated.relevance, 625.0, epsilon = 1e-8);
        // scipy: F ≈ 0.0516 for noise
        assert_relative_eq!(noise.relevance, 0.0516, epsilon = 1e-4);
    }
}
