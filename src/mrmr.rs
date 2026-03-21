use indicatif::{ProgressBar, ProgressStyle};
use polars::error::PolarsError;
use polars::prelude::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_polars::{PyDataFrame, PySeries};
use rayon::prelude::*;

use crate::stats::{f_classification, f_regression, pearson_corr, Feature};
use crate::utils::get_numeric_columns;

const MIN_PEARSON_CORR: f64 = 0.001;
const MIN_VALID_SCORE: f64 = 0.0;

#[derive(Debug)]
enum TaskType {
    Classification,
    Regression,
}

impl TaskType {
    fn from_str(task_type: &str) -> PyResult<Self> {
        match task_type {
            "classification" => Ok(TaskType::Classification),
            "regression" => Ok(TaskType::Regression),
            _ => Err(PyValueError::new_err(format!(
                "Invalid task_type '{task_type}', supported values are 'classification' or 'regression'.",
            ))),
        }
    }
}

// Helper function to convert PolarsError to PyErr
fn polars_to_py_err(err: PolarsError) -> PyErr {
    PyValueError::new_err(format!("Polars error: {err}"))
}

/// Performs feature selection using the mRMR (minimum Redundancy Maximum Relevance) algorithm.
///
/// # Arguments
/// * `x` - Input dataframe containing feature columns
/// * `y` - Target variable series
/// * `number_of_features` - Maximum number of features to select
/// * `task_type` - Either "classification" or "regression"
///
/// # Returns
/// Vector of selected features ordered by selection priority
///
/// # Errors
/// Returns error if task_type is invalid or data processing fails
#[pyfunction]
pub fn mrmr(
    x: PyDataFrame,
    y: PySeries,
    number_of_features: usize,
    task_type: &str,
) -> PyResult<Vec<Feature>> {
    let task_type = TaskType::from_str(task_type)?;

    let df: DataFrame = x.into();
    let df = get_numeric_columns(&df)
        .map_err(|e| PyValueError::new_err(format!("Failed to filter numeric columns: {e}")))?;
    let y: Series = y.into();

    let df_cols: &[Column] = df.get_columns();

    // Initialize with all features unselected
    let mut selected_mask: Vec<bool> = vec![false; df_cols.len()];
    let mut selected_features: Vec<Feature> = Vec::with_capacity(number_of_features);

    // Compute relevance
    let mut features: Vec<Feature> = match task_type {
        TaskType::Classification => f_classification(&df, &y).map_err(polars_to_py_err)?,
        TaskType::Regression => f_regression(&df, &y).map_err(polars_to_py_err)?,
    };

    // Create progress bar for feature selection
    let pb = ProgressBar::new(number_of_features as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} features selected ({msg})")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );

    // Select the features
    for num_selected in 1..=number_of_features {
        // update the redundance for each feature
        // compute the score for each feature (relevance / redundance)
        let last_added_feat = if let Some(last_feat) = selected_features.last() {
            Some(
                df_cols[last_feat.idx]
                    .cast(&DataType::Float64)
                    .map_err(polars_to_py_err)?,
            )
        } else {
            None
        };

        // update redundance scores and find the highest scoring feature
        let max_score_feature = features
            .par_iter_mut()
            .filter(|feat| !selected_mask[feat.idx])
            .try_fold(Feature::empty_feature, |max_feat, feat| {
                if let Some(ref last_added_feat) = last_added_feat {
                    let unselected_feat_f64 = df_cols[feat.idx]
                        .cast(&DataType::Float64)
                        .map_err(polars_to_py_err)?;
                    let unselected_feat_f64_vals =
                        unselected_feat_f64.f64().map_err(polars_to_py_err)?;
                    let last_added_feat_f64_vals =
                        last_added_feat.f64().map_err(polars_to_py_err)?;

                    let redundance =
                        pearson_corr(unselected_feat_f64_vals, last_added_feat_f64_vals)?.abs();

                    // without this features which are not very relevant would get boosted too
                    // much just because they are not redundant, so we limit the minimum
                    // redundance metric, this is again set for pearson correlation and as such
                    // it would have to be modified when other methods such as MI are added
                    feat.redundance_sum += redundance.max(MIN_PEARSON_CORR);
                    feat.redundance = feat.redundance_sum / num_selected as f64;
                    feat.score = feat.relevance / feat.redundance;
                }
                if feat.score > max_feat.score {
                    Ok(feat.clone())
                } else {
                    Ok(max_feat)
                }
            })
            .try_reduce(Feature::empty_feature, |a, b| {
                Ok(if a.score > b.score { a } else { b })
            })
            .map_err(polars_to_py_err)?;

        // add the feature with max score to selected features and mark it as selected
        //
        // relevance and redundance are both assumed to be positive so score should be either a
        // positive float for a valid feature or f64::NEG_INFINITY from the empty_feature in which
        // case there are no more relevant features and we exit early
        //
        // this will probably need to get reworked if other scoring schemes than FCQ (f-test
        // correlation quotient) are used, namely the difference scores such as FCD
        if max_score_feature.score > MIN_VALID_SCORE {
            selected_mask[max_score_feature.idx] = true;
            selected_features.push(max_score_feature.clone());

            // Update progress bar
            pb.set_position(selected_features.len() as u64);
            pb.set_message(format!("Selected: {}", max_score_feature.name));
        } else {
            break;
        }
    }

    // Finish the progress bar
    pb.finish_with_message(format!(
        "Completed: {} features selected",
        selected_features.len(),
    ));

    Ok(selected_features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_from_str_classification() {
        let result = TaskType::from_str("classification");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), TaskType::Classification));
    }

    #[test]
    fn test_task_type_from_str_regression() {
        let result = TaskType::from_str("regression");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), TaskType::Regression));
    }

    #[test]
    fn test_task_type_from_str_invalid() {
        let result = TaskType::from_str("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_task_type_from_str_case_sensitive() {
        // Should fail for uppercase
        let result = TaskType::from_str("Classification");
        assert!(result.is_err());

        let result = TaskType::from_str("REGRESSION");
        assert!(result.is_err());
    }

    #[test]
    fn test_task_type_from_str_empty() {
        let result = TaskType::from_str("");
        assert!(result.is_err());
    }

    #[test]
    fn test_constants() {
        // Verify the constants have expected values
        assert_eq!(MIN_PEARSON_CORR, 0.001);
        assert_eq!(MIN_VALID_SCORE, 0.0);
    }
}
