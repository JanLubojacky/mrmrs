use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Debug, Clone)]
pub struct Feature {
    // private fields to rust
    pub idx: usize,
    pub redundance_sum: f64,
    // fields exposed to python
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub relevance: f64,
    #[pyo3(get)]
    pub redundance: f64,
    #[pyo3(get)]
    pub score: f64,
}

impl Feature {
    /// Creates a new Feature with specified initial values.
    ///
    /// # Arguments
    /// * `idx` - The feature index
    /// * `name` - The feature name
    /// * `relevance` - Initial relevance score (default: 0.0)
    /// * `redundance` - Initial redundance score (default: 0.0)
    /// * `score` - Initial overall score (default: 0.0)
    #[must_use]
    pub fn empty_feature() -> Self {
        Self {
            idx: 0,
            name: "empty_feature".to_string(),
            relevance: f64::NEG_INFINITY,
            redundance: f64::NEG_INFINITY,
            redundance_sum: f64::NEG_INFINITY,
            score: f64::NEG_INFINITY,
        }
    }
}

#[pymethods]
impl Feature {
    fn __repr__(&self) -> String {
        format!(
            "Feature(name='{}', relevance={}, redundance={}, score={})",
            self.name, self.relevance, self.redundance, self.score
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_empty_feature_defaults() {
        let feat = Feature::empty_feature();
        assert_eq!(feat.idx, 0);
        assert_eq!(feat.name, "empty_feature");
        assert_eq!(feat.relevance, f64::NEG_INFINITY);
        assert_eq!(feat.redundance, f64::NEG_INFINITY);
        assert_eq!(feat.redundance_sum, f64::NEG_INFINITY);
        assert_eq!(feat.score, f64::NEG_INFINITY);
    }

    #[test]
    fn test_feature_repr() {
        let feat = Feature {
            idx: 0,
            name: "test_feature".to_string(),
            relevance: 1.5,
            redundance: 0.5,
            redundance_sum: 0.5,
            score: 3.0,
        };
        let repr = feat.__repr__();
        assert!(repr.contains("test_feature"));
        assert!(repr.contains("1.5"));
        assert!(repr.contains("0.5"));
        assert!(repr.contains("3"));
    }
}
