use polars::prelude::*;

pub fn get_numeric_columns(df: &DataFrame) -> PolarsResult<DataFrame> {
    let numerical_cols: Vec<&str> = df
        .get_columns()
        .iter()
        .filter(|col| col.dtype().is_numeric())
        .map(|col| col.name().as_str())
        .collect();

    df.select(numerical_cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_numeric_columns_mixed_types() {
        let df = df! {
            "int_col" => [1, 2, 3],
            "float_col" => [1.0, 2.0, 3.0],
            "str_col" => ["a", "b", "c"],
            "bool_col" => [true, false, true]
        }
        .unwrap();

        let result = get_numeric_columns(&df).unwrap();

        // Should only have numeric columns (int and float)
        assert_eq!(result.width(), 2);
        let col_names: Vec<String> = result
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(col_names.contains(&"int_col".to_string()));
        assert!(col_names.contains(&"float_col".to_string()));
        assert!(!col_names.contains(&"str_col".to_string()));
        assert!(!col_names.contains(&"bool_col".to_string()));
    }

    #[test]
    fn test_get_numeric_columns_all_numeric() {
        let df = df! {
            "int_col" => [1, 2, 3],
            "float_col" => [1.0, 2.0, 3.0],
            "another_int" => [4, 5, 6]
        }
        .unwrap();

        let result = get_numeric_columns(&df).unwrap();

        // All columns should be included
        assert_eq!(result.width(), 3);
    }

    #[test]
    fn test_get_numeric_columns_no_numeric() {
        let df = df! {
            "str_col" => ["a", "b", "c"],
            "another_str" => ["x", "y", "z"]
        }
        .unwrap();

        let result = get_numeric_columns(&df).unwrap();

        // Should return empty dataframe (no numeric columns)
        assert_eq!(result.width(), 0);
    }

    #[test]
    fn test_get_numeric_columns_empty_dataframe() {
        let df = DataFrame::empty();

        let result = get_numeric_columns(&df).unwrap();

        assert_eq!(result.width(), 0);
        assert_eq!(result.height(), 0);
    }

    #[test]
    fn test_get_numeric_columns_preserves_data() {
        let df = df! {
            "values" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "text" => ["a", "b", "c", "d", "e"]
        }
        .unwrap();

        let result = get_numeric_columns(&df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 5);

        // Verify the values are preserved
        let values = result.column("values").unwrap().f64().unwrap();
        assert_eq!(values.get(0), Some(1.0));
        assert_eq!(values.get(4), Some(5.0));
    }

    #[test]
    fn test_get_numeric_columns_different_int_types() {
        let df = df! {
            "i32_col" => [1i32, 2, 3],
            "i64_col" => [1i64, 2, 3],
            "u32_col" => [1u32, 2, 3]
        }
        .unwrap();

        let result = get_numeric_columns(&df).unwrap();

        // All integer types should be considered numeric
        assert_eq!(result.width(), 3);
    }
}
