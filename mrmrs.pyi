from typing import Literal

from polars import DataFrame, Series

class Feature:
    """
    A struct holding information about the selected feature

    Available fields are:
        name: name of the selected feature
        relevance: as returned by the specified relevance method (e.g. ANOVA, MI)
        redundance:
            average redundance score vs all already selected features at the
            step when this feature was added to the set of selected features
        score:
            calculated from relevance and redundance based on which method is used
            FCQ: score = relevance / redundance (default)
    """

    name: str
    relevance: float
    redundance: float
    score: float

def mrmr(
    X: DataFrame,
    y: Series,
    number_of_features: int,
    task_type: Literal["classification", "regression"],
) -> list[Feature]:
    """
    Perform mrmr feature selection on a polars dataframe.

    The returned list of features is sorted in the order of the features being
    added to it. So if a smaller list is desired it can be created by taking a
    slice of the one returned from this function.

    Inputs:
        X - a polars dataframe with the features as columns
        y - a polars series with the target variable
        number_of_features - number of features to select
        task_type - task to perform the selection for, either regression or classification
    """
