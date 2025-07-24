import pandas as pd


def produce_prediction_publication_table(
    consolidated: pd.DataFrame,
) -> pd.DataFrame:
    # consolidated data without band_type index level!
    rounded = consolidated.map(lambda x: f"{x:.3f}")

    column_groups = [
        "coverage",
        "maximum_width_statistic",
        "band_score",
    ]
    combined = {}

    for column in column_groups:
        mean_col = rounded[column]
        std_col = rounded[f"{column}_std"]
        combined[column] = mean_col.astype(str) + " (" + std_col.astype(str) + ")"

    result = pd.DataFrame(combined)

    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "band_score": "Band Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "covariance_type": r"$\gamma_{st}$",
    }

    result = result.reset_index()
    if "fair" in result["method"].values:
        cats = ["fair", "ci"]
    else:
        cats = ["min_width", "ci"]
    result["method"] = (
        result["method"]
        .astype(pd.CategoricalDtype(cats, ordered=True))
        .cat.rename_categories(
            {
                "fair": "Fair",
                "ci": "Conf. Inf.",
                "min_width": "Min. Width",
            }
        )
    )
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(["method", "$n$", r"$\nu$"])
    result = result.unstack(level="method")  # type: ignore[assignment]
    columns = result.columns
    columns.names = [None, None]  # type: ignore[list-item]
    result.columns = columns
    return result


def produce_confidence_publication_table(
    consolidated: pd.DataFrame,
) -> pd.DataFrame:
    # consolidated data without band_type index level!
    rounded = consolidated.map(lambda x: f"{x:.3f}")

    column_groups = [
        "coverage",
        "maximum_width_statistic",
        "band_score",
    ]
    combined = {}

    for column in column_groups:
        mean_col = rounded[column]
        std_col = rounded[f"{column}_std"]
        combined[column] = mean_col.astype(str) + " (" + std_col.astype(str) + ")"

    result = pd.DataFrame(combined).reset_index()

    result["method"] = (
        result["method"]
        .astype(pd.CategoricalDtype(["min_width", "fair"], ordered=True))
        .cat.rename_categories(
            {
                "fair": "Fair",
                "min_width": "Min. Width",
            }
        )
    )
    result = result.set_index(["n_samples", "dof"])

    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "band_score": "Band Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "method": "Method",
    }

    result = result.reset_index()
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(["$n$", r"$\nu$", "Method"]).sort_index()
    result = result.unstack(level="Method")  # type: ignore[assignment]
    columns = result.columns
    columns.names = [None, None]  # type: ignore[list-item]
    result.columns = columns

    return result
