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
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(["Method", "$n$", r"$\nu$"])
    result = result.unstack(level="Method")  # type: ignore[return-value]

    columns = pd.MultiIndex.from_tuples(
        [
            (
                metric,
                "CI" if method == "CI (Linear)" else method,
            )
            for (metric, method) in result.columns.to_list()
        ]
    )
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
    covariance_type_dtype = pd.CategoricalDtype(
        categories=["stationary", "non_stationary"], ordered=True
    )
    result["covariance_type"] = (
        result["covariance_type"]
        .astype(covariance_type_dtype)
        .cat.rename_categories(
            {
                "stationary": "S",
                "non_stationary": "NS",
            },
        )
    )
    result["band_method"] = (
        result["band_method"]
        .astype(pd.CategoricalDtype(["fair", "min_width"], ordered=True))
        .cat.rename_categories(
            {
                "fair": "Fair",
                "min_width": "Min. Width",
            }
        )
    )
    result = result.set_index(["covariance_type", "n_samples", "dof"])

    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "band_score": "Band Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "covariance_type": r"$\gamma_{st}$",
        "band_method": "Band Method",
    }

    result = result.reset_index()
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(
        [r"$\gamma_{st}$", "$n$", r"$\nu$", "Band Method"]
    ).sort_index()
    result = result.unstack(level="Band Method")
    columns = result.columns
    columns.names = [None, None]
    result.columns = columns

    return result
