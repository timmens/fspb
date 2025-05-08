import pandas as pd


def produce_publication_table(
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
