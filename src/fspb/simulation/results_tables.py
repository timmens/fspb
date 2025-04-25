import pandas as pd


def produce_publication_table(
    consolidated: pd.DataFrame,
) -> pd.DataFrame:
    # consolidated data without band_type index level!
    rounded = consolidated.map(lambda x: f"{x:.3f}").drop(columns="coverage_std")

    column_groups = [
        "maximum_width_statistic",
        "band_score",
    ]
    combined = {"coverage": rounded["coverage"]}

    for column in column_groups:
        mean_col = rounded[column]
        std_col = rounded[f"{column}_std"]

        combined[column] = mean_col.astype(str) + " (" + std_col.astype(str) + ")"

    result = pd.DataFrame(combined)

    val_rename_mapping = {
        "covariance_type": {
            "non_stationary": "NS",
            "stationary": "S",
        },
    }

    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "band_score": "Band Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "covariance_type": r"$\gamma_{st}$",
    }

    result = result.reset_index()
    result = result.replace(val_rename_mapping)  # type: ignore[arg-type]
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(["Method", "$n$", r"$\nu$", r"$\gamma_{st}$"])
    return result.unstack(level="Method")  # type: ignore[return-value]
