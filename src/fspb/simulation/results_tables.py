import pandas as pd


def produce_prediction_publication_table(
    consolidated: pd.DataFrame,
) -> pd.DataFrame:
    # consolidated data without band_type index level!
    rounded = consolidated.map(lambda x: f"{x:.2f}")

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
        "coverage": r"\textsc{Coverage}",
        "maximum_width_statistic": r"\textsc{Maximum Width}",
        "band_score": r"\textsc{Band Score}",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "covariance_type": r"$\gamma_{st}$",
    }

    result = result.reset_index()
    methods = set(result["method"].values)
    if methods == {"fair", "ci"}:
        cats = ["fair", "ci"]
    else:
        raise ValueError(f"Unexpected methods in result: {methods}")

    result["method"] = (
        result["method"]
        .astype(pd.CategoricalDtype(cats, ordered=True))
        .cat.rename_categories(
            {
                "fair": "Fair",
                "ci": "Conf. Inf.",
            }
        )
    )
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(["method", "$n$", r"$\nu$"])
    result = result.unstack(level="method")  # type: ignore[assignment]
    columns = result.columns
    columns.names = ["Metric", "Band"]
    result.columns = columns
    return result


def produce_confidence_publication_table(consolidated: pd.DataFrame) -> pd.DataFrame:
    # consolidated data without band_type index level!
    rounded = consolidated.map(lambda x: f"{x:.2f}")  # safe for numeric cols

    column_groups = ["coverage", "maximum_width_statistic", "band_score"]
    combined = {}
    for column in column_groups:
        mean_col = rounded[column]
        std_col = rounded[f"{column}_std"]
        combined[column] = mean_col.astype(str) + " (" + std_col.astype(str) + ")"

    result = pd.DataFrame(combined).reset_index()

    # Only Fair method for confidence bands
    result["method"] = (
        result["method"]
        .astype(pd.CategoricalDtype(["fair"], ordered=True))
        .cat.rename_categories({"fair": "Fair"})
    )

    result["covariance_type"] = (
        result["covariance_type"]
        .astype(pd.CategoricalDtype(["stationary", "non_stationary"], ordered=False))
        .cat.rename_categories(
            {
                "stationary": "Stat.",
                "non_stationary": "Non-Stat.",
            }
        )
    )

    # Rename variables to publication-friendly labels
    var_rename_mapping = {
        "coverage": r"\textsc{Coverage}",
        "maximum_width_statistic": r"\textsc{Maximum Width}",
        "band_score": r"\textsc{Band Score}",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "method": "Method",
    }

    result = result.rename(columns=var_rename_mapping)

    # Pivot to MultiIndex columns: (Metric, Covariance Type)
    result = (
        result.set_index(["$n$", r"$\nu$", "covariance_type"])  # type: ignore[assignment]
        .sort_index()
        .drop(columns=["Method"])
        .unstack(level="covariance_type")
    )

    # Tidy column index (remove names)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns.to_flat_index(), names=["Metric", "Band"]
    )
    return result


template_start_fair = r"""
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} rrcccccc}
\toprule
 &  & \multicolumn{2}{c}{\textsc{Coverage}} & \multicolumn{2}{c}{\textsc{Maximum Width}} & \multicolumn{2}{c}{\textsc{Band Score}} \\
$n$ & $\nu$ & Stat. & Non-Stat. & Stat. & Non-Stat. & Stat. & Non-Stat. \\
\midrule
"""

template_start_fair_vs_ci = r"""
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} rrcccccc}
\toprule
 &  & \multicolumn{2}{c}{\textsc{Coverage}} & \multicolumn{2}{c}{\textsc{Maximum Width}} & \multicolumn{2}{c}{\textsc{Band Score}} \\
$n$ & $\nu$ & Fair & Conf. Inf. & Fair & Conf. Inf. & Fair & Conf. Inf. \\
\midrule
"""

template_end = r"""
\bottomrule
\end{tabular*}
"""


def fill_template(df: pd.DataFrame, type: str) -> str:
    if type == "confidence":
        col_order = [
            (r"\textsc{Coverage}", "Stat."),
            (r"\textsc{Coverage}", "Non-Stat."),
            (r"\textsc{Maximum Width}", "Stat."),
            (r"\textsc{Maximum Width}", "Non-Stat."),
            (r"\textsc{Band Score}", "Stat."),
            (r"\textsc{Band Score}", "Non-Stat."),
        ]
        template_start = template_start_fair
    elif type == "prediction":
        col_order = [
            (r"\textsc{Coverage}", "Fair"),
            (r"\textsc{Coverage}", "Conf. Inf."),
            (r"\textsc{Maximum Width}", "Fair"),
            (r"\textsc{Maximum Width}", "Conf. Inf."),
            (r"\textsc{Band Score}", "Fair"),
            (r"\textsc{Band Score}", "Conf. Inf."),
        ]
        template_start = template_start_fair_vs_ci
    else:
        raise ValueError(f"Unknown type: {type}")

    rows = []
    for n, sub in df.groupby(level=0):
        first = True
        for (nn, nu), row in sub.iterrows():  # type: ignore[misc]
            n_cell = f"{n}" if first else ""
            first = False
            vals = [str(row[c]) for c in col_order]
            rows.append(" & ".join([n_cell, str(nu), *vals]) + r" \\")  # type: ignore[has-type]
    return template_start + "\n".join(rows) + template_end
