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
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "band_score": "Band Score",
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

    # Rename variables to publication-friendly labels
    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "band_score": "Band Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "method": "Method",
    }

    result = result.rename(columns=var_rename_mapping)

    # Pivot to MultiIndex columns: (Metric, Band)
    result = (
        result.set_index(["$n$", r"$\nu$", "Method"])
        .sort_index()
        .unstack(level="Method")  # -> columns like (Coverage, Fair) etc.
    )

    # Tidy column index (remove names)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns.to_flat_index(), names=["Metric", "Band"]
    )
    return result


template_start_fair = r"""
\begin{tabular}{rrccc}
\toprule
 &  & Coverage & Maximum Width & Band Score \\
$n$ & $\nu$ & Fair & Fair & Fair \\
\midrule
"""

template_start_fair_vs_ci = r"""
\begin{tabular}{rrcccccc}
\toprule
 &  & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Maximum Width} & \multicolumn{2}{c}{Band Score} \\
$n$ & $\nu$ & Fair & Conf. Inf. & Fair & Conf. Inf. & Fair & Conf. Inf. \\
\midrule
"""

template_end = r"""
\bottomrule
\end{tabular}
"""


def fill_template(df: pd.DataFrame, type: str) -> str:
    if type == "confidence":
        col_order = [
            ("Coverage", "Fair"),
            ("Maximum Width", "Fair"),
            ("Band Score", "Fair"),
        ]
        template_start = template_start_fair
    elif type == "prediction":
        col_order = [
            ("Coverage", "Fair"),
            ("Coverage", "Conf. Inf."),
            ("Maximum Width", "Fair"),
            ("Maximum Width", "Conf. Inf."),
            ("Band Score", "Fair"),
            ("Band Score", "Conf. Inf."),
        ]
        template_start = template_start_fair_vs_ci
    else:
        raise ValueError(f"Unknown type: {type}")

    rows = []
    for n, sub in df.groupby(level=0):
        first = True
        for (nn, nu), row in sub.iterrows():
            n_cell = f"{n}" if first else ""
            first = False
            vals = [str(row[c]) for c in col_order]
            rows.append(" & ".join([n_cell, str(nu), *vals]) + r" \\")
    return template_start + "\n".join(rows) + template_end
