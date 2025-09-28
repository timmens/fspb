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
    if methods == {"min_width", "ci"}:
        cats = ["min_width", "ci"]
    elif methods == {"min_width", "fair"}:
        cats = ["min_width", "fair"]
    else:
        raise ValueError(f"Unexpected methods in result: {methods}")

    result["method"] = (
        result["method"]
        .astype(pd.CategoricalDtype(cats, ordered=True))
        .cat.rename_categories(
            {
                "fair": "Fair",
                "ci": "Conf. Inf.",
                "min_width": "Min.-Width",
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

    # Ordered methods and final labels
    result["method"] = (
        result["method"]
        .astype(pd.CategoricalDtype(["min_width", "fair"], ordered=True))
        .cat.rename_categories({"fair": "Fair", "min_width": "Min.-Width"})
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


template_start = r"""
\begin{tabular}{rrcccccc}
\toprule
 &  & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Maximum Width} & \multicolumn{2}{c}{Band Score} \\
$n$ & $\nu$ & Min.-Width & Fair & Min.-Width & Fair & Min.-Width & Fair \\
\midrule
"""
template_end = r"""
\bottomrule
\end{tabular}
"""


def fill_template(df: pd.DataFrame, type: str) -> str:
    if type == "confidence":
        col_order = [
            ("Coverage", "Min.-Width"),
            ("Coverage", "Fair"),
            ("Maximum Width", "Min.-Width"),
            ("Maximum Width", "Fair"),
            ("Band Score", "Min.-Width"),
            ("Band Score", "Fair"),
        ]
    else:
        col_order = [
            ("Coverage", "Min.-Width"),
            ("Coverage", "Conf. Inf."),
            ("Maximum Width", "Min.-Width"),
            ("Maximum Width", "Conf. Inf."),
            ("Band Score", "Min.-Width"),
            ("Band Score", "Conf. Inf."),
        ]
    rows = []
    for n, sub in df.groupby(level=0):
        first = True
        for (nn, nu), row in sub.iterrows():
            n_cell = f"{n}" if first else ""
            first = False
            vals = [str(row[c]) for c in col_order]
            rows.append(" & ".join([n_cell, str(nu), *vals]) + r" \\")
    return template_start + "\n".join(rows) + template_end
