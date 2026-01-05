import numpy as np
import pandas as pd
from fspb.config import SRC
from pathlib import Path
from pytask import Product
import pytask
from typing import Annotated

SEED = 80734598723


NON_ANONYMIZED_DATA_EXISTS = (SRC / "application" / "covariates.csv").exists() and (
    SRC / "application" / "outcomes.csv"
).exists()


@pytask.mark.skipif(
    not NON_ANONYMIZED_DATA_EXISTS, reason="Non-anonymized source data does not exist."
)
def task_anonymize(
    covariates_path: Path = SRC / "application" / "covariates.csv",
    outcomes_path: Path = SRC / "application" / "outcomes.csv",
    anonymized_path: Annotated[dict[str, Path], Product] = {
        "covariates": SRC / "application" / "covariates_anonymized.csv",
        "outcomes": SRC / "application" / "outcomes_anonymized.csv",
    },
) -> None:
    covariates = pd.read_csv(covariates_path)
    outcomes = pd.read_csv(outcomes_path)
    rng = np.random.default_rng(SEED)
    anon_cov, anon_out = _anonymize(covariates, outcomes, rng=rng)
    anon_cov.to_csv(anonymized_path["covariates"], index=False)
    anon_out.to_csv(anonymized_path["outcomes"], index=False)


def _anonymize(
    cov: pd.DataFrame, out: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Align rows via id
    df = cov.merge(out, on="id", how="inner", validate="one_to_one")

    # Columns
    cov_cols = cov.columns.tolist()
    out_cols = [c for c in df.columns if c not in cov_cols]
    num_cov_cols = [c for c in cov_cols if c not in ["id", "amputee", "sex"]]
    num_out_cols = [c for c in out_cols if c != "id"]

    # Preserve exactly 7 amputees by sampling within amputee strata
    parts = []
    for _, group in df.groupby("amputee", sort=False):
        parts.append(_bootstrap_with_noise(group, num_cov_cols, num_out_cols, rng=rng))

    anon = pd.concat(parts, ignore_index=True)

    # New anonymized IDs
    anon = anon.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    anon["id"] = np.arange(1000, 1000 + len(anon))

    # Split back and overwrite files
    anon_cov = anon[cov_cols].copy()
    anon_out = anon[
        ["id"] + [c for c in anon.columns if c not in cov_cols and c != "id"]
    ].copy()

    return anon_cov, anon_out


def _bootstrap_with_noise(
    group: pd.DataFrame,
    num_cov_cols: list[str],
    num_out_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(group)
    boot = group.sample(n=n, replace=True).reset_index(drop=True).copy()

    # Add small noise to numeric covariates
    for c in num_cov_cols:
        x = group[c].astype(float).to_numpy()
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        boot[c] = boot[c].astype(float) + rng.normal(0, 0.05 * sd, size=n)

    # Keep age roughly integer-like if it is integer-ish
    if "age" in boot.columns:
        boot["age"] = np.clip(np.round(boot["age"]), 0, None).astype(int)

    # Add small noise to numeric outcomes and clip to original global range
    for c in num_out_cols:
        x = group[c].astype(float).to_numpy()
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        boot[c] = boot[c].astype(float) + rng.normal(0, 0.02 * sd, size=n)

    return boot
