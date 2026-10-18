from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from google.cloud import bigquery
from lifelines import CoxPHFitter
from scipy import stats


CDR = os.environ["WORKSPACE_CDR"]
os.environ["WORKSPACE_CDR"] = CDR

DATA_DIR = Path(os.environ["CAPSTONE_DATA_DIR"]).expanduser()
if DATA_DIR.exists():
    os.chdir(DATA_DIR)

OUTPUT_DIR = Path(os.environ["CAPSTONE_OUTPUT_DIR"]).expanduser()
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def table_path(filename: str) -> Path:
    return TABLE_DIR / filename


def figure_path(filename: str) -> Path:
    return FIGURE_DIR / filename

client = bigquery.Client()

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 120)

WCM_RED = "#B31B1B"
WCM_NAVY = "#1f3a5f"
NMOL_PER_NG = 2.496

Q1_COVARS = [
    "age_at_index",
    "female",
    "race_black",
    "race_hispanic",
    "race_asian",
    "race_other",
    "baseline_diabetes",
    "baseline_hypertension",
    "baseline_hyperlipidemia",
]

CRP_CONCEPTS = (3020460, 3010156, 4208414, 46272908)
CRP_UNIT_FACTOR = {
    8751: 1.0,
    4122414: 1.0,
    8840: 10.0,
    4121396: 10.0,
}

def print_section(title: str, width: int = 88) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_query(sql: str) -> pd.DataFrame:
    df = client.query(sql).to_dataframe()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def as_person_id(df: pd.DataFrame, col: str = "person_id") -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def first_existing(paths: Iterable[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def vitamin_d_group(value: float | int | None) -> str | None:
    if pd.isna(value):
        return None
    if value < 10:
        return "Deficient (<10)"
    if value < 20:
        return "Insufficient (10-20)"
    if value < 30:
        return "Adequate (20-30)"
    return "Sufficient (>=30)"


def vitamin_d_trend(value: float | int | None) -> float | None:
    if pd.isna(value):
        return None
    if value >= 30:
        return 0
    if value >= 20:
        return 1
    if value >= 10:
        return 2
    return 3


def add_standard_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gender" in df.columns:
        df["female"] = (df["gender"] == "Female").astype(int)
    if "race_eth_simplified" in df.columns:
        race = df["race_eth_simplified"]
        df["race_black"] = (race == "Black or African American").astype(int)
        df["race_hispanic"] = (race == "Hispanic or Latino").astype(int)
        df["race_asian"] = (race == "Asian").astype(int)
        df["race_other"] = race.isin(["Other/Multiracial", "Unknown/Other"]).astype(int)
    return df


def normalize_index_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "index_date" not in df.columns:
        candidates = [c for c in df.columns if c.startswith("index_date")]
        if candidates:
            df["index_date"] = df[candidates[0]]
    if "index_date" in df.columns:
        df["index_date"] = parse_datetime(df["index_date"])
    return df

def load_q1_analysis_base() -> pd.DataFrame:
    fu_path = first_existing(["q1_followup_person_level_v2.csv"])
    cov_path = first_existing(["q1_covariate_table_v3_bmi.csv", "q1_covariate_table_v2.csv"])

    if fu_path and cov_path:
        fu = pd.read_csv(fu_path)
        cov = pd.read_csv(cov_path)
        fu = as_person_id(fu)
        cov = as_person_id(cov)
        base = fu.merge(cov, on="person_id", how="inner", suffixes=("_fu", "_cov"))

        for col in ["index_date", "follow_up_years", "event_mi_or_stroke"]:
            if col not in base.columns:
                for suffix in ["_fu", "_cov"]:
                    src = f"{col}{suffix}"
                    if src in base.columns:
                        base[col] = base[src]
                        break
        if "vitd_main" not in base.columns:
            if "vitd_ng_ml" not in base.columns:
                raise ValueError("Need vitd_main or vitd_ng_ml in covariate table")
            base["vitd_main"] = base["vitd_ng_ml"]
        if "vitd_ng_ml" not in base.columns:
            base["vitd_ng_ml"] = base["vitd_main"]
    else:
        cached_base_path = first_existing(["analysis_base_clean.csv", str(table_path("analysis_base_clean.csv"))])
        if cached_base_path is None:
            raise FileNotFoundError(
                "Need q1_followup_person_level_v2.csv + q1_covariate_table_v2/v3_bmi.csv, "
                "or an existing analysis_base_clean.csv"
            )
        base = pd.read_csv(cached_base_path)
        base = as_person_id(base)

    base = normalize_index_date(base)
    base = add_standard_covariates(base)
    base["vitd_main"] = pd.to_numeric(base["vitd_main"], errors="coerce")
    base["vitd_ng_ml"] = pd.to_numeric(base["vitd_ng_ml"], errors="coerce")
    base["follow_up_years"] = pd.to_numeric(base["follow_up_years"], errors="coerce")
    base["event_mi_or_stroke"] = pd.to_numeric(base["event_mi_or_stroke"], errors="coerce")

    if "gender" in base.columns:
        base = base[base["gender"].isin(["Female", "Male"])].copy()
    base = base[base["follow_up_years"] > 0].copy()
    base = base.dropna(subset=["vitd_main", "follow_up_years", "event_mi_or_stroke"] + Q1_COVARS)

    if "bmi_full" in base.columns:
        base["bmi_full"] = pd.to_numeric(base["bmi_full"], errors="coerce")

    keep = [
        "person_id",
        "index_date",
        "vitd_main",
        "vitd_ng_ml",
        "follow_up_years",
        "event_mi_or_stroke",
        "gender",
        "race_eth_simplified",
        "bmi_full",
    ] + Q1_COVARS
    keep = [c for c in keep if c in base.columns]
    base = base[keep].copy()
    analysis_base_out = table_path("analysis_base_clean.csv")
    base.to_csv(analysis_base_out, index=False)

    print_section("Analysis base")
    print(f"Rows:   {len(base):,}")
    print(f"Events: {int(base['event_mi_or_stroke'].sum()):,}")
    print(f"Saved {analysis_base_out}")
    return base

def fetch_crp_records() -> pd.DataFrame:
    concepts = ",".join(map(str, CRP_CONCEPTS))
    return run_query(
        f"""
        SELECT person_id,
               measurement_datetime AS dt,
               value_as_number AS val,
               unit_concept_id AS unit_id,
               measurement_concept_id AS cid
        FROM `{CDR}.measurement`
        WHERE measurement_concept_id IN ({concepts})
          AND value_as_number IS NOT NULL
          AND value_as_number > 0
        """
    )


def clean_crp_records(crp_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    crp = crp_raw.copy()
    crp = as_person_id(crp)
    crp["val"] = pd.to_numeric(crp["val"], errors="coerce")
    crp["unit_key"] = pd.to_numeric(crp["unit_id"], errors="coerce").fillna(-1).astype(int)
    raw_rows = len(crp)

    crp["factor"] = crp["unit_key"].map(CRP_UNIT_FACTOR)
    group_median = crp.groupby(["cid", "unit_key"])["val"].transform("median")
    ambiguous = crp["unit_key"].isin([0, -1])
    crp.loc[ambiguous, "factor"] = np.where(group_median[ambiguous] < 1.5, 10.0, 1.0)

    no_factor = int(crp["factor"].isna().sum())
    crp = crp.dropna(subset=["factor", "val"]).copy()
    rows_before_range = len(crp)
    crp["crp_mgL"] = crp["val"] * crp["factor"]
    crp["ambiguous"] = crp["unit_key"].isin([0, -1])
    crp = crp[(crp["crp_mgL"] >= 0.1) & (crp["crp_mgL"] <= 300)].copy()
    audit = {
        "raw_rows": raw_rows,
        "no_factor_rows": no_factor,
        "out_of_range_rows": rows_before_range - len(crp),
        "clean_rows": len(crp),
        "clean_participants": crp["person_id"].nunique(),
    }
    return crp, audit


def build_baseline_crp(base: pd.DataFrame, crp_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_ids = set(base["person_id"].dropna())
    matched = crp_clean[crp_clean["person_id"].isin(cohort_ids)].merge(
        base[["person_id", "index_date"]],
        on="person_id",
        how="inner",
    )
    matched["dt"] = parse_datetime(matched["dt"])
    matched["days_before"] = (matched["index_date"] - matched["dt"]).dt.days
    pre_index = matched[matched["days_before"] >= 0].copy()
    baseline = (
        pre_index.sort_values(["person_id", "days_before"])
        .groupby("person_id", as_index=False)
        .first()[["person_id", "crp_mgL", "days_before", "ambiguous"]]
    )
    return baseline, matched


def run_q2_crp_build(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    crp_raw = fetch_crp_records()
    crp_clean, audit = clean_crp_records(crp_raw)
    baseline_crp, matched = build_baseline_crp(base, crp_clean)
    baseline_crp.to_csv(table_path("q2_crp_baseline_clean.csv"), index=False)
    crp_clean[["person_id", "dt", "cid", "unit_key", "crp_mgL", "ambiguous"]].to_csv(table_path("crp_long_clean.csv"), index=False)

    print_section("Q2 CRP rebuild")
    for key, value in audit.items():
        print(f"{key}: {value:,}")
    print(f"Baseline/pre-index CRP participants: {len(baseline_crp):,}")
    print(f"Ambiguous baseline unit rows: {int(baseline_crp['ambiguous'].sum()):,}")
    print("Saved q2_crp_baseline_clean.csv and crp_long_clean.csv")
    return baseline_crp, crp_clean

def build_q2_pair(base: pd.DataFrame, baseline_crp: pd.DataFrame) -> pd.DataFrame:
    pair = base[["person_id", "vitd_main"]].merge(
        baseline_crp[["person_id", "crp_mgL", "days_before", "ambiguous"]],
        on="person_id",
        how="inner",
    )
    pair["vitd_main"] = pd.to_numeric(pair["vitd_main"], errors="coerce")
    pair["crp_mgL"] = pd.to_numeric(pair["crp_mgL"], errors="coerce")
    pair = pair.dropna(subset=["vitd_main", "crp_mgL"])
    pair = pair[(pair["vitd_main"] > 0) & (pair["crp_mgL"] > 0)].copy()
    pair["log_vitd"] = np.log(pair["vitd_main"])
    pair["log_crp"] = np.log(pair["crp_mgL"])
    pair.to_csv(table_path("q2_pair_loglog.csv"), index=False)
    return pair

def resolve_input_file(filename):
    """Prefer this run's tables, then the configured authorized input directory."""
    for p in [TABLE_DIR / filename, DATA_DIR / filename]:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing required input: {filename}")
