# Source companion of 01_q1_q2_workflow.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 3
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


CDR = os.environ.get("WORKSPACE_CDR", "YOUR_GOOGLE_PROJECT.C2024Q3R9")
os.environ["WORKSPACE_CDR"] = CDR

DATA_DIR = Path(
    os.environ.get(
        "CAPSTONE_DATA_DIR",
        "/home/jupyter/capstone/data",
    )
)
if DATA_DIR.exists():
    os.chdir(DATA_DIR)

OUTPUT_DIR = Path(os.environ.get("CAPSTONE_OUTPUT_DIR", "0729_clean_outputs"))
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

# %% Cell 5
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

# %% Cell 7
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

# %% Cell 9
def median_iqr(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def n_pct(mask: pd.Series) -> str:
    mask = mask.fillna(False)
    return f"{int(mask.sum()):,} ({mask.mean() * 100:.1f}%)"


def make_q1_table1(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["vitd_group"] = df["vitd_main"].apply(vitamin_d_group)
    order = [
        "Deficient (<10)",
        "Insufficient (10-20)",
        "Adequate (20-30)",
        "Sufficient (>=30)",
    ]

    rows: list[list[str]] = []

    def add(label: str, fn) -> None:
        rows.append([label] + [fn(df[df["vitd_group"] == g]) for g in order] + [fn(df)])

    add("N", lambda d: f"{len(d):,}")
    add("Age, median (IQR)", lambda d: median_iqr(d["age_at_index"]))
    if "gender" in df.columns:
        add("Female, n (%)", lambda d: n_pct(d["gender"] == "Female"))
    if "race_eth_simplified" in df.columns:
        add("White, n (%)", lambda d: n_pct(d["race_eth_simplified"] == "White (Non-Hispanic)"))
        add("Black, n (%)", lambda d: n_pct(d["race_eth_simplified"] == "Black or African American"))
        add("Hispanic, n (%)", lambda d: n_pct(d["race_eth_simplified"] == "Hispanic or Latino"))
        add("Asian, n (%)", lambda d: n_pct(d["race_eth_simplified"] == "Asian"))
    add("Diabetes, n (%)", lambda d: n_pct(d["baseline_diabetes"] == 1))
    add("Hypertension, n (%)", lambda d: n_pct(d["baseline_hypertension"] == 1))
    add("Hyperlipidemia, n (%)", lambda d: n_pct(d["baseline_hyperlipidemia"] == 1))
    if "bmi_full" in df.columns:
        add("BMI, median (IQR)", lambda d: median_iqr(d["bmi_full"]))
        add("BMI available, n (%)", lambda d: n_pct(d["bmi_full"].notna()))
    add("Incident MI/stroke, n (%)", lambda d: n_pct(d["event_mi_or_stroke"] == 1))
    add("Follow-up years, median (IQR)", lambda d: median_iqr(d["follow_up_years"]))

    table = pd.DataFrame(rows, columns=["Characteristic"] + order + ["Overall"])
    table.to_csv(table_path("q1_table1_by_vitd.csv"), index=False)
    print_section("Q1 Table 1 by vitamin D category")
    print(table.to_string(index=False))
    print("Saved q1_table1_by_vitd.csv")
    return table

# %% Cell 11
def lifelines_rows(model: CoxPHFitter, label: str, terms: list[str]) -> pd.DataFrame:
    rows = []
    for term in terms:
        if term not in model.summary.index:
            continue
        row = model.summary.loc[term]
        rows.append(
            {
                "model": label,
                "term": term,
                "n": int(model._n_examples),
                "events": int(model.event_observed.sum()),
                "hr": row["exp(coef)"],
                "lo": row["exp(coef) lower 95%"],
                "hi": row["exp(coef) upper 95%"],
                "p": row["p"],
            }
        )
    return pd.DataFrame(rows)


def run_q1_main_cox(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = base.copy()
    d["vitd_deficient"] = (d["vitd_main"] < 10).astype(int)
    d["vitd_insufficient"] = ((d["vitd_main"] >= 10) & (d["vitd_main"] < 20)).astype(int)
    d["vitd_adequate"] = ((d["vitd_main"] >= 20) & (d["vitd_main"] < 30)).astype(int)
    d["vitd_trend"] = d["vitd_main"].apply(vitamin_d_trend)
    d["vitd_neg10"] = -d["vitd_main"] / 10.0

    exposure_terms = ["vitd_deficient", "vitd_insufficient", "vitd_adequate"]
    base_cols = ["follow_up_years", "event_mi_or_stroke"] + exposure_terms + Q1_COVARS
    model_a_df = d[base_cols].dropna()
    model_a = CoxPHFitter().fit(model_a_df, "follow_up_years", "event_mi_or_stroke")

    tables = [lifelines_rows(model_a, "categorical_no_bmi", exposure_terms)]

    if "bmi_full" in d.columns:
        bmi_df = d[(d["bmi_full"] >= 12) & (d["bmi_full"] <= 70)].copy()
        model_b_df = bmi_df[base_cols + ["bmi_full"]].dropna()
        model_b = CoxPHFitter().fit(model_b_df, "follow_up_years", "event_mi_or_stroke")
        tables.append(lifelines_rows(model_b, "categorical_plus_bmi", exposure_terms + ["bmi_full"]))

    trend_cols = ["follow_up_years", "event_mi_or_stroke", "vitd_trend"] + Q1_COVARS
    trend_df = d[trend_cols].dropna()
    trend_model = CoxPHFitter().fit(trend_df, "follow_up_years", "event_mi_or_stroke")
    trend_table = lifelines_rows(trend_model, "ordinal_trend", ["vitd_trend"])
    trend_table.to_csv(table_path("q1_trend_test.csv"), index=False)

    cont_cols = ["follow_up_years", "event_mi_or_stroke", "vitd_neg10"] + Q1_COVARS
    cont_df = d[cont_cols].dropna()
    cont_model = CoxPHFitter().fit(cont_df, "follow_up_years", "event_mi_or_stroke")
    overall = lifelines_rows(cont_model, "continuous_per_10_ng_ml_lower", ["vitd_neg10"])
    overall.rename(columns={"hr": "hr", "lo": "lo", "hi": "hi", "p": "p"}, inplace=True)
    overall[["hr", "lo", "hi", "p", "n", "events"]].to_csv(table_path("q1_overall_hr.csv"), index=False)

    cox_table = pd.concat(tables + [trend_table, overall], ignore_index=True)
    cox_table.to_csv(table_path("q1_cox_models_clean.csv"), index=False)

    print_section("Q1 Cox models")
    print(cox_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Saved q1_cox_models_clean.csv, q1_trend_test.csv, q1_overall_hr.csv")
    return cox_table, overall

# %% Cell 13
def add_income_band_if_available(df: pd.DataFrame) -> pd.DataFrame:
    try:
        qc = run_query(
            f"""
            SELECT o.observation_concept_id AS cid,
                   c.concept_name AS name,
                   COUNT(DISTINCT o.person_id) AS n
            FROM `{CDR}.observation` o
            JOIN `{CDR}.concept` c
              ON o.observation_concept_id = c.concept_id
            WHERE LOWER(c.concept_name) LIKE '%income%'
            GROUP BY 1, 2
            ORDER BY n DESC
            LIMIT 1
            """
        )
        if qc.empty:
            return df
        income_cid = int(qc.iloc[0]["cid"])
        inc = run_query(
            f"""
            SELECT o.person_id, c.concept_name AS ans
            FROM `{CDR}.observation` o
            JOIN `{CDR}.concept` c
              ON o.value_as_concept_id = c.concept_id
            WHERE o.observation_concept_id = {income_cid}
            """
        )
        inc = as_person_id(inc)

        def band(answer: object) -> str | None:
            text = str(answer).lower()
            if any(k in text for k in ["less 10", "10k 25", "25k 35", "less than 10"]):
                return "Income <35k"
            if any(k in text for k in ["35k 50", "50k 75", "75k 100"]):
                return "Income 35-100k"
            if any(k in text for k in ["100k 150", "150k 200", "more 200"]):
                return "Income >=100k"
            return None

        inc["g_income"] = inc["ans"].apply(band)
        inc = inc.dropna(subset=["g_income"]).drop_duplicates("person_id")
        out = df.merge(inc[["person_id", "g_income"]], on="person_id", how="left")
        print(f"Income coverage in Q1 base: {out['g_income'].notna().sum():,}")
        return out
    except Exception as exc:
        print(f"Income subgroup skipped: {exc}")
        return df


def run_q1_subgroups(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = base.copy()
    d["vitd_neg10"] = -d["vitd_main"] / 10.0
    d["g_age"] = np.where(d["age_at_index"] >= 65, "Age >=65", "Age <65")
    d["g_sex"] = np.where(d["female"] == 1, "Female", "Male")
    d["g_diabetes"] = np.where(d["baseline_diabetes"] == 1, "Yes", "No")
    d["g_hypertension"] = np.where(d["baseline_hypertension"] == 1, "Yes", "No")
    d["g_hyperlipidemia"] = np.where(d["baseline_hyperlipidemia"] == 1, "Yes", "No")
    if "race_eth_simplified" in d.columns:
        d["g_race"] = d["race_eth_simplified"]
    if "bmi_full" in d.columns:
        d["g_bmi"] = pd.cut(
            d["bmi_full"],
            [0, 25, 30, 200],
            labels=["BMI <25", "BMI 25-30", "BMI >=30"],
        ).astype(object)
    d = add_income_band_if_available(d)

    exclude = {
        "g_age": ["age_at_index"],
        "g_sex": ["female"],
        "g_race": ["race_black", "race_hispanic", "race_asian", "race_other"],
        "g_diabetes": ["baseline_diabetes"],
        "g_hypertension": ["baseline_hypertension"],
        "g_hyperlipidemia": ["baseline_hyperlipidemia"],
        "g_bmi": [],
        "g_income": [],
    }

    specs = [
        ("g_age", "Age"),
        ("g_sex", "Sex"),
        ("g_race", "Race/ethnicity"),
        ("g_bmi", "BMI"),
        ("g_diabetes", "Diabetes"),
        ("g_hypertension", "Hypertension"),
        ("g_hyperlipidemia", "Hyperlipidemia"),
        ("g_income", "Household income"),
    ]
    specs = [(col, label) for col, label in specs if col in d.columns]

    def covars_for(group_col: str) -> list[str]:
        return [c for c in Q1_COVARS if c not in exclude[group_col]]

    def fit_hr(sub: pd.DataFrame, adj: list[str]) -> tuple[float, float, float, float]:
        cols = ["follow_up_years", "event_mi_or_stroke", "vitd_neg10"] + adj
        m = sub[cols].dropna()
        m = m.loc[:, m.nunique() > 1]
        model = CoxPHFitter().fit(m, "follow_up_years", "event_mi_or_stroke")
        row = model.summary.loc["vitd_neg10"]
        return row["exp(coef)"], row["exp(coef) lower 95%"], row["exp(coef) upper 95%"], row["p"]

    def interaction_p(group_col: str) -> float:
        dd = d.dropna(subset=[group_col]).copy()
        dum = pd.get_dummies(dd[group_col].astype(str), prefix="g", drop_first=True).astype(float)
        dd = pd.concat([dd.reset_index(drop=True), dum.reset_index(drop=True)], axis=1)
        dum_cols = list(dum.columns)
        interaction_cols = []
        for dum_col in dum_cols:
            col = "ix_" + dum_col
            dd[col] = dd["vitd_neg10"] * dd[dum_col]
            interaction_cols.append(col)
        adj = covars_for(group_col)
        cols = ["follow_up_years", "event_mi_or_stroke", "vitd_neg10"] + adj + dum_cols
        reduced = dd[cols].dropna()
        full = dd[cols + interaction_cols].dropna()
        cr = CoxPHFitter(penalizer=0.0).fit(reduced, "follow_up_years", "event_mi_or_stroke")
        cf = CoxPHFitter(penalizer=0.0).fit(full, "follow_up_years", "event_mi_or_stroke")
        lr = 2 * (cf.log_likelihood_ - cr.log_likelihood_)
        return stats.chi2.sf(lr, len(interaction_cols))

    rows = []
    interactions = []
    for group_col, label in specs:
        for level in sorted(d[group_col].dropna().unique()):
            sub = d[d[group_col] == level]
            events = int(sub["event_mi_or_stroke"].sum())
            if len(sub) < 300 or events < 25:
                continue
            try:
                hr, lo, hi, p = fit_hr(sub, covars_for(group_col))
                rows.append(
                    {
                        "subgroup": label,
                        "level": str(level),
                        "n": len(sub),
                        "events": events,
                        "hr": hr,
                        "lo": lo,
                        "hi": hi,
                        "p": p,
                    }
                )
            except Exception as exc:
                print(f"Subgroup failed: {label}={level}: {exc}")
        try:
            interactions.append({"subgroup": label, "p_interaction": interaction_p(group_col)})
        except Exception as exc:
            print(f"Interaction failed: {label}: {exc}")
            interactions.append({"subgroup": label, "p_interaction": np.nan})

    res = pd.DataFrame(rows)
    itab = pd.DataFrame(interactions)
    res.to_csv(table_path("q1_subgroups_all.csv"), index=False)
    itab.to_csv(table_path("q1_interaction_tests.csv"), index=False)
    if not itab.empty:
        bonf = itab.copy()
        bonf["n_tests"] = len(bonf)
        bonf["bonferroni_threshold"] = 0.05 / len(bonf)
        bonf["bonferroni_pass"] = bonf["p_interaction"] < bonf["bonferroni_threshold"]
        bonf.to_csv(table_path("q1_interaction_bonferroni.csv"), index=False)
    print_section("Q1 subgroup analysis")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nInteraction tests")
    print(itab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if not itab.empty:
        print("\nBonferroni correction")
        print(bonf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Saved q1_subgroups_all.csv, q1_interaction_tests.csv, and q1_interaction_bonferroni.csv")
    return res, itab


def plot_q1_subgroup_forest() -> None:
    required = [
        table_path("q1_subgroups_all.csv"),
        table_path("q1_interaction_tests.csv"),
        table_path("q1_overall_hr.csv"),
    ]
    if not all(path.exists() for path in required):
        return
    res = pd.read_csv(table_path("q1_subgroups_all.csv"))
    itab = pd.read_csv(table_path("q1_interaction_tests.csv")).set_index("subgroup")
    ov = pd.read_csv(table_path("q1_overall_hr.csv")).iloc[0]

    level_order = {
        "Age": ["Age <65", "Age >=65"],
        "Sex": ["Male", "Female"],
        "Race/ethnicity": [
            "White (Non-Hispanic)",
            "Black or African American",
            "Hispanic or Latino",
            "Asian",
            "Other/Multiracial",
            "Unknown/Other",
        ],
        "BMI": ["BMI <25", "BMI 25-30", "BMI >=30"],
        "Diabetes": ["No", "Yes"],
        "Hypertension": ["No", "Yes"],
        "Hyperlipidemia": ["No", "Yes"],
        "Household income": ["Income <35k", "Income 35-100k", "Income >=100k"],
    }
    order = [s for s in level_order if s in set(res["subgroup"])]

    rows = [
        (
            "Overall",
            None,
            int(ov["n"]),
            int(ov["events"]),
            ov["hr"],
            ov["lo"],
            ov["hi"],
            True,
        )
    ]
    for subgroup in order:
        p_int = itab.loc[subgroup, "p_interaction"] if subgroup in itab.index else np.nan
        rows.append((subgroup, p_int, None, None, None, None, None, False))
        sub = res[res["subgroup"] == subgroup].set_index("level")
        for level in level_order[subgroup]:
            if level not in sub.index:
                continue
            row = sub.loc[level]
            rows.append(
                (
                    "    " + level,
                    None,
                    int(row["n"]),
                    int(row["events"]),
                    row["hr"],
                    row["lo"],
                    row["hi"],
                    False,
                )
            )

    xmin, xmax = 0.92, 1.62
    fig, ax = plt.subplots(figsize=(11.5, max(7, 0.36 * len(rows))))
    y = np.arange(len(rows))[::-1]
    for yy, (label, p_int, n, events, hr, lo, hi, is_overall) in zip(y, rows):
        if hr is None:
            ax.text(-0.60, yy, label, transform=ax.get_yaxis_transform(), fontweight="bold", fontsize=9.5, va="center")
            if pd.notna(p_int):
                ax.text(
                    1.30,
                    yy,
                    f"{p_int:.3f}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=9,
                    va="center",
                    ha="center",
                    fontweight="bold" if p_int < 0.05 else "normal",
                    color=WCM_RED if p_int < 0.05 else "black",
                )
            continue
        hi_clip = min(hi, xmax * 0.985)
        ax.plot([lo, hi_clip], [yy, yy], color=WCM_NAVY, lw=1.7, solid_capstyle="butt")
        if hi > xmax * 0.985:
            ax.annotate("", xy=(xmax, yy), xytext=(hi_clip, yy), arrowprops=dict(arrowstyle="->", color=WCM_NAVY, lw=1.7))
        ax.plot(hr, yy, "s" if is_overall else "o", color=WCM_RED, ms=9 if is_overall else 6, zorder=3)
        ax.text(-0.60, yy, label, transform=ax.get_yaxis_transform(), fontsize=9, va="center", fontweight="bold" if is_overall else "normal")
        ax.text(-0.14, yy, f"{n:,}", transform=ax.get_yaxis_transform(), fontsize=8.5, va="center", ha="right")
        ax.text(-0.02, yy, f"{events:,}", transform=ax.get_yaxis_transform(), fontsize=8.5, va="center", ha="right")
        ax.text(1.03, yy, f"{hr:.2f} ({lo:.2f}-{hi:.2f})", transform=ax.get_yaxis_transform(), fontsize=8.5, va="center")

    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-1, len(rows))
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("Hazard ratio per 10 ng/mL lower 25(OH)D (95% CI)", fontsize=11)
    top = len(rows) - 0.2
    for xx, text, ha in [(-0.14, "N", "right"), (-0.02, "Events", "right"), (1.03, "HR (95% CI)", "left"), (1.30, "P-inter", "center")]:
        ax.text(xx, top, text, transform=ax.get_yaxis_transform(), fontsize=9, ha=ha, fontweight="bold")
    ax.set_title(
        "Subgroup analysis: baseline vitamin D and incident MI or stroke\n"
        f"All of Us cohort, n = {int(ov['n']):,}, {int(ov['events']):,} events",
        fontsize=12,
        pad=22,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.subplots_adjust(left=0.34, right=0.80)
    plt.savefig(figure_path("q1_forest_subgroups.png"), dpi=170, bbox_inches="tight")
    plt.show()
    print("Saved q1_forest_subgroups.png")

# %% Cell 15
def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    t = np.sort(np.asarray(knots, float))
    k = len(t)
    cube = lambda v: np.where(v > 0, v**3, 0.0)
    den = (t[k - 1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        tj = t[j]
        cols.append(
            (
                cube(x - tj)
                - cube(x - t[k - 2]) * (t[k - 1] - tj) / (t[k - 1] - t[k - 2])
                + cube(x - t[k - 1]) * (t[k - 2] - tj) / (t[k - 1] - t[k - 2])
            )
            / den
        )
    return np.column_stack(cols)


def run_q1_spline_and_threshold(base: pd.DataFrame) -> None:
    d = base.copy()
    d["vitd"] = pd.to_numeric(d["vitd_main"], errors="coerce")
    d = d.dropna(subset=["vitd", "follow_up_years", "event_mi_or_stroke"] + Q1_COVARS)

    knot_options = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}
    fits = {}
    for k, percentiles in knot_options.items():
        knots = np.percentile(d["vitd"], percentiles)
        basis = rcs_basis(d["vitd"].values, knots)
        spline_cols = [f"s{i}" for i in range(basis.shape[1])]
        m = pd.DataFrame(basis, columns=spline_cols)
        m["follow_up_years"] = d["follow_up_years"].values
        m["event_mi_or_stroke"] = d["event_mi_or_stroke"].values
        for covar in Q1_COVARS:
            m[covar] = d[covar].values
        cph = CoxPHFitter().fit(m, "follow_up_years", "event_mi_or_stroke")
        aic = -2 * cph.log_likelihood_ + 2 * (m.shape[1] - 2)
        fits[k] = {"model": cph, "knots": knots, "cols": spline_cols, "aic": aic, "ll": cph.log_likelihood_}

    best_k = min(fits, key=lambda k: fits[k]["aic"])
    fit = fits[best_k]

    linear = pd.DataFrame(
        {
            "vitd": d["vitd"].values,
            "follow_up_years": d["follow_up_years"].values,
            "event_mi_or_stroke": d["event_mi_or_stroke"].values,
        }
    )
    for covar in Q1_COVARS:
        linear[covar] = d[covar].values
    cph_linear = CoxPHFitter().fit(linear, "follow_up_years", "event_mi_or_stroke")
    lr_nonlinear = 2 * (fit["ll"] - cph_linear.log_likelihood_)
    p_nonlinear = stats.chi2.sf(lr_nonlinear, len(fit["cols"]) - 1)

    beta = fit["model"].params_[fit["cols"]].values
    var = fit["model"].variance_matrix_.loc[fit["cols"], fit["cols"]].values
    reference = 30.0
    grid = np.linspace(np.percentile(d["vitd"], 1), np.percentile(d["vitd"], 99), 300)
    diff = rcs_basis(grid, fit["knots"]) - rcs_basis(np.array([reference]), fit["knots"])
    log_hr = diff @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", diff, var, diff))
    hr = np.exp(log_hr)
    lo = np.exp(log_hr - 1.96 * se)
    hi = np.exp(log_hr + 1.96 * se)

    xiao_ng = 27.70 / NMOL_PER_NG
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(grid, lo, hi, color=WCM_NAVY, alpha=0.18, label="95% CI")
    ax.plot(grid, hr, color=WCM_RED, lw=2.4, label=f"Adjusted HR ({best_k}-knot RCS)")
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.axvline(reference, color="grey", ls=":", lw=1)
    ax.axvline(xiao_ng, color="green", ls="-.", lw=1.6, label=f"Xiao 2022 = {xiao_ng:.1f} ng/mL")
    ax.set_yscale("log")
    ax.set_xlabel("Baseline serum 25(OH)D, ng/mL", fontsize=12)
    ax.set_ylabel(f"Adjusted hazard ratio (reference = {reference:.0f} ng/mL)", fontsize=12)
    ax.set_title(
        "Dose-response: vitamin D and incident MI or stroke\n"
        f"Restricted cubic spline Cox, n = {len(d):,}, nonlinearity p = {p_nonlinear:.1e}",
        fontsize=11.5,
    )
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(figure_path("q1_spline_doseresponse.png"), dpi=170, bbox_inches="tight")
    plt.show()
    pd.DataFrame({"vitd": grid, "hr": hr, "lo": lo, "hi": hi}).to_csv(table_path("q1_spline_curve.csv"), index=False)

    candidates = np.arange(15, 46, 0.5)
    log_likelihoods = []
    for threshold in candidates:
        m = pd.DataFrame(
            {
                "seg1": np.minimum(d["vitd"].values, threshold),
                "seg2": np.maximum(d["vitd"].values - threshold, 0),
                "follow_up_years": d["follow_up_years"].values,
                "event_mi_or_stroke": d["event_mi_or_stroke"].values,
            }
        )
        for covar in Q1_COVARS:
            m[covar] = d[covar].values
        try:
            log_likelihoods.append(CoxPHFitter().fit(m, "follow_up_years", "event_mi_or_stroke").log_likelihood_)
        except Exception:
            log_likelihoods.append(np.nan)

    log_likelihoods = np.array(log_likelihoods)
    best = int(np.nanargmax(log_likelihoods))
    threshold_hat = candidates[best]
    inside = candidates[log_likelihoods >= np.nanmax(log_likelihoods) - 1.92]

    segmented = pd.DataFrame(
        {
            "seg1": np.minimum(d["vitd"].values, threshold_hat),
            "seg2": np.maximum(d["vitd"].values - threshold_hat, 0),
            "follow_up_years": d["follow_up_years"].values,
            "event_mi_or_stroke": d["event_mi_or_stroke"].values,
        }
    )
    for covar in Q1_COVARS:
        segmented[covar] = d[covar].values
    cph_segmented = CoxPHFitter().fit(segmented, "follow_up_years", "event_mi_or_stroke")
    cph_segmented.summary.to_csv(table_path("q1_twopiecewise_cox.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(candidates, log_likelihoods, color=WCM_NAVY, lw=2)
    ax.axvline(threshold_hat, color=WCM_RED, ls="--", lw=1.8, label=f"Estimate {threshold_hat:.1f} ng/mL")
    ax.axhline(np.nanmax(log_likelihoods) - 1.92, color="grey", ls=":", lw=1.4, label="95% profile likelihood bound")
    ax.axvspan(inside.min(), inside.max(), color=WCM_RED, alpha=0.10)
    ax.axvline(xiao_ng, color="green", ls="-.", lw=1.6, label=f"Xiao 2022 {xiao_ng:.1f} ng/mL")
    ax.set_xlabel("Candidate threshold, ng/mL", fontsize=11)
    ax.set_ylabel("Partial log-likelihood", fontsize=11)
    ax.set_title("Threshold estimation for the L-shaped association", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path("q1_threshold_profile.png"), dpi=170, bbox_inches="tight")
    plt.show()
    pd.DataFrame({"threshold": candidates, "loglik": log_likelihoods}).to_csv(table_path("q1_threshold_profile.csv"), index=False)

    print_section("Q1 spline and threshold")
    print(f"RCS selected knots: {best_k}")
    print(f"Nonlinearity p: {p_nonlinear:.3e}")
    print(f"Threshold estimate: {threshold_hat:.1f} ng/mL, profile CI {inside.min():.1f}-{inside.max():.1f}")
    print("Saved q1_spline_doseresponse.png, q1_spline_curve.csv, q1_threshold_profile.png, q1_twopiecewise_cox.csv")

# %% Cell 17
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

# %% Cell 19
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


def run_q2_correlation(pair: pd.DataFrame) -> pd.DataFrame:
    n = len(pair)
    specs = [
        ("spearman_raw", pair["vitd_main"], pair["crp_mgL"], stats.spearmanr),
        ("spearman_log_log", pair["log_vitd"], pair["log_crp"], stats.spearmanr),
        ("pearson_raw", pair["vitd_main"], pair["crp_mgL"], stats.pearsonr),
        ("pearson_vitd_logcrp", pair["vitd_main"], pair["log_crp"], stats.pearsonr),
        ("pearson_log_log", pair["log_vitd"], pair["log_crp"], stats.pearsonr),
    ]
    corr = pd.DataFrame(
        [{"method": name, "r": func(x, y)[0], "p_value": func(x, y)[1], "n": n} for name, x, y, func in specs]
    )

    slope, intercept, _, p_value, se = stats.linregress(pair["log_vitd"], pair["log_crp"])
    result = pd.DataFrame(
        [
            {
                "n": n,
                "intercept": intercept,
                "loglog_slope": slope,
                "loglog_se": se,
                "loglog_ci_low": slope - 1.96 * se,
                "loglog_ci_high": slope + 1.96 * se,
                "loglog_p": p_value,
                "pearson_loglog": corr.loc[corr["method"] == "pearson_log_log", "r"].iloc[0],
                "spearman_raw": corr.loc[corr["method"] == "spearman_raw", "r"].iloc[0],
            }
        ]
    )
    corr.to_csv(table_path("q2_correlation_table.csv"), index=False)
    result.to_csv(table_path("q2_loglog_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.6, 6))
    ax.scatter(pair["vitd_main"], pair["crp_mgL"], s=4, alpha=0.10, color=WCM_RED, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = np.linspace(pair["log_vitd"].min(), pair["log_vitd"].max(), 200)
    resid = pair["log_crp"] - (intercept + slope * pair["log_vitd"])
    s_err = np.sqrt((resid**2).sum() / (n - 2))
    xm = pair["log_vitd"].mean()
    sxx = ((pair["log_vitd"] - xm) ** 2).sum()
    ci = 1.96 * s_err * np.sqrt(1 / n + (xs - xm) ** 2 / sxx)
    ax.plot(np.exp(xs), np.exp(intercept + slope * xs), color=WCM_NAVY, lw=2.2, label="Log-log linear fit")
    ax.fill_between(
        np.exp(xs),
        np.exp(intercept + slope * xs - ci),
        np.exp(intercept + slope * xs + ci),
        color=WCM_NAVY,
        alpha=0.22,
        label="95% CI",
    )
    ax.set_xlabel("Baseline serum 25(OH)D, ng/mL (log scale)", fontsize=12)
    ax.set_ylabel("C-reactive protein, mg/L (log scale)", fontsize=12)
    ax.set_title(
        f"Log-log association between vitamin D and CRP (n = {n:,})\n"
        f"Pearson r = {result['pearson_loglog'].iloc[0]:.3f}, "
        f"Spearman rho = {result['spearman_raw'].iloc[0]:.3f}, "
        f"slope = {slope:.3f}",
        fontsize=11.5,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(figure_path("q2_loglog_scatter.png"), dpi=170, bbox_inches="tight")
    plt.show()

    print_section("Q2 log-log correlation")
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Saved q2_correlation_table.csv, q2_loglog_results.csv, q2_loglog_scatter.png")
    return result


def run_q2_multivariable(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    d = pair.merge(base[["person_id", "index_date", "age_at_index", "female", "bmi_full"]], on="person_id", how="inner")
    d["crp_date"] = d["index_date"] - pd.to_timedelta(d["days_before"], unit="D")
    month = d["crp_date"].dt.month
    d["winter"] = ((month >= 11) | (month <= 4)).astype(int)
    d = d[(d["bmi_full"] >= 12) & (d["bmi_full"] <= 70)].copy()
    model_df = d[["log_crp", "log_vitd", "age_at_index", "female", "winter", "bmi_full"]].dropna()

    formulas = [
        ("M0 unadjusted", "log_crp ~ log_vitd"),
        ("M1 age + sex", "log_crp ~ log_vitd + age_at_index + female"),
        ("M2 age + sex + winter", "log_crp ~ log_vitd + age_at_index + female + winter"),
        ("M3 age + sex + BMI", "log_crp ~ log_vitd + age_at_index + female + bmi_full"),
        ("M4 age + sex + winter + BMI", "log_crp ~ log_vitd + age_at_index + female + winter + bmi_full"),
    ]
    rows = []
    for label, formula in formulas:
        model = smf.ols(formula, data=model_df).fit()
        ci = model.conf_int().loc["log_vitd"]
        rows.append(
            {
                "model": label,
                "n": int(model.nobs),
                "slope": model.params["log_vitd"],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
                "p_value": model.pvalues["log_vitd"],
                "r2": model.rsquared,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(table_path("q2_multivariable_loglog.csv"), index=False)
    print_section("Q2 multivariable log-log OLS")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Saved q2_multivariable_loglog.csv")
    return table


def run_q2_quartile_plot(pair: pd.DataFrame) -> pd.DataFrame:
    df = pair.copy()
    df["vitd_quartile_num"] = pd.qcut(df["vitd_main"], 4, labels=False, duplicates="drop")
    quartiles = sorted(df["vitd_quartile_num"].dropna().astype(int).unique())

    rows = []
    data = []
    tick_labels = []
    display_labels = []
    for quartile in quartiles:
        sub = df[df["vitd_quartile_num"] == quartile]
        crp_values = sub["crp_mgL"].dropna().values
        vitd_values = sub["vitd_main"].dropna()
        if len(crp_values) == 0 or vitd_values.empty:
            continue
        label = f"Q{quartile + 1}"
        if quartile == min(quartiles):
            display_label = "Q1 lowest"
        elif quartile == max(quartiles):
            display_label = f"Q{quartile + 1} highest"
        else:
            display_label = label
        data.append(crp_values)
        display_labels.append(display_label)
        tick_labels.append(f"{label}\n{vitd_values.min():.0f}-{vitd_values.max():.0f} ng/mL\n(n={len(sub):,})")
        rows.append(
            {
                "vitd_quartile": display_label,
                "n": len(sub),
                "vitd_min": vitd_values.min(),
                "vitd_max": vitd_values.max(),
                "crp_median": np.median(crp_values),
                "crp_q1": np.percentile(crp_values, 25),
                "crp_q3": np.percentile(crp_values, 75),
            }
        )
    if len(data) < 2:
        raise ValueError("Need at least two vitamin D quartiles with CRP values for Q2 quartile plot")
    kw = stats.kruskal(*data)
    summary = pd.DataFrame(rows)
    summary["kruskal_p"] = kw.pvalue
    summary.to_csv(table_path("q2_crp_by_vitd_quartile_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        bp = ax.boxplot(data, tick_labels=tick_labels, showfliers=False, patch_artist=True, widths=0.55)
    except TypeError:
        bp = ax.boxplot(data, labels=tick_labels, showfliers=False, patch_artist=True, widths=0.55)
    for box in bp["boxes"]:
        box.set_facecolor("#f3d6d6")
        box.set_edgecolor(WCM_NAVY)
        box.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_color(WCM_RED)
        median.set_linewidth(2.4)
    for element in bp["whiskers"] + bp["caps"]:
        element.set_color(WCM_NAVY)
    ax.set_yscale("log")
    ax.set_xlabel("Baseline serum 25(OH)D quartile", fontsize=12)
    ax.set_ylabel("C-reactive protein, mg/L (log scale)", fontsize=12)
    ax.set_title(f"CRP by vitamin D quartile (n = {len(pair):,})\nKruskal-Wallis p = {kw.pvalue:.1e}", fontsize=11.5)
    for idx, values in enumerate(data, 1):
        ax.text(idx, np.median(values) * 1.28, f"{np.median(values):.2f}", ha="center", fontsize=11, color=WCM_RED, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(figure_path("q2_crp_by_vitd_quartile.png"), dpi=170, bbox_inches="tight")
    plt.savefig(figure_path("q2_crp_by_vitd_quartile_labeled.png"), dpi=170, bbox_inches="tight")
    plt.show()

    print_section("Q2 CRP by vitamin D quartile")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Saved q2_crp_by_vitd_quartile_summary.csv, q2_crp_by_vitd_quartile.png, and q2_crp_by_vitd_quartile_labeled.png")
    return summary

# %% Cell 21
from pathlib import Path
import numpy as np
import pandas as pd

_needed = [
    "client",
    "bigquery",
    "CDR",
    "table_path",
    "as_person_id",
    "normalize_index_date",
    "parse_datetime",
    "print_section",
]
_missing = [name for name in _needed if name not in globals()]
if _missing:
    raise RuntimeError("Run Setup and Shared Helper Functions before this cell: " + ", ".join(_missing))

SMOKING_OBS_CONCEPTS = (4282779, 43054909)
ALCOHOL_OBS_CONCEPTS = (45766930, 45772705)

INCOME_CONCEPT_ID = 46235933
EDUCATION_CONCEPT_ID = 40771091
INSURANCE_COVERAGE_CONCEPT_ID = 40766240
HEALTHCARE_ACCESS_BARRIER_CONCEPTS = (
    43530268,  # Delayed Medical Care: Rural Area
    43528664,  # Can't Afford Care: Healthcare Provider
    43530415,  # Can't Afford Care: Delayed Filling Rx To Save Money
    43530594,  # Delayed Medical Care: Nervous
    43529904,  # Delayed Medical Care: Elderly Care
    43529906,  # Delayed Medical Care: Transportation
    43530583,  # Delayed Medical Care: Can't Afford Co-pay
    43529903,  # Delayed Medical Care: Child Care
    43529905,  # Delayed Medical Care: Time Off Work
    43530584,  # Delayed Medical Care: Had To Pay Out Of Pocket
    43530585,  # Delayed Medical Care: Deductible Too High
)

CREATININE_SERUM_PLASMA_CONCEPT = 3016723
AST_CONCEPT_ID = 3013721
BILIRUBIN_TOTAL_CONCEPT_ID = 3024128
ALT_CONCEPT_ID = 3006923
LIVER_LAB_CONCEPTS = {
    "ast": AST_CONCEPT_ID,
    "alt": ALT_CONCEPT_ID,
    "bilirubin_total": BILIRUBIN_TOTAL_CONCEPT_ID,
}


def q2_id_list(pair: pd.DataFrame) -> list[int]:
    values = pd.to_numeric(pair["person_id"], errors="coerce").dropna().astype("int64")
    return [int(x) for x in values.unique().tolist()]


def query_ids(sql: str, ids: list[int]) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", ids)])
    df = client.query(sql, job_config=job_config).to_dataframe()
    if "person_id" in df.columns:
        df = as_person_id(df)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def q2_dates(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    keep = [col for col in ["person_id", "index_date", "age_at_index", "gender", "female"] if col in base.columns]
    dates = pair[["person_id", "days_before", "crp_mgL"]].merge(base[keep], on="person_id", how="left")
    dates = normalize_index_date(dates)
    dates["days_before"] = pd.to_numeric(dates["days_before"], errors="coerce")
    dates["crp_date"] = dates["index_date"] - pd.to_timedelta(dates["days_before"], unit="D")
    dates["target_date"] = dates["crp_date"].combine_first(dates["index_date"])
    return dates


def write_empty(filename: str, columns: list[str], reason: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=columns)
    out.to_csv(table_path(filename), index=False)
    print(f"{filename}: no rows written ({reason})")
    return out


def nearest_to_target(records: pd.DataFrame, dates: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if records.empty:
        return records
    d = records.merge(dates[["person_id", "target_date"]], on="person_id", how="left")
    d[dt_col] = parse_datetime(d[dt_col])
    d["gap_days"] = (d[dt_col] - d["target_date"]).abs().dt.days
    return d.sort_values(["person_id", "gap_days"]).groupby("person_id", as_index=False).first()


def summarize_observation_concepts(ids: list[int], concept_ids: list[int], output_file: str) -> pd.DataFrame:
    if not ids or not concept_ids:
        return write_empty(output_file, ["concept_id", "question", "n_rows", "n_people", "n_answer_concepts"], "empty ids/concepts")
    concept_list = ",".join(map(str, concept_ids))
    sql = f"""
    SELECT o.observation_concept_id AS concept_id,
           q.concept_name AS question,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT o.person_id) AS n_people,
           COUNT(DISTINCT o.value_as_concept_id) AS n_answer_concepts
    FROM `{CDR}.observation` o
    JOIN `{CDR}.concept` q ON o.observation_concept_id = q.concept_id
    WHERE o.person_id IN UNNEST(@ids)
      AND o.observation_concept_id IN ({concept_list})
      AND o.value_as_concept_id IS NOT NULL
    GROUP BY 1, 2
    ORDER BY n_people DESC, n_rows DESC
    """
    out = query_ids(sql, ids)
    out.to_csv(table_path(output_file), index=False)
    return out


def fetch_observation_answers(ids: list[int], concept_ids: list[int]) -> pd.DataFrame:
    if not ids or not concept_ids:
        return pd.DataFrame()
    concept_list = ",".join(map(str, concept_ids))
    sql = f"""
    SELECT o.person_id,
           COALESCE(CAST(o.observation_datetime AS TIMESTAMP), TIMESTAMP(o.observation_date)) AS observation_datetime,
           o.observation_concept_id,
           q.concept_name AS question,
           o.value_as_concept_id,
           v.concept_name AS answer
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` q ON o.observation_concept_id = q.concept_id
    LEFT JOIN `{CDR}.concept` v ON o.value_as_concept_id = v.concept_id
    WHERE o.person_id IN UNNEST(@ids)
      AND o.observation_concept_id IN ({concept_list})
      AND o.value_as_concept_id IS NOT NULL
    """
    return query_ids(sql, ids)


def build_q2_smoking_status(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    if not ids:
        return write_empty("q2_smoking_status.csv", ["person_id", "smk_status"], "empty Q2 pair")
    concepts = ",".join(map(str, SMOKING_OBS_CONCEPTS))
    sql = f"""
    SELECT o.person_id,
           COALESCE(CAST(o.observation_datetime AS TIMESTAMP), TIMESTAMP(o.observation_date)) AS observation_datetime,
           o.observation_concept_id,
           q.concept_name AS smoking_question,
           o.value_as_concept_id,
           v.concept_name AS smoking_answer,
           CASE
             WHEN o.value_as_concept_id IN (4144272, 45879404, 45883458, 4222303, 903653, 903651) THEN 'never'
             WHEN o.value_as_concept_id IN (4310250) THEN 'former'
             WHEN o.value_as_concept_id IN (4276526, 42709996, 45881517, 45884037, 37395605, 45881518,
                                            903657, 45878118, 762498, 45884038, 762499, 903659,
                                            4298794, 4034855, 4219336, 903654, 4188539) THEN 'current'
             ELSE NULL
           END AS smk_status
    FROM `{CDR}.observation` o
    LEFT JOIN `{CDR}.concept` q ON o.observation_concept_id = q.concept_id
    LEFT JOIN `{CDR}.concept` v ON o.value_as_concept_id = v.concept_id
    WHERE o.person_id IN UNNEST(@ids)
      AND o.observation_concept_id IN ({concepts})
      AND o.value_as_concept_id IS NOT NULL
    """
    raw = query_ids(sql, ids)
    raw.to_csv(table_path("q2_smoking_status_long.csv"), index=False)
    raw = raw.dropna(subset=["smk_status"])
    if raw.empty:
        return write_empty("q2_smoking_status.csv", ["person_id", "smk_status", "smoking_answer"], "no mapped smoking status")
    out = nearest_to_target(raw, q2_dates(base, pair), "observation_datetime")
    cols = ["person_id", "smk_status", "smoking_answer", "smoking_question", "observation_datetime", "observation_concept_id", "value_as_concept_id"]
    out[cols].to_csv(table_path("q2_smoking_status.csv"), index=False)
    print(f"q2_smoking_status.csv: {len(out):,} participants")
    return out[cols]


def map_alcohol_row(question: object, answer: object) -> str | None:
    qtext = str(question).lower()
    atext = str(answer).lower()
    if "denies alcohol use" in qtext:
        return "none_or_never"
    if "admits alcohol use" in qtext:
        return "current"
    if any(k in atext for k in ["never", "no alcohol", "not at all", "none of the time"]):
        return "none_or_never"
    if any(k in atext for k in ["former", "used to", "previous", "quit"]):
        return "former"
    if any(k in atext for k in ["daily", "weekly", "monthly", "some days", "every day", "yes"]):
        return "current"
    return None


def build_q2_alcohol_use(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    if not ids:
        return write_empty("q2_alcohol_use.csv", ["person_id", "alcohol_use"], "empty Q2 pair")
    summarize_observation_concepts(ids, list(ALCOHOL_OBS_CONCEPTS), "q2_alcohol_use_concept_candidates.csv")
    raw = fetch_observation_answers(ids, list(ALCOHOL_OBS_CONCEPTS))
    if raw.empty:
        return write_empty("q2_alcohol_use.csv", ["person_id", "alcohol_use", "alcohol_answer"], "no alcohol records")
    raw = raw.rename(columns={"question": "alcohol_question", "answer": "alcohol_answer"})
    raw["alcohol_use"] = raw.apply(lambda row: map_alcohol_row(row["alcohol_question"], row["alcohol_answer"]), axis=1)
    raw.to_csv(table_path("q2_alcohol_use_long.csv"), index=False)
    raw = raw.dropna(subset=["alcohol_use"])
    if raw.empty:
        return write_empty("q2_alcohol_use.csv", ["person_id", "alcohol_use", "alcohol_answer"], "no mapped alcohol status")
    out = nearest_to_target(raw, q2_dates(base, pair), "observation_datetime")
    cols = ["person_id", "alcohol_use", "alcohol_answer", "alcohol_question", "observation_datetime", "observation_concept_id", "value_as_concept_id"]
    out[cols].to_csv(table_path("q2_alcohol_use.csv"), index=False)
    print(f"q2_alcohol_use.csv: {len(out):,} participants")
    return out[cols]


def map_income(answer: object) -> str | None:
    text = str(answer).lower()
    if any(k in text for k in ["less 10", "less than 10", "10k 25", "25k 35"]):
        return "Income <35k"
    if any(k in text for k in ["35k 50", "50k 75", "75k 100"]):
        return "Income 35-100k"
    if any(k in text for k in ["100k 150", "150k 200", "more 200", "200k"]):
        return "Income >=100k"
    return None


def map_education(answer: object) -> str | None:
    text = str(answer).lower()
    if any(k in text for k in ["never attended", "kindergarten", "elementary", "junior high", "nine through eleven", "9th", "10th", "11th"]):
        return "Less than high school"
    if "high school" in text or "ged" in text:
        return "High school/GED"
    if "vocational" in text or "technical" in text:
        return "Some college/vocational"
    if "university" in text or "higher education" in text or "college education" in text:
        return "College or higher"
    return None


def map_insurance_coverage(answer: object) -> str | None:
    text = str(answer).lower()
    if any(k in text for k in ["pmi: skip", "prefer not", "don't know", "dont know"]):
        return None
    if any(k in text for k in ["i don't have health insurance", "do not have health insurance", "self-pay", "uninsured"]):
        return "Uninsured"
    if text.strip() == "no":
        return "Uninsured"
    insured_terms = [
        "yes",
        "medicare",
        "medicaid",
        "employer",
        "union",
        "insurance purchased",
        "purchased directly",
        "tricare",
        "military",
        "veterans affairs",
        " va ",
        "government-assistance",
        "other type of health insurance",
        "health coverage plan",
        "indian health service",
    ]
    if any(k in f" {text} " for k in insured_terms):
        return "Insured"
    return None


def map_barrier_answer(answer: object) -> int | None:
    text = str(answer).lower()
    if any(k in text for k in ["pmi: skip", "prefer not", "don't know", "dont know"]):
        return None
    if text.strip() == "yes":
        return 1
    if text.strip() == "no":
        return 0
    return None


def build_fixed_observation_covariate(
    base: pd.DataFrame,
    pair: pd.DataFrame,
    output_file: str,
    target_col: str,
    concept_ids: list[int],
    mapper,
    summary_file: str,
) -> pd.DataFrame:
    ids = q2_id_list(pair)
    if not ids:
        return write_empty(output_file, ["person_id", target_col], "empty Q2 pair")
    summarize_observation_concepts(ids, concept_ids, summary_file)
    raw = fetch_observation_answers(ids, concept_ids)
    if raw.empty:
        return write_empty(output_file, ["person_id", target_col, f"{target_col}_answer"], "no records for fixed concept")
    raw[target_col] = raw["answer"].apply(mapper)
    stem = output_file.replace(".csv", "")
    raw.to_csv(table_path(f"{stem}_long.csv"), index=False)
    raw = raw.dropna(subset=[target_col])
    if raw.empty:
        return write_empty(output_file, ["person_id", target_col, f"{target_col}_answer"], "answers did not map to usable categories")
    out = nearest_to_target(raw, q2_dates(base, pair), "observation_datetime")
    out = out[["person_id", target_col, "answer", "question", "observation_datetime", "observation_concept_id", "value_as_concept_id"]]
    out = out.rename(columns={"answer": f"{target_col}_answer", "question": f"{target_col}_question"})
    out.to_csv(table_path(output_file), index=False)
    print(f"{output_file}: {len(out):,} participants")
    return out


def build_q2_healthcare_access(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    if not ids:
        return write_empty("q2_healthcare_access.csv", ["person_id", "healthcare_access_barrier"], "empty Q2 pair")
    concept_ids = list(HEALTHCARE_ACCESS_BARRIER_CONCEPTS)
    summarize_observation_concepts(ids, concept_ids, "q2_healthcare_access_concept_candidates.csv")
    raw = fetch_observation_answers(ids, concept_ids)
    if raw.empty:
        return write_empty("q2_healthcare_access.csv", ["person_id", "healthcare_access_barrier"], "no access-barrier records")
    raw["healthcare_access_barrier"] = raw["answer"].apply(map_barrier_answer)
    raw.to_csv(table_path("q2_healthcare_access_long.csv"), index=False)
    raw = raw.dropna(subset=["healthcare_access_barrier"])
    if raw.empty:
        return write_empty("q2_healthcare_access.csv", ["person_id", "healthcare_access_barrier"], "no mapped access-barrier answers")
    out = raw.groupby("person_id", as_index=False)["healthcare_access_barrier"].max()
    out.to_csv(table_path("q2_healthcare_access.csv"), index=False)
    print(f"q2_healthcare_access.csv: {len(out):,} participants")
    return out


def build_q2_ses_covariates(base: pd.DataFrame, pair: pd.DataFrame) -> dict[str, pd.DataFrame]:
    income = build_fixed_observation_covariate(
        base, pair, "q2_income_band.csv", "income_band", [INCOME_CONCEPT_ID], map_income, "q2_income_band_concept_candidates.csv"
    )
    education = build_fixed_observation_covariate(
        base, pair, "q2_education.csv", "education_level", [EDUCATION_CONCEPT_ID], map_education, "q2_education_concept_candidates.csv"
    )
    insurance = build_fixed_observation_covariate(
        base, pair, "q2_insurance.csv", "insurance_status", [INSURANCE_COVERAGE_CONCEPT_ID], map_insurance_coverage, "q2_insurance_concept_candidates.csv"
    )
    access = build_q2_healthcare_access(base, pair)
    return {"income": income, "education": education, "insurance": insurance, "healthcare_access": access}


def ckd_epi_2021(scr: float, age: float, female: bool) -> float:
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = scr / kappa
    return 142 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.200) * (0.9938 ** age) * (1.012 if female else 1.0)


def build_q2_egfr_ckd(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    if not ids:
        return write_empty("q2_egfr_baseline.csv", ["person_id", "egfr", "baseline_ckd"], "empty Q2 pair")
    sql = f"""
    SELECT person_id, measurement_datetime AS dt, value_as_number AS creatinine
    FROM `{CDR}.measurement`
    WHERE person_id IN UNNEST(@ids)
      AND measurement_concept_id = {CREATININE_SERUM_PLASMA_CONCEPT}
      AND value_as_number BETWEEN 0.2 AND 20
    """
    cr = query_ids(sql, ids)
    if cr.empty:
        return write_empty("q2_egfr_baseline.csv", ["person_id", "egfr", "baseline_ckd"], "no creatinine records")
    dates = q2_dates(base, pair)
    cr = cr.merge(dates[["person_id", "index_date"]], on="person_id", how="left")
    cr["dt"] = parse_datetime(cr["dt"])
    cr["days_before_index"] = (cr["index_date"] - cr["dt"]).dt.days
    cr = cr[cr["days_before_index"] >= 0].copy()
    if cr.empty:
        return write_empty("q2_egfr_baseline.csv", ["person_id", "egfr", "baseline_ckd"], "no pre-index creatinine records")
    nearest = cr.sort_values(["person_id", "days_before_index"]).groupby("person_id", as_index=False).first()
    covars = base[[col for col in ["person_id", "age_at_index", "gender", "female"] if col in base.columns]].copy()
    nearest = nearest.merge(covars, on="person_id", how="left")
    if "female" not in nearest.columns and "gender" in nearest.columns:
        nearest["female"] = (nearest["gender"] == "Female").astype(int)
    nearest = nearest.dropna(subset=["creatinine", "age_at_index", "female"])
    nearest["egfr"] = nearest.apply(lambda row: ckd_epi_2021(float(row["creatinine"]), float(row["age_at_index"]), bool(row["female"] == 1)), axis=1)
    nearest["baseline_ckd"] = (nearest["egfr"] < 60).astype(int)
    out = nearest[["person_id", "egfr", "baseline_ckd", "creatinine", "dt", "days_before_index"]]
    out.to_csv(table_path("q2_egfr_baseline.csv"), index=False)
    print(f"q2_egfr_baseline.csv: {len(out):,} participants")
    return out


VITD_SUPPLEMENT_LOOKBACK_DAYS = 120

VITD_SUPPLEMENT_TERMS = (
    "vitamin d",
    "vitamin d2",
    "vitamin d3",
    "cholecalciferol",
    "ergocalciferol",
    "calcifediol",
)

ACTIVE_VITD_ANALOG_TERMS = (
    "calcitriol",
    "paricalcitol",
    "doxercalciferol",
)

LOCAL_ROUTE_TERMS = (
    "topical",
    "cutaneous",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "patch",
    "ophthalmic",
    "otic",
    "nasal",
    "intranasal",
    "inhalation",
    "inhaler",
    "rectal",
    "vaginal",
)

ORAL_RELEVANT_TERMS = (
    "oral",
    "tablet",
    "capsule",
    "chewable",
    "solution",
    "drops",
    "drop",
    "liquid",
    "suspension",
)


def classify_vitd_drug_name(text: object) -> dict[str, int | str]:
    lower = str(text).lower()
    has_supplement_term = any(term in lower for term in VITD_SUPPLEMENT_TERMS)
    has_active_analog_term = any(term in lower for term in ACTIVE_VITD_ANALOG_TERMS)
    has_local_route = any(term in lower for term in LOCAL_ROUTE_TERMS)
    has_oral_relevant_route = any(term in lower for term in ORAL_RELEVANT_TERMS)
    route_class = "local_excluded" if has_local_route else ("oral_or_relevant" if has_oral_relevant_route else "unspecified_relevant")
    return {
        "vitd_supplement_name_match": int(has_supplement_term),
        "active_vitd_analog_name_match": int(has_active_analog_term),
        "local_route_excluded": int(has_local_route),
        "oral_or_relevant_route": int(not has_local_route and (has_oral_relevant_route or has_supplement_term)),
        "vitd_supplement_product": int(has_supplement_term and not has_local_route),
        "active_vitd_analog_product": int(has_active_analog_term and not has_local_route),
        "vitd_route_class": route_class,
    }


def add_drug_end_from_days_supply(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["drug_start"] = parse_datetime(df["drug_start"])
    df["drug_end"] = parse_datetime(df["drug_end"]) if "drug_end" in df.columns else pd.NaT
    df["days_supply"] = pd.to_numeric(df.get("days_supply"), errors="coerce")
    implied_end = df["drug_start"] + pd.to_timedelta(df["days_supply"].clip(lower=1) - 1, unit="D")
    df["drug_end_imputed_from_days_supply"] = df["drug_end"].isna() & df["days_supply"].notna() & df["drug_start"].notna()
    df["drug_end"] = df["drug_end"].combine_first(implied_end)
    return df


def build_q2_medication_use(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    cols = [
        "person_id",
        "vitd_supplement_use",
        "vitd_supplement_120d",
        "vitd_supplement_active_on_target",
        "vitd_supplement_history_pre_target",
        "active_vitd_analog_use_120d",
    ]
    if not ids:
        return write_empty("q2_medication_use.csv", cols, "empty Q2 pair")
    name_condition = " OR ".join([
        "LOWER(drug.concept_name) LIKE '%vitamin d%'", "LOWER(drug.concept_name) LIKE '%cholecalciferol%'",
        "LOWER(drug.concept_name) LIKE '%ergocalciferol%'", "LOWER(drug.concept_name) LIKE '%calcifediol%'",
        "LOWER(drug.concept_name) LIKE '%calcitriol%'", "LOWER(drug.concept_name) LIKE '%paricalcitol%'",
        "LOWER(drug.concept_name) LIKE '%doxercalciferol%'",
    ])
    sql = f"""
    SELECT de.person_id,
           COALESCE(CAST(de.drug_exposure_start_datetime AS TIMESTAMP), TIMESTAMP(de.drug_exposure_start_date)) AS drug_start,
           COALESCE(CAST(de.drug_exposure_end_datetime AS TIMESTAMP), TIMESTAMP(de.drug_exposure_end_date)) AS drug_end,
           de.days_supply,
           de.drug_concept_id,
           drug.concept_name AS drug_name
    FROM `{CDR}.drug_exposure` de
    LEFT JOIN `{CDR}.concept` drug ON de.drug_concept_id = drug.concept_id
    WHERE de.person_id IN UNNEST(@ids)
      AND ({name_condition})
    """
    raw = query_ids(sql, ids)
    if raw.empty:
        return write_empty("q2_medication_use.csv", cols, "no matching drug exposure records")
    dates = q2_dates(base, pair)
    raw = raw.merge(dates[["person_id", "target_date"]], on="person_id", how="left")
    raw = add_drug_end_from_days_supply(raw)
    raw = raw[(raw["drug_start"].isna()) | (raw["target_date"].isna()) | (raw["drug_start"] <= raw["target_date"])].copy()
    if raw.empty:
        return write_empty("q2_medication_use.csv", cols, "no pre-target vitamin D drug exposure records")
    flags = raw["drug_name"].fillna("").apply(classify_vitd_drug_name).apply(pd.Series)
    raw = pd.concat([raw.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    raw["days_before_target"] = (raw["target_date"] - raw["drug_start"]).dt.days
    raw["within_120d"] = raw["days_before_target"].between(0, VITD_SUPPLEMENT_LOOKBACK_DAYS, inclusive="both")
    raw["active_on_target"] = (
        raw["drug_start"].notna()
        & raw["target_date"].notna()
        & raw["drug_end"].notna()
        & (raw["drug_start"] <= raw["target_date"])
        & (raw["drug_end"] >= raw["target_date"])
    )
    raw["vitd_supplement_120d"] = (raw["vitd_supplement_product"].eq(1) & raw["within_120d"]).astype(int)
    raw["vitd_supplement_active_on_target"] = (raw["vitd_supplement_product"].eq(1) & raw["active_on_target"]).astype(int)
    raw["vitd_supplement_use"] = (
        raw["vitd_supplement_120d"].eq(1) | raw["vitd_supplement_active_on_target"].eq(1)
    ).astype(int)
    raw["vitd_supplement_history_pre_target"] = raw["vitd_supplement_product"].eq(1).astype(int)
    raw["active_vitd_analog_use_120d"] = (
        raw["active_vitd_analog_product"].eq(1) & (raw["within_120d"] | raw["active_on_target"])
    ).astype(int)
    drug_counts = (
        raw.groupby(["drug_concept_id", "drug_name", "vitd_route_class"], dropna=False)
        .agg(
            n_rows=("person_id", "size"),
            n_people=("person_id", "nunique"),
            vitd_supplement_use=("vitd_supplement_use", "max"),
            vitd_supplement_120d=("vitd_supplement_120d", "max"),
            vitd_supplement_active_on_target=("vitd_supplement_active_on_target", "max"),
            vitd_supplement_history_pre_target=("vitd_supplement_history_pre_target", "max"),
            active_vitd_analog_use_120d=("active_vitd_analog_use_120d", "max"),
            local_route_excluded=("local_route_excluded", "max"),
        )
        .reset_index()
        .sort_values(["n_people", "n_rows"], ascending=False)
    )
    drug_counts.to_csv(table_path("q2_medication_drug_name_counts.csv"), index=False)
    raw.head(5000).to_csv(table_path("q2_medication_use_long_sample.csv"), index=False)
    out = raw.groupby("person_id", as_index=False)[cols[1:]].max()
    out.to_csv(table_path("q2_medication_use.csv"), index=False)
    print(f"q2_medication_use.csv: {len(out):,} participants")
    print(f"vitd_supplement_use uses oral/relevant products within {VITD_SUPPLEMENT_LOOKBACK_DAYS} days before target_date or active on target_date")
    print("Saved q2_medication_drug_name_counts.csv and q2_medication_use_long_sample.csv")
    return out


def build_q2_acute_proxy(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    dates = q2_dates(base, pair)
    out = dates[["person_id", "crp_mgL", "target_date"]].copy()
    out["crp_gt10"] = (pd.to_numeric(out["crp_mgL"], errors="coerce") > 10).astype("Int64")
    out["infection_30d"] = 0
    out["hospitalization_30d"] = 0
    out["ed_visit_30d"] = 0
    if ids:
        infection_condition = " OR ".join([
            "LOWER(c.concept_name) LIKE '%sepsis%'",
            "LOWER(c.concept_name) LIKE '%pneumonia%'",
            "LOWER(c.concept_name) LIKE '%influenza%'",
            "LOWER(c.concept_name) LIKE '%covid%'",
            "LOWER(c.concept_name) LIKE '%cellulitis%'",
            "LOWER(c.concept_name) LIKE '%acute upper respiratory infection%'",
        ])
        infection_sql = f"""
        SELECT co.person_id,
               COALESCE(CAST(co.condition_start_datetime AS TIMESTAMP), TIMESTAMP(co.condition_start_date)) AS condition_start,
               co.condition_concept_id,
               c.concept_name AS condition_name
        FROM `{CDR}.condition_occurrence` co
        LEFT JOIN `{CDR}.concept` c ON co.condition_concept_id = c.concept_id
        WHERE co.person_id IN UNNEST(@ids)
          AND ({infection_condition})
        """
        infections = query_ids(infection_sql, ids)
        if not infections.empty:
            infections = infections.merge(dates[["person_id", "target_date"]], on="person_id", how="left")
            infections["condition_start"] = parse_datetime(infections["condition_start"])
            infections["within_30d"] = (infections["condition_start"] - infections["target_date"]).abs().dt.days <= 30
            true_ids = set(infections.loc[infections["within_30d"], "person_id"].dropna().astype(int))
            out["infection_30d"] = out["person_id"].astype(int).isin(true_ids).astype(int)
            infections.to_csv(table_path("q2_infection_proxy_long.csv"), index=False)
        visit_sql = f"""
        SELECT vo.person_id,
               COALESCE(CAST(vo.visit_start_datetime AS TIMESTAMP), TIMESTAMP(vo.visit_start_date)) AS visit_start,
               vo.visit_concept_id,
               c.concept_name AS visit_name
        FROM `{CDR}.visit_occurrence` vo
        LEFT JOIN `{CDR}.concept` c ON vo.visit_concept_id = c.concept_id
        WHERE vo.person_id IN UNNEST(@ids)
          AND vo.visit_concept_id IN (9201, 9203)
        """
        visits = query_ids(visit_sql, ids)
        if not visits.empty:
            visits = visits.merge(dates[["person_id", "target_date"]], on="person_id", how="left")
            visits["visit_start"] = parse_datetime(visits["visit_start"])
            visits["within_30d"] = (visits["visit_start"] - visits["target_date"]).abs().dt.days <= 30
            hosp_ids = set(visits.loc[(visits["within_30d"]) & (visits["visit_concept_id"] == 9201), "person_id"].dropna().astype(int))
            ed_ids = set(visits.loc[(visits["within_30d"]) & (visits["visit_concept_id"] == 9203), "person_id"].dropna().astype(int))
            out["hospitalization_30d"] = out["person_id"].astype(int).isin(hosp_ids).astype(int)
            out["ed_visit_30d"] = out["person_id"].astype(int).isin(ed_ids).astype(int)
            visits.to_csv(table_path("q2_hospitalization_proxy_long.csv"), index=False)
    final_cols = ["person_id", "crp_gt10", "infection_30d", "hospitalization_30d", "ed_visit_30d"]
    out[final_cols].to_csv(table_path("q2_acute_inflammation_proxy.csv"), index=False)
    print(f"q2_acute_inflammation_proxy.csv: {len(out):,} participants")
    return out[final_cols]


def build_q2_liver_covariates(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    ids = q2_id_list(pair)
    dates = q2_dates(base, pair)
    out = dates[["person_id", "target_date"]].copy()
    out["liver_disease"] = 0
    out["alt"] = np.nan
    out["ast"] = np.nan
    out["bilirubin_total"] = np.nan
    if not ids:
        out.to_csv(table_path("q2_liver_function_disease.csv"), index=False)
        return out
    liver_condition = " OR ".join([
        "LOWER(c.concept_name) LIKE '%liver disease%'",
        "LOWER(c.concept_name) LIKE '%hepatic%'",
        "LOWER(c.concept_name) LIKE '%cirrhosis%'",
        "LOWER(c.concept_name) LIKE '%hepatitis%'",
        "LOWER(c.concept_name) LIKE '%fatty liver%'",
    ])
    condition_sql = f"""
    SELECT co.person_id,
           COALESCE(CAST(co.condition_start_datetime AS TIMESTAMP), TIMESTAMP(co.condition_start_date)) AS condition_start,
           co.condition_concept_id,
           c.concept_name AS condition_name
    FROM `{CDR}.condition_occurrence` co
    LEFT JOIN `{CDR}.concept` c ON co.condition_concept_id = c.concept_id
    WHERE co.person_id IN UNNEST(@ids)
      AND ({liver_condition})
    """
    liver = query_ids(condition_sql, ids)
    if not liver.empty:
        liver = liver.merge(dates[["person_id", "target_date"]], on="person_id", how="left")
        liver["condition_start"] = parse_datetime(liver["condition_start"])
        liver = liver[(liver["condition_start"].isna()) | (liver["target_date"].isna()) | (liver["condition_start"] <= liver["target_date"])]
        out["liver_disease"] = out["person_id"].astype(int).isin(set(liver["person_id"].dropna().astype(int))).astype(int)
        liver.to_csv(table_path("q2_liver_disease_long.csv"), index=False)
    lab_concept_list = ",".join(map(str, LIVER_LAB_CONCEPTS.values()))
    lab_summary_sql = f"""
    SELECT m.measurement_concept_id AS concept_id,
           c.concept_name AS measurement_name,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT m.person_id) AS n_people
    FROM `{CDR}.measurement` m
    JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE m.person_id IN UNNEST(@ids)
      AND m.measurement_concept_id IN ({lab_concept_list})
      AND m.value_as_number IS NOT NULL
    GROUP BY 1, 2
    ORDER BY n_people DESC, n_rows DESC
    """
    lab_summary = query_ids(lab_summary_sql, ids)
    lab_summary.to_csv(table_path("q2_liver_lab_concept_candidates.csv"), index=False)
    for analyte, concept_id in LIVER_LAB_CONCEPTS.items():
        value_sql = f"""
        SELECT person_id, measurement_datetime AS lab_dt, value_as_number AS value
        FROM `{CDR}.measurement`
        WHERE person_id IN UNNEST(@ids)
          AND measurement_concept_id = {concept_id}
          AND value_as_number IS NOT NULL
        """
        lab = query_ids(value_sql, ids)
        if lab.empty:
            continue
        lab = lab.merge(dates[["person_id", "target_date"]], on="person_id", how="left")
        lab["lab_dt"] = parse_datetime(lab["lab_dt"])
        lab = lab[(lab["lab_dt"].isna()) | (lab["target_date"].isna()) | (lab["lab_dt"] <= lab["target_date"])]
        if lab.empty:
            continue
        lab["gap_days"] = (lab["target_date"] - lab["lab_dt"]).dt.days
        nearest = lab.sort_values(["person_id", "gap_days"]).groupby("person_id", as_index=False).first()
        out = out.drop(columns=[analyte]).merge(nearest[["person_id", "value"]].rename(columns={"value": analyte}), on="person_id", how="left")
    out.to_csv(table_path("q2_liver_function_disease.csv"), index=False)
    print(f"q2_liver_function_disease.csv: {len(out):,} participants")
    return out


def run_q2_candidate_covariate_construction(base: pd.DataFrame, pair: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    print_section("Q2 candidate covariate construction")
    builders = [
        ("smoking", ["q2_smoking_status.csv"], build_q2_smoking_status),
        ("alcohol", ["q2_alcohol_use.csv"], build_q2_alcohol_use),
        ("ses", ["q2_income_band.csv", "q2_education.csv", "q2_insurance.csv", "q2_healthcare_access.csv"], build_q2_ses_covariates),
        ("egfr_ckd", ["q2_egfr_baseline.csv"], build_q2_egfr_ckd),
        ("medication_supplement", ["q2_medication_use.csv", "q2_medication_drug_name_counts.csv"], build_q2_medication_use),
        ("acute_inflammation_proxy", ["q2_acute_inflammation_proxy.csv"], build_q2_acute_proxy),
        ("liver", ["q2_liver_function_disease.csv"], build_q2_liver_covariates),
    ]
    rows = []
    for name, output_files, builder in builders:
        out_paths = [table_path(filename) for filename in output_files]
        primary_output = "; ".join(str(path) for path in out_paths)
        if all(path.exists() for path in out_paths) and not rebuild:
            print(f"{name}: cached")
            rows.append({"construction_step": name, "status": "cached", "primary_output": primary_output, "error": ""})
            continue
        try:
            builder(base, pair)
            rows.append({"construction_step": name, "status": "built", "primary_output": primary_output, "error": ""})
        except Exception as exc:
            print(f"{name} construction failed: {exc}")
            rows.append({"construction_step": name, "status": "failed", "primary_output": primary_output, "error": str(exc)})
    status = pd.DataFrame(rows)
    status.to_csv(table_path("q2_candidate_covariate_construction_status.csv"), index=False)
    print("Saved q2_candidate_covariate_construction_status.csv")
    return status

# %% Cell 23
from pathlib import Path
import numpy as np
import pandas as pd

_needed = ["OUTPUT_DIR", "table_path", "as_person_id", "normalize_index_date", "print_section"]
_missing = [name for name in _needed if name not in globals()]
if _missing:
    raise RuntimeError("Run Setup and Shared Helper Functions before this cell: " + ", ".join(_missing))


def candidate_file_paths(filenames: list[str]) -> list[Path]:
    paths = []
    for filename in filenames:
        for path in [Path(filename), table_path(filename), OUTPUT_DIR / filename]:
            if path not in paths:
                paths.append(path)
    return paths


def read_optional_person_table(filenames: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    for path in candidate_file_paths(filenames):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Optional covariate table skipped ({path}): {exc}")
            continue
        if "person_id" not in df.columns:
            print(f"Optional covariate table skipped ({path}): missing person_id")
            continue
        return as_person_id(df), str(path)
    return None, None


def merge_optional_covariate_table(
    df: pd.DataFrame,
    filenames: list[str],
    column_map: dict[str, str],
    source_key: str,
    sources: dict[str, str],
) -> pd.DataFrame:
    table, source = read_optional_person_table(filenames)
    if table is None:
        return df
    rename = {old: new for old, new in column_map.items() if old in table.columns}
    if not rename:
        print(f"Optional covariate table skipped ({source}): none of {list(column_map)} found")
        return df
    table = table.rename(columns=rename)
    keep = ["person_id"] + sorted(set(rename.values()))
    table = table[keep].drop_duplicates("person_id")
    duplicate_cols = [col for col in keep if col != "person_id" and col in df.columns]
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)
    sources[source_key] = source or ""
    return df.merge(table, on="person_id", how="left")


def availability_status(source_key: str, sources: dict[str, str], base_status: str) -> tuple[str, str]:
    if source_key == "base":
        return base_status, "analysis base"
    if source_key == "derived":
        return base_status, "derived from Q2 pair/base dates"
    if source_key == "parked":
        return base_status, "parked by current Q2 scope"
    if source_key in sources:
        return "constructed/reusable table found", sources[source_key]
    return "construction output not found", "no readable person-level table found"


def coverage_row(
    df: pd.DataFrame,
    covariate: str,
    fields: list[str],
    source_key: str,
    sources: dict[str, str],
    q2_role: str,
    mode: str = "all",
    base_status: str = "available in analysis base",
    note: str = "",
) -> dict[str, object]:
    present = [field for field in fields if field in df.columns]
    missing = [field for field in fields if field not in df.columns]
    if not present:
        mask = pd.Series(False, index=df.index)
    elif mode == "any":
        mask = df[present].notna().any(axis=1)
    else:
        mask = df[present].notna().all(axis=1)
    status, source = availability_status(source_key, sources, base_status)
    n_available = int(mask.sum())
    denominator = len(df)
    return {
        "covariate": covariate,
        "fields_checked": ", ".join(fields),
        "present_fields": ", ".join(present),
        "missing_fields": ", ".join(missing),
        "availability_status": status,
        "q2_role_now": q2_role,
        "source_or_construction": source,
        "n_available_in_q2_pair": n_available,
        "q2_pair_n": denominator,
        "coverage_pct": 100 * n_available / denominator if denominator else np.nan,
        "note": note,
    }


def build_q2_covariate_level_counts(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        if field not in df.columns:
            continue
        counts = df[field].astype("object").where(df[field].notna(), "<missing>").value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"field": field, "level": str(value), "n": int(count), "pct_of_q2_pair": 100 * int(count) / len(df) if len(df) else np.nan})
    return pd.DataFrame(rows)


def build_q2_covariate_continuous_summary(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        if field not in df.columns:
            continue
        values = pd.to_numeric(df[field], errors="coerce")
        rows.append({
            "field": field,
            "n_nonmissing": int(values.notna().sum()),
            "mean": values.mean(),
            "sd": values.std(),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "min": values.min(),
            "max": values.max(),
        })
    return pd.DataFrame(rows)


def run_q2_covariate_availability_audit(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "person_id", "index_date", "age_at_index", "female", "gender", "bmi_full",
        "race_eth_simplified", "race_black", "race_hispanic", "race_asian", "race_other",
        "baseline_diabetes", "baseline_hypertension", "baseline_hyperlipidemia",
    ]
    audit = pair.merge(base[[col for col in base_cols if col in base.columns]], on="person_id", how="left")
    audit = normalize_index_date(audit)
    audit["days_before"] = pd.to_numeric(audit.get("days_before"), errors="coerce")
    audit["crp_date"] = audit["index_date"] - pd.to_timedelta(audit["days_before"], unit="D")
    audit["blood_draw_month"] = audit["crp_date"].dt.month
    audit["winter"] = audit["blood_draw_month"].isin([11, 12, 1, 2, 3, 4]).astype("Int64")
    audit["crp_gt10"] = (pd.to_numeric(audit["crp_mgL"], errors="coerce") > 10).astype("Int64")

    sources: dict[str, str] = {}
    merge_specs = [
        ("smoking", ["q2_smoking_status.csv", "q1_smoking_status.csv"], {"smk_status": "smk_status", "smoking_status": "smk_status"}),
        ("alcohol", ["q2_alcohol_use.csv", "q1_alcohol_use.csv"], {"alcohol_use": "alcohol_use"}),
        ("egfr_ckd", ["q2_egfr_baseline.csv", "q1_egfr_baseline.csv"], {"egfr": "egfr", "baseline_ckd": "baseline_ckd"}),
        ("income", ["q2_income_band.csv", "q1_income_band.csv", "income_band.csv", "household_income.csv"], {"income_band": "income_band", "g_income": "income_band"}),
        ("education", ["q2_education.csv", "education.csv"], {"education_level": "education_level", "education": "education_level"}),
        ("insurance", ["q2_insurance.csv", "insurance.csv"], {"insurance_status": "insurance_status", "insurance": "insurance_status"}),
        ("healthcare_access", ["q2_healthcare_access.csv", "healthcare_access.csv"], {"healthcare_access_barrier": "healthcare_access_barrier", "healthcare_access": "healthcare_access_barrier"}),
        ("medication", ["q2_medication_use.csv"], {
            "vitd_supplement_use": "vitd_supplement_use",
            "vitd_supplement_120d": "vitd_supplement_120d",
            "vitd_supplement_active_on_target": "vitd_supplement_active_on_target",
            "vitd_supplement_history_pre_target": "vitd_supplement_history_pre_target",
            "active_vitd_analog_use_120d": "active_vitd_analog_use_120d",
        }),
        ("acute_proxy", ["q2_acute_inflammation_proxy.csv"], {"infection_30d": "infection_30d", "hospitalization_30d": "hospitalization_30d", "ed_visit_30d": "ed_visit_30d"}),
        ("liver", ["q2_liver_function_disease.csv"], {"liver_disease": "liver_disease", "alt": "alt", "ast": "ast", "bilirubin_total": "bilirubin_total"}),
    ]
    for key, filenames, cmap in merge_specs:
        audit = merge_optional_covariate_table(audit, filenames, cmap, key, sources)

    current_cols = ["log_crp", "log_vitd", "age_at_index", "female", "winter", "bmi_full"]
    current_mask = audit[current_cols].notna().all(axis=1)
    current_mask = current_mask & pd.to_numeric(audit["bmi_full"], errors="coerce").between(12, 70)

    specs = [
        ("Vitamin D exposure", ["vitd_main", "log_vitd"], "base", "exposure", "all", "already in Q2 pair", ""),
        ("CRP outcome", ["crp_mgL", "log_crp"], "base", "outcome", "all", "already in Q2 pair", ""),
        ("Age", ["age_at_index"], "base", "current covariate", "all", "already in current Q2 model", ""),
        ("Sex", ["female"], "base", "current covariate", "all", "already in current Q2 model", ""),
        ("BMI", ["bmi_full"], "base", "current covariate", "all", "already in current Q2 model", "Q2 OLS applies BMI range filter 12-70"),
        ("Season/month", ["blood_draw_month", "winter"], "derived", "current covariate", "all", "already in current Q2 model", "winter is Nov-Apr"),
        ("Race/ethnicity", ["race_black", "race_hispanic", "race_asian", "race_other"], "base", "available to add", "all", "available in analysis base, not in Q2 model", ""),
        ("Diabetes", ["baseline_diabetes"], "base", "available to add", "all", "available in analysis base, not in Q2 model", ""),
        ("Hypertension", ["baseline_hypertension"], "base", "available to add or sensitivity", "all", "available in analysis base, not in Q2 model", ""),
        ("Hyperlipidemia", ["baseline_hyperlipidemia"], "base", "available to add or sensitivity", "all", "available in analysis base, not in Q2 model", ""),
        ("Smoking", ["smk_status"], "smoking", "candidate covariate", "all", "", "constructed before investigating coverage shortfall"),
        ("Alcohol", ["alcohol_use"], "alcohol", "candidate covariate", "all", "", "constructed before investigating coverage shortfall"),
        ("Kidney function", ["egfr", "baseline_ckd"], "egfr_ckd", "candidate/sensitivity covariate", "any", "", "constructed from creatinine"),
        ("Household income", ["income_band"], "income", "candidate SES covariate", "all", "", "SES component"),
        ("Education", ["education_level"], "education", "candidate SES covariate", "all", "", "SES component"),
        ("Insurance", ["insurance_status"], "insurance", "candidate SES/access covariate", "all", "", "SES/access component"),
        ("Healthcare access", ["healthcare_access_barrier"], "healthcare_access", "candidate SES/access covariate", "all", "", "SES/access component"),
        ("Vitamin D supplement use", ["vitd_supplement_use"], "medication", "candidate covariate", "all", "", "120-day near-target oral/relevant vitamin D supplement definition"),
        ("Vitamin D supplement detail", ["vitd_supplement_120d", "vitd_supplement_active_on_target", "vitd_supplement_history_pre_target"], "medication", "audit detail", "any", "", "documents current/recent versus historical vitamin D supplement records"),
        ("Active vitamin D analog use", ["active_vitd_analog_use_120d"], "medication", "audit detail", "all", "", "not counted as ordinary vitamin D supplement use"),
        ("Any supplement use", ["any_supplement_use"], "parked", "parked", "all", "parked in current supplement-only pass", "current construction only targets vitamin D supplement use"),
        ("Medication use", ["statin_use", "systemic_steroid_use", "nsaid_use"], "parked", "parked", "any", "parked in current supplement-only pass", "steroid/NSAID parked; statin not required for this supplement-focused pass"),
        ("Acute inflammation flag", ["crp_gt10"], "derived", "sensitivity-ready", "all", "derivable from Q2 CRP", ""),
        ("Infection/hospitalization proxy", ["infection_30d", "hospitalization_30d", "ed_visit_30d"], "acute_proxy", "sensitivity candidate", "any", "", "constructed around CRP date"),
        ("Liver function/disease", ["liver_disease", "alt", "ast"], "liver", "sensitivity candidate", "any", "", "diagnosis or lab proxy"),
    ]
    coverage = pd.DataFrame([
        coverage_row(audit, covariate, fields, source_key, sources, role, mode, status, note)
        for covariate, fields, source_key, role, mode, status, note in specs
    ])
    coverage.insert(7, "current_q2_model_complete_case_n", int(current_mask.sum()))

    categorical_fields = [
        "female", "winter", "blood_draw_month", "race_eth_simplified",
        "baseline_diabetes", "baseline_hypertension", "baseline_hyperlipidemia",
        "smk_status", "alcohol_use", "baseline_ckd", "income_band", "education_level",
        "insurance_status", "healthcare_access_barrier", "vitd_supplement_use",
        "vitd_supplement_120d", "vitd_supplement_active_on_target", "vitd_supplement_history_pre_target",
        "active_vitd_analog_use_120d", "any_supplement_use", "statin_use", "systemic_steroid_use", "nsaid_use",
        "crp_gt10", "infection_30d", "hospitalization_30d", "ed_visit_30d", "liver_disease",
    ]
    continuous_fields = ["vitd_main", "crp_mgL", "log_vitd", "log_crp", "age_at_index", "bmi_full", "days_before", "egfr", "alt", "ast", "bilirubin_total"]
    level_counts = build_q2_covariate_level_counts(audit, categorical_fields)
    continuous_summary = build_q2_covariate_continuous_summary(audit, continuous_fields)

    audit.to_csv(table_path("q2_covariate_exploration_dataset.csv"), index=False)
    coverage.to_csv(table_path("q2_covariate_coverage.csv"), index=False)
    level_counts.to_csv(table_path("q2_covariate_level_counts.csv"), index=False)
    continuous_summary.to_csv(table_path("q2_covariate_continuous_summary.csv"), index=False)

    print_section("Q2 covariate availability exploration")
    print(f"Q2 pair n: {len(audit):,}")
    print(f"Current Q2 M4 complete-case n: {int(current_mask.sum()):,}")
    display_cols = ["covariate", "q2_role_now", "n_available_in_q2_pair", "coverage_pct", "availability_status"]
    print(coverage[display_cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("Saved q2_covariate_exploration_dataset.csv, q2_covariate_coverage.csv, q2_covariate_level_counts.csv, q2_covariate_continuous_summary.csv")
    return coverage

# %% Cell 25
def find_q2_covariate_file(filename: str) -> Path | None:
    for path in [Path(filename), table_path(filename), OUTPUT_DIR / filename]:
        if path.exists():
            return path
    return None


def read_q2_model_covariate_table(filename: str, columns: list[str]) -> pd.DataFrame:
    path = find_q2_covariate_file(filename)
    if path is None:
        raise FileNotFoundError(f"Required Q2 covariate table not found: {filename}")
    table = pd.read_csv(path)
    if "person_id" not in table.columns:
        raise ValueError(f"{filename} is missing person_id")
    table = as_person_id(table)
    keep = [col for col in columns if col in table.columns]
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")
    return table[keep].drop_duplicates("person_id")


def prepare_q2_extended_model_df(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "person_id",
        "index_date",
        "age_at_index",
        "female",
        "gender",
        "bmi_full",
        "race_eth_simplified",
        "race_black",
        "race_hispanic",
        "race_asian",
        "race_other",
        "baseline_diabetes",
    ]
    d = pair.merge(base[[col for col in base_cols if col in base.columns]], on="person_id", how="inner")
    d = normalize_index_date(d)
    d["crp_date"] = d["index_date"] - pd.to_timedelta(d["days_before"], unit="D")
    month = d["crp_date"].dt.month
    d["winter"] = ((month >= 11) | (month <= 4)).astype(int)
    d["bmi_full"] = pd.to_numeric(d["bmi_full"], errors="coerce")
    d = d[(d["bmi_full"] >= 12) & (d["bmi_full"] <= 70)].copy()

    if "female" not in d.columns and "gender" in d.columns:
        d["female"] = (d["gender"] == "Female").astype(int)
    if "race_black" not in d.columns and "race_eth_simplified" in d.columns:
        race = d["race_eth_simplified"]
        d["race_black"] = (race == "Black or African American").astype(int)
        d["race_hispanic"] = (race == "Hispanic or Latino").astype(int)
        d["race_asian"] = (race == "Asian").astype(int)
        d["race_other"] = race.isin(["Other/Multiracial", "Unknown/Other"]).astype(int)

    merge_specs = [
        ("q2_income_band.csv", ["person_id", "income_band"]),
        ("q2_education.csv", ["person_id", "education_level"]),
        ("q2_insurance.csv", ["person_id", "insurance_status"]),
        (
            "q2_medication_use.csv",
            [
                "person_id",
                "vitd_supplement_use",
                "vitd_supplement_120d",
                "vitd_supplement_active_on_target",
                "vitd_supplement_history_pre_target",
                "active_vitd_analog_use_120d",
            ],
        ),
    ]
    for filename, columns in merge_specs:
        try:
            covariates = read_q2_model_covariate_table(filename, columns)
        except FileNotFoundError as exc:
            print(f"Extended model covariate skipped: {exc}")
            continue
        d = d.merge(covariates, on="person_id", how="left")

    if "vitd_supplement_use" in d.columns:
        d["vitd_supplement_use"] = pd.to_numeric(d["vitd_supplement_use"], errors="coerce").fillna(0).astype(int)

    d.to_csv(table_path("q2_extended_model_dataset.csv"), index=False)
    return d


def classify_q2_model_term(term: str) -> str:
    if term == "Intercept":
        return "intercept"
    if term == "log_vitd":
        return "exposure"
    if term in {"age_at_index", "female", "winter", "bmi_full"}:
        return "core_covariate"
    if term.startswith("race_") or term == "baseline_diabetes":
        return "extended_covariate"
    if term.startswith("C(income_band)") or term.startswith("C(education_level)") or term.startswith("C(insurance_status)"):
        return "SES_covariate"
    if term == "vitd_supplement_use":
        return "supplement_covariate"
    return "other_covariate"


def fit_q2_extended_ols_rows(model_df: pd.DataFrame, model_specs: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for spec in model_specs:
        label = str(spec["model"])
        formula = str(spec["formula"])
        fields = list(spec["fields"])
        missing_fields = [field for field in fields if field not in model_df.columns]
        if missing_fields:
            rows.append(
                {
                    "model": label,
                    "status": "missing_fields",
                    "n": 0,
                    "formula": formula,
                    "complete_case_fields": ", ".join(fields),
                    "slope": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "r2": np.nan,
                    "error": f"Missing fields: {missing_fields}",
                }
            )
            continue
        d = model_df[fields].dropna().copy()
        if d.empty:
            rows.append(
                {
                    "model": label,
                    "status": "empty_complete_case",
                    "n": 0,
                    "formula": formula,
                    "complete_case_fields": ", ".join(fields),
                    "slope": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "r2": np.nan,
                    "error": "No complete-case rows",
                }
            )
            continue
        try:
            model = smf.ols(formula, data=d).fit()
            coefficient_ci = model.conf_int()
            for term in model.params.index:
                coefficient_rows.append(
                    {
                        "model": label,
                        "status": "fit",
                        "n": int(model.nobs),
                        "formula": formula,
                        "term": term,
                        "term_role": classify_q2_model_term(term),
                        "estimate": model.params[term],
                        "std_error": model.bse[term],
                        "ci_low": coefficient_ci.loc[term].iloc[0],
                        "ci_high": coefficient_ci.loc[term].iloc[1],
                        "p_value": model.pvalues[term],
                        "r2": model.rsquared,
                    }
                )
            ci = model.conf_int().loc["log_vitd"]
            rows.append(
                {
                    "model": label,
                    "status": "fit",
                    "n": int(model.nobs),
                    "formula": formula,
                    "complete_case_fields": ", ".join(fields),
                    "slope": model.params["log_vitd"],
                    "ci_low": ci.iloc[0],
                    "ci_high": ci.iloc[1],
                    "p_value": model.pvalues["log_vitd"],
                    "r2": model.rsquared,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": label,
                    "status": "failed",
                    "n": len(d),
                    "formula": formula,
                    "complete_case_fields": ", ".join(fields),
                    "slope": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "r2": np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(coefficient_rows)


def plot_q2_extended_logvitd_forest(table: pd.DataFrame) -> None:
    plot_data = table[(table["status"] == "fit") & table["slope"].notna()].copy()
    if plot_data.empty:
        print("Skipped q2_extended_model_logvitd_forest.png because no fitted extended models were available.")
        return

    plot_data["label"] = plot_data["model"].astype(str) + " (n=" + plot_data["n"].astype(int).map("{:,}".format) + ")"
    y = np.arange(len(plot_data))[::-1]
    x = plot_data["slope"].to_numpy(dtype=float)
    lower = plot_data["ci_low"].to_numpy(dtype=float)
    upper = plot_data["ci_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.62 * len(plot_data) + 1.5)))
    ax.errorbar(
        x,
        y,
        xerr=[x - lower, upper - x],
        fmt="o",
        color=WCM_RED,
        ecolor=WCM_NAVY,
        elinewidth=2,
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["label"])
    ax.set_xlabel("Coefficient for log 25(OH)D in log-CRP model")
    ax.set_title("Q2 extended models: log vitamin D association with log CRP")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    plt.savefig(figure_path("q2_extended_model_logvitd_forest.png"), dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def q2_reference_value_for_adjusted_plot(d: pd.DataFrame, field: str) -> object:
    if field in {"age_at_index", "bmi_full"}:
        return pd.to_numeric(d[field], errors="coerce").median()
    if field in {"race_black", "race_hispanic", "race_asian", "race_other", "baseline_diabetes", "vitd_supplement_use"}:
        return 0
    if field == "winter":
        return 0 if (d[field] == 0).any() else d[field].dropna().mode().iloc[0]
    if field == "income_band":
        return "Income 35-100k" if (d[field] == "Income 35-100k").any() else d[field].dropna().mode().iloc[0]
    if field == "education_level":
        return "College or higher" if (d[field] == "College or higher").any() else d[field].dropna().mode().iloc[0]
    if field == "insurance_status":
        return "Insured" if (d[field] == "Insured").any() else d[field].dropna().mode().iloc[0]
    mode = d[field].dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def q2_adjusted_prediction_grid(d: pd.DataFrame, fields: list[str], x_grid: np.ndarray) -> pd.DataFrame:
    grid = pd.DataFrame({"log_vitd": x_grid})
    for field in fields:
        if field in {"log_crp", "log_vitd"}:
            continue
        if field in d.columns:
            grid[field] = q2_reference_value_for_adjusted_plot(d, field)
    return grid


def plot_q2_extended_adjusted_fit_lines(model_df: pd.DataFrame, model_specs: list[dict[str, object]]) -> None:
    scatter = model_df[["vitd_main", "crp_mgL", "log_vitd"]].dropna().copy()
    scatter = scatter[(scatter["vitd_main"] > 0) & (scatter["crp_mgL"] > 0)]
    if scatter.empty:
        print("Skipped q2_extended_model_adjusted_fit_lines.png because no positive vitamin D/CRP values were available.")
        return

    x_low, x_high = scatter["log_vitd"].quantile([0.01, 0.99])
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_low >= x_high:
        x_low, x_high = scatter["log_vitd"].min(), scatter["log_vitd"].max()
    x_grid = np.linspace(x_low, x_high, 120)

    fitted_panels: list[dict[str, object]] = []
    for spec in model_specs:
        label = str(spec["model"])
        formula = str(spec["formula"])
        fields = list(spec["fields"])
        if any(field not in model_df.columns for field in fields):
            continue
        d = model_df[fields].dropna().copy()
        if d.empty:
            continue
        try:
            model = smf.ols(formula, data=d).fit()
            grid = q2_adjusted_prediction_grid(d, fields, x_grid)
            pred = model.get_prediction(grid).summary_frame(alpha=0.05)
            fitted_panels.append(
                {
                    "model": label,
                    "n": int(model.nobs),
                    "x": np.exp(x_grid),
                    "mean": np.exp(pred["mean"].to_numpy(dtype=float)),
                    "low": np.exp(pred["mean_ci_lower"].to_numpy(dtype=float)),
                    "high": np.exp(pred["mean_ci_upper"].to_numpy(dtype=float)),
                }
            )
        except Exception as exc:
            print(f"Skipped adjusted fitted line for {label}: {exc}")

    if not fitted_panels:
        print("Skipped q2_extended_model_adjusted_fit_lines.png because no adjusted fitted lines could be generated.")
        return

    panel_count = len(fitted_panels)
    fig, axes = plt.subplots(1, panel_count, figsize=(5.1 * panel_count, 4.2), sharex=True, sharey=True)
    if panel_count == 1:
        axes = [axes]

    rng = np.random.default_rng(714)
    max_points = min(len(scatter), 6000)
    scatter_plot = scatter.iloc[rng.choice(len(scatter), size=max_points, replace=False)] if len(scatter) > max_points else scatter

    for ax, panel in zip(axes, fitted_panels):
        ax.scatter(scatter_plot["vitd_main"], scatter_plot["crp_mgL"], s=5, alpha=0.08, color="0.25", edgecolors="none")
        ax.plot(panel["x"], panel["mean"], color=WCM_RED, lw=2.3)
        ax.fill_between(panel["x"], panel["low"], panel["high"], color=WCM_NAVY, alpha=0.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f'{panel["model"]}\n(n={panel["n"]:,})', fontsize=10)
        ax.grid(alpha=0.25, which="both")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("CRP, mg/L (log scale)")
    for ax in axes:
        ax.set_xlabel("Serum/plasma 25(OH)D, ng/mL (log scale)")
    fig.suptitle("Q2 adjusted log-log fitted lines at reference covariate profile", y=1.04, fontsize=14)
    fig.tight_layout()
    plt.savefig(figure_path("q2_extended_model_adjusted_fit_lines.png"), dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_q2_extended_models(base: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    d = prepare_q2_extended_model_df(base, pair)

    core_fields = ["log_crp", "log_vitd", "age_at_index", "female", "winter", "bmi_full"]
    extended_fields = core_fields + [
        "race_black",
        "race_hispanic",
        "race_asian",
        "race_other",
        "baseline_diabetes",
        "income_band",
        "education_level",
        "insurance_status",
    ]
    supplement_fields = extended_fields + ["vitd_supplement_use"]

    model_specs = [
        {
            "model": "A_core_current",
            "formula": "log_crp ~ log_vitd + age_at_index + female + winter + bmi_full",
            "fields": core_fields,
        },
        {
            "model": "B_extended_race_diabetes_SES",
            "formula": (
                "log_crp ~ log_vitd + age_at_index + female + winter + bmi_full "
                "+ race_black + race_hispanic + race_asian + race_other "
                "+ baseline_diabetes + C(income_band) + C(education_level) + C(insurance_status)"
            ),
            "fields": extended_fields,
        },
        {
            "model": "C_extended_plus_vitd_supplement",
            "formula": (
                "log_crp ~ log_vitd + age_at_index + female + winter + bmi_full "
                "+ race_black + race_hispanic + race_asian + race_other "
                "+ baseline_diabetes + C(income_band) + C(education_level) + C(insurance_status) "
                "+ vitd_supplement_use"
            ),
            "fields": supplement_fields,
        },
    ]

    table, coefficient_table = fit_q2_extended_ols_rows(d, model_specs)
    table.to_csv(table_path("q2_extended_model_loglog.csv"), index=False)
    coefficient_table.to_csv(table_path("q2_extended_model_full_coefficients.csv"), index=False)
    plot_q2_extended_logvitd_forest(table)
    plot_q2_extended_adjusted_fit_lines(d, model_specs)

    print_section("Q2 extended log-log OLS models")
    print(table[["model", "status", "n", "slope", "ci_low", "ci_high", "p_value", "r2"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Saved q2_extended_model_dataset.csv, q2_extended_model_loglog.csv, q2_extended_model_full_coefficients.csv, q2_extended_model_logvitd_forest.png, and q2_extended_model_adjusted_fit_lines.png")
    return table

# %% Cell 27
def print_workflow_context() -> None:
    print_section("Setup")
    print(f"CDR: {CDR}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Output tables: {TABLE_DIR.resolve()}")
    print(f"Output figures: {FIGURE_DIR.resolve()}")


def run_common_preparation() -> pd.DataFrame:
    print_workflow_context()
    base = load_q1_analysis_base()
    print_section("Common preparation complete")
    print(f"Analysis base rows: {len(base):,}")
    return base


def run_q1_workflow(base: pd.DataFrame) -> None:
    print_section("Q1 workflow")
    make_q1_table1(base)
    run_q1_main_cox(base)
    run_q1_subgroups(base)
    plot_q1_subgroup_forest()
    run_q1_spline_and_threshold(base)
    print_section("Q1 done")
    print("Q1 workflow completed.")


def run_q2_workflow(
    base: pd.DataFrame,
    build_candidate_covariates: bool = True,
    rebuild_candidate_covariates: bool = False,
) -> None:
    print_section("Q2 workflow")
    baseline_crp, _ = run_q2_crp_build(base)
    pair = build_q2_pair(base, baseline_crp)
    if build_candidate_covariates:
        run_q2_candidate_covariate_construction(base, pair, rebuild=rebuild_candidate_covariates)
    run_q2_correlation(pair)
    run_q2_multivariable(base, pair)
    run_q2_extended_models(base, pair)
    run_q2_covariate_availability_audit(base, pair)
    run_q2_quartile_plot(pair)
    print_section("Q2 done")
    print("Q2 workflow completed.")


def run_full_workflow() -> None:
    base = run_common_preparation()
    run_q1_workflow(base)
    run_q2_workflow(base)
    print_section("Done")
    print("Q1 and Q2 clean workflow completed.")

# %% Cell 29
base = run_common_preparation()

# %% Cell 31
# Run only when Q1 outputs need to be refreshed.
# run_q1_workflow(base)

# %% Cell 33
run_q2_workflow(base)
