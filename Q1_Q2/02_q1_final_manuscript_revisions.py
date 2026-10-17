# Source companion of 02_q1_final_manuscript_revisions.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 1
# ============================================================
# FINAL Q1 TABLE 1 WITH P-VALUES
# Includes Other/Unknown race/ethnicity category
# Primary analytic cohort: N = 90,636
#
# Continuous variables: Kruskal-Wallis test
# Categorical variables: Chi-square test
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, chi2_contingency


# ============================================================
# 1. LOAD FINAL Q1 FILES
# ============================================================

vitd = pd.read_csv("vitd_baseline_versions.csv")
cov  = pd.read_csv("q1_covariate_table_v3_bmi.csv")
fu   = pd.read_csv("q1_followup_person_level_v2.csv")

print("=" * 90)
print("FILES LOADED")
print("=" * 90)

print("Vitamin D file N:", len(vitd))
print("Covariate file N:", len(cov))
print("Follow-up file N:", len(fu))


# ============================================================
# 2. IDENTIFY FINAL VITAMIN D COLUMN
# ============================================================

if "vitd_final" in vitd.columns:
    VITD_COL = "vitd_final"
elif "vitd_main" in vitd.columns:
    VITD_COL = "vitd_main"
else:
    raise ValueError(
        "Cannot find final vitamin D column.\n"
        f"Available columns: {list(vitd.columns)}"
    )

print("\nVitamin D column used:", VITD_COL)


# ============================================================
# 3. CREATE FINAL VITAMIN D CATEGORIES
# ============================================================

cat_order = [
    "Deficient (<10)",
    "Insufficient (10-<20)",
    "Adequate (20-<30)",
    "Sufficient (>=30)"
]

vitd["vitd_category_final"] = pd.cut(
    vitd[VITD_COL],
    bins=[-np.inf, 10, 20, 30, np.inf],
    labels=cat_order,
    right=False
)


# ============================================================
# 4. BUILD FINAL TABLE 1 DATASET
# ============================================================

needed_cov = [
    "person_id",
    "age_at_index",
    "gender",
    "race_eth_simplified",
    "baseline_diabetes",
    "baseline_hypertension",
    "baseline_hyperlipidemia",
    "bmi_full"
]

needed_fu = [
    "person_id",
    "event_mi_or_stroke",
    "follow_up_years"
]

t1 = (
    vitd[
        ["person_id", VITD_COL, "vitd_category_final"]
    ]
    .merge(
        cov[needed_cov],
        on="person_id",
        how="left"
    )
    .merge(
        fu[needed_fu],
        on="person_id",
        how="left"
    )
)


# ============================================================
# 5. VERIFY FINAL COHORT
# ============================================================

print("\n" + "=" * 90)
print("FINAL TABLE 1 COHORT")
print("=" * 90)

print("Table 1 N =", len(t1))
print("Unique participants =", t1["person_id"].nunique())

assert len(t1) == 90636
assert t1["person_id"].nunique() == 90636

print("\nVitamin D category counts:")
print(
    t1["vitd_category_final"]
    .value_counts(sort=False)
)


# ============================================================
# 6. CREATE MANUSCRIPT RACE/ETHNICITY VARIABLE
#
# Methods defines:
# White
# Black
# Hispanic
# Asian
# Other/Unknown
#
# Therefore collapse:
# Other/Multiracial + Unknown/Other -> Other/Unknown
# ============================================================

def collapse_race(x):

    if pd.isna(x):
        return "Other/Unknown"

    x = str(x)

    if x == "White (Non-Hispanic)":
        return "White"

    elif x == "Black or African American":
        return "Black"

    elif x == "Hispanic or Latino":
        return "Hispanic"

    elif x == "Asian":
        return "Asian"

    else:
        return "Other/Unknown"


t1["race_table"] = (
    t1["race_eth_simplified"]
    .apply(collapse_race)
)

race_order = [
    "White",
    "Black",
    "Hispanic",
    "Asian",
    "Other/Unknown"
]

t1["race_table"] = pd.Categorical(
    t1["race_table"],
    categories=race_order,
    ordered=True
)


print("\n" + "=" * 90)
print("RACE/ETHNICITY USED FOR FINAL TABLE 1")
print("=" * 90)

print(t1["race_table"].value_counts(sort=False))

print("\nOverall percentages:")
print(
    (
        t1["race_table"]
        .value_counts(sort=False)
        / len(t1)
        * 100
    ).round(1)
)


# ============================================================
# 7. DERIVED VARIABLES
# ============================================================

t1["female"] = (
    t1["gender"] == "Female"
).astype(int)

t1["bmi_available"] = (
    t1["bmi_full"].notna()
).astype(int)


# ============================================================
# 8. STATISTICAL TEST FUNCTIONS
# ============================================================

def kw_p(column):

    groups = [
        t1.loc[
            t1["vitd_category_final"] == group,
            column
        ].dropna()
        for group in cat_order
    ]

    stat, p = kruskal(*groups)

    return stat, p


def chi_p(column):

    tab = pd.crosstab(
        t1["vitd_category_final"],
        t1[column]
    )

    chi2, p, dof, expected = chi2_contingency(tab)

    return chi2, p, dof


def fmt_p(p):

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


# ============================================================
# 9. SUMMARY FUNCTIONS
# ============================================================

def med_iqr(series, decimals=0):

    x = series.dropna()

    med = x.median()
    q1  = x.quantile(0.25)
    q3  = x.quantile(0.75)

    if decimals == 0:
        return f"{med:.0f} ({q1:.0f}–{q3:.0f})"

    return (
        f"{med:.{decimals}f} "
        f"({q1:.{decimals}f}–{q3:.{decimals}f})"
    )


def pct_binary(df, col):

    return (
        pd.to_numeric(df[col], errors="coerce")
        .mean()
        * 100
    )


def pct_race(df, race):

    return (
        (df["race_table"] == race).sum()
        / len(df)
        * 100
    )


def f1(x):
    return f"{x:.1f}"


# ============================================================
# 10. RUN TABLE 1 TESTS
# ============================================================

age_stat, p_age = kw_p("age_at_index")
bmi_stat, p_bmi = kw_p("bmi_full")
follow_stat, p_follow = kw_p("follow_up_years")

female_stat, p_female, _ = chi_p("female")

# IMPORTANT:
# Race P-value now uses the manuscript five-category variable
race_stat, p_race, _ = chi_p("race_table")

diabetes_stat, p_diabetes, _ = chi_p(
    "baseline_diabetes"
)

htn_stat, p_htn, _ = chi_p(
    "baseline_hypertension"
)

hlip_stat, p_hlip, _ = chi_p(
    "baseline_hyperlipidemia"
)

bmi_avail_stat, p_bmi_avail, _ = chi_p(
    "bmi_available"
)

event_stat, p_event, _ = chi_p(
    "event_mi_or_stroke"
)


# ============================================================
# 11. PRINT P-VALUES
# ============================================================

print("\n" + "=" * 90)
print("FINAL TABLE 1 P-VALUES")
print("=" * 90)

print("Age:", fmt_p(p_age))
print("Sex:", fmt_p(p_female))
print("Race/ethnicity overall:", fmt_p(p_race))
print("Diabetes:", fmt_p(p_diabetes))
print("Hypertension:", fmt_p(p_htn))
print("Hyperlipidemia:", fmt_p(p_hlip))
print("BMI:", fmt_p(p_bmi))
print("BMI available:", fmt_p(p_bmi_avail))
print("Incident MI/stroke:", fmt_p(p_event))
print("Follow-up:", fmt_p(p_follow))


# ============================================================
# 12. PREPARE GROUP DATA
# ============================================================

groups = {
    group: t1[
        t1["vitd_category_final"] == group
    ].copy()
    for group in cat_order
}

overall = t1.copy()


# ============================================================
# 13. HELPER FOR ROW VALUES
# ============================================================

def values_by_group(func):

    vals = []

    for group in cat_order:
        vals.append(
            func(groups[group])
        )

    vals.append(
        func(overall)
    )

    return vals


# ============================================================
# 14. BUILD FINAL TABLE ROWS
# ============================================================

table_rows = []


# N
table_rows.append([
    "N",
    f"{len(groups[cat_order[0]]):,}",
    f"{len(groups[cat_order[1]]):,}",
    f"{len(groups[cat_order[2]]):,}",
    f"{len(groups[cat_order[3]]):,}",
    f"{len(overall):,}",
    "—"
])


# Age
age_vals = values_by_group(
    lambda d: med_iqr(d["age_at_index"], 0)
)

table_rows.append(
    ["Age, median (IQR)"] +
    age_vals +
    [fmt_p(p_age)]
)


# Female
female_vals = values_by_group(
    lambda d: f1(
        (d["gender"] == "Female").mean() * 100
    )
)

table_rows.append(
    ["Female, %"] +
    female_vals +
    [fmt_p(p_female)]
)


# ============================================================
# RACE / ETHNICITY PARENT ROW
# ============================================================

table_rows.append([
    "Race/ethnicity, %",
    "",
    "",
    "",
    "",
    "",
    fmt_p(p_race)
])


# White
white_vals = values_by_group(
    lambda d: f1(
        pct_race(d, "White")
    )
)

table_rows.append(
    ["   White, %"] +
    white_vals +
    [""]
)


# Black
black_vals = values_by_group(
    lambda d: f1(
        pct_race(d, "Black")
    )
)

table_rows.append(
    ["   Black, %"] +
    black_vals +
    [""]
)


# Hispanic
hisp_vals = values_by_group(
    lambda d: f1(
        pct_race(d, "Hispanic")
    )
)

table_rows.append(
    ["   Hispanic, %"] +
    hisp_vals +
    [""]
)


# Asian
asian_vals = values_by_group(
    lambda d: f1(
        pct_race(d, "Asian")
    )
)

table_rows.append(
    ["   Asian, %"] +
    asian_vals +
    [""]
)


# NEW: Other/Unknown
other_vals = values_by_group(
    lambda d: f1(
        pct_race(d, "Other/Unknown")
    )
)

table_rows.append(
    ["   Other/Unknown, %"] +
    other_vals +
    [""]
)


# Diabetes
diab_vals = values_by_group(
    lambda d: f1(
        pd.to_numeric(
            d["baseline_diabetes"],
            errors="coerce"
        ).mean() * 100
    )
)

table_rows.append(
    ["Diabetes, %"] +
    diab_vals +
    [fmt_p(p_diabetes)]
)


# Hypertension
htn_vals = values_by_group(
    lambda d: f1(
        pd.to_numeric(
            d["baseline_hypertension"],
            errors="coerce"
        ).mean() * 100
    )
)

table_rows.append(
    ["Hypertension, %"] +
    htn_vals +
    [fmt_p(p_htn)]
)


# Hyperlipidemia
hlip_vals = values_by_group(
    lambda d: f1(
        pd.to_numeric(
            d["baseline_hyperlipidemia"],
            errors="coerce"
        ).mean() * 100
    )
)

table_rows.append(
    ["Hyperlipidemia, %"] +
    hlip_vals +
    [fmt_p(p_hlip)]
)


# BMI
bmi_vals = values_by_group(
    lambda d: med_iqr(
        d["bmi_full"],
        1
    )
)

table_rows.append(
    ["BMI, median (IQR)"] +
    bmi_vals +
    [fmt_p(p_bmi)]
)


# BMI available
bmi_avail_vals = values_by_group(
    lambda d: f1(
        d["bmi_full"].notna().mean() * 100
    )
)

table_rows.append(
    ["BMI available, %"] +
    bmi_avail_vals +
    [fmt_p(p_bmi_avail)]
)


# Incident MI/stroke
event_vals = values_by_group(
    lambda d: f1(
        pd.to_numeric(
            d["event_mi_or_stroke"],
            errors="coerce"
        ).mean() * 100
    )
)

table_rows.append(
    ["Incident MI/stroke, %"] +
    event_vals +
    [fmt_p(p_event)]
)


# Follow-up
follow_vals = values_by_group(
    lambda d: f"{d['follow_up_years'].median():.1f}"
)

table_rows.append(
    ["Follow-up yr, median"] +
    follow_vals +
    [fmt_p(p_follow)]
)


# ============================================================
# 15. CREATE DATAFRAME
# ============================================================

columns = [
    "Characteristic",
    "Deficient (<10)",
    "Insufficient (10–<20)",
    "Adequate (20–<30)",
    "Sufficient (>=30)",
    "Overall",
    "P value"
]

table1_final = pd.DataFrame(
    table_rows,
    columns=columns
)


# ============================================================
# 16. DISPLAY FINAL TABLE
# ============================================================

print("\n" + "=" * 90)
print("FINAL TABLE 1")
print("=" * 90)

display(table1_final)


# ============================================================
# 17. SAVE CSV
# ============================================================

table1_final.to_csv(
    "Table1_final_with_pvalues_and_other_unknown.csv",
    index=False
)


# ============================================================
# 18. CREATE HIGH-RESOLUTION IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(19, 10)
)

ax.axis("off")

tbl = ax.table(
    cellText=table1_final.values,
    colLabels=table1_final.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
    colWidths=[
        0.20,
        0.14,
        0.18,
        0.17,
        0.17,
        0.10,
        0.08
    ]
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.65)


# Header
for j in range(len(columns)):
    tbl[(0, j)].set_text_props(
        weight="bold"
    )


# Left-align characteristic column
for i in range(
    1,
    len(table1_final) + 1
):
    tbl[(i, 0)].get_text().set_ha(
        "left"
    )


# Remove most borders
for (row, col), cell in tbl.get_celld().items():
    cell.set_linewidth(0)


# Header rule
for j in range(len(columns)):
    tbl[(0, j)].set_linewidth(0.8)


# Bold race/ethnicity parent row
race_index = (
    table1_final.index[
        table1_final["Characteristic"]
        == "Race/ethnicity, %"
    ][0]
    + 1
)

for j in range(len(columns)):
    tbl[(race_index, j)].set_text_props(
        weight="bold"
    )


# Footnote
plt.figtext(
    0.02,
    0.025,
    (
        "P values were calculated using Kruskal–Wallis tests "
        "for continuous variables and chi-square tests for categorical variables. "
        "The race/ethnicity P value represents the overall comparison across "
        "White, Black, Hispanic, Asian, and Other/Unknown categories."
    ),
    ha="left",
    fontsize=10
)

plt.tight_layout(
    rect=[0, 0.07, 1, 1]
)


# ============================================================
# 19. SAVE PNG
# ============================================================

plt.savefig(
    "Table1_final_with_pvalues_and_other_unknown.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. DONE
# ============================================================

print("\n" + "=" * 90)
print("SAVED FILES")
print("=" * 90)

print("Table1_final_with_pvalues_and_other_unknown.csv")
print("Table1_final_with_pvalues_and_other_unknown.png")

# %% Cell 2
# ============================================================
# FINAL FIG. 5 — SUBGROUP FOREST PLOT
# Reads existing subgroup results only; does NOT rerun models
#
# Fixes:
# 1) P-int shown to 4 decimal places
# 2) Bonferroni threshold = 0.00625
# 3) Only Bonferroni-significant P-int shown in red/bold
# 4) Correct subgroup level labels
# 5) Income renamed to Household income
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Working directory
# ------------------------------------------------------------

os.chdir(
    '/home/jupyter/capstone/data'
)


# ------------------------------------------------------------
# 2. Load existing subgroup results
# ------------------------------------------------------------

res = pd.read_csv('q1_subgroups_all.csv')

itab = (
    pd.read_csv('q1_interaction_tests.csv')
    .set_index('subgroup')
)

ov = pd.read_csv('q1_overall_hr.csv').iloc[0]

print("Loaded:")
print(" - q1_subgroups_all.csv")
print(" - q1_interaction_tests.csv")
print(" - q1_overall_hr.csv")


# ------------------------------------------------------------
# 3. Rename Income -> Household income
# ------------------------------------------------------------

res['subgroup'] = res['subgroup'].replace({
    'Income': 'Household income'
})

itab = itab.rename(index={
    'Income': 'Household income'
})


# ------------------------------------------------------------
# 4. Bonferroni threshold
# 8 interaction tests -> 0.05 / 8 = 0.00625
# ------------------------------------------------------------

BONF = 0.00625


# ------------------------------------------------------------
# 5. Exact subgroup level order
# ------------------------------------------------------------

LEVEL_ORDER = {

    'Age': [
        'Age <65',
        'Age >=65'
    ],

    'Sex': [
        'Female',
        'Male'
    ],

    'Race/ethnicity': [
        'White (Non-Hispanic)',
        'Black or African American',
        'Hispanic or Latino',
        'Asian',
        'Other/Multiracial',
        'Unknown/Other'
    ],

    'BMI': [
        'BMI <25',
        'BMI 25-30',
        'BMI >=30'
    ],

    'Diabetes': [
        'Diabetes: No',
        'Diabetes: Yes'
    ],

    'Hypertension': [
        'Hypertension: No',
        'Hypertension: Yes'
    ],

    'Hyperlipidemia': [
        'Hyperlipidemia: No',
        'Hyperlipidemia: Yes'
    ],

    'Household income': [
        'Income <35k',
        'Income 35-100k',
        'Income >=100k'
    ]
}


# ------------------------------------------------------------
# 6. Overall order
# ------------------------------------------------------------

ORDER = [
    'Age',
    'Sex',
    'Race/ethnicity',
    'BMI',
    'Diabetes',
    'Hypertension',
    'Hyperlipidemia',
    'Household income'
]


# ------------------------------------------------------------
# 7. Print interaction P-values
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("INTERACTION P-VALUES")
print(f"Bonferroni threshold = {BONF:.5f}")
print("=" * 62)

for sg in ORDER:

    p = float(
        itab.loc[sg, 'p_interaction']
    )

    status = (
        "Bonferroni significant"
        if p < BONF
        else "not Bonferroni significant"
    )

    print(
        f"{sg:22s} "
        f"P = {p:.4f}   "
        f"{status}"
    )


# ------------------------------------------------------------
# 8. Construct plotting rows
# ------------------------------------------------------------

rows = []

# Overall
rows.append({
    'type': 'overall',
    'label': 'Overall',
    'p_int': np.nan,
    'n': int(ov['n']),
    'events': int(ov['events']),
    'hr': float(ov['hr']),
    'lo': float(ov['lo']),
    'hi': float(ov['hi'])
})


for sg in ORDER:

    # Header row
    p_int = float(
        itab.loc[sg, 'p_interaction']
    )

    rows.append({
        'type': 'header',
        'label': sg,
        'p_int': p_int,
        'n': np.nan,
        'events': np.nan,
        'hr': np.nan,
        'lo': np.nan,
        'hi': np.nan
    })

    # Data rows
    sub = (
        res[res['subgroup'] == sg]
        .set_index('level')
    )

    for lv in LEVEL_ORDER[sg]:

        if lv not in sub.index:
            print(
                f"WARNING: missing level: "
                f"{sg} -> {lv}"
            )
            continue

        r = sub.loc[lv]

        rows.append({
            'type': 'level',
            'label': lv,
            'p_int': np.nan,
            'n': int(r['n']),
            'events': int(r['events']),
            'hr': float(r['hr']),
            'lo': float(r['lo']),
            'hi': float(r['hi'])
        })


plot_df = pd.DataFrame(rows)


# ------------------------------------------------------------
# 9. Plot settings
# ------------------------------------------------------------

WCM_RED = '#B31B1B'
WCM_NAVY = '#1f3a5f'

XMIN = 0.92
XMAX = 1.62

n_rows = len(plot_df)

fig, ax = plt.subplots(
    figsize=(12, max(9, n_rows * 0.36))
)

y_positions = np.arange(n_rows)[::-1]


# ------------------------------------------------------------
# 10. Draw rows
# ------------------------------------------------------------

for yy, (_, r) in zip(
    y_positions,
    plot_df.iterrows()
):

    row_type = r['type']
    label = r['label']


    # ========================================================
    # HEADER ROW
    # ========================================================

    if row_type == 'header':

        ax.text(
            -0.62,
            yy,
            label,
            transform=ax.get_yaxis_transform(),
            fontsize=10,
            fontweight='bold',
            va='center'
        )

        p = float(r['p_int'])

        sig = p < BONF

        ax.text(
            1.31,
            yy,
            f'{p:.4f}',
            transform=ax.get_yaxis_transform(),
            fontsize=9.5,
            va='center',
            ha='center',
            color=WCM_RED if sig else 'black',
            fontweight='bold' if sig else 'normal'
        )

        continue


    # ========================================================
    # OVERALL / LEVEL ROW
    # ========================================================

    hr = float(r['hr'])
    lo = float(r['lo'])
    hi = float(r['hi'])

    is_overall = row_type == 'overall'


    # CI line
    hi_clip = min(
        hi,
        XMAX * 0.99
    )

    lo_clip = max(
        lo,
        XMIN
    )

    ax.plot(
        [lo_clip, hi_clip],
        [yy, yy],
        color=WCM_NAVY,
        lw=1.8,
        solid_capstyle='butt'
    )


    # Right arrow if CI exceeds plot range
    if hi > XMAX:

        ax.annotate(
            '',
            xy=(XMAX, yy),
            xytext=(hi_clip, yy),
            arrowprops=dict(
                arrowstyle='->',
                color=WCM_NAVY,
                lw=1.8
            )
        )


    # Left arrow if CI below plot range
    if lo < XMIN:

        ax.annotate(
            '',
            xy=(XMIN, yy),
            xytext=(lo_clip, yy),
            arrowprops=dict(
                arrowstyle='<-',
                color=WCM_NAVY,
                lw=1.8
            )
        )


    # Point estimate
    ax.plot(
        hr,
        yy,
        marker='s' if is_overall else 'o',
        color=WCM_RED,
        markersize=9 if is_overall else 6.5,
        zorder=3
    )


    # Label
    display_label = (
        label
        if is_overall
        else '    ' + label
    )

    ax.text(
        -0.62,
        yy,
        display_label,
        transform=ax.get_yaxis_transform(),
        fontsize=9.5,
        va='center',
        fontweight='bold' if is_overall else 'normal'
    )


    # N
    ax.text(
        -0.17,
        yy,
        f"{int(r['n']):,}",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        ha='right',
        va='center'
    )


    # Events
    ax.text(
        -0.04,
        yy,
        f"{int(r['events']):,}",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        ha='right',
        va='center'
    )


    # HR text
    ax.text(
        1.03,
        yy,
        f"{hr:.2f} ({lo:.2f}–{hi:.2f})",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        va='center'
    )


# ------------------------------------------------------------
# 11. Reference line
# ------------------------------------------------------------

ax.axvline(
    1.0,
    color='grey',
    linestyle='--',
    linewidth=1
)


# ------------------------------------------------------------
# 12. Axes
# ------------------------------------------------------------

ax.set_yticks([])

ax.set_ylim(
    -1,
    n_rows
)

ax.set_xscale('log')

ax.set_xlim(
    XMIN,
    XMAX
)

ax.set_xticks([
    0.95,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50
])

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda x, pos: f'{x:.2f}'
    )
)

ax.xaxis.set_minor_formatter(
    plt.NullFormatter()
)

ax.tick_params(
    axis='x',
    which='minor',
    length=0
)

ax.set_xlabel(
    'Hazard ratio per 10 ng/mL lower 25(OH)D (95% CI)',
    fontsize=11
)


# ------------------------------------------------------------
# 13. Column headers
# ------------------------------------------------------------

top_y = n_rows - 0.05

ax.text(
    -0.17,
    top_y,
    'N',
    transform=ax.get_yaxis_transform(),
    fontsize=9.5,
    fontweight='bold',
    ha='right'
)

ax.text(
    -0.04,
    top_y,
    'Events',
    transform=ax.get_yaxis_transform(),
    fontsize=9.5,
    fontweight='bold',
    ha='right'
)

ax.text(
    1.03,
    top_y,
    'HR (95% CI)',
    transform=ax.get_yaxis_transform(),
    fontsize=9.5,
    fontweight='bold',
    ha='left'
)

ax.text(
    1.31,
    top_y,
    'P-int',
    transform=ax.get_yaxis_transform(),
    fontsize=9.5,
    fontweight='bold',
    ha='center'
)


# ------------------------------------------------------------
# 14. Title
# ------------------------------------------------------------

ax.set_title(
    'Subgroup analysis: baseline vitamin D and incident MI or stroke\n'
    f'All of Us cohort, n = {int(ov["n"]):,}, '
    f'{int(ov["events"]):,} events',
    fontsize=12.5,
    pad=24
)


# ------------------------------------------------------------
# 15. Clean style
# ------------------------------------------------------------

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.tick_params(
    axis='y',
    left=False
)

plt.subplots_adjust(
    left=0.34,
    right=0.81,
    top=0.92,
    bottom=0.08
)


# ------------------------------------------------------------
# 16. Save final figure
# ------------------------------------------------------------

outfile = 'q1_forest_subgroups_FINAL.png'

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n" + "=" * 62)
print("FINAL FIGURE SAVED")
print("=" * 62)
print(outfile)

# %% Cell 3
# ============================================================
# Q1 main Cox model table
# Re-run adjusted HR table in AoU
# ============================================================

import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# ---- 1. set working directory ----
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')

# ---- 2. read final Cox cohort ----
df = pd.read_csv("analysis_base_clean.csv")

print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

# ---- 3. basic checks ----
needed = [
    'vitd_main',
    'follow_up_years',
    'event_mi_or_stroke',
    'age_at_index',
    'female',
    'race_black',
    'race_hispanic',
    'race_asian',
    'race_other',
    'baseline_diabetes',
    'baseline_hypertension',
    'baseline_hyperlipidemia'
]

missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    print("Missing columns:", missing_cols)
else:
    print("All required columns are present.")

# ---- 4. define vitamin D categories ----
# Reference group = sufficient (>=30 ng/mL)
df['vitd_deficient'] = (df['vitd_main'] < 10).astype(int)
df['vitd_insufficient'] = ((df['vitd_main'] >= 10) & (df['vitd_main'] < 20)).astype(int)
df['vitd_adequate'] = ((df['vitd_main'] >= 20) & (df['vitd_main'] < 30)).astype(int)

# ---- 5. keep complete cases for the model ----
model_vars = [
    'follow_up_years',
    'event_mi_or_stroke',
    'vitd_deficient',
    'vitd_insufficient',
    'vitd_adequate',
    'age_at_index',
    'female',
    'race_black',
    'race_hispanic',
    'race_asian',
    'race_other',
    'baseline_diabetes',
    'baseline_hypertension',
    'baseline_hyperlipidemia'
]

d = df.dropna(subset=model_vars).copy()

print("Final modeling N =", len(d))
print("Events =", int(d['event_mi_or_stroke'].sum()))

# ---- 6. fit Cox model ----
cph = CoxPHFitter()
cph.fit(
    d[model_vars],
    duration_col='follow_up_years',
    event_col='event_mi_or_stroke'
)

# ---- 7. display raw Cox summary ----
print("\n==================== RAW COX SUMMARY ====================")
display(cph.summary)

# ---- 8. format output table like your screenshot ----
summary = cph.summary.copy()

# map variable names to display names
name_map = {
    'vitd_deficient': 'Vitamin D: Deficient (<10)',
    'vitd_insufficient': 'Vitamin D: Insufficient (10–<20)',
    'vitd_adequate': 'Vitamin D: Adequate (20–<30)',
    'age_at_index': 'Age (per year)',
    'female': 'Female (vs Male)',
    'race_black': 'Black (vs White)',
    'race_hispanic': 'Hispanic (vs White)',
    'race_asian': 'Asian (vs White)',
    'race_other': 'Other/Unknown (vs White)',
    'baseline_diabetes': 'Diabetes',
    'baseline_hypertension': 'Hypertension',
    'baseline_hyperlipidemia': 'Hyperlipidemia'
}

order = [
    'vitd_deficient',
    'vitd_insufficient',
    'vitd_adequate',
    'age_at_index',
    'female',
    'race_black',
    'race_hispanic',
    'race_asian',
    'race_other',
    'baseline_diabetes',
    'baseline_hypertension',
    'baseline_hyperlipidemia'
]

out = summary.loc[order, ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
out = out.reset_index().rename(columns={'covariate': 'CovariateRaw', 'index': 'CovariateRaw'})

# if reset_index used 'covariate' or 'index'
if 'covariate' in out.columns:
    out['Covariate'] = out['covariate'].map(name_map)
    out = out.drop(columns=['covariate'])
elif 'CovariateRaw' in out.columns:
    out['Covariate'] = out['CovariateRaw'].map(name_map)
    out = out.drop(columns=['CovariateRaw'])

out['HR (95% CI)'] = out.apply(
    lambda r: f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}–{r['exp(coef) upper 95%']:.2f})",
    axis=1
)

def fmt_p(x):
    if x < 0.001:
        return "<0.001"
    else:
        return f"{x:.3f}"

out['p'] = out['p'].apply(fmt_p)

final_table = out[['Covariate', 'HR (95% CI)', 'p']].copy()

print("\n==================== FORMATTED TABLE ====================")
display(final_table)

# ---- 9. save outputs ----
final_table.to_csv("q1_main_cox_table.csv", index=False)
summary.to_csv("q1_main_cox_raw_summary.csv")

print("\nSaved:")
print(" - q1_main_cox_table.csv")
print(" - q1_main_cox_raw_summary.csv")

# %% Cell 4
# ============================================================
# Supplementary Table 1
# OMOP concept sets used for baseline CVD exclusion
# and incident outcome ascertainment
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Working directory
# ------------------------------------------------------------

os.chdir(
    '/home/jupyter/capstone/data'
)


# ------------------------------------------------------------
# 2. Locked Q1 concept sets
# ------------------------------------------------------------

# Myocardial infarction:
# baseline exclusion + incident outcome
MI_PRIMARY_IDS = [
    4270024,
    312327,
    4296653,
    444406,
    434376,
    4329847,
    438170,
    438438,
    43020460,
    46270163,
    438447,
    441579,
    436706,
    46270162
]


# Old myocardial infarction:
# baseline exclusion only
OLD_MI_BASELINE_IDS = [
    314666
]


# Coronary artery disease / coronary heart disease:
# baseline exclusion only
CAD_CHD_BASELINE_IDS = [
    764123,
    40481919,
    317576,
    36712983,
    36712982,
    4242670
]


# Stroke:
# baseline exclusion + incident outcome
STROKE_PRIMARY_IDS = [
    443454,
    4110192,
    4108356,
    381316,
    45772786,
    4110190,
    4111711,
    45767658,
    46270031,
    4153352,
    4111710,
    4310996,
    46273649,
    43531605
]


# ------------------------------------------------------------
# 3. Build Supplementary Table 1
# ------------------------------------------------------------

supp_table1 = pd.DataFrame({
    'Category': [
        'CAD/CHD',
        'Myocardial infarction',
        'Old myocardial infarction',
        'Stroke'
    ],

    'Use in analysis': [
        'Baseline exclusion only',
        'Baseline exclusion + incident outcome',
        'Baseline exclusion only',
        'Baseline exclusion + incident outcome'
    ],

    'Number of concepts': [
        len(CAD_CHD_BASELINE_IDS),
        len(MI_PRIMARY_IDS),
        len(OLD_MI_BASELINE_IDS),
        len(STROKE_PRIMARY_IDS)
    ]
})


print("=" * 80)
print("SUPPLEMENTARY TABLE 1")
print("=" * 80)

display(
    supp_table1.style.hide(axis='index')
)


# ------------------------------------------------------------
# 4. Sanity checks
# ------------------------------------------------------------

print("\nConcept-set counts:")
print("CAD/CHD                =", len(CAD_CHD_BASELINE_IDS))
print("Myocardial infarction  =", len(MI_PRIMARY_IDS))
print("Old myocardial infarct =", len(OLD_MI_BASELINE_IDS))
print("Stroke                 =", len(STROKE_PRIMARY_IDS))


# Expected:
assert len(CAD_CHD_BASELINE_IDS) == 6
assert len(MI_PRIMARY_IDS) == 14
assert len(OLD_MI_BASELINE_IDS) == 1
assert len(STROKE_PRIMARY_IDS) == 14

print("\n✓ All concept-set counts match Supplementary Table 1.")


# ------------------------------------------------------------
# 5. Save clean CSV
# ------------------------------------------------------------

supp_table1.to_csv(
    'Supplementary_Table1_concept_sets.csv',
    index=False
)


# ------------------------------------------------------------
# 6. Create clean publication-style PNG
#    No pandas index
#    No blue selection
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 3.2))

ax.axis('off')

table = ax.table(
    cellText=supp_table1.values,
    colLabels=supp_table1.columns,
    cellLoc='left',
    colLoc='left',
    loc='center',
    colWidths=[0.30, 0.52, 0.18]
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)


# ------------------------------------------------------------
# 7. Style
# ------------------------------------------------------------

for (row, col), cell in table.get_celld().items():

    # remove vertical/grid borders
    cell.set_linewidth(0)

    # header
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('white')

    # alternating rows
    else:
        if row % 2 == 0:
            cell.set_facecolor('#f3f3f3')
        else:
            cell.set_facecolor('white')


plt.tight_layout()


# ------------------------------------------------------------
# 8. Save high-resolution image
# ------------------------------------------------------------

outfile = 'Supplementary_Table1_FINAL.png'

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\nSaved:")
print(" - Supplementary_Table1_concept_sets.csv")
print(" - Supplementary_Table1_FINAL.png")
