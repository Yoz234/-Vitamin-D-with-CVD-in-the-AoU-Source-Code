# Source companion of 18_q3_prior_manuscript_figures.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 3
from pathlib import Path
import os
import sys
import math
import json
import textwrap
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception as exc:
    raise RuntimeError(
        "This notebook needs numpy, pandas, and matplotlib. "
        "Run it in AoU/Jupyter or another scientific Python environment. "
        f"Original import error: {exc}"
    )

# Optional overrides. Leave blank when running in the standard AoU workspace output layout.
HAIL_DIR_OVERRIDE = os.environ.get("Q3_VDR_HAIL_DIR", "")
MODEL15_DIR_OVERRIDE = os.environ.get("Q3_VDR_MODEL15_DIR", "")
CRP16_DIR_OVERRIDE = os.environ.get("Q3_VDR_CRP16_DIR", "")
STEP17_DIR_OVERRIDE = os.environ.get("Q3_VDR_STEP17_DIR", "")
OUTPUT_DIR_OVERRIDE = os.environ.get("Q3_VDR_FIGURE_OUTPUT_DIR", "")

def existing_dir(path):
    if not path:
        return None
    p = Path(path).expanduser()
    return p if p.exists() and p.is_dir() else None

def first_existing_dir(candidates, required_files=None):
    required_files = required_files or []
    for candidate in candidates:
        p = existing_dir(candidate)
        if p is None:
            continue
        if all((p / name).exists() for name in required_files):
            return p
    return None

data_dir_candidates = [
    os.environ.get("Q3_VDR_DATA_DIR", ""),
    "/home/dataproc/workspace/VDR",
    Path.cwd(),
    Path.cwd().parent,
]
DATA_DIR = first_existing_dir([p for p in data_dir_candidates if p]) or Path.cwd()

HAIL_DIR = first_existing_dir([
    HAIL_DIR_OVERRIDE,
    DATA_DIR / "Q3_VDR_tier1_hail_genotype_outputs_2026_08_07",
], ["14_hail_stage_once_local_validation.csv", "14_hail_stage_once_status.csv"])
MODEL15_DIR = first_existing_dir([
    MODEL15_DIR_OVERRIDE,
    DATA_DIR / "Q3_VDR_tier1_additive_genotype_outputs_2026_08_08",
], ["14_additive_primary_interaction_results.csv", "09_additive_vitd_dosage_cell_counts.csv"])
CRP16_DIR = first_existing_dir([
    CRP16_DIR_OVERRIDE,
    DATA_DIR / "Q3_VDR_tier1_crp_sensitivity_outputs_2026_08_09",
], ["16_crp_sensitivity_model_comparison.csv", "16_crp_distribution_extreme_value_summary.csv", "16_crp_timing_relative_to_vitd_summary.csv"])
STEP17_DIR = first_existing_dir([
    STEP17_DIR_OVERRIDE,
    DATA_DIR / "Q3_VDR_step5_remaining_validation_outputs_2026_08_09",
], ["17_tier1_vdr_aou_ref_alt_records_for_harmonization.csv", "17_tier1_vdr_pooled_genotype_qc_hwe_descriptive.csv", "17_tier1_vdr_pairwise_ld_r2_dprime_unphased_em.csv"])

missing_dirs = [name for name, value in {
    "HAIL_DIR": HAIL_DIR,
    "MODEL15_DIR": MODEL15_DIR,
    "CRP16_DIR": CRP16_DIR,
    "STEP17_DIR": STEP17_DIR,
}.items() if value is None]
if missing_dirs:
    raise FileNotFoundError(
        "Could not infer required result directories: " + ", ".join(missing_dirs) +
        ". Set Q3_VDR_HAIL_DIR, Q3_VDR_MODEL15_DIR, Q3_VDR_CRP16_DIR, and/or Q3_VDR_STEP17_DIR."
    )

if OUTPUT_DIR_OVERRIDE:
    OUTPUT_DIR = Path(OUTPUT_DIR_OVERRIDE).expanduser()
else:
    OUTPUT_DIR = DATA_DIR / "Q3_VDR_manuscript_figures_2026_08_09"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SNP_ORDER = ["rs2228570", "rs1544410", "rs7975232", "rs731236"]
SNP_LABELS = {
    "rs2228570": "FokI\nrs2228570",
    "rs1544410": "BsmI\nrs1544410",
    "rs7975232": "ApaI\nrs7975232",
    "rs731236": "TaqI\nrs731236",
}
SNP_SHORT = {"rs2228570": "FokI", "rs1544410": "BsmI", "rs7975232": "ApaI", "rs731236": "TaqI"}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGURE_MANIFEST = []

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path

def read_csv(path):
    return pd.read_csv(require_file(path))

def save_figure(fig, stem, caption, figure_type="main_or_supplement"):
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    FIGURE_MANIFEST.append({
        "figure_stem": stem,
        "png": str(png),
        "pdf": str(pdf),
        "figure_type": figure_type,
        "caption": caption,
    })
    print(f"Saved {png}")
    print(f"Saved {pdf}")

print("DATA_DIR:", DATA_DIR)
print("HAIL_DIR:", HAIL_DIR)
print("MODEL15_DIR:", MODEL15_DIR)
print("CRP16_DIR:", CRP16_DIR)
print("STEP17_DIR:", STEP17_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% Cell 5
hail_validation = read_csv(HAIL_DIR / "14_hail_stage_once_local_validation.csv")
hail_status = read_csv(HAIL_DIR / "14_hail_stage_once_status.csv")
cvd_primary = read_csv(MODEL15_DIR / "14_additive_primary_interaction_results.csv")
cell_counts = read_csv(MODEL15_DIR / "09_additive_vitd_dosage_cell_counts.csv")
crp_model = read_csv(CRP16_DIR / "16_crp_sensitivity_model_comparison.csv")
crp_dist = read_csv(CRP16_DIR / "16_crp_distribution_extreme_value_summary.csv")
crp_timing = read_csv(CRP16_DIR / "16_crp_timing_relative_to_vitd_summary.csv")
ref_alt = read_csv(STEP17_DIR / "17_tier1_vdr_aou_ref_alt_records_for_harmonization.csv")
pooled_qc = read_csv(STEP17_DIR / "17_tier1_vdr_pooled_genotype_qc_hwe_descriptive.csv")
ld_pairs = read_csv(STEP17_DIR / "17_tier1_vdr_pairwise_ld_r2_dprime_unphased_em.csv")

for df in [cvd_primary, crp_model, pooled_qc, ld_pairs]:
    for col in df.columns:
        if col not in {"rsid", "snp1", "snp2", "status", "model_dataset", "term", "analysis_family", "analysis_role", "model_name", "dosage", "interaction", "backend", "multiple_testing_family", "scope", "group", "run_id"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")

print("Loaded inputs:")
print("  hail_validation", hail_validation.shape)
print("  cvd_primary", cvd_primary.shape)
print("  crp_model", crp_model.shape)
print("  pooled_qc", pooled_qc.shape)
print("  ld_pairs", ld_pairs.shape)

# %% Cell 7
availability_rows = []
for rsid in SNP_ORDER:
    row = hail_validation.loc[hail_validation["check"] == f"{rsid}_nonmissing"]
    if not row.empty:
        availability_rows.append({"rsid": rsid, "nonmissing": int(row.iloc[0]["detail"])})
availability = pd.DataFrame(availability_rows)
wide_n = int(hail_validation.loc[hail_validation["check"] == "wide_row_count", "detail"].iloc[0])
complete_n = int(hail_validation.loc[hail_validation["check"] == "complete_four_snp_count", "detail"].iloc[0])
availability["missing"] = wide_n - availability["nonmissing"]
availability["label"] = availability["rsid"].map(SNP_LABELS)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(availability))
ax.bar(x, availability["nonmissing"], color="#4C78A8", label="Non-missing genotype")
ax.bar(x, availability["missing"], bottom=availability["nonmissing"], color="#D9D9D9", label="Missing genotype")
ax.axhline(complete_n, color="#E45756", linestyle="--", linewidth=1.6, label=f"Complete all four SNPs: {complete_n:,}")
ax.set_xticks(x)
ax.set_xticklabels(availability["label"])
ax.set_ylabel("Participants")
ax.set_title("AoU VDR genotype extraction completeness")
ax.set_ylim(0, wide_n * 1.08)
for i, row in availability.iterrows():
    ax.text(i, row["nonmissing"] + wide_n * 0.012, f"{row['nonmissing']:,}", ha="center", va="bottom", fontsize=8)
ax.text(-0.45, wide_n * 1.02, f"Wide genotype table: {wide_n:,} participants", ha="left", va="center", fontsize=9)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
save_figure(
    fig,
    "fig01_genotype_extraction_completeness",
    "Genotype extraction completeness for the four Tier 1 VDR SNPs. Additive ALT dosage was available for over 112,000 analytic participants at each SNP, with 112,252 participants complete for all four SNPs.",
    "supplement"
)
plt.show()

# %% Cell 9
forest = cvd_primary.copy()
forest["rsid"] = pd.Categorical(forest["rsid"], categories=SNP_ORDER, ordered=True)
forest = forest.sort_values("rsid")
forest["label"] = forest["rsid"].astype(str).map(SNP_LABELS)

y = np.arange(len(forest))[::-1]
fig, ax = plt.subplots(figsize=(7.4, 4.6))
hr = forest["hr"].astype(float).values
lo = forest["hr_ci_lower"].astype(float).values
hi = forest["hr_ci_upper"].astype(float).values
xerr = np.vstack([hr - lo, hi - hr])
colors = ["#4C78A8" if p >= 0.05 else "#F58518" for p in forest["p"].astype(float)]
ax.errorbar(hr, y, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=1.4, capsize=3, zorder=1)
ax.scatter(hr, y, s=72, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(forest["label"])
ax.set_xlabel("Hazard-ratio ratio for vitamin D deficiency per additional AoU ALT allele")
ax.set_title("Primary additive VDR-by-vitamin D deficiency CVD interaction models")
ax.set_xlim(0.78, 1.28)
for i, row in enumerate(forest.iloc[::-1].itertuples(index=False)):
    pass
for ypos, row in zip(y, forest.itertuples(index=False)):
    text = f"HR {row.hr:.3f} ({row.hr_ci_lower:.3f}-{row.hr_ci_upper:.3f}); p={row.p:.3f}; Bonf. p={row.p_bonferroni_4:.3f}"
    ax.text(1.285, ypos, text, ha="left", va="center", fontsize=8, clip_on=False)
ax.text(0.78, -0.72, "All models adjusted for age, female sex, winter, and BMI; no Bonferroni-significant interaction.", ha="left", va="center", fontsize=8)
fig.subplots_adjust(right=0.62)
save_figure(
    fig,
    "fig02_primary_additive_cvd_interaction_forest",
    "Primary additive Cox interaction results for vitamin D deficiency by VDR ALT dosage. Estimates are hazard-ratio ratios per additional selected AoU ALT allele; none were significant after correction for four Tier 1 interaction tests.",
    "main"
)
plt.show()

# %% Cell 11
dataset_order = [
    "all_crp_model_base",
    "exclude_crp_gt10",
    "exclude_ambiguous_unit",
    "exclude_crp_gt10_and_ambiguous_unit",
    "same_day_or_365d_before",
    "same_day_or_5y_before",
]
dataset_labels = {
    "all_crp_model_base": "Full base",
    "exclude_crp_gt10": "Exclude\nCRP >10",
    "exclude_ambiguous_unit": "Exclude\nambiguous unit",
    "exclude_crp_gt10_and_ambiguous_unit": "Exclude both",
    "same_day_or_365d_before": "Same day or\n<=365d before",
    "same_day_or_5y_before": "Same day or\n<=5y before",
}
crp_plot = crp_model[crp_model["model_dataset"].isin(dataset_order)].copy()
crp_plot["model_dataset"] = pd.Categorical(crp_plot["model_dataset"], categories=dataset_order, ordered=True)
crp_plot["rsid"] = pd.Categorical(crp_plot["rsid"], categories=SNP_ORDER, ordered=True)
crp_plot = crp_plot.sort_values(["rsid", "model_dataset"])

fig, ax = plt.subplots(figsize=(9.4, 5.2))
palette = {"rs2228570": "#4C78A8", "rs1544410": "#F58518", "rs7975232": "#54A24B", "rs731236": "#B279A2"}
x_positions = np.arange(len(dataset_order))
offsets = np.linspace(-0.18, 0.18, len(SNP_ORDER))
for offset, rsid in zip(offsets, SNP_ORDER):
    tmp = crp_plot[crp_plot["rsid"] == rsid].set_index("model_dataset").reindex(dataset_order)
    xs = x_positions + offset
    ys = tmp["exp_coef"].astype(float).values
    ylo = tmp["exp_ci_lower"].astype(float).values
    yhi = tmp["exp_ci_upper"].astype(float).values
    ax.plot(xs, ys, marker="o", linewidth=1.6, color=palette[rsid], label=f"{SNP_SHORT[rsid]} {rsid}")
    ax.vlines(xs, ylo, yhi, color=palette[rsid], linewidth=1, alpha=0.65)
ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
ax.set_xticks(x_positions)
ax.set_xticklabels([dataset_labels[x] for x in dataset_order])
ax.set_ylabel("exp(coef) for log(25[OH]D)-by-ALT dosage interaction")
ax.set_title("CRP interaction sensitivity to acute-inflammation and timing restrictions")
ax.set_ylim(0.90, 1.17)
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.text(0.02, 0.92, "Nominal full-base rs731236/rs1544410 signals attenuate after excluding CRP >10 mg/L.", transform=ax.transAxes, fontsize=9)
fig.tight_layout()
save_figure(
    fig,
    "fig03_crp_interaction_sensitivity",
    "CRP sensitivity analysis for log(25[OH]D)-by-additive ALT-dosage interactions. Nominal full-base signals for rs731236 and rs1544410 attenuated after excluding CRP >10 mg/L, supporting an exploratory rather than mechanistic interpretation.",
    "main_or_supplement"
)
plt.show()

# %% Cell 13
timing_order = ["same_day", "1_30d_before", "31_365d_before", "1_5y_before", "gt5y_before", "after_vitd_index"]
timing_labels = {
    "same_day": "Same day",
    "1_30d_before": "1-30d\nbefore",
    "31_365d_before": "31-365d\nbefore",
    "1_5y_before": "1-5y\nbefore",
    "gt5y_before": ">5y\nbefore",
    "after_vitd_index": "After\nindex",
}
timing = crp_timing[crp_timing["timing_group"].isin(timing_order)].copy()
timing["timing_group"] = pd.Categorical(timing["timing_group"], categories=timing_order, ordered=True)
timing = timing.sort_values("timing_group")

dist_row = crp_dist.iloc[0]
n_total = int(dist_row["n"])
n_gt10 = int(dist_row["n_gt10"])
n_ambig = int(dist_row["n_ambiguous_unit"])
summary_labels = ["CRP >10\nmg/L", "Ambiguous\nunit", "Neither flag\n(approx.)"]
summary_counts = [n_gt10, n_ambig, max(n_total - n_gt10 - n_ambig, 0)]
summary_pct = [100 * x / n_total for x in summary_counts]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
ax.bar(np.arange(len(timing)), timing["percent"].astype(float), color="#4C78A8")
ax.set_xticks(np.arange(len(timing)))
ax.set_xticklabels([timing_labels[x] for x in timing["timing_group"].astype(str)])
ax.set_ylabel("Percent of CRP analytic base")
ax.set_title("CRP timing relative to vitamin D index")
for i, val in enumerate(timing["percent"].astype(float)):
    ax.text(i, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max(timing["percent"].astype(float)) * 1.25)

ax = axes[1]
colors = ["#E45756", "#F58518", "#54A24B"]
ax.bar(np.arange(len(summary_labels)), summary_pct, color=colors)
ax.set_xticks(np.arange(len(summary_labels)))
ax.set_xticklabels(summary_labels)
ax.set_ylabel("Percent")
ax.set_title("CRP exclusion flags")
for i, (pct, cnt) in enumerate(zip(summary_pct, summary_counts)):
    ax.text(i, pct + 1.0, f"{pct:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max(summary_pct) * 1.25)
fig.suptitle("CRP measurement context for exploratory inflammatory phenotype", y=1.03)
fig.tight_layout()
save_figure(
    fig,
    "fig04_crp_timing_and_exclusion_context",
    "CRP timing and exclusion context. CRP values were same-day or pre-index, but many were remote from vitamin D assessment; CRP >10 mg/L was common, supporting sensitivity analyses and cautious interpretation.",
    "supplement"
)
plt.show()

# %% Cell 15
def pair_matrix(value_col):
    mat = pd.DataFrame(np.eye(len(SNP_ORDER)), index=SNP_ORDER, columns=SNP_ORDER, dtype=float)
    for row in ld_pairs.itertuples(index=False):
        mat.loc[row.snp1, row.snp2] = float(getattr(row, value_col))
        mat.loc[row.snp2, row.snp1] = float(getattr(row, value_col))
    return mat

r2_mat = pair_matrix("r2_corr")
dp_mat = pair_matrix("d_prime_em")
labels = [SNP_SHORT[x] + "\n" + x for x in SNP_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, mat, title, vmax in [
    (axes[0], r2_mat, "Allele-count r2", 1.0),
    (axes[1], dp_mat, "Unphased EM D-prime", 1.0),
]:
    im = ax.imshow(mat.values, vmin=0, vmax=vmax, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat.values[i, j]
            text_color = "white" if val > 0.55 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("AoU VDR pairwise LD among four Tier 1 SNPs", y=1.03)
fig.tight_layout()
save_figure(
    fig,
    "fig05_vdr_pairwise_ld_heatmaps",
    "Pairwise LD among the four Tier 1 VDR SNPs. BsmI/ApaI/TaqI are correlated 3-prime markers, whereas FokI/rs2228570 is nearly independent of that cluster. D-prime values are unphased EM estimates and do not imply phased haplotype testing.",
    "main_or_supplement"
)
plt.show()

# %% Cell 17
qc = pooled_qc.copy()
qc["rsid"] = pd.Categorical(qc["rsid"], categories=SNP_ORDER, ordered=True)
qc = qc.sort_values("rsid")
qc["label"] = qc["rsid"].astype(str).map(SNP_LABELS)

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
x = np.arange(len(qc))
axes[0].bar(x, qc["maf"].astype(float), color="#4C78A8")
axes[0].set_xticks(x)
axes[0].set_xticklabels(qc["label"])
axes[0].set_ylabel("Minor allele frequency")
axes[0].set_title("Pooled MAF")
axes[0].set_ylim(0, 0.55)
for i, val in enumerate(qc["maf"].astype(float)):
    axes[0].text(i, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

axes[1].bar(x, qc["missing_pct"].astype(float), color="#F58518")
axes[1].set_xticks(x)
axes[1].set_xticklabels(qc["label"])
axes[1].set_ylabel("Missing genotype (%)")
axes[1].set_title("Pooled missingness")
axes[1].set_ylim(0, max(qc["missing_pct"].astype(float)) * 1.8)
for i, val in enumerate(qc["missing_pct"].astype(float)):
    axes[1].text(i, val + 0.001, f"{val:.3f}%", ha="center", va="bottom", fontsize=8)

fig.suptitle("Descriptive genotype QC for selected AoU VDR records", y=1.03)
fig.text(0.5, -0.02, "Pooled HWE p-values are descriptive only; ancestry-unrelated/sample-QC HWE was not computed in the current analysis.", ha="center", va="top", fontsize=8)
fig.tight_layout()
save_figure(
    fig,
    "fig06_pooled_genotype_qc_maf_missingness",
    "Descriptive pooled genotype QC for the selected AoU VDR records. Missingness was low and MAF was common for all four SNPs; pooled HWE was not interpreted as pass/fail because genetic ancestry, unrelated-set, and sample-QC filters were not applied.",
    "supplement"
)
plt.show()

# %% Cell 19
manifest = pd.DataFrame(FIGURE_MANIFEST)
manifest_path = OUTPUT_DIR / "18_q3_vdr_manuscript_figure_manifest.csv"
manifest.to_csv(manifest_path, index=False)

caption_path = OUTPUT_DIR / "18_q3_vdr_manuscript_figure_captions.md"
with open(caption_path, "w", encoding="utf-8") as f:
    f.write("# Q3 VDR Manuscript Figure Captions\n\n")
    f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    for i, row in manifest.iterrows():
        f.write(f"## {row['figure_stem']}\n\n")
        f.write(row["caption"] + "\n\n")
        f.write(f"PNG: `{row['png']}`\n\n")
        f.write(f"PDF: `{row['pdf']}`\n\n")

print("Wrote manifest:", manifest_path)
print("Wrote captions:", caption_path)
manifest
