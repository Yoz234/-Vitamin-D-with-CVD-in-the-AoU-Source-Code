# Source companion of 15_q3_vdr_additive_genotype_analytic_modeling_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# ============================================================
# Block 0. Setup, paths, and output policy
# ============================================================
import os
import sys
import math
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TARGET_SNPS = [
    {"rsid": "rs731236", "label": "TaqI", "contig": "12", "position": 47844974, "ref": "A", "alt": "G", "expected_vid": "12-47844974-A-G"},
    {"rsid": "rs7975232", "label": "ApaI", "contig": "12", "position": 47845054, "ref": "C", "alt": "A", "expected_vid": "12-47845054-C-A"},
    {"rsid": "rs1544410", "label": "BsmI", "contig": "12", "position": 47846052, "ref": "C", "alt": "T", "expected_vid": "12-47846052-C-T"},
    {"rsid": "rs2228570", "label": "FokI", "contig": "12", "position": 47879112, "ref": "A", "alt": "G", "expected_vid": "12-47879112-A-G"},
]
TIER1_RSIDS = [x["rsid"] for x in TARGET_SNPS]
RUN_ID = os.environ.get("Q3_VDR_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        Path("/home/dataproc/workspace/VDR"),
        Path("/home/jupyter/workspace/VDR"),
        Path("/home/jupyter/workspace/VDR/data"),
        Path("/home/jupyter/workspace/Q3_VDR"),
        cwd,
        cwd / "data",
        cwd.parent / "data",
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            pass
    return cwd

DATA_DIR = Path(os.environ.get("Q3_VDR_DATA_DIR", str(infer_default_data_dir()))).expanduser()

def resolve_existing_dir(env_name, default_dir_name):
    manual = os.environ.get(env_name, "").strip()
    if manual:
        return Path(manual).expanduser()
    roots = []
    for root in [DATA_DIR, Path("/home/jupyter/workspace/VDR"), Path("/home/dataproc/workspace/VDR"), Path("/home/jupyter/workspace"), Path("/home/dataproc/workspace"), Path.cwd(), Path.cwd().parent]:
        if root not in roots:
            roots.append(root)
    dir_names = [default_dir_name]
    if default_dir_name == "Q3_VDR_v9_model_prep_outputs":
        dir_names.append("prepresult")
    marker = "02_cvd_model_dataset_v9.csv" if default_dir_name == "Q3_VDR_v9_model_prep_outputs" else "14_hail_extraction_status.csv"
    for root in roots:
        for dir_name in dir_names:
            candidate = root / dir_name
            try:
                if candidate.exists() and (candidate / marker).exists():
                    return candidate
            except Exception:
                pass
    for root in roots:
        try:
            if root.exists():
                hits = sorted(root.rglob(marker))
                if hits:
                    return hits[0].parent
        except Exception:
            pass
    return DATA_DIR / default_dir_name

MODEL_PREP_DIR = resolve_existing_dir("Q3_VDR_MODEL_OUTPUT_DIR", "Q3_VDR_v9_model_prep_outputs")
HAIL_OUTPUT_DIR = resolve_existing_dir("Q3_VDR_HAIL_OUTPUT_DIR", "Q3_VDR_tier1_hail_genotype_outputs_2026_08_07")
OUTPUT_DIR = Path(os.environ.get("Q3_VDR_ADDITIVE_OUTPUT_DIR", str(DATA_DIR / "Q3_VDR_tier1_additive_genotype_outputs_2026_08_08"))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERWRITE_OUTPUTS = os.environ.get("Q3_VDR_OVERWRITE_OUTPUTS", "true").strip().lower() in {"1", "true", "yes", "y"}
USE_EXISTING_GENOTYPE_WIDE = os.environ.get("Q3_VDR_15_USE_EXISTING_GENOTYPE_WIDE", "true").strip().lower() in {"1", "true", "yes", "y"}
RUN_TARGETED_HAIL_EXTRACTION = os.environ.get("Q3_VDR_15_RUN_TARGETED_HAIL_EXTRACTION", "true").strip().lower() in {"1", "true", "yes", "y"}
STOP_IF_NO_GENOTYPE = os.environ.get("Q3_VDR_15_STOP_IF_NO_GENOTYPE", "false").strip().lower() in {"1", "true", "yes", "y"}
MIN_MODEL_N = int(os.environ.get("Q3_VDR_15_MIN_MODEL_N", "100"))
MIN_CVD_EVENTS = int(os.environ.get("Q3_VDR_15_MIN_CVD_EVENTS", "50"))

GOOGLE_PROJECT = os.environ.get("GOOGLE_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
WORKSPACE_BUCKET = os.environ.get("WORKSPACE_BUCKET", "").strip()
HAIL_TMP_DIR = os.environ.get("Q3_VDR_HAIL_TMP_DIR", (WORKSPACE_BUCKET.rstrip("/") + "/hail_tmp/q3_vdr_15") if WORKSPACE_BUCKET.startswith("gs://") else "").strip()
HAIL_PATH_CANDIDATES = []
for source, path in [
    ("Q3_VDR_HAIL_MT_PATH", os.environ.get("Q3_VDR_HAIL_MT_PATH", "").strip()),
    ("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", os.environ.get("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", "").strip()),
    ("WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH", os.environ.get("WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH", "").strip()),
    ("AOU_V9_WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH_FALLBACK", "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/acaf_threshold/splitMT/hail.mt"),
    ("AOU_V9_WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH_FALLBACK", "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/acaf_threshold/multiMT/hail.mt"),
]:
    if path and path not in {p for _, p in HAIL_PATH_CANDIDATES}:
        HAIL_PATH_CANDIDATES.append((source, path))

written_outputs = []
status_rows = []

def save(df, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")
    df.to_csv(path, index=False)
    written_outputs.append(path)
    print(f"Saved {filename}: {len(df):,} rows")
    return path

def save_text(text, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")
    path.write_text(text, encoding="utf-8")
    written_outputs.append(path)
    print(f"Saved {filename}: {len(text):,} characters")
    return path

def add_status(step, status, detail=""):
    row = {"run_id": RUN_ID, "step": step, "status": status, "detail": str(detail)[:4000]}
    status_rows.append(row)
    return row

def pct(n, denom):
    return round(100 * n / denom, 3) if denom else np.nan

save(pd.DataFrame([{
    "run_id": RUN_ID,
    "data_dir": str(DATA_DIR),
    "model_prep_dir": str(MODEL_PREP_DIR),
    "hail_output_dir": str(HAIL_OUTPUT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "use_existing_genotype_wide": USE_EXISTING_GENOTYPE_WIDE,
    "run_targeted_hail_extraction": RUN_TARGETED_HAIL_EXTRACTION,
    "stop_if_no_genotype": STOP_IF_NO_GENOTYPE,
    "hail_path_candidates": ";".join([f"{s}={p}" for s, p in HAIL_PATH_CANDIDATES]),
}]), "00_additive_run_configuration.csv")
print("MODEL_PREP_DIR:", MODEL_PREP_DIR)
print("HAIL_OUTPUT_DIR:", HAIL_OUTPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% Cell 3
# ============================================================
# Block 1. Load model-prep CVD/CRP analytic cohorts
# ============================================================
def read_csv_optional(path, label):
    row = {"run_id": RUN_ID, "label": label, "path": str(path), "exists": path.exists(), "rows": 0, "cols": 0, "status": "missing"}
    if not path.exists():
        return pd.DataFrame(), row
    try:
        df = pd.read_csv(path)
        row.update({"rows": len(df), "cols": len(df.columns), "status": "loaded"})
        return df, row
    except Exception:
        row.update({"status": "read_error", "detail": traceback.format_exc()[:2000]})
        return pd.DataFrame(), row

cvd, cvd_input = read_csv_optional(MODEL_PREP_DIR / "02_cvd_model_dataset_v9.csv", "CVD model-prep dataset")
crp, crp_input = read_csv_optional(MODEL_PREP_DIR / "03_crp_model_dataset_v9.csv", "CRP model-prep dataset")
save(pd.DataFrame([cvd_input, crp_input]), "01_additive_input_dataset_status.csv")

for name, df in [("cvd", cvd), ("crp", crp)]:
    if not df.empty and "person_id" in df.columns:
        df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
    elif not df.empty:
        add_status(f"{name}_person_id", "missing", "person_id column not found")

analytic_ids = set()
for df in [cvd, crp]:
    if not df.empty and "person_id" in df.columns:
        analytic_ids.update([str(int(x)) for x in df["person_id"].dropna().unique()])

if not analytic_ids:
    add_status("load_model_prep_inputs", "blocked", "No analytic person_id found. Check Q3_VDR_MODEL_OUTPUT_DIR and model-prep outputs.")
else:
    add_status("load_model_prep_inputs", "pass", f"cvd_rows={len(cvd)}; crp_rows={len(crp)}; analytic_ids={len(analytic_ids)}")

save(pd.DataFrame([{
    "run_id": RUN_ID,
    "cvd_rows": len(cvd),
    "crp_rows": len(crp),
    "analytic_person_id_union_n": len(analytic_ids),
    "cvd_has_person_id": (not cvd.empty and "person_id" in cvd.columns),
    "crp_has_person_id": (not crp.empty and "person_id" in crp.columns),
}]), "02_additive_analytic_person_id_summary.csv")

# %% Cell 4
# ============================================================
# Block 2. Get person-level additive genotype dosage
# ============================================================
def normalize_genotype_wide(df):
    if df.empty:
        return df
    d = df.copy()
    if "person_id" not in d.columns:
        if "sample_id" in d.columns:
            d["person_id"] = pd.to_numeric(d["sample_id"], errors="coerce").astype("Int64")
        else:
            d["person_id"] = pd.Series([pd.NA] * len(d), dtype="Int64")
    else:
        d["person_id"] = pd.to_numeric(d["person_id"], errors="coerce").astype("Int64")
    for rsid in TIER1_RSIDS:
        if rsid in d.columns and f"{rsid}_alt_dosage" not in d.columns:
            d[f"{rsid}_alt_dosage"] = pd.to_numeric(d[rsid], errors="coerce")
        elif f"{rsid}_alt_dosage" in d.columns:
            d[f"{rsid}_alt_dosage"] = pd.to_numeric(d[f"{rsid}_alt_dosage"], errors="coerce")
    keep = ["person_id"] + (["sample_id"] if "sample_id" in d.columns else []) + [f"{rsid}_alt_dosage" for rsid in TIER1_RSIDS if f"{rsid}_alt_dosage" in d.columns]
    return d[keep].drop_duplicates(subset=["person_id"]).copy()

def find_existing_genotype_wide():
    manual = os.environ.get("Q3_VDR_15_GENOTYPE_WIDE_PATH", "").strip()
    candidates = []
    if manual:
        candidates.append(Path(manual).expanduser())
    candidates.extend([HAIL_OUTPUT_DIR / "14_tier1_vdr_genotype_wide.csv", OUTPUT_DIR / "15_tier1_vdr_genotype_wide_analytic.csv"])
    for p in candidates:
        if p.exists():
            return p
    return None

genotype_source = "none"
genotype_wide = pd.DataFrame()
genotype_long = pd.DataFrame()
existing_wide_path = find_existing_genotype_wide() if USE_EXISTING_GENOTYPE_WIDE else None

if existing_wide_path is not None:
    try:
        genotype_wide = normalize_genotype_wide(pd.read_csv(existing_wide_path))
        genotype_source = f"existing_wide:{existing_wide_path}"
        add_status("load_existing_genotype_wide", "pass", f"rows={len(genotype_wide)}; path={existing_wide_path}")
    except Exception:
        add_status("load_existing_genotype_wide", "fail", traceback.format_exc()[:2000])

if genotype_wide.empty and RUN_TARGETED_HAIL_EXTRACTION and analytic_ids:
    try:
        import hail as hl
        add_status("import_hail", "pass", getattr(hl, "__version__", ""))
        spark_conf = {}
        if GOOGLE_PROJECT:
            spark_conf.update({
                "spark.hadoop.fs.gs.requester.pays.mode": "CUSTOM",
                "spark.hadoop.fs.gs.requester.pays.project.id": GOOGLE_PROJECT,
                "spark.hadoop.fs.gs.requester.pays.buckets": "vwb-aou-datasets-controlled",
            })
        init_kwargs = {"spark_conf": spark_conf}
        if HAIL_TMP_DIR:
            init_kwargs["tmp_dir"] = HAIL_TMP_DIR
        hl.init(**init_kwargs)
        try:
            hl.default_reference("GRCh38")
        except Exception:
            pass
        add_status("hail_init", "pass", f"tmp_dir={HAIL_TMP_DIR}; requester_project={GOOGLE_PROJECT}")

        mt = None
        selected_source = ""
        selected_path = ""
        for source, path in HAIL_PATH_CANDIDATES:
            try:
                mt = hl.read_matrix_table(path)
                selected_source = source
                selected_path = path
                add_status("read_matrix_table", "pass", f"{source}={path}")
                break
            except Exception:
                add_status("read_matrix_table", "fail", f"{source}={path}\n" + traceback.format_exc()[:2000])
        if mt is None:
            raise RuntimeError("No Hail MatrixTable could be read.")

        target_expr = None
        for snp in TARGET_SNPS:
            contig_match = (mt.locus.contig == snp["contig"]) | (mt.locus.contig == ("chr" + snp["contig"]))
            one = contig_match & (mt.locus.position == snp["position"]) & (mt.alleles[0] == snp["ref"]) & (mt.alleles[1] == snp["alt"])
            target_expr = one if target_expr is None else (target_expr | one)
        mt_target = mt.filter_rows(target_expr)

        rsid_case = hl.case()
        vid_case = hl.case()
        for snp in TARGET_SNPS:
            contig_match = (mt_target.locus.contig == snp["contig"]) | (mt_target.locus.contig == ("chr" + snp["contig"]))
            cond = contig_match & (mt_target.locus.position == snp["position"]) & (mt_target.alleles[0] == snp["ref"]) & (mt_target.alleles[1] == snp["alt"])
            rsid_case = rsid_case.when(cond, snp["rsid"])
            vid_case = vid_case.when(cond, snp["expected_vid"])
        mt_target = mt_target.annotate_rows(target_rsid=rsid_case.default("unmatched"), expected_vid=vid_case.default("unmatched"))
        mt_target = mt_target.filter_rows(mt_target.target_rsid != "unmatched")
        add_status("target_row_filter", "pass", f"target_rows={mt_target.count_rows()}")

        entry_fields = set(mt_target.entry)
        if "GT" in entry_fields:
            mt_target = mt_target.annotate_entries(q3_alt_dosage=hl.or_missing(hl.is_defined(mt_target.GT), mt_target.GT.n_alt_alleles()))
            genotype_field = "GT.n_alt_alleles"
        elif "LGT" in entry_fields:
            mt_target = mt_target.annotate_entries(q3_alt_dosage=hl.or_missing(hl.is_defined(mt_target.LGT), mt_target.LGT.n_alt_alleles()))
            genotype_field = "LGT.n_alt_alleles"
        elif "DS" in entry_fields:
            mt_target = mt_target.annotate_entries(q3_alt_dosage=hl.float64(mt_target.DS))
            genotype_field = "DS"
        elif "dosage" in entry_fields:
            mt_target = mt_target.annotate_entries(q3_alt_dosage=hl.float64(mt_target.dosage))
            genotype_field = "dosage"
        else:
            raise RuntimeError(f"No supported genotype dosage field found: {sorted(entry_fields)}")
        add_status("define_dosage", "pass", genotype_field)

        cols_table = mt_target.cols()
        sample_source = ""
        for field in ["s", "sample_id", "person_id", "sample_name"]:
            if field in cols_table.row:
                sample_source = field
                break
        if not sample_source:
            col_key_fields = list(mt_target.col_key)
            sample_source = col_key_fields[0] if col_key_fields else ""
        if not sample_source:
            raise RuntimeError("Could not identify Hail sample/person column.")
        mt_target = mt_target.filter_cols(hl.literal(set(analytic_ids)).contains(hl.str(mt_target[sample_source])))
        add_status("filter_cols_to_analytic_person_ids", "pass", f"sample_source={sample_source}; cols={mt_target.count_cols()}; analytic_ids={len(analytic_ids)}")

        entries = mt_target.entries()
        if sample_source not in entries.row:
            raise RuntimeError(f"Sample source {sample_source} not present after entries().")
        genotype_long = entries.select(
            sample_id=hl.str(entries[sample_source]),
            rsid=entries.target_rsid,
            expected_vid=entries.expected_vid,
            q3_alt_dosage=entries.q3_alt_dosage,
        ).to_pandas()
        genotype_long["person_id"] = pd.to_numeric(genotype_long["sample_id"], errors="coerce").astype("Int64")
        genotype_long["q3_alt_dosage"] = pd.to_numeric(genotype_long["q3_alt_dosage"], errors="coerce")
        save(genotype_long, "15_tier1_vdr_genotype_long_analytic.csv")
        tmp = genotype_long.pivot_table(index=["sample_id", "person_id"], columns="rsid", values="q3_alt_dosage", aggfunc="first").reset_index()
        tmp.columns.name = None
        genotype_wide = normalize_genotype_wide(tmp)
        save(genotype_wide, "15_tier1_vdr_genotype_wide_analytic.csv")
        genotype_source = f"targeted_hail:{selected_source}={selected_path}; {genotype_field}"
        add_status("targeted_hail_genotype_export", "pass", f"long_rows={len(genotype_long)}; wide_rows={len(genotype_wide)}")
    except Exception:
        tb = traceback.format_exc()
        save_text(tb, "15_targeted_hail_genotype_extraction_failure.txt")
        add_status("targeted_hail_genotype_export", "fail", tb[:2000])

if genotype_wide.empty:
    add_status("genotype_available_for_modeling", "blocked", "No person-level additive genotype wide data available.")
else:
    genotype_wide = normalize_genotype_wide(genotype_wide)
    add_status("genotype_available_for_modeling", "pass", f"rows={len(genotype_wide)}; source={genotype_source}")

save(pd.DataFrame(status_rows), "15_additive_extraction_status.csv")
save(pd.DataFrame([{
    "run_id": RUN_ID,
    "genotype_source": genotype_source,
    "genotype_wide_rows": len(genotype_wide),
    "has_all_four_dosage_columns": all(f"{rsid}_alt_dosage" in genotype_wide.columns for rsid in TIER1_RSIDS),
}]), "03_additive_genotype_source_status.csv")
if genotype_wide.empty and STOP_IF_NO_GENOTYPE:
    raise RuntimeError("No person-level genotype is available for additive modeling.")

# %% Cell 5
# ============================================================
# Block 3. Merge genotype dosage and summarize analytic QC
# ============================================================
def add_dosage_terms(df):
    if df.empty or genotype_wide.empty or "person_id" not in df.columns:
        return df.copy()
    gcols = ["person_id"] + [f"{rsid}_alt_dosage" for rsid in TIER1_RSIDS if f"{rsid}_alt_dosage" in genotype_wide.columns]
    d = df.merge(genotype_wide[gcols], on="person_id", how="left", validate="many_to_one")
    for col in ["vitd_lt20", "log_vitd", "vitd_per10", "age_at_index", "female", "winter", "bmi_full", "follow_up_years", "event_mi_or_stroke", "log_crp"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    for rsid in TIER1_RSIDS:
        dosage = f"{rsid}_alt_dosage"
        if dosage in d.columns:
            d[dosage] = pd.to_numeric(d[dosage], errors="coerce")
            d[f"{rsid}_alt_carrier_from_dosage"] = np.where(d[dosage].notna(), (d[dosage] > 0).astype(int), np.nan)
            d[f"{rsid}_x_vitd_lt20_additive"] = d[dosage] * d["vitd_lt20"] if "vitd_lt20" in d.columns else np.nan
            d[f"{rsid}_x_log_vitd_additive"] = d[dosage] * d["log_vitd"] if "log_vitd" in d.columns else np.nan
    return d

cvd_add = add_dosage_terms(cvd)
crp_add = add_dosage_terms(crp)
if not cvd_add.empty:
    save(cvd_add, "04_cvd_additive_genotype_model_dataset.csv")
if not crp_add.empty:
    save(crp_add, "05_crp_additive_genotype_model_dataset.csv")

join_rows = []
concordance_rows = []
qc_rows = []
cell_rows = []
for dataset_name, d in [("cvd", cvd_add), ("crp", crp_add)]:
    if d.empty:
        join_rows.append({"run_id": RUN_ID, "dataset": dataset_name, "rows": 0, "status": "missing_dataset"})
        continue
    join_row = {"run_id": RUN_ID, "dataset": dataset_name, "rows": len(d), "status": "merged" if not genotype_wide.empty else "no_genotype"}
    for rsid in TIER1_RSIDS:
        dosage = f"{rsid}_alt_dosage"
        if dosage not in d.columns:
            join_row[f"{rsid}_dosage_nonmissing_n"] = 0
            qc_rows.append({"run_id": RUN_ID, "dataset": dataset_name, "rsid": rsid, "status": "missing_dosage_col"})
            continue
        x = pd.to_numeric(d[dosage], errors="coerce")
        called = x.dropna()
        join_row[f"{rsid}_dosage_nonmissing_n"] = int(x.notna().sum())
        join_row[f"{rsid}_dosage_nonmissing_pct"] = pct(int(x.notna().sum()), len(d))
        n_called = int(len(called))
        alt_af = float(called.sum() / (2 * n_called)) if n_called else np.nan
        qc_rows.append({
            "run_id": RUN_ID, "dataset": dataset_name, "rsid": rsid,
            "status": "pass" if n_called else "no_called_genotype",
            "n_total": len(d), "n_called": n_called, "n_missing": int(x.isna().sum()),
            "missing_pct": pct(int(x.isna().sum()), len(d)),
            "genotype_0_ref_ref": int((called == 0).sum()),
            "genotype_1_ref_alt": int((called == 1).sum()),
            "genotype_2_alt_alt": int((called == 2).sum()),
            "alt_af": alt_af, "maf": min(alt_af, 1 - alt_af) if pd.notna(alt_af) else np.nan,
        })
        if "vitd_lt20" in d.columns:
            tmp = d.assign(alt_dosage=pd.to_numeric(d[dosage], errors="coerce"))
            for (vitd_lt20, alt_dosage), sub in tmp.groupby(["vitd_lt20", "alt_dosage"], dropna=False):
                row = {"run_id": RUN_ID, "dataset": dataset_name, "rsid": rsid, "vitd_lt20": vitd_lt20, "alt_dosage": alt_dosage, "n": len(sub)}
                if dataset_name == "cvd" and "event_mi_or_stroke" in sub.columns:
                    row["events"] = int(pd.to_numeric(sub["event_mi_or_stroke"], errors="coerce").fillna(0).sum())
                cell_rows.append(row)
        old = f"{rsid}_carrier"
        new = f"{rsid}_alt_carrier_from_dosage"
        if old in d.columns and new in d.columns:
            comp = d[[old, new]].dropna().copy()
            if len(comp):
                mismatch = int((pd.to_numeric(comp[old], errors="coerce").astype(int) != pd.to_numeric(comp[new], errors="coerce").astype(int)).sum())
                concordance_rows.append({"run_id": RUN_ID, "dataset": dataset_name, "rsid": rsid, "compared_n": len(comp), "mismatch_n": mismatch, "mismatch_pct": pct(mismatch, len(comp))})
    join_rows.append(join_row)

save(pd.DataFrame(join_rows), "06_additive_genotype_join_summary.csv")
save(pd.DataFrame(concordance_rows), "07_dosage_vs_existing_carrier_concordance.csv")
save(pd.DataFrame(qc_rows), "08_additive_genotype_analytic_qc.csv")
save(pd.DataFrame(cell_rows), "09_additive_vitd_dosage_cell_counts.csv")

# %% Cell 6
# ============================================================
# Block 4. Fit additive CVD Cox and CRP OLS models
# ============================================================
statsmodels_available = False
phreg_available = False
try:
    import statsmodels.api as sm
    from statsmodels.duration.hazard_regression import PHReg
    statsmodels_available = True
    phreg_available = True
except Exception:
    sm = None
    PHReg = None
    save_text(traceback.format_exc(), "15_statsmodels_import_failure.txt")
save(pd.DataFrame([{"run_id": RUN_ID, "package": "statsmodels", "available": statsmodels_available, "phreg_available": phreg_available, "python": sys.version}]), "10_additive_model_package_status.csv")

def complete_case(df, needed, require_positive_followup=False):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame(), "missing_terms", ";".join(missing)
    d = df.dropna(subset=needed).copy()
    if require_positive_followup:
        d = d[pd.to_numeric(d["follow_up_years"], errors="coerce") > 0].copy()
    return d, "ready" if len(d) else "no_complete_cases", ""

def normal_p(z):
    return math.erfc(abs(float(z)) / math.sqrt(2)) if pd.notna(z) else np.nan

def tidy_phreg(res, terms):
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    conf = np.asarray(res.conf_int(), dtype=float)
    pvalues = getattr(res, "pvalues", None)
    if pvalues is None:
        pvalues = [normal_p(p / s) if s else np.nan for p, s in zip(params, bse)]
    rows = []
    for i, term in enumerate(terms):
        coef = params[i]
        se = bse[i]
        lo = conf[i, 0] if conf.ndim == 2 else coef - 1.96 * se
        hi = conf[i, 1] if conf.ndim == 2 else coef + 1.96 * se
        rows.append({"term": term, "coef": coef, "se": se, "z_or_t": coef / se if se else np.nan, "p": float(pvalues[i]) if i < len(pvalues) else np.nan, "ci_lower_coef": lo, "ci_upper_coef": hi, "hr": math.exp(coef), "hr_ci_lower": math.exp(lo), "hr_ci_upper": math.exp(hi)})
    return pd.DataFrame(rows)

def tidy_ols(res, terms):
    rows = []
    for term in terms:
        if term not in res.params.index:
            continue
        coef = float(res.params[term])
        se = float(res.bse[term])
        lo = coef - 1.96 * se
        hi = coef + 1.96 * se
        rows.append({"term": term, "coef": coef, "se": se, "z_or_t": float(res.tvalues[term]), "p": float(res.pvalues[term]), "ci_lower_coef": lo, "ci_upper_coef": hi, "exp_coef": math.exp(coef), "percent_change": (math.exp(coef) - 1) * 100})
    return pd.DataFrame(rows)

specs = []
for rsid in TIER1_RSIDS:
    dosage = f"{rsid}_alt_dosage"
    ix_bin = f"{rsid}_x_vitd_lt20_additive"
    ix_log = f"{rsid}_x_log_vitd_additive"
    specs.extend([
        {"analysis_family": "cvd_cox", "analysis_role": "primary_additive", "model_name": "cvd_binary_additive_core_with_bmi", "dataset": "cvd", "rsid": rsid, "exposure": "vitd_lt20", "dosage": dosage, "interaction": ix_bin, "terms": ["vitd_lt20", dosage, ix_bin, "age_at_index", "female", "winter", "bmi_full"]},
        {"analysis_family": "cvd_cox", "analysis_role": "missingness_sensitivity", "model_name": "cvd_binary_additive_core_no_bmi", "dataset": "cvd", "rsid": rsid, "exposure": "vitd_lt20", "dosage": dosage, "interaction": ix_bin, "terms": ["vitd_lt20", dosage, ix_bin, "age_at_index", "female", "winter"]},
        {"analysis_family": "cvd_cox", "analysis_role": "continuous_vitd_sensitivity", "model_name": "cvd_logvitd_additive_core_with_bmi", "dataset": "cvd", "rsid": rsid, "exposure": "log_vitd", "dosage": dosage, "interaction": ix_log, "terms": ["log_vitd", dosage, ix_log, "age_at_index", "female", "winter", "bmi_full"]},
        {"analysis_family": "crp_ols", "analysis_role": "crp_extension", "model_name": "crp_loglog_additive_core_with_bmi", "dataset": "crp", "rsid": rsid, "exposure": "log_vitd", "dosage": dosage, "interaction": ix_log, "terms": ["log_vitd", dosage, ix_log, "age_at_index", "female", "winter", "bmi_full"], "outcome": "log_crp"},
    ])
save(pd.DataFrame([{**s, "terms": ";".join(s["terms"])} for s in specs]), "11_additive_model_specifications.csv")

term_rows = []
fit_rows = []
for spec in specs:
    d0 = cvd_add if spec["dataset"] == "cvd" else crp_add
    status = {"run_id": RUN_ID, "analysis_family": spec["analysis_family"], "analysis_role": spec["analysis_role"], "model_name": spec["model_name"], "rsid": spec["rsid"], "status": "not_run", "detail": "", "n": 0, "events": np.nan}
    if d0.empty:
        status.update({"status": "dataset_missing", "detail": spec["dataset"]})
        fit_rows.append(status)
        continue
    if not statsmodels_available:
        status.update({"status": "package_missing", "detail": "statsmodels"})
        fit_rows.append(status)
        continue
    if spec["analysis_family"] == "cvd_cox":
        needed = ["follow_up_years", "event_mi_or_stroke"] + spec["terms"]
        d, cc_status, detail = complete_case(d0, needed, True)
        events = int(pd.to_numeric(d.get("event_mi_or_stroke", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not d.empty else 0
        status.update({"status": cc_status, "detail": detail, "n": len(d), "events": events})
        if cc_status != "ready" or len(d) < MIN_MODEL_N or events < MIN_CVD_EVENTS or not phreg_available:
            if cc_status == "ready":
                status.update({"status": "too_few_rows_or_events_or_phreg_missing", "detail": f"n={len(d)}; events={events}; phreg={phreg_available}"})
            fit_rows.append(status)
            continue
        try:
            X = d[spec["terms"]].astype(float)
            res = PHReg(endog=pd.to_numeric(d["follow_up_years"], errors="coerce"), exog=X, status=pd.to_numeric(d["event_mi_or_stroke"], errors="coerce").astype(int), ties="efron").fit()
            out = tidy_phreg(res, spec["terms"])
            out["hr_ratio_or_exp_coef"] = out["hr"]
            out["run_id"] = RUN_ID
            out["analysis_family"] = spec["analysis_family"]
            out["analysis_role"] = spec["analysis_role"]
            out["model_name"] = spec["model_name"]
            out["rsid"] = spec["rsid"]
            out["n"] = len(d)
            out["events"] = events
            out["dosage"] = spec["dosage"]
            out["interaction"] = spec["interaction"]
            term_rows.append(out)
            status.update({"status": "pass", "detail": ""})
        except Exception:
            status.update({"status": "fit_error", "detail": traceback.format_exc()[:2000]})
        fit_rows.append(status)
    else:
        needed = [spec["outcome"]] + spec["terms"]
        d, cc_status, detail = complete_case(d0, needed, False)
        status.update({"status": cc_status, "detail": detail, "n": len(d), "events": np.nan})
        if cc_status != "ready" or len(d) < MIN_MODEL_N:
            fit_rows.append(status)
            continue
        try:
            X = sm.add_constant(d[spec["terms"]].astype(float), has_constant="add")
            y = pd.to_numeric(d[spec["outcome"]], errors="coerce").astype(float)
            res = sm.OLS(y, X).fit(cov_type="HC3")
            out = tidy_ols(res, spec["terms"])
            out["hr_ratio_or_exp_coef"] = out["exp_coef"]
            out["run_id"] = RUN_ID
            out["analysis_family"] = spec["analysis_family"]
            out["analysis_role"] = spec["analysis_role"]
            out["model_name"] = spec["model_name"]
            out["rsid"] = spec["rsid"]
            out["n"] = len(d)
            out["events"] = np.nan
            out["dosage"] = spec["dosage"]
            out["interaction"] = spec["interaction"]
            term_rows.append(out)
            status.update({"status": "pass", "detail": ""})
        except Exception:
            status.update({"status": "fit_error", "detail": traceback.format_exc()[:2000]})
        fit_rows.append(status)

fit_status = pd.DataFrame(fit_rows)
term_results = pd.concat(term_rows, ignore_index=True) if term_rows else pd.DataFrame()
save(fit_status, "12_additive_model_fit_status.csv")
save(term_results, "13_additive_all_model_term_results.csv")
if not term_results.empty:
    primary_ix = term_results[(term_results["analysis_family"] == "cvd_cox") & (term_results["analysis_role"] == "primary_additive") & (term_results["term"] == term_results["interaction"])].copy()
    if not primary_ix.empty:
        p = pd.to_numeric(primary_ix["p"], errors="coerce")
        primary_ix["p_bonferroni_4"] = np.minimum(p * len(TIER1_RSIDS), 1.0)
        primary_ix["multiple_testing_family"] = "four Tier 1 VDR additive ALT-dosage primary interaction tests"
else:
    primary_ix = pd.DataFrame()
save(primary_ix, "14_additive_primary_interaction_results.csv")

# %% Cell 7
# ============================================================
# Block 4B. R fallback for model fitting if statsmodels/scipy is broken
# ============================================================
import shutil
import subprocess

primary_pass_current = 0
if "fit_status" in globals() and not fit_status.empty and "analysis_role" in fit_status and "status" in fit_status:
    primary_pass_current = int(((fit_status["analysis_role"] == "primary_additive") & (fit_status["status"] == "pass")).sum())

rscript_path = shutil.which("Rscript")
need_r_fallback = primary_pass_current < len(TIER1_RSIDS)
save(pd.DataFrame([{"run_id": RUN_ID, "rscript_path": rscript_path or "", "need_r_fallback": need_r_fallback}]), "10b_additive_r_fallback_status.csv")

if need_r_fallback and rscript_path:
    r_code = r'''
run_id <- Sys.getenv("Q3_VDR_RUN_ID")
out_dir <- Sys.getenv("Q3_VDR_R_FALLBACK_OUTPUT_DIR")
rsids <- strsplit(Sys.getenv("Q3_VDR_R_FALLBACK_TIER1"), ";", fixed=TRUE)[[1]]
min_n <- as.integer(Sys.getenv("Q3_VDR_15_MIN_MODEL_N", "100"))
min_events <- as.integer(Sys.getenv("Q3_VDR_15_MIN_CVD_EVENTS", "50"))
read_opt <- function(filename) {
  path <- file.path(out_dir, filename)
  if (!file.exists(path)) return(data.frame())
  read.csv(path, check.names=FALSE, stringsAsFactors=FALSE)
}
cvd <- read_opt("04_cvd_additive_genotype_model_dataset.csv")
crp <- read_opt("05_crp_additive_genotype_model_dataset.csv")
status_rows <- list()
term_rows <- list()
add_status <- function(fam, role, model_name, rsid, status, detail, n=0, events=NA, backend="R") {
  status_rows[[length(status_rows)+1]] <<- data.frame(run_id=run_id, analysis_family=fam, analysis_role=role, model_name=model_name, rsid=rsid, status=status, detail=detail, n=n, events=events, backend=backend, stringsAsFactors=FALSE)
}
append_term <- function(row) term_rows[[length(term_rows)+1]] <<- row
complete_data <- function(d, needed, positive_followup=FALSE) {
  miss <- setdiff(needed, names(d))
  if (length(miss) > 0) return(list(data=data.frame(), status="missing_terms", detail=paste(miss, collapse=";")))
  x <- d[complete.cases(d[, needed, drop=FALSE]), , drop=FALSE]
  if (positive_followup) x <- x[as.numeric(x$follow_up_years) > 0, , drop=FALSE]
  if (nrow(x) == 0) return(list(data=x, status="no_complete_cases", detail=""))
  list(data=x, status="ready", detail="")
}
fit_cox <- function(d0, role, model_name, rsid, exposure, dosage, interaction, terms) {
  if (nrow(d0) == 0) { add_status("cvd_cox", role, model_name, rsid, "dataset_missing", "cvd"); return() }
  if (!requireNamespace("survival", quietly=TRUE)) { add_status("cvd_cox", role, model_name, rsid, "package_missing", "survival"); return() }
  cc <- complete_data(d0, c("follow_up_years", "event_mi_or_stroke", terms), TRUE)
  ev <- if (nrow(cc$data) > 0) sum(as.numeric(cc$data$event_mi_or_stroke), na.rm=TRUE) else 0
  if (cc$status != "ready" || nrow(cc$data) < min_n || ev < min_events) { add_status("cvd_cox", role, model_name, rsid, ifelse(cc$status == "ready", "too_few_rows_or_events", cc$status), cc$detail, nrow(cc$data), ev, "R survival::coxph"); return() }
  f <- as.formula(paste("survival::Surv(follow_up_years, event_mi_or_stroke) ~", paste(terms, collapse=" + ")))
  fit <- tryCatch(survival::coxph(f, data=cc$data, ties="efron", x=FALSE, model=FALSE), error=function(e) e)
  if (inherits(fit, "error")) { add_status("cvd_cox", role, model_name, rsid, "fit_error", conditionMessage(fit), nrow(cc$data), ev, "R survival::coxph"); return() }
  co <- summary(fit)$coefficients
  for (term in rownames(co)) {
    coef <- as.numeric(co[term, "coef"]); se <- as.numeric(co[term, "se(coef)"]); lo <- coef - 1.96*se; hi <- coef + 1.96*se
    append_term(data.frame(run_id=run_id, term=term, coef=coef, se=se, z_or_t=as.numeric(co[term, "z"]), p=as.numeric(co[term, "Pr(>|z|)"]), ci_lower_coef=lo, ci_upper_coef=hi, hr=exp(coef), hr_ci_lower=exp(lo), hr_ci_upper=exp(hi), hr_ratio_or_exp_coef=exp(coef), analysis_family="cvd_cox", analysis_role=role, model_name=model_name, rsid=rsid, n=nrow(cc$data), events=ev, dosage=dosage, interaction=interaction, backend="R survival::coxph", stringsAsFactors=FALSE))
  }
  add_status("cvd_cox", role, model_name, rsid, "pass", "", nrow(cc$data), ev, "R survival::coxph")
}
fit_lm <- function(d0, role, model_name, rsid, exposure, dosage, interaction, terms) {
  if (nrow(d0) == 0) { add_status("crp_ols", role, model_name, rsid, "dataset_missing", "crp"); return() }
  cc <- complete_data(d0, c("log_crp", terms), FALSE)
  if (cc$status != "ready" || nrow(cc$data) < min_n) { add_status("crp_ols", role, model_name, rsid, ifelse(cc$status == "ready", "too_few_rows", cc$status), cc$detail, nrow(cc$data), NA, "R stats::lm"); return() }
  f <- as.formula(paste("log_crp ~", paste(terms, collapse=" + ")))
  fit <- tryCatch(stats::lm(f, data=cc$data), error=function(e) e)
  if (inherits(fit, "error")) { add_status("crp_ols", role, model_name, rsid, "fit_error", conditionMessage(fit), nrow(cc$data), NA, "R stats::lm"); return() }
  co <- summary(fit)$coefficients
  for (term in setdiff(rownames(co), "(Intercept)")) {
    coef <- as.numeric(co[term, "Estimate"]); se <- as.numeric(co[term, "Std. Error"]); lo <- coef - 1.96*se; hi <- coef + 1.96*se
    append_term(data.frame(run_id=run_id, term=term, coef=coef, se=se, z_or_t=as.numeric(co[term, "t value"]), p=as.numeric(co[term, "Pr(>|t|)"]), ci_lower_coef=lo, ci_upper_coef=hi, exp_coef=exp(coef), percent_change=(exp(coef)-1)*100, hr_ratio_or_exp_coef=exp(coef), analysis_family="crp_ols", analysis_role=role, model_name=model_name, rsid=rsid, n=nrow(cc$data), events=NA, dosage=dosage, interaction=interaction, backend="R stats::lm", stringsAsFactors=FALSE))
  }
  add_status("crp_ols", role, model_name, rsid, "pass", "", nrow(cc$data), NA, "R stats::lm")
}
for (rsid in rsids) {
  dosage <- paste0(rsid, "_alt_dosage"); ix_bin <- paste0(rsid, "_x_vitd_lt20_additive"); ix_log <- paste0(rsid, "_x_log_vitd_additive")
  fit_cox(cvd, "primary_additive", "cvd_binary_additive_core_with_bmi", rsid, "vitd_lt20", dosage, ix_bin, c("vitd_lt20", dosage, ix_bin, "age_at_index", "female", "winter", "bmi_full"))
  fit_cox(cvd, "missingness_sensitivity", "cvd_binary_additive_core_no_bmi", rsid, "vitd_lt20", dosage, ix_bin, c("vitd_lt20", dosage, ix_bin, "age_at_index", "female", "winter"))
  fit_cox(cvd, "continuous_vitd_sensitivity", "cvd_logvitd_additive_core_with_bmi", rsid, "log_vitd", dosage, ix_log, c("log_vitd", dosage, ix_log, "age_at_index", "female", "winter", "bmi_full"))
  fit_lm(crp, "crp_extension", "crp_loglog_additive_core_with_bmi", rsid, "log_vitd", dosage, ix_log, c("log_vitd", dosage, ix_log, "age_at_index", "female", "winter", "bmi_full"))
}
status_df <- if (length(status_rows)) do.call(rbind, status_rows) else data.frame()
term_df <- if (length(term_rows)) do.call(rbind, term_rows) else data.frame()
primary_df <- if (nrow(term_df)) term_df[term_df$analysis_family == "cvd_cox" & term_df$analysis_role == "primary_additive" & term_df$term == term_df$interaction, , drop=FALSE] else data.frame()
if (nrow(primary_df)) { primary_df$p_bonferroni_4 <- pmin(as.numeric(primary_df$p) * length(rsids), 1); primary_df$multiple_testing_family <- "four Tier 1 VDR additive ALT-dosage primary interaction tests" }
write.csv(status_df, file.path(out_dir, "12_additive_model_fit_status.csv"), row.names=FALSE, na="")
write.csv(term_df, file.path(out_dir, "13_additive_all_model_term_results.csv"), row.names=FALSE, na="")
write.csv(primary_df, file.path(out_dir, "14_additive_primary_interaction_results.csv"), row.names=FALSE, na="")
'''
    env = os.environ.copy()
    env["Q3_VDR_RUN_ID"] = RUN_ID
    env["Q3_VDR_R_FALLBACK_OUTPUT_DIR"] = str(OUTPUT_DIR)
    env["Q3_VDR_R_FALLBACK_TIER1"] = ";".join(TIER1_RSIDS)
    r_run = subprocess.run([rscript_path, "-"], input=r_code, text=True, capture_output=True, env=env, timeout=3600)
    save_text(r_run.stdout, "10b_additive_r_fallback_stdout.txt")
    save_text(r_run.stderr, "10b_additive_r_fallback_stderr.txt")
    save(pd.DataFrame([{"run_id": RUN_ID, "r_returncode": r_run.returncode, "rscript_path": rscript_path}]), "10c_additive_r_fallback_run_status.csv")
    if r_run.returncode == 0:
        fit_status = pd.read_csv(OUTPUT_DIR / "12_additive_model_fit_status.csv") if (OUTPUT_DIR / "12_additive_model_fit_status.csv").exists() else pd.DataFrame()
        term_results = pd.read_csv(OUTPUT_DIR / "13_additive_all_model_term_results.csv") if (OUTPUT_DIR / "13_additive_all_model_term_results.csv").exists() else pd.DataFrame()
        primary_ix = pd.read_csv(OUTPUT_DIR / "14_additive_primary_interaction_results.csv") if (OUTPUT_DIR / "14_additive_primary_interaction_results.csv").exists() else pd.DataFrame()
        add_status("r_model_fallback", "pass", f"primary_rows={len(primary_ix)}")
    else:
        add_status("r_model_fallback", "fail", r_run.stderr[:2000])
elif need_r_fallback:
    add_status("r_model_fallback", "blocked", "Rscript not found and Python statsmodels model fitting did not complete.")

# %% Cell 8
# ============================================================
# Block 5. Decision note and current-run manifest
# ============================================================
primary_status = fit_status[fit_status["analysis_role"].eq("primary_additive")].copy() if "fit_status" in globals() and not fit_status.empty else pd.DataFrame()
primary_pass_n = int(primary_status["status"].eq("pass").sum()) if not primary_status.empty else 0
all_four_cols = (not genotype_wide.empty) and all(f"{rsid}_alt_dosage" in genotype_wide.columns for rsid in TIER1_RSIDS)
merged_cvd_ready = (not cvd_add.empty) and all(f"{rsid}_alt_dosage" in cvd_add.columns for rsid in TIER1_RSIDS)
merged_crp_ready = (not crp_add.empty) and all(f"{rsid}_alt_dosage" in crp_add.columns for rsid in TIER1_RSIDS)
decision = pd.DataFrame([
    {"run_id": RUN_ID, "question": "Is person-level additive ALT dosage available for all four SNPs?", "status": "yes" if all_four_cols else "no", "evidence": "03_additive_genotype_source_status.csv"},
    {"run_id": RUN_ID, "question": "Was additive dosage joined to the CVD analytic cohort?", "status": "yes" if merged_cvd_ready else "no_or_missing_cvd", "evidence": "04_cvd_additive_genotype_model_dataset.csv; 06_additive_genotype_join_summary.csv"},
    {"run_id": RUN_ID, "question": "Was additive dosage joined to the CRP analytic cohort?", "status": "yes" if merged_crp_ready else "no_or_missing_crp", "evidence": "05_crp_additive_genotype_model_dataset.csv; 06_additive_genotype_join_summary.csv"},
    {"run_id": RUN_ID, "question": "Did all four primary additive CVD Cox models fit?", "status": "yes" if primary_pass_n == len(TIER1_RSIDS) else "no_or_partial", "evidence": "12_additive_model_fit_status.csv"},
    {"run_id": RUN_ID, "question": "Can ALT dosage be interpreted as functional/risk allele dosage?", "status": "no", "evidence": "Allele harmonization remains outside this notebook; ALT dosage is a coding label only."},
])
save(decision, "15_additive_modeling_decision_matrix.csv")

note = f"""# Q3 VDR Additive Genotype Analytic Modeling Note

Run ID: `{RUN_ID}`

## Input Status

- Model-prep dir: `{MODEL_PREP_DIR}`
- Layer 14 Hail output dir: `{HAIL_OUTPUT_DIR}`
- CVD rows loaded: {len(cvd):,}
- CRP rows loaded: {len(crp):,}
- Analytic person_id union: {len(analytic_ids):,}
- Genotype source: `{genotype_source}`
- Genotype wide rows: {len(genotype_wide):,}
- All four dosage columns present: {all_four_cols}

## Model Status

- Primary additive CVD Cox models passed: {primary_pass_n}/{len(TIER1_RSIDS)}

## Interpretation Rule

If all primary additive models pass, Q3 may report additive ALT dosage as a sensitivity layer. Use wording such as `per additional AoU ALT allele` or `ALT dosage`. Do not call the ALT allele functional, risk, vitamin-D-response, or causal unless a separate allele-harmonization step proves that mapping.
"""
save_text(note, "16_additive_modeling_interpretation_note.md")
manifest = pd.DataFrame([{"run_id": RUN_ID, "filename": p.name, "path": str(p), "size_bytes": p.stat().st_size} for p in written_outputs if p.exists()])
save(manifest, "99_additive_current_run_manifest.csv")
save(pd.DataFrame(status_rows), "15_additive_extraction_status.csv")
print("DONE")
print("Output dir:", OUTPUT_DIR)
