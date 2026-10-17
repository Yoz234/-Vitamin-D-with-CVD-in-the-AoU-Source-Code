# Source companion of 13_q3_vdr_aou_followup_stepwise_search_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None


# ============================================================
# Block 0. Setup
# ============================================================

DEFAULT_CDR = "YOUR_GOOGLE_PROJECT.C2025Q4R6"
OUTPUT_FOLDER_NAME = "Q3_VDR_AoU_lit_db_followup_outputs_2026_08_07"
TIER1_SNPS = [
    {"tier": "Tier 1", "gene": "VDR", "rsid": "rs1544410", "label": "BsmI", "expected_vid": "12-47846052-C-T"},
    {"tier": "Tier 1", "gene": "VDR", "rsid": "rs2228570", "label": "FokI", "expected_vid": "12-47879112-A-G"},
    {"tier": "Tier 1", "gene": "VDR", "rsid": "rs731236", "label": "TaqI", "expected_vid": "12-47844974-A-G"},
    {"tier": "Tier 1", "gene": "VDR", "rsid": "rs7975232", "label": "ApaI", "expected_vid": "12-47845054-C-A"},
]
CRP_CONCEPTS = [3020460, 3010156, 4208414, 46272908]
CRP_UNIT_FACTOR = {8751: 1.0, 4122414: 1.0, 8840: 10.0, 4121396: 10.0}


def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "data",
        cwd.parent / "data",
        Path("/home/jupyter/workspace") / "VDR",
        Path("/home/jupyter/workspace") / "VDR" / "data",
        Path("/home/jupyter/workspace") / "Q3_VDR" / "data",
    ]
    if Path("/home/jupyter/workspace").exists():
        candidates.extend(sorted(Path("/home/jupyter/workspace").glob("*/data")))
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            pass
    return cwd


def split_project_dataset(value):
    value = str(value).strip()
    return value.split(".", 1) if "." in value else ("", value)


CDR = os.environ.get("WORKSPACE_CDR", DEFAULT_CDR).strip()
CDR_PROJECT, CDR_DATASET = split_project_dataset(CDR)
DEFAULT_PREP_CDR = f"{CDR_PROJECT}.prep_{CDR_DATASET}" if CDR_PROJECT and not CDR_DATASET.startswith("prep_") else ""
PREP_CDR = os.environ.get("WORKSPACE_PREP_CDR", DEFAULT_PREP_CDR).strip()
DATA_DIR = Path(os.environ.get("Q3_VDR_DATA_DIR", str(infer_default_data_dir()))).expanduser()
OUTPUT_DIR = Path(os.environ.get("Q3_VDR_AOU_FOLLOWUP_OUTPUT_DIR", str(DATA_DIR / OUTPUT_FOLDER_NAME))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERWRITE_OUTPUTS = os.environ.get("Q3_VDR_OVERWRITE_OUTPUTS", "true").strip().lower() in {"1", "true", "yes", "y"}
REQUESTER_PAYS_PROJECT = os.environ.get("GOOGLE_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
EXTRA_BQ_DATASETS_RAW = os.environ.get("Q3_VDR_EXTRA_BQ_DATASETS", "").strip()
GENOTYPE_FILE = os.environ.get("Q3_VDR_TIER1_GENOTYPE_CSV", "").strip()

client = None
query_status_rows = []
if bigquery is not None:
    try:
        client = bigquery.Client()
        query_status_rows.append({"label": "bigquery_client", "status": "pass", "detail": ""})
    except Exception as exc:
        query_status_rows.append({"label": "bigquery_client", "status": "fail", "detail": repr(exc)[:2000]})


def save(df, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")
    df.to_csv(path, index=False)
    print(f"Saved {filename}: {len(df):,} rows")
    return path


def save_text(text, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")
    path.write_text(text, encoding="utf-8")
    print(f"Saved {filename}: {len(text):,} chars")
    return path


def q(sql, label):
    if client is None:
        query_status_rows.append({"label": label, "status": "skip_no_bigquery_client", "detail": ""})
        return pd.DataFrame()
    try:
        df = client.query(sql).to_dataframe()
        query_status_rows.append({"label": label, "status": "pass", "detail": f"n_rows={len(df)}"})
        print(f"Query pass [{label}]: {len(df):,} rows")
        return df
    except Exception as exc:
        query_status_rows.append({"label": label, "status": "fail", "detail": repr(exc)[:2000]})
        print(f"QUERY FAILED [{label}]: {repr(exc)[:700]}")
        return pd.DataFrame()


def bq_quote(value):
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def int_list(values):
    values = [int(v) for v in values]
    return "(" + ",".join(str(v) for v in values) + ")"


def classify_text(value):
    text = str(value).lower()
    tags = []
    if any(k in text for k in ["pgen", "pvar", "psam", "plink", ".bed", ".bim", ".fam", "bgen", ".sample", "vcf", "bcf", "hail", ".mt", "vds", "genotype", "dosage", "hard_call"]):
        tags.append("genotype_resource")
    if any(k in text for k in ["ancestry", "pca", "principal", "admixture", "pc1", "pc_1"]):
        tags.append("ancestry_pc")
    if any(k in text for k in ["related", "kinship", "unrelated"]):
        tags.append("relatedness")
    if any(k in text for k in ["qc", "flagged", "metrics", "callrate", "call_rate", "missingness"]):
        tags.append("genomic_qc")
    if any(k in text for k in ["crp", "c-reactive"]):
        tags.append("crp")
    return ";".join(sorted(set(tags)))


def run_args(args, timeout=60):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": repr(exc)}


def gsutil_ls(uri, recursive=False, timeout=120, max_lines=2000):
    if shutil.which("gsutil") is None:
        return {"status": "gsutil_not_found", "lines": [], "stderr": "gsutil not found"}
    args = ["gsutil"]
    if REQUESTER_PAYS_PROJECT:
        args.extend(["-u", REQUESTER_PAYS_PROJECT])
    args.append("ls")
    args.append("-r" if recursive else "")
    args.append(uri.rstrip("/") + ("/**" if recursive else ""))
    args = [a for a in args if a]
    result = run_args(args, timeout=timeout)
    lines = [x.strip() for x in result["stdout"].splitlines() if x.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    status = "pass" if result["returncode"] == 0 else f"fail_returncode_{result['returncode']}"
    return {"status": status, "lines": lines, "stderr": result["stderr"]}


def find_first_existing(filenames, search_dirs):
    for directory in search_dirs:
        d = Path(directory)
        for filename in filenames:
            p = d / filename
            if p.exists():
                return p
    return None


def pct(n, d):
    return round(float(n) / float(d) * 100, 2) if d else np.nan


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 160)

config_df = pd.DataFrame([{
    "cdr": CDR,
    "prep_cdr": PREP_CDR,
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "overwrite_outputs": OVERWRITE_OUTPUTS,
    "requester_pays_project": REQUESTER_PAYS_PROJECT,
    "genotype_file_env": GENOTYPE_FILE,
    "purpose": "Q3 VDR AoU database and literature/database-needed follow-up checks",
}])
save(config_df, "00_run_configuration.csv")



# %% Cell 4
known_output_dirs = [
    DATA_DIR,
    DATA_DIR / "Q3_VDR_v9_rebuild_outputs",
    DATA_DIR / "Q3_VDR_v9_model_prep_outputs",
    DATA_DIR / "Q3_VDR_genomic_resource_detector_outputs_2026_08_03",
    DATA_DIR / "Q3_VDR_tier1_model_outputs_R",
    DATA_DIR / "Q3_VDR_tier1_ph_sensitivity_outputs_R",
]

expected_existing_files = [
    "06_genomics_denominator_summary_v9.csv",
    "06_vdr_candidate_carrier_counts_v9.csv",
    "07_snp_model_readiness_v9.csv",
    "04_model_dataset_summary_v9.csv",
    "08_pc_relatedness_schema_search_v9.csv",
    "08_mr_readiness_status_v9.csv",
    "02_crp_unit_concept_audit_v9.csv",
    "03_crp_model_dataset_v9.csv",
    "05_q2_crp_pair_v9.csv",
    "09_tier1_R_crp_lm_term_results.csv",
]

existing_rows = []
for filename in expected_existing_files:
    hit = find_first_existing([filename], known_output_dirs)
    existing_rows.append({
        "expected_file": filename,
        "status": "found" if hit else "missing",
        "path": str(hit) if hit else "",
        "size_bytes": hit.stat().st_size if hit else np.nan,
    })
existing_audit = pd.DataFrame(existing_rows)
save(existing_audit, "00_existing_q3_output_file_audit.csv")
print(existing_audit.to_string(index=False))



# %% Cell 6
env_rows = []
for key, value in sorted(os.environ.items()):
    if any(token in key.upper() for token in ["WORKSPACE", "CDR", "BUCKET", "GOOGLE", "AOU", "GENOMIC", "HAIL", "TERRA", "LEONARDO", "PROJECT", "VDR"]) or str(value).startswith("gs://"):
        env_rows.append({"key": key, "value": value, "tags": classify_text(value)})
env_df = pd.DataFrame(env_rows)
save(env_df, "01_environment_variables_relevant_to_genomics.csv")

tool_rows = []
for tool in ["python", "gcloud", "gsutil", "java"]:
    path = shutil.which(tool) or ""
    tool_rows.append({"tool": tool, "path": path, "available": bool(path), "stdout": "", "stderr": "", "returncode": 0 if path else -1})
for args in [["gcloud", "config", "get-value", "project"], ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], ["gsutil", "version", "-l"]]:
    if shutil.which(args[0]) is None:
        tool_rows.append({"tool": " ".join(args), "path": "", "available": False, "stdout": "", "stderr": f"{args[0]} not found", "returncode": -1})
    else:
        r = run_args(args, timeout=60)
        tool_rows.append({"tool": " ".join(args), "path": shutil.which(args[0]), "available": True, **r})
tool_df = pd.DataFrame(tool_rows)
save(tool_df, "01_runtime_tool_status.csv")

bq_datasets = []
for raw in [CDR, PREP_CDR] + [x.strip() for x in re.split(r"[\\n,;]+", EXTRA_BQ_DATASETS_RAW) if x.strip()]:
    if not raw:
        continue
    full = raw if "." in raw else f"{CDR_PROJECT}.{raw}"
    if full not in bq_datasets:
        bq_datasets.append(full)

table_parts = []
column_parts = []
for bq_dataset in bq_datasets:
    table_sql = f"""
    SELECT {bq_quote(bq_dataset)} AS bq_dataset, table_name, table_type, creation_time
    FROM `{bq_dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE LOWER(table_name) LIKE '%variant%'
       OR LOWER(table_name) LIKE '%genomic%'
       OR LOWER(table_name) LIKE '%genome%'
       OR LOWER(table_name) LIKE '%wgs%'
       OR LOWER(table_name) LIKE '%array%'
       OR LOWER(table_name) LIKE '%sample%'
       OR LOWER(table_name) LIKE '%ancestry%'
       OR LOWER(table_name) LIKE '%related%'
       OR LOWER(table_name) LIKE '%kinship%'
       OR LOWER(table_name) LIKE '%qc%'
       OR LOWER(table_name) LIKE '%plink%'
       OR LOWER(table_name) LIKE '%bgen%'
       OR LOWER(table_name) LIKE '%hail%'
       OR LOWER(table_name) LIKE '%dosage%'
       OR LOWER(table_name) LIKE '%genotype%'
    ORDER BY table_name
    """
    part = q(table_sql, f"table_resource_search:{bq_dataset}")
    if not part.empty:
        part["tags"] = part["table_name"].map(classify_text)
        table_parts.append(part)

    column_sql = f"""
    SELECT {bq_quote(bq_dataset)} AS bq_dataset, table_name, column_name, data_type
    FROM `{bq_dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE LOWER(column_name) LIKE '%dosage%'
       OR LOWER(column_name) LIKE '%genotype%'
       OR LOWER(column_name) LIKE '%hard_call%'
       OR LOWER(column_name) LIKE '%principal%'
       OR LOWER(column_name) LIKE '%pc1%'
       OR LOWER(column_name) LIKE '%pc_1%'
       OR LOWER(column_name) LIKE '%ancestry%'
       OR LOWER(column_name) LIKE '%related%'
       OR LOWER(column_name) LIKE '%kinship%'
       OR LOWER(column_name) LIKE '%qc%'
       OR LOWER(column_name) LIKE '%call_rate%'
       OR LOWER(column_name) LIKE '%missing%'
       OR LOWER(column_name) LIKE '%rs_number%'
       OR LOWER(column_name) LIKE '%vid%'
       OR LOWER(column_name) LIKE '%sample%'
       OR LOWER(column_name) LIKE '%participant%'
    ORDER BY table_name, column_name
    """
    part = q(column_sql, f"column_resource_search:{bq_dataset}")
    if not part.empty:
        part["tags"] = part.apply(lambda r: classify_text(f"{r.get('table_name')} {r.get('column_name')}"), axis=1)
        column_parts.append(part)

table_search = pd.concat(table_parts, ignore_index=True) if table_parts else pd.DataFrame(columns=["bq_dataset", "table_name", "table_type", "creation_time", "tags"])
column_search = pd.concat(column_parts, ignore_index=True) if column_parts else pd.DataFrame(columns=["bq_dataset", "table_name", "column_name", "data_type", "tags"])
save(table_search, "01_bigquery_genomic_table_search.csv")
save(column_search, "01_bigquery_genotype_pc_qc_column_search.csv")

# Candidate GCS roots. Listing is shallow/limited and records permission errors.
candidate_roots = [
    {"source": "official_default_cdr_bucket", "uri": "gs://fc-aou-datasets-controlled", "recursive": False},
    {"source": "v9_genomics_resource_direct", "uri": "gs://fc-aou-datasets-controlled/vwb-aou-datasets-controlled-v9", "recursive": True},
    {"source": "v9_allxall_resource_direct", "uri": "gs://fc-aou-datasets-controlled/vwb-aou-allxall-datasets-v9", "recursive": True},
    {"source": "v8_genomics_resource_fallback", "uri": "gs://fc-aou-datasets-controlled/vwb-aou-datasets-controlled", "recursive": True},
    {"source": "v8_allxall_resource_fallback", "uri": "gs://fc-aou-datasets-controlled/vwb-aou-allxall-datasets", "recursive": True},
]
workspace_bucket = os.environ.get("WORKSPACE_BUCKET", "").strip()
if workspace_bucket.startswith("gs://"):
    candidate_roots.append({"source": "WORKSPACE_BUCKET", "uri": workspace_bucket, "recursive": True})
for key in ["Q3_VDR_GENOMIC_ROOTS", "Q3_VDR_CDR_DIRECTORY_URI", "AOU_GENOMIC_DATA_PATH", "AOU_CDR_DIRECTORY_URI", "WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH", "WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", "WGS_ACAF_THRESHOLD_VCF_PATH"]:
    raw = os.environ.get(key, "")
    for part in re.split(r"[\\n,;]+", raw):
        part = part.strip()
        if part.startswith("gs://"):
            candidate_roots.append({"source": key, "uri": part, "recursive": True})
for _, row in env_df.iterrows():
    value = str(row.get("value", "")).strip()
    if value.startswith("gs://") and any(k in value.lower() for k in ["genomic", "genome", "wgs", "array", "cdr", "hail", "vds", "workspace"]):
        candidate_roots.append({"source": "env:" + str(row.get("key")), "uri": value, "recursive": True})

candidate_roots_df = pd.DataFrame(candidate_roots).drop_duplicates()
save(candidate_roots_df, "01_candidate_gcs_roots_to_test.csv")

gcs_rows = []
for _, row in candidate_roots_df.iterrows():
    result = gsutil_ls(row["uri"], recursive=bool(row.get("recursive", True)), max_lines=1200, timeout=120)
    if result["lines"]:
        hits = [line for line in result["lines"] if classify_text(line)]
        if hits:
            for line in hits[:500]:
                gcs_rows.append({"source": row["source"], "root_uri": row["uri"], "hit_uri": line, "tags": classify_text(line), "list_status": result["status"], "stderr": result["stderr"][:1000]})
        else:
            gcs_rows.append({"source": row["source"], "root_uri": row["uri"], "hit_uri": "", "tags": "", "list_status": "listed_no_relevant_hits", "stderr": result["stderr"][:1000]})
    else:
        gcs_rows.append({"source": row["source"], "root_uri": row["uri"], "hit_uri": "", "tags": "", "list_status": result["status"], "stderr": result["stderr"][:1000]})
gcs_hits = pd.DataFrame(gcs_rows)
save(gcs_hits, "01_gcs_genomic_resource_listing_hits.csv")

hail_status = []
try:
    import hail as hl
    hail_status.append({"check": "import_hail", "status": "pass", "detail": getattr(hl, "__version__", "")})
except Exception as exc:
    hail_status.append({"check": "import_hail", "status": "fail", "detail": repr(exc)[:1000]})
hail_df = pd.DataFrame(hail_status)
save(hail_df, "01_hail_import_smoke_test.csv")

def has_tag(df, tag):
    return df is not None and not df.empty and "tags" in df.columns and df["tags"].fillna("").str.contains(tag, regex=False).any()

resource_summary = pd.DataFrame([
    {"requirement": "BigQuery cb_variant_to_person carrier route", "status": "available" if table_search["table_name"].astype(str).eq("cb_variant_to_person").any() else "not_seen", "evidence": "01_bigquery_genomic_table_search.csv"},
    {"requirement": "Person-level genotype dosage or hard calls", "status": "possible" if (has_tag(column_search, "genotype_resource") or has_tag(gcs_hits, "genotype_resource")) else "not_found_in_search", "evidence": "01_bigquery_genotype_pc_qc_column_search.csv; 01_gcs_genomic_resource_listing_hits.csv"},
    {"requirement": "Ancestry PCs or inferred genetic ancestry", "status": "possible" if (has_tag(column_search, "ancestry_pc") or has_tag(gcs_hits, "ancestry_pc")) else "not_found_in_search", "evidence": "01_bigquery_genotype_pc_qc_column_search.csv; 01_gcs_genomic_resource_listing_hits.csv"},
    {"requirement": "Relatedness or unrelated set", "status": "possible" if (has_tag(column_search, "relatedness") or has_tag(gcs_hits, "relatedness")) else "not_found_in_search", "evidence": "01_bigquery_genotype_pc_qc_column_search.csv; 01_gcs_genomic_resource_listing_hits.csv"},
    {"requirement": "Sample/variant genetic QC fields", "status": "possible" if (has_tag(column_search, "genomic_qc") or has_tag(gcs_hits, "genomic_qc")) else "not_found_in_search", "evidence": "01_bigquery_genotype_pc_qc_column_search.csv; 01_gcs_genomic_resource_listing_hits.csv"},
    {"requirement": "Hail runtime import", "status": "available" if hail_df["status"].eq("pass").any() else "not_available", "evidence": "01_hail_import_smoke_test.csv"},
])
save(resource_summary, "01_aou_genotype_pc_qc_resource_summary.csv")
print(resource_summary.to_string(index=False))



# %% Cell 8
candidate_structs = ",\n    ".join([
    "STRUCT("
    + bq_quote(r["tier"]) + " AS tier, "
    + bq_quote(r["gene"]) + " AS gene, "
    + bq_quote(r["rsid"]) + " AS rsid, "
    + bq_quote(r["label"]) + " AS label, "
    + bq_quote(r["expected_vid"]) + " AS expected_vid)"
    for r in TIER1_SNPS
])

variant_records = q(f"""
WITH candidate AS (
  SELECT * FROM UNNEST([
    {candidate_structs}
  ])
)
SELECT
  candidate.tier,
  candidate.gene,
  candidate.rsid,
  candidate.label,
  candidate.expected_vid,
  rs.vid,
  va.contig,
  va.position,
  va.ref_allele,
  va.alt_allele,
  va.allele_count,
  va.allele_number,
  va.allele_frequency,
  va.participant_count,
  IF(rs.vid = candidate.expected_vid, TRUE, FALSE) AS matches_expected_vid
FROM candidate
LEFT JOIN `{CDR}.cb_variant_attribute_rs_number` rs
  ON rs.rs_number = candidate.rsid
LEFT JOIN `{CDR}.cb_variant_attribute` va
  ON va.vid = rs.vid
ORDER BY candidate.rsid, matches_expected_vid DESC, va.participant_count DESC, va.allele_frequency DESC
""", "tier1_vdr_all_aou_variant_records")

if not variant_records.empty:
    variant_records["allele_frequency"] = pd.to_numeric(variant_records["allele_frequency"], errors="coerce")
    variant_records["participant_count"] = pd.to_numeric(variant_records["participant_count"], errors="coerce")
    variant_records["is_snp_record"] = variant_records["ref_allele"].fillna("").str.len().eq(1) & variant_records["alt_allele"].fillna("").str.len().eq(1)
    variant_records["is_common_af_001_999"] = variant_records["allele_frequency"].between(0.001, 0.999, inclusive="both")
    variant_records["eligible_common_biallelic"] = variant_records["vid"].notna() & variant_records["is_snp_record"] & variant_records["is_common_af_001_999"] & variant_records["participant_count"].fillna(0).gt(0)
save(variant_records, "02_tier1_vdr_all_aou_variant_records.csv")

selected_rows = []
for r in TIER1_SNPS:
    sub = variant_records[variant_records["rsid"].eq(r["rsid"])].copy() if not variant_records.empty else pd.DataFrame()
    if sub.empty:
        selected_rows.append({**r, "selected_vid": "", "ref_allele": "", "alt_allele": "", "selection_status": "missing", "selection_reason": "No AoU records returned"})
        continue
    expected = sub[sub["vid"].eq(r["expected_vid"])].copy()
    if not expected.empty:
        chosen = expected.iloc[0]
        status = "selected_expected_vid"
        reason = "Expected v9/v8 validated VID found"
    else:
        eligible = sub[sub.get("eligible_common_biallelic", False)].copy()
        if eligible.empty:
            chosen = sub.sort_values("participant_count", ascending=False).iloc[0]
            status = "fallback_noneligible_record"
            reason = "No eligible common biallelic record; largest participant_count record shown for review"
        else:
            chosen = eligible.sort_values(["participant_count", "allele_frequency"], ascending=False).iloc[0]
            status = "fallback_largest_common_biallelic"
            reason = "Expected VID absent; selected largest eligible common biallelic record"
    selected_rows.append({
        **r,
        "selected_vid": chosen.get("vid", ""),
        "ref_allele": chosen.get("ref_allele", ""),
        "alt_allele": chosen.get("alt_allele", ""),
        "allele_frequency": chosen.get("allele_frequency", np.nan),
        "participant_count": chosen.get("participant_count", np.nan),
        "selection_status": status,
        "selection_reason": reason,
        "n_records_for_rsid": len(sub),
        "excluded_records": ";".join([str(x) for x in sub["vid"].dropna().unique() if str(x) != str(chosen.get("vid", ""))]),
    })
selected_records = pd.DataFrame(selected_rows)
save(selected_records, "02_tier1_vdr_selected_variant_records_final.csv")

harmonization = selected_records.copy()
harmonization["literature_allele_names_to_check"] = harmonization["label"].map({
    "FokI": "F/f and T/C nomenclature; map to AoU A/G before functional wording",
    "BsmI": "B/b nomenclature; check 3-prime marker and risk/effect allele in each source",
    "ApaI": "A/a nomenclature; check 3-prime marker and risk/effect allele in each source",
    "TaqI": "T/t nomenclature; check silent exon 9/3-prime marker and risk/effect allele in each source",
})
harmonization["effect_or_function_allele"] = "not_verified"
harmonization["allowed_wording"] = "Use ALT-carrier as exposure coding only; do not call ALT a risk/function allele until literature-strand mapping is verified."
save(harmonization[[
    "rsid", "label", "selected_vid", "ref_allele", "alt_allele",
    "literature_allele_names_to_check", "effect_or_function_allele",
    "selection_status", "allowed_wording"
]], "02_tier1_vdr_allele_harmonization_status.csv")
print(selected_records.to_string(index=False))



# %% Cell 10
def load_genotype_matrix(path):
    if not path:
        return pd.DataFrame(), "no_env_path"
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(), "path_not_found"
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    if "person_id" not in df.columns:
        return pd.DataFrame(), "missing_person_id"
    rsids = [r["rsid"] for r in TIER1_SNPS]
    lower_cols = {c.lower(): c for c in df.columns}
    if {"rsid", "dosage"}.issubset(lower_cols):
        tmp = df.rename(columns={lower_cols["rsid"]: "rsid", lower_cols["dosage"]: "dosage"})
        tmp = tmp[tmp["rsid"].isin(rsids)].copy()
        wide = tmp.pivot_table(index="person_id", columns="rsid", values="dosage", aggfunc="first").reset_index()
    elif {"rsid", "genotype"}.issubset(lower_cols):
        tmp = df.rename(columns={lower_cols["rsid"]: "rsid", lower_cols["genotype"]: "genotype"}).copy()
        tmp = tmp[tmp["rsid"].isin(rsids)].copy()
        tmp["dosage"] = pd.to_numeric(tmp["genotype"], errors="coerce")
        wide = tmp.pivot_table(index="person_id", columns="rsid", values="dosage", aggfunc="first").reset_index()
    else:
        wide = df.copy()
        rename = {}
        for c in wide.columns:
            for rsid in rsids:
                if c == rsid or c.lower() in {f"{rsid}_dosage", f"{rsid}_gt", f"{rsid}_genotype"}:
                    rename[c] = rsid
        wide = wide.rename(columns=rename)
    for rsid in rsids:
        if rsid in wide.columns:
            wide[rsid] = pd.to_numeric(wide[rsid], errors="coerce")
    keep = ["person_id"] + [rsid for rsid in rsids if rsid in wide.columns]
    return wide[keep], "loaded"


def hwe_chisq(n0, n1, n2):
    n0, n1, n2 = float(n0), float(n1), float(n2)
    n = n0 + n1 + n2
    if n <= 0:
        return np.nan
    p = (2 * n2 + n1) / (2 * n)
    q_ = 1 - p
    exp0 = n * q_ * q_
    exp1 = 2 * n * p * q_
    exp2 = n * p * p
    if min(exp0, exp1, exp2) <= 0:
        return np.nan
    chi = ((n0 - exp0) ** 2 / exp0) + ((n1 - exp1) ** 2 / exp1) + ((n2 - exp2) ** 2 / exp2)
    try:
        import scipy.stats as stats
        return float(stats.chi2.sf(chi, df=1))
    except Exception:
        return np.nan


genotype_matrix, genotype_status = load_genotype_matrix(GENOTYPE_FILE)
genotype_status_df = pd.DataFrame([{"genotype_file": GENOTYPE_FILE, "status": genotype_status, "n_rows": len(genotype_matrix), "columns": ";".join(genotype_matrix.columns)}])
save(genotype_status_df, "03_genotype_input_status.csv")

if genotype_matrix.empty:
    save(pd.DataFrame(columns=["rsid", "n_called", "n_missing", "missing_pct", "genotype_0", "genotype_1", "genotype_2", "alt_allele_frequency", "maf", "hwe_p"]), "03_tier1_vdr_genotype_qc_summary.csv")
    save(pd.DataFrame(columns=["snp1", "snp2", "n_pair_called", "r2", "correlation"]), "03_tier1_vdr_pairwise_ld.csv")
    save(pd.DataFrame([{"decision": "skip_genotype_qc", "reason": "Set Q3_VDR_TIER1_GENOTYPE_CSV to a genotype dosage/hard-call CSV to run MAF/HWE/LD."}]), "03_tier1_vdr_haplotype_feasibility.csv")
else:
    rsids = [r["rsid"] for r in TIER1_SNPS if r["rsid"] in genotype_matrix.columns]
    qc_rows = []
    for rsid in rsids:
        s = pd.to_numeric(genotype_matrix[rsid], errors="coerce")
        n = len(s)
        called = s.dropna()
        g0 = int((called == 0).sum())
        g1 = int((called == 1).sum())
        g2 = int((called == 2).sum())
        alt_af = (2 * g2 + g1) / (2 * len(called)) if len(called) else np.nan
        maf = min(alt_af, 1 - alt_af) if pd.notna(alt_af) else np.nan
        qc_rows.append({
            "rsid": rsid,
            "n_called": len(called),
            "n_missing": int(s.isna().sum()),
            "missing_pct": pct(int(s.isna().sum()), n),
            "genotype_0": g0,
            "genotype_1": g1,
            "genotype_2": g2,
            "alt_allele_frequency": alt_af,
            "maf": maf,
            "hwe_p": hwe_chisq(g0, g1, g2),
        })
    save(pd.DataFrame(qc_rows), "03_tier1_vdr_genotype_qc_summary.csv")

    ld_rows = []
    for i, snp1 in enumerate(rsids):
        for snp2 in rsids[i + 1:]:
            pair = genotype_matrix[[snp1, snp2]].dropna()
            if len(pair) < 10:
                corr = np.nan
            else:
                corr = float(pair[snp1].corr(pair[snp2]))
            ld_rows.append({"snp1": snp1, "snp2": snp2, "n_pair_called": len(pair), "correlation": corr, "r2": corr ** 2 if pd.notna(corr) else np.nan})
    save(pd.DataFrame(ld_rows), "03_tier1_vdr_pairwise_ld.csv")

    complete = genotype_matrix[["person_id"] + rsids].dropna()
    save(pd.DataFrame([{
        "haplotype_set": "tier1_four_snp_complete",
        "required_snps": ";".join(rsids),
        "n_complete_genotypes": len(complete),
        "feasible": len(complete) > 0 and len(rsids) == 4,
        "note": "This only checks complete genotype availability; formal haplotype modeling still needs ancestry/relatedness decisions.",
    }]), "03_tier1_vdr_haplotype_feasibility.csv")



# %% Cell 12
search_dirs = known_output_dirs + [OUTPUT_DIR]
crp_unit_path = find_first_existing(["02_crp_unit_concept_audit_v9.csv"], search_dirs)
crp_model_path = find_first_existing(["03_crp_model_dataset_v9.csv", "05_q2_crp_pair_v9.csv"], search_dirs)

if crp_unit_path:
    crp_unit_audit = pd.read_csv(crp_unit_path)
else:
    crp_unit_audit = q(f"""
    SELECT
      m.measurement_concept_id AS concept_id,
      c.concept_name AS concept_name,
      m.unit_concept_id,
      cu.concept_name AS unit_name,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT m.person_id) AS n_persons,
      COUNTIF(m.value_as_number IS NOT NULL) AS n_numeric_rows,
      COUNT(DISTINCT IF(m.value_as_number IS NOT NULL, m.person_id, NULL)) AS n_numeric_persons,
      APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(50)] AS median_value,
      MIN(m.measurement_date) AS min_date,
      MAX(m.measurement_date) AS max_date
    FROM `{CDR}.measurement` m
    LEFT JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
    LEFT JOIN `{CDR}.concept` cu ON m.unit_concept_id = cu.concept_id
    WHERE m.measurement_concept_id IN {int_list(CRP_CONCEPTS)}
    GROUP BY concept_id, concept_name, unit_concept_id, unit_name
    ORDER BY concept_id, n_rows DESC
    """, "crp_unit_harmonization_audit")
save(crp_unit_audit, "04_crp_unit_harmonization_audit.csv")

if crp_model_path:
    crp_df = pd.read_csv(crp_model_path)
else:
    crp_df = pd.DataFrame()

crp_status = pd.DataFrame([{"crp_model_dataset_path": str(crp_model_path) if crp_model_path else "", "status": "loaded" if not crp_df.empty else "missing_run_06_or_09_first", "n_rows": len(crp_df)}])
save(crp_status, "04_crp_model_dataset_status.csv")

if crp_df.empty or "crp_mgL" not in crp_df.columns:
    save(pd.DataFrame(columns=["n", "min", "p1", "p5", "median", "p95", "p99", "max", "n_gt10", "pct_gt10", "n_ambiguous_unit", "pct_ambiguous_unit"]), "04_crp_distribution_extreme_value_summary.csv")
    save(pd.DataFrame(columns=["timing_group", "n", "percent", "median_days_before", "min_days_before", "max_days_before"]), "04_crp_timing_relative_to_vitd_summary.csv")
    save(pd.DataFrame([{"decision": "crp_dataset_missing", "next_action": "Run 06_v9_q3_rebuild_from_raw_cdr.ipynb and 09_v9_vdr_model_prep.ipynb first, then rerun this notebook."}]), "04_crp_sensitivity_dataset_counts.csv")
else:
    crp = pd.to_numeric(crp_df["crp_mgL"], errors="coerce")
    ambiguous = pd.to_numeric(crp_df.get("ambiguous_unit", pd.Series([0] * len(crp_df))), errors="coerce").fillna(0).astype(int)
    quant = crp.quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).to_dict()
    dist = pd.DataFrame([{
        "n": int(crp.notna().sum()),
        "min": quant.get(0, np.nan),
        "p1": quant.get(0.01, np.nan),
        "p5": quant.get(0.05, np.nan),
        "median": quant.get(0.5, np.nan),
        "p95": quant.get(0.95, np.nan),
        "p99": quant.get(0.99, np.nan),
        "max": quant.get(1, np.nan),
        "n_gt10": int((crp > 10).sum()),
        "pct_gt10": pct(int((crp > 10).sum()), int(crp.notna().sum())),
        "n_ambiguous_unit": int(ambiguous.sum()),
        "pct_ambiguous_unit": pct(int(ambiguous.sum()), len(crp_df)),
    }])
    save(dist, "04_crp_distribution_extreme_value_summary.csv")

    timing_rows = []
    if "days_before" in crp_df.columns:
        days = pd.to_numeric(crp_df["days_before"], errors="coerce")
        groups = pd.cut(days, bins=[-np.inf, -1, 0, 30, 365, 1825, np.inf], labels=["after_vitd_index", "same_day", "1_30d_before", "31_365d_before", "1_5y_before", "gt5y_before"])
        for group, sub in crp_df.assign(timing_group=groups, days_before_num=days).groupby("timing_group", dropna=False, observed=False):
            timing_rows.append({
                "timing_group": str(group),
                "n": len(sub),
                "percent": pct(len(sub), len(crp_df)),
                "median_days_before": float(sub["days_before_num"].median()) if len(sub) else np.nan,
                "min_days_before": float(sub["days_before_num"].min()) if len(sub) else np.nan,
                "max_days_before": float(sub["days_before_num"].max()) if len(sub) else np.nan,
            })
    save(pd.DataFrame(timing_rows), "04_crp_timing_relative_to_vitd_summary.csv")

    sens_rows = []
    base_n = len(crp_df)
    masks = {
        "all_crp_model_base": pd.Series(True, index=crp_df.index),
        "exclude_crp_gt10": ~(crp > 10),
        "exclude_ambiguous_unit": ambiguous.eq(0),
        "exclude_crp_gt10_and_ambiguous_unit": (~(crp > 10)) & ambiguous.eq(0),
    }
    for name, mask in masks.items():
        sub = crp_df[mask].copy()
        sens_rows.append({
            "dataset": name,
            "n": len(sub),
            "n_removed": base_n - len(sub),
            "pct_removed": pct(base_n - len(sub), base_n),
            "crp_gt10_remaining": int((pd.to_numeric(sub.get("crp_mgL", pd.Series(dtype=float)), errors="coerce") > 10).sum()) if len(sub) else 0,
            "ambiguous_unit_remaining": int(pd.to_numeric(sub.get("ambiguous_unit", pd.Series([0] * len(sub))), errors="coerce").fillna(0).sum()) if len(sub) else 0,
        })
    save(pd.DataFrame(sens_rows), "04_crp_sensitivity_dataset_counts.csv")



# %% Cell 14
resource_summary_path = OUTPUT_DIR / "01_aou_genotype_pc_qc_resource_summary.csv"
harmonization_path = OUTPUT_DIR / "02_tier1_vdr_allele_harmonization_status.csv"
genotype_status_path = OUTPUT_DIR / "03_genotype_input_status.csv"
crp_dist_path = OUTPUT_DIR / "04_crp_distribution_extreme_value_summary.csv"

resource_summary = pd.read_csv(resource_summary_path) if resource_summary_path.exists() else pd.DataFrame()
harmonization = pd.read_csv(harmonization_path) if harmonization_path.exists() else pd.DataFrame()
genotype_status_df = pd.read_csv(genotype_status_path) if genotype_status_path.exists() else pd.DataFrame()
crp_dist = pd.read_csv(crp_dist_path) if crp_dist_path.exists() else pd.DataFrame()

def status_for(requirement_keyword):
    if resource_summary.empty:
        return "unknown"
    sub = resource_summary[resource_summary["requirement"].str.contains(requirement_keyword, case=False, regex=False, na=False)]
    return str(sub.iloc[0]["status"]) if not sub.empty else "unknown"

genotype_loaded = (not genotype_status_df.empty and str(genotype_status_df.iloc[0].get("status", "")) == "loaded")
allele_all_selected = (not harmonization.empty and harmonization["selection_status"].astype(str).str.contains("selected|fallback", regex=True).all())
crp_gt10_pct = float(crp_dist.iloc[0]["pct_gt10"]) if not crp_dist.empty and "pct_gt10" in crp_dist.columns and pd.notna(crp_dist.iloc[0]["pct_gt10"]) else np.nan

decision_rows = [
    {
        "question": "Can Q3 use additive genotype models?",
        "current_status": "yes_if_genotype_qc_outputs_reviewed" if genotype_loaded else "no_currently_carrier_only",
        "evidence": "03_genotype_input_status.csv; 03_tier1_vdr_genotype_qc_summary.csv",
        "q3_action": "Add additive sensitivity only if genotype QC is valid; otherwise keep dominant ALT-carrier wording.",
    },
    {
        "question": "Can Q3 say ALT is a functional/risk allele?",
        "current_status": "not_yet",
        "evidence": "02_tier1_vdr_allele_harmonization_status.csv",
        "q3_action": "Keep ALT-carrier as exposure coding until FokI/BsmI/ApaI/TaqI literature allele names are harmonized to AoU REF/ALT.",
    },
    {
        "question": "Can Q3 control genetic population structure with PCs?",
        "current_status": status_for("Ancestry PCs"),
        "evidence": "01_aou_genotype_pc_qc_resource_summary.csv; 01_bigquery_genotype_pc_qc_column_search.csv",
        "q3_action": "Use PC-adjusted models only if PC resource is confirmed and joinable; otherwise race/ethnicity is proxy sensitivity only.",
    },
    {
        "question": "Can Q3 handle relatedness/unrelated set?",
        "current_status": status_for("Relatedness"),
        "evidence": "01_aou_genotype_pc_qc_resource_summary.csv; 01_gcs_genomic_resource_listing_hits.csv",
        "q3_action": "Add unrelated-only or kinship-aware sensitivity only if relatedness resource is confirmed.",
    },
    {
        "question": "Can Q3 report MAF/HWE/LD/haplotype?",
        "current_status": "yes_if_genotype_loaded" if genotype_loaded else "not_from_carrier_only_data",
        "evidence": "03_tier1_vdr_genotype_qc_summary.csv; 03_tier1_vdr_pairwise_ld.csv",
        "q3_action": "Report MAF/HWE/LD only from genotype dosage/hard calls, not binary carrier correlations.",
    },
    {
        "question": "Does CRP require acute-inflammation sensitivity?",
        "current_status": "yes" if pd.notna(crp_gt10_pct) and crp_gt10_pct > 5 else "unknown_or_low",
        "evidence": "04_crp_distribution_extreme_value_summary.csv; 04_crp_sensitivity_dataset_counts.csv",
        "q3_action": "Run/report CRP >10 mg/L sensitivity; do not treat CRP as mediator unless timing supports it.",
    },
]
decision_matrix = pd.DataFrame(decision_rows)
save(decision_matrix, "05_final_followup_decision_matrix.csv")

note = f"""# Q3 VDR Follow-up Wording Decision Note

Generated: 2026-08-07

## Current Decision

Keep Q3 as a VDR candidate ALT-carrier x measured 25(OH)D interaction analysis unless the follow-up outputs confirm genotype dosage/hard calls, allele harmonization, PCs/relatedness/QC, and genotype-level LD.

## Evidence From This Run

- Genotype input status: {genotype_status_df.iloc[0].get('status', 'unknown') if not genotype_status_df.empty else 'unknown'}
- Ancestry PC resource search: {status_for('Ancestry PCs')}
- Relatedness resource search: {status_for('Relatedness')}
- Genetic QC resource search: {status_for('QC')}
- Allele records selected: {'yes' if allele_all_selected else 'not fully confirmed'}
- CRP >10 mg/L percent: {crp_gt10_pct if pd.notna(crp_gt10_pct) else 'unknown'}

## Report-ready Conservative Wording

The Tier 1 AoU analysis supports a measured vitamin D deficiency-CVD association more strongly than VDR ALT-carrier effect modification. The four classic VDR polymorphisms remain biologically plausible candidate markers, but allele direction, additive genotype effects, local LD/haplotype structure, ancestry/relatedness control, and CRP acute-inflammation sensitivity require additional AoU validation before stronger mechanism or methods claims are made.

## Stop Rules

- Do not call an AoU ALT allele a functional or risk allele until literature allele names are harmonized to AoU REF/ALT.
- Do not report HWE, MAF, LD, or haplotypes from carrier-only data.
- Do not claim population stratification is fully solved without genetic PCs or an equivalent ancestry control.
- Do not describe CRP as a mediator unless timing supports vitamin D -> CRP -> CVD ordering.
"""
save_text(note, "05_q3_final_mechanism_methods_wording_decision.md")

save(pd.DataFrame(query_status_rows), "99_bigquery_query_status.csv")
manifest = pd.DataFrame([{"filename": p.name, "path": str(p), "size_bytes": p.stat().st_size} for p in sorted(OUTPUT_DIR.glob("*")) if p.is_file()])
save(manifest, "99_followup_output_manifest.csv")
print("Done. Upload/review OUTPUT_DIR:", OUTPUT_DIR)
