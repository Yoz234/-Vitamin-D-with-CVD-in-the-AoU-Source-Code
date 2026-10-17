# Source companion of 14_q3_vdr_hail_tier1_genotype_extraction_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
import sys
import math
import traceback
from pathlib import Path


TARGET_SNPS = [
    {"rsid": "rs731236", "label": "TaqI", "contig": "12", "position": 47844974, "ref": "A", "alt": "G", "expected_vid": "12-47844974-A-G"},
    {"rsid": "rs7975232", "label": "ApaI", "contig": "12", "position": 47845054, "ref": "C", "alt": "A", "expected_vid": "12-47845054-C-A"},
    {"rsid": "rs1544410", "label": "BsmI", "contig": "12", "position": 47846052, "ref": "C", "alt": "T", "expected_vid": "12-47846052-C-T"},
    {"rsid": "rs2228570", "label": "FokI", "contig": "12", "position": 47879112, "ref": "A", "alt": "G", "expected_vid": "12-47879112-A-G"},
]


def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "data",
        cwd.parent / "data",
        Path("/home/jupyter/workspace") / "VDR",
        Path("/home/jupyter/workspace") / "VDR" / "data",
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            pass
    return cwd


DATA_DIR = Path(os.environ.get("Q3_VDR_DATA_DIR", str(infer_default_data_dir()))).expanduser()
OUTPUT_DIR = Path(os.environ.get("Q3_VDR_HAIL_OUTPUT_DIR", str(DATA_DIR / "Q3_VDR_tier1_hail_genotype_outputs_2026_08_07"))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERWRITE_OUTPUTS = os.environ.get("Q3_VDR_OVERWRITE_OUTPUTS", "true").strip().lower() in {"1", "true", "yes", "y"}
GOOGLE_PROJECT = os.environ.get("GOOGLE_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
WORKSPACE_BUCKET = os.environ.get("WORKSPACE_BUCKET", "").strip()
HAIL_TMP_DIR = os.environ.get("Q3_VDR_HAIL_TMP_DIR", (WORKSPACE_BUCKET.rstrip("/") + "/hail_tmp/q3_vdr") if WORKSPACE_BUCKET.startswith("gs://") else "").strip()

MANUAL_HAIL_PATH = os.environ.get("Q3_VDR_HAIL_MT_PATH", "").strip()
DEFAULT_AOU_V9_HAIL_PATH_CANDIDATES = [
    ("AOU_V9_WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH_FALLBACK", "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/acaf_threshold/splitMT/hail.mt"),
    ("AOU_V9_WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH_FALLBACK", "gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/acaf_threshold/multiMT/hail.mt"),
    ("AOU_V9_MICROARRAY_HAIL_STORAGE_PATH_FALLBACK", "gs://vwb-aou-datasets-controlled/v9/microarray/hail.mt"),
]
env_hail_path_candidates = [
    ("Q3_VDR_HAIL_MT_PATH", MANUAL_HAIL_PATH),
    ("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", os.environ.get("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", "").strip()),
    ("WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH", os.environ.get("WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH", "").strip()),
    ("MICROARRAY_HAIL_STORAGE_PATH", os.environ.get("MICROARRAY_HAIL_STORAGE_PATH", "").strip()),
]
HAIL_PATH_CANDIDATES = []
seen_hail_paths = set()
for source, path in env_hail_path_candidates + DEFAULT_AOU_V9_HAIL_PATH_CANDIDATES:
    if path and path not in seen_hail_paths:
        HAIL_PATH_CANDIDATES.append((source, path))
        seen_hail_paths.add(path)

EXPORT_WIDE_GENOTYPE = os.environ.get("Q3_VDR_EXPORT_WIDE_GENOTYPE", "true").strip().lower() in {"1", "true", "yes", "y"}
RUN_ENTRY_LEVEL_EXPORT = os.environ.get("Q3_VDR_RUN_ENTRY_LEVEL_EXPORT", "false").strip().lower() in {"1", "true", "yes", "y"}
EXPORT_POSITION_CANDIDATES = os.environ.get("Q3_VDR_EXPORT_POSITION_CANDIDATES", "false").strip().lower() in {"1", "true", "yes", "y"}
MAX_TARGET_ENTRY_ROWS_TO_PANDAS = int(os.environ.get("Q3_VDR_MAX_TARGET_ENTRY_ROWS_TO_PANDAS", "3000000"))
# Change this to True only if this block stops with `_ARRAY_API not found` or a NumPy 1.x/2.x pyarrow mismatch.
REPAIR_NUMPY_PYARROW_IN_CURRENT_KERNEL = False or (os.environ.get("Q3_VDR_REPAIR_NUMPY_PYARROW", "false").strip().lower() in {"1", "true", "yes", "y"})


try:
    import numpy as np
    import pandas as pd
    package_status_text = "\n".join([
        "package,status,detail",
        f"numpy,pass,{getattr(np, '__version__', '')}",
        f"pandas,pass,{getattr(pd, '__version__', '')}",
    ]) + "\n"
    (OUTPUT_DIR / "14_python_package_import_status.csv").write_text(package_status_text, encoding="utf-8")
except Exception as import_exc:
    import_tb = traceback.format_exc()
    (OUTPUT_DIR / "14_python_package_import_failure.txt").write_text(import_tb, encoding="utf-8")
    numpy_binary_mismatch = ("_ARRAY_API" in import_tb) or ("compiled using NumPy 1.x" in import_tb)
    if REPAIR_NUMPY_PYARROW_IN_CURRENT_KERNEL and numpy_binary_mismatch:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--force-reinstall", "--no-cache-dir", "numpy==1.26.4"])
        raise RuntimeError("Installed numpy==1.26.4 to repair the pandas/pyarrow NumPy binary mismatch. Restart the kernel, then rerun from Block 0.")
    if numpy_binary_mismatch:
        raise RuntimeError("Python package mismatch before analysis starts: pandas imports pyarrow compiled for NumPy 1.x, but this kernel is using NumPy 2.x. Set Q3_VDR_REPAIR_NUMPY_PYARROW=true before running this cell, or manually run `%pip install --user --force-reinstall --no-cache-dir numpy==1.26.4`, then restart the kernel and rerun from Block 0. Details saved to 14_python_package_import_failure.txt.") from import_exc
    raise


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


def pct(n, d):
    return round(float(n) / float(d) * 100, 3) if d else np.nan


def is_hail_backend_stopped_error(text):
    return any(token in str(text) for token in [
        "SparkContext was shut down",
        "Cannot call methods on a stopped SparkContext",
        "Could not find CoarseGrainedScheduler",
        "RemoteDisconnected",
        "Py4JNetworkError",
        "ConnectionError",
        "Answer from Java side is empty",
    ])


def hwe_chisq_p(n0, n1, n2):
    n0, n1, n2 = float(n0), float(n1), float(n2)
    n = n0 + n1 + n2
    if n <= 0:
        return np.nan
    p = (2 * n2 + n1) / (2 * n)
    q = 1 - p
    exp0 = n * q * q
    exp1 = 2 * n * p * q
    exp2 = n * p * p
    if min(exp0, exp1, exp2) <= 0:
        return np.nan
    chi = ((n0 - exp0) ** 2 / exp0) + ((n1 - exp1) ** 2 / exp1) + ((n2 - exp2) ** 2 / exp2)
    try:
        import scipy.stats as stats
        return float(stats.chi2.sf(chi, df=1))
    except Exception:
        # Wilson-Hilferty approximation fallback is unnecessary for decision use.
        return np.nan


target_df = pd.DataFrame(TARGET_SNPS)
save(target_df, "14_tier1_vdr_target_snp_list.csv")

save(pd.DataFrame([{
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "hail_path_candidates": ";".join([f"{source}={path}" for source, path in HAIL_PATH_CANDIDATES]),
    "export_wide_genotype": EXPORT_WIDE_GENOTYPE,
    "max_target_entry_rows_to_pandas": MAX_TARGET_ENTRY_ROWS_TO_PANDAS,
    "run_entry_level_export": RUN_ENTRY_LEVEL_EXPORT,
    "export_position_candidates": EXPORT_POSITION_CANDIDATES,
}]), "14_hail_run_configuration.csv")



# %% Cell 4
INSTALL_HAIL_IN_CURRENT_KERNEL = False  # Change to True only if Block 1 says: No module named 'hail'.
HAIL_VERSION_TO_INSTALL = os.environ.get("Q3_VDR_HAIL_VERSION", "0.2.137")

if INSTALL_HAIL_IN_CURRENT_KERNEL:
    import subprocess
    import sys
    print(f"Installing hail=={HAIL_VERSION_TO_INSTALL} with numpy==1.26.4 into this kernel: {sys.executable}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--force-reinstall", "--no-cache-dir", "numpy==1.26.4", f"hail=={HAIL_VERSION_TO_INSTALL}"])
    print("Install finished. Restart this kernel before rerunning Block 0 and Block 1.")
else:
    print("Skipped Hail install. If Block 1 fails at import_hail with ModuleNotFoundError, change INSTALL_HAIL_IN_CURRENT_KERNEL to True, run this cell once, restart the kernel, then rerun Block 1.")


# %% Cell 6
status_rows = []
selected_source = ""
selected_path = ""
mt = None
hl = None

try:
    import hail as hl
    status_rows.append({"step": "import_hail", "status": "pass", "detail": getattr(hl, "__version__", "")})
except Exception as exc:
    status_rows.append({"step": "import_hail", "status": "fail", "detail": repr(exc)[:2000]})

if hl is not None:
    try:
        try:
            hl.stop()
        except Exception:
            pass
        spark_conf = {
            "spark.hadoop.fs.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "spark.hadoop.fs.AbstractFileSystem.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
        }
        if GOOGLE_PROJECT:
            spark_conf.update({
                "spark.hadoop.fs.gs.requester.pays.mode": "CUSTOM",
                "spark.hadoop.fs.gs.requester.pays.project.id": GOOGLE_PROJECT,
                "spark.hadoop.fs.gs.requester.pays.buckets": "vwb-aou-datasets-controlled,fc-aou-datasets-controlled,genomics-public-data",
            })
        init_kwargs = {"quiet": True, "spark_conf": spark_conf}
        if HAIL_TMP_DIR:
            init_kwargs["tmp_dir"] = HAIL_TMP_DIR
        hl.init(**init_kwargs)
        try:
            hl.default_reference("GRCh38")
        except Exception as ref_exc:
            status_rows.append({"step": "hail_default_reference", "status": "warning", "detail": repr(ref_exc)[:1000]})
        status_rows.append({"step": "hail_init", "status": "pass", "detail": f"tmp_dir={HAIL_TMP_DIR}; requester_project={GOOGLE_PROJECT}; spark_conf_keys={';'.join(sorted(spark_conf))}"})
        gcs_probe_rows = []
        try:
            from pyspark import SparkContext
            sc = SparkContext._active_spark_context
        except Exception as sc_exc:
            sc = None
            gcs_probe_rows.append({"check": "spark_context", "status": "fail", "detail": repr(sc_exc)[:1000]})
        if sc is None:
            gcs_probe_rows.append({"check": "spark_context", "status": "missing", "detail": "No active SparkContext after hl.init"})
        else:
            hconf = sc._jsc.hadoopConfiguration()
            for key in [
                "fs.gs.impl",
                "fs.AbstractFileSystem.gs.impl",
                "fs.gs.requester.pays.mode",
                "fs.gs.requester.pays.project.id",
                "fs.gs.requester.pays.buckets",
            ]:
                try:
                    gcs_probe_rows.append({"check": key, "status": "value", "detail": str(hconf.get(key))})
                except Exception as key_exc:
                    gcs_probe_rows.append({"check": key, "status": "fail", "detail": repr(key_exc)[:1000]})
            for class_name in [
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            ]:
                try:
                    sc._jvm.java.lang.Class.forName(class_name)
                    gcs_probe_rows.append({"check": "jvm_class", "status": "present", "detail": class_name})
                except Exception as class_exc:
                    gcs_probe_rows.append({"check": "jvm_class", "status": "missing", "detail": f"{class_name}: {repr(class_exc)[:1000]}"})
            try:
                uri = sc._jvm.java.net.URI("gs://vwb-aou-datasets-controlled/")
                fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hconf)
                gcs_probe_rows.append({"check": "filesystem_get_gs", "status": "pass", "detail": str(fs.getClass().getName())})
            except Exception as fs_exc:
                gcs_probe_rows.append({"check": "filesystem_get_gs", "status": "fail", "detail": repr(fs_exc)[:2000]})
        save(pd.DataFrame(gcs_probe_rows), "14_hail_runtime_gcs_connector_probe.csv")
    except Exception as exc:
        status_rows.append({"step": "hail_init", "status": "fail", "detail": traceback.format_exc()[:4000]})

if hl is not None:
    for source, path in HAIL_PATH_CANDIDATES:
        try:
            print(f"Trying Hail MatrixTable: {source} = {path}")
            mt_candidate = hl.read_matrix_table(path)
            # Touch schema only; this should be lightweight.
            _ = str(mt_candidate.row.dtype)
            mt = mt_candidate
            selected_source = source
            selected_path = path
            status_rows.append({"step": "read_matrix_table", "status": "pass", "detail": f"{source}={path}"})
            break
        except Exception as exc:
            status_rows.append({"step": "read_matrix_table", "status": "fail", "detail": f"{source}={path}\n" + traceback.format_exc()[:5000]})

if mt is None:
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    save_text("\n\n".join([str(r) for r in status_rows]), "14_hail_read_failure_details.txt")
    failure_text = "\n".join([str(r.get("detail", "")) for r in status_rows])
    import_hail_failed = any((row.get("step") == "import_hail" and row.get("status") == "fail") for row in status_rows)
    no_hail_path_candidates = len(HAIL_PATH_CANDIDATES) == 0
    no_gs_filesystem = "No FileSystem for scheme" in failure_text and "gs" in failure_text
    missing_gcs_class = False
    try:
        missing_gcs_class = any((row.get("check") == "jvm_class" and row.get("status") == "missing") for row in gcs_probe_rows)
    except Exception:
        missing_gcs_class = False
    decision_lines = [
        "# Q3 VDR Hail Runtime Repair Decision",
        "",
        "Observed failure: Hail imported and initialized, but no candidate MatrixTable could be opened.",
        "This file is generated by Block 1 before genotype extraction stops.",
        "",
        "## Diagnosis",
    ]
    if import_hail_failed:
        decision_lines.extend([
            "- Primary issue: the selected Jupyter kernel cannot import the Python `hail` package.",
            "- No Hail initialization or MatrixTable read was attempted in this run.",
            "- In the notebook, run Optional Block 0b with `INSTALL_HAIL_IN_CURRENT_KERNEL = True`, then restart the kernel and rerun Blocks 0-1.",
            "- If installation is not allowed or fails, switch to an AoU app/kernel that already includes Hail or use an official AoU Hail example notebook in this workspace to identify the correct kernel.",
        ])
    elif no_hail_path_candidates:
        decision_lines.extend([
            "- Primary issue: no Hail MatrixTable path candidates were available.",
            "- Set `Q3_VDR_HAIL_MT_PATH` manually or verify the AoU v9 fallback paths in the setup cell.",
        ])
    elif no_gs_filesystem:
        decision_lines.extend([
            "- Primary issue: the Spark/Hadoop layer used by Hail cannot resolve the `gs://` filesystem scheme.",
            "- This is a runtime connector problem, not a VDR rsID/position problem.",
            "- `gsutil` access alone is not enough; Hail needs the Google Cloud Storage Hadoop connector in the JVM classpath.",
        ])
    elif missing_gcs_class:
        decision_lines.extend([
            "- Primary issue: Google Cloud Storage Hadoop connector classes are missing from the JVM classpath.",
            "- The notebook sets `fs.gs.impl`, but the classes must also exist in the runtime image.",
        ])
    else:
        decision_lines.extend([
            "- Primary issue was not uniquely classified by the notebook.",
            "- Review `14_hail_extraction_status.csv`, `14_hail_read_failure_details.txt`, and `14_hail_runtime_gcs_connector_probe.csv`.",
        ])
    decision_lines.extend([
        "",
        "## Next action in AoU",
        "1. Rerun this notebook in an AoU runtime/image where Hail can read `gs://` paths, preferably the genomics/Hail-enabled runtime.",
        "2. Confirm `GOOGLE_PROJECT` is set to the workspace billing project and, if available, `WORKSPACE_BUCKET` starts with `gs://`.",
        "3. Reopen `14_hail_runtime_gcs_connector_probe.csv`; `filesystem_get_gs` should be `pass` before MatrixTable extraction can work.",
        "4. After that passes, continue Blocks 2 onward to produce VDR MAF, missingness, HWE, LD, and dosage/hard-call outputs.",
        "",
        "## Interpretation impact",
        "- Do not interpret this failure as evidence that the four VDR SNPs are absent from AoU.",
        "- Method and mechanism wording that depends on REF/ALT, dosage, MAF, HWE, missingness, or LD must wait until the MatrixTable can be read successfully.",
    ])
    save_text("\n".join(decision_lines), "14_hail_runtime_repair_decision.md")
    if import_hail_failed:
        raise RuntimeError("Current kernel cannot import hail. Run Optional Block 0b to install hail in this kernel, restart the kernel, then rerun Blocks 0-1; or switch to an AoU Hail-enabled kernel/app.")
    if no_hail_path_candidates:
        raise RuntimeError("No Hail MatrixTable path candidates are available. Set Q3_VDR_HAIL_MT_PATH or use the AoU v9 fallback paths in the setup cell.")
    if no_gs_filesystem or missing_gcs_class:
        raise RuntimeError("Hail cannot read gs:// in this runtime. Review 14_hail_runtime_gcs_connector_probe.csv and 14_hail_runtime_repair_decision.md, then rerun in a Hail/genomics runtime with the Google Cloud Storage Hadoop connector available.")
    raise RuntimeError("No Hail MatrixTable could be read. Review 14_hail_extraction_status.csv, 14_hail_read_failure_details.txt, and 14_hail_runtime_gcs_connector_probe.csv.")

schema_text = "\n".join([
    f"selected_source: {selected_source}",
    f"selected_path: {selected_path}",
    "",
    "row_key:",
    str(mt.row_key),
    "",
    "col_key:",
    str(mt.col_key),
    "",
    "entry fields:",
    str(mt.entry.dtype),
    "",
    "row fields:",
    str(mt.row.dtype),
    "",
    "col fields:",
    str(mt.col.dtype),
])
save_text(schema_text, "14_hail_schema_summary.txt")
save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
runtime_success_note = f"""# Q3 VDR Hail Runtime Read Status

Current Block 1 status: success.

Selected Hail source: `{selected_source}`

Selected Hail path:

```text
{selected_path}
```

The current runtime can import Hail, initialize Spark/Hail, resolve `gs://`, and read the selected MatrixTable schema. No runtime repair is needed for this run. If later blocks fail, debug the block-specific Hail expression or data logic rather than the GCS connector.
"""
save_text(runtime_success_note, "14_hail_runtime_read_success.md")
save_text(runtime_success_note, "14_hail_runtime_repair_decision.md")



# %% Cell 8
target_min = min(x["position"] for x in TARGET_SNPS) - 100
target_max = max(x["position"] for x in TARGET_SNPS) + 100
target_positions = {int(x["position"]) for x in TARGET_SNPS}
target_by_expected_vid = {x["expected_vid"]: x["rsid"] for x in TARGET_SNPS}

has_locus = "locus" in mt.row
has_alleles = "alleles" in mt.row
if not has_locus or not has_alleles:
    save(pd.DataFrame(status_rows + [{"step": "target_row_filter", "status": "fail", "detail": "MatrixTable does not expose row fields named locus and alleles."}]), "14_hail_extraction_status.csv")
    raise RuntimeError("Expected row fields `locus` and `alleles` were not found.")

intervals = []
interval_probe_rows = []
try:
    rg = mt.locus.dtype.reference_genome
    rg_name = rg.name
    rg_contigs = set(getattr(rg, "contigs", []) or [])
except Exception as rg_exc:
    rg = None
    rg_name = "GRCh38"
    rg_contigs = set()
    interval_probe_rows.append({"step": "reference_genome", "status": "fallback", "detail": repr(rg_exc)[:1000]})

candidate_contigs = [contig for contig in ["chr12", "12"] if (not rg_contigs or contig in rg_contigs)]
if not candidate_contigs:
    candidate_contigs = ["chr12", "12"]
interval_probe_rows.append({"step": "reference_genome", "status": "selected", "detail": f"rg_name={rg_name}; candidate_contigs={candidate_contigs}; has_chr12={'chr12' in rg_contigs}; has_12={'12' in rg_contigs}"})

for contig in candidate_contigs:
    try:
        start_locus = hl.Locus(contig, target_min, reference_genome=rg_name)
        end_locus = hl.Locus(contig, target_max + 1, reference_genome=rg_name)
        intervals.append(hl.Interval(start_locus, end_locus, includes_start=True, includes_end=False))
        interval_probe_rows.append({"step": "construct_interval", "status": "pass", "detail": f"[{contig}:{target_min}-{contig}:{target_max + 1})"})
    except Exception as interval_exc:
        interval_probe_rows.append({"step": "construct_interval", "status": "fail", "detail": f"{contig}: {repr(interval_exc)[:1000]}"})

try:
    save(pd.DataFrame(interval_probe_rows), "14_hail_block2_interval_probe.csv")
except Exception:
    pass

if intervals:
    try:
        mt_region = hl.filter_intervals(mt, intervals)
        status_rows.append({"step": "region_filter", "status": "interval", "detail": f"intervals={len(intervals)}; rg={rg_name}; contigs={candidate_contigs}"})
    except Exception as interval_filter_exc:
        status_rows.append({"step": "region_filter", "status": "interval_failed_using_expression_fallback", "detail": traceback.format_exc()[:2000]})
        mt_region = mt.filter_rows(((mt.locus.contig == "chr12") | (mt.locus.contig == "12")) & (mt.locus.position >= target_min) & (mt.locus.position <= target_max))
else:
    status_rows.append({"step": "region_filter", "status": "expression_fallback", "detail": f"No valid interval objects; rg={rg_name}; contigs={candidate_contigs}"})
    mt_region = mt.filter_rows(((mt.locus.contig == "chr12") | (mt.locus.contig == "12")) & (mt.locus.position >= target_min) & (mt.locus.position <= target_max))

target_vid_literal = hl.literal(target_by_expected_vid)
position_literal = hl.literal(target_positions)

mt_region_positions = mt_region.filter_rows(position_literal.contains(mt_region.locus.position) & (hl.len(mt_region.alleles) >= 2))
region_candidate_rows = mt_region_positions.rows()
region_candidate_rows = region_candidate_rows.annotate(
    candidate_contig=hl.str(region_candidate_rows.locus.contig),
    candidate_position=region_candidate_rows.locus.position,
    candidate_ref=region_candidate_rows.alleles[0],
    candidate_alt=region_candidate_rows.alleles[1],
    expected_vid=hl.str(region_candidate_rows.locus.contig).replace("chr", "") + "-" + hl.str(region_candidate_rows.locus.position) + "-" + region_candidate_rows.alleles[0] + "-" + region_candidate_rows.alleles[1],
)
region_candidate_rows = region_candidate_rows.select(
    "candidate_contig",
    "candidate_position",
    "candidate_ref",
    "candidate_alt",
    "expected_vid",
)
if EXPORT_POSITION_CANDIDATES:
    try:
        save(region_candidate_rows.to_pandas(), "14_tier1_vdr_hail_position_candidate_rows.csv")
    except Exception as candidate_export_exc:
        candidate_export_tb = traceback.format_exc()
        save_text(candidate_export_tb, "14_hail_block2_position_candidates_export_failure.txt")
        save(pd.DataFrame([{"status": "failed_to_export_position_candidate_rows", "detail": repr(candidate_export_exc)[:1500]}]), "14_tier1_vdr_hail_position_candidate_rows.csv")
        if is_hail_backend_stopped_error(candidate_export_tb):
            status_rows.append({"step": "position_candidate_export", "status": "fail_backend_stopped", "detail": candidate_export_tb[:2000]})
            save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
            raise RuntimeError("Hail/Spark backend stopped during optional Block 2 candidate-row export. Restart the AoU Spark/Hail app or kernel, rerun from Block 1, and leave Q3_VDR_EXPORT_POSITION_CANDIDATES=false. Details saved to 14_hail_block2_position_candidates_export_failure.txt.") from candidate_export_exc
else:
    save(pd.DataFrame([{"status": "skipped_disabled_by_default", "detail": "Set Q3_VDR_EXPORT_POSITION_CANDIDATES=true only for debugging target-position rows."}]), "14_tier1_vdr_hail_position_candidate_rows.csv")

mt_target = mt_region_positions.annotate_rows(
    expected_vid=hl.str(mt_region_positions.locus.contig).replace("chr", "") + "-" + hl.str(mt_region_positions.locus.position) + "-" + mt_region_positions.alleles[0] + "-" + mt_region_positions.alleles[1],
)
mt_target = mt_target.annotate_rows(target_rsid=target_vid_literal.get(mt_target.expected_vid))
mt_target = mt_target.filter_rows(hl.is_defined(mt_target.target_rsid))

row_table = mt_target.rows()
row_table = row_table.select(
    target_rsid=row_table.target_rsid,
    expected_vid=row_table.expected_vid,
    contig=hl.str(row_table.locus.contig),
    position=row_table.locus.position,
    ref=row_table.alleles[0],
    alt=row_table.alleles[1],
    a_index=row_table.a_index if "a_index" in row_table.row else hl.missing(hl.tint32),
    was_split=row_table.was_split if "was_split" in row_table.row else hl.missing(hl.tbool),
)
try:
    row_df = row_table.to_pandas()
except Exception as row_export_exc:
    row_export_tb = traceback.format_exc()
    save_text(row_export_tb, "14_hail_block2_target_rows_export_failure.txt")
    status_rows.append({"step": "target_row_export", "status": "fail_backend_stopped" if is_hail_backend_stopped_error(row_export_tb) else "fail", "detail": row_export_tb[:2000]})
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    if is_hail_backend_stopped_error(row_export_tb):
        raise RuntimeError("Hail/Spark backend stopped during Block 2 target-row export. Restart the AoU Spark/Hail app or kernel, rerun from Block 1, and avoid running package-install cells after Hail is initialized. Details saved to 14_hail_block2_target_rows_export_failure.txt.") from row_export_exc
    raise
target_count_rows = int(len(row_df))
target_count_cols = np.nan
status_rows.append({"step": "target_row_filter", "status": "pass_lazy_cols_not_counted", "detail": f"target_rows={target_count_rows}; cols=not_counted_in_block2"})
save(row_df, "14_tier1_vdr_hail_target_rows.csv")
save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

if target_count_rows == 0:
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    raise RuntimeError("No exact target rows found. Review contig/position/ref/alt in 14_tier1_vdr_hail_target_rows.csv and schema summary.")



# %% Cell 10
entry_fields = set(mt_target.entry)

if "GT" in entry_fields:
    dosage_expr = hl.or_missing(hl.is_defined(mt_target.GT), mt_target.GT.n_alt_alleles())
    genotype_source = "GT.n_alt_alleles"
elif "LGT" in entry_fields:
    dosage_expr = hl.or_missing(hl.is_defined(mt_target.LGT), mt_target.LGT.n_alt_alleles())
    genotype_source = "LGT.n_alt_alleles"
elif "DS" in entry_fields:
    dosage_expr = mt_target.DS
    genotype_source = "DS"
elif "dosage" in entry_fields:
    dosage_expr = mt_target.dosage
    genotype_source = "dosage"
else:
    save(pd.DataFrame(status_rows + [{"step": "define_dosage", "status": "fail", "detail": f"No supported genotype field found in entry fields: {sorted(entry_fields)}"}]), "14_hail_extraction_status.csv")
    raise RuntimeError("No supported entry genotype field found. Expected GT, LGT, DS, or dosage.")

mt_target = mt_target.annotate_entries(q3_alt_dosage=dosage_expr)
status_rows.append({"step": "define_dosage", "status": "pass", "detail": genotype_source})
save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")



# %% Cell 12
# Use row-level AoU/Hail precomputed QC fields by default. Re-aggregating all entries caused Py4J/Hail backend disconnects in this workspace.
qc_row_table = mt_target.rows()
has_variant_qc = "variant_qc" in qc_row_table.row
has_info = "info" in qc_row_table.row
if not has_variant_qc or not has_info:
    save(pd.DataFrame(status_rows + [{"step": "genotype_qc_summary", "status": "fail", "detail": "Expected row fields `variant_qc` and `info` were not found for lightweight QC."}]), "14_hail_extraction_status.csv")
    raise RuntimeError("Block 4 needs row-level `variant_qc` and `info` fields for lightweight QC in this runtime.")

qc_row_table = qc_row_table.select(
    rsid=qc_row_table.target_rsid,
    expected_vid=qc_row_table.expected_vid,
    n_called=qc_row_table.variant_qc.n_called,
    variant_qc_n_not_called=qc_row_table.variant_qc.n_not_called,
    variant_qc_n_filtered=qc_row_table.variant_qc.n_filtered,
    variant_qc_call_rate=qc_row_table.variant_qc.call_rate,
    variant_qc_n_het=qc_row_table.variant_qc.n_het,
    variant_qc_p_value_hwe=qc_row_table.variant_qc.p_value_hwe,
    info_AC=qc_row_table.info.AC,
    info_AF=qc_row_table.info.AF,
    info_AN=qc_row_table.info.AN,
    info_homozygote_count=qc_row_table.info.homozygote_count,
)

try:
    qc_df = qc_row_table.to_pandas()
except Exception as qc_export_exc:
    qc_export_tb = traceback.format_exc()
    save_text(qc_export_tb, "14_hail_block4_qc_export_failure.txt")
    status_rows.append({"step": "genotype_qc_summary", "status": "fail", "detail": qc_export_tb[:2000]})
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    if is_hail_backend_stopped_error(qc_export_tb):
        raise RuntimeError("Hail/Spark backend disconnected during Block 4 lightweight row-QC export. Restart the AoU Spark/Hail app or kernel, rerun from Block 1, and avoid package-install cells after Hail is initialized. Details saved to 14_hail_block4_qc_export_failure.txt.") from qc_export_exc
    raise

def list_numeric(value):
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        if not cleaned:
            return []
        out = []
        for item in cleaned.split(","):
            try:
                out.append(float(item.strip()))
            except Exception:
                pass
        return out
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            return [float(x) for x in value]
    except Exception:
        return []
    try:
        return [float(value)]
    except Exception:
        return []

def first_numeric(value):
    values = list_numeric(value)
    return values[0] if values else np.nan

if not qc_df.empty:
    qc_df["n_called"] = pd.to_numeric(qc_df["n_called"], errors="coerce")
    qc_df["variant_qc_n_not_called"] = pd.to_numeric(qc_df["variant_qc_n_not_called"], errors="coerce")
    qc_df["variant_qc_n_filtered"] = pd.to_numeric(qc_df["variant_qc_n_filtered"], errors="coerce")
    qc_df["variant_qc_call_rate"] = pd.to_numeric(qc_df["variant_qc_call_rate"], errors="coerce")
    qc_df["n_total_cols"] = qc_df["n_called"] + qc_df["variant_qc_n_not_called"].fillna(0) + qc_df["variant_qc_n_filtered"].fillna(0)
    n_from_call_rate = (qc_df["n_called"] / qc_df["variant_qc_call_rate"].replace(0, np.nan)).round()
    qc_df["n_total_cols"] = qc_df["n_total_cols"].where(qc_df["n_total_cols"].notna() & (qc_df["n_total_cols"] > 0), n_from_call_rate)
    qc_df["n_missing"] = qc_df["n_total_cols"] - qc_df["n_called"]
    qc_df["genotype_1"] = pd.to_numeric(qc_df["variant_qc_n_het"], errors="coerce")
    qc_df["_ac_first"] = qc_df["info_AC"].map(first_numeric)
    qc_df["_af_first"] = qc_df["info_AF"].map(first_numeric)
    qc_df["_hom_counts"] = qc_df["info_homozygote_count"].map(list_numeric)
    qc_df["_hom_ref"] = qc_df["_hom_counts"].map(lambda xs: xs[0] if len(xs) > 0 else np.nan)
    qc_df["_hom_alt"] = qc_df["_hom_counts"].map(lambda xs: xs[1] if len(xs) > 1 else np.nan)
    qc_df["genotype_2"] = qc_df["_hom_alt"]
    fallback_g2 = (qc_df["_ac_first"] - qc_df["genotype_1"]) / 2
    qc_df["genotype_2"] = qc_df["genotype_2"].where(qc_df["genotype_2"].notna(), fallback_g2)
    qc_df["genotype_0"] = qc_df["_hom_ref"]
    fallback_g0 = qc_df["n_called"] - qc_df["genotype_1"] - qc_df["genotype_2"]
    qc_df["genotype_0"] = qc_df["genotype_0"].where(qc_df["genotype_0"].notna(), fallback_g0)
    qc_df["dosage_mean"] = (2 * qc_df["genotype_2"] + qc_df["genotype_1"]) / qc_df["n_called"].replace(0, np.nan)
    dosage_af = qc_df["dosage_mean"] / 2
    qc_df["alt_allele_frequency"] = qc_df["_af_first"].where(qc_df["_af_first"].notna(), dosage_af)
    qc_df["af_dosage_abs_diff"] = (qc_df["alt_allele_frequency"] - dosage_af).abs()
    qc_df["hom_count_orientation"] = "info_homozygote_count[0]=hom_ref;[1]=hom_alt"
    qc_df["maf"] = qc_df["alt_allele_frequency"].map(lambda x: min(x, 1 - x) if pd.notna(x) else np.nan)
    qc_df["missing_pct"] = qc_df.apply(lambda r: pct(r["n_missing"], r["n_total_cols"]), axis=1)
    qc_df["hwe_p_chisq"] = qc_df["variant_qc_p_value_hwe"]
    missing_hwe = qc_df["hwe_p_chisq"].isna()
    if missing_hwe.any():
        qc_df.loc[missing_hwe, "hwe_p_chisq"] = qc_df.loc[missing_hwe].apply(lambda r: hwe_chisq_p(r["genotype_0"], r["genotype_1"], r["genotype_2"]), axis=1)
    qc_df["genotype_source"] = genotype_source
    qc_df["qc_method"] = "precomputed_variant_qc_info_row_fields"
    qc_df = qc_df[[
        "rsid", "expected_vid", "n_total_cols", "n_called", "n_missing", "missing_pct",
        "genotype_0", "genotype_1", "genotype_2", "dosage_mean", "alt_allele_frequency", "maf",
        "hwe_p_chisq", "genotype_source", "qc_method", "af_dosage_abs_diff", "hom_count_orientation", "variant_qc_call_rate", "variant_qc_n_not_called", "variant_qc_n_filtered", "info_AC", "info_AF", "info_AN", "info_homozygote_count",
    ]]
save(qc_df, "14_tier1_vdr_genotype_qc_summary.csv")
save_text("Block 4 used precomputed Hail row fields `variant_qc` and `info` instead of entry-level aggregation to avoid Py4J/Spark backend crashes in this AoU runtime.\n", "14_hail_block4_qc_method_note.md")
n_cols_values = pd.to_numeric(qc_df.get("n_total_cols", pd.Series(dtype=float)), errors="coerce")
n_cols_summary = int(n_cols_values.max()) if len(n_cols_values) and n_cols_values.notna().any() else 0
target_count_cols = n_cols_summary if n_cols_summary else np.nan
status_rows.append({"step": "genotype_qc_summary", "status": "pass_precomputed_row_qc", "detail": f"rows={len(qc_df)}; inferred_max_cols={n_cols_summary}"})
save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")



# %% Cell 15
# Block 5A. Column and entry preflight diagnostics.
block5_status_rows = []


def block5_status(step, status, detail):
    row = {"step": step, "status": status, "detail": str(detail)}
    block5_status_rows.append(row)
    status_rows.append(row)
    return row


cols_table = mt_target.cols()
cols_fields = list(cols_table.row)
col_key_fields = list(mt_target.col_key)

sample_source = ""
for field in ["s", "sample_id", "person_id", "sample_name"]:
    if field in cols_table.row:
        sample_source = field
        break

if not sample_source:
    sample_source = col_key_fields[0] if col_key_fields else "col_idx_unavailable"

if sample_source != "col_idx_unavailable" and sample_source in cols_table.row:
    col_id_table = cols_table.select(sample_id=hl.str(cols_table[sample_source]))
else:
    col_id_table = cols_table.select(sample_id=hl.str(""))

try:
    col_probe_rows = col_id_table.take(20)
    col_probe_df = pd.DataFrame([{"sample_source": sample_source, "sample_id": str(row.sample_id)} for row in col_probe_rows])
    if not col_probe_df.empty:
        col_probe_df["sample_id_is_numeric_person_id"] = col_probe_df["sample_id"].str.match(r"^\d+$")
    save(col_probe_df, "14_tier1_vdr_column_id_probe.csv")
    col_id_summary = col_id_table.aggregate(hl.struct(
        n_samples=hl.agg.count(),
        n_numeric_person_id=hl.agg.count_where(hl.is_defined(col_id_table.sample_id) & col_id_table.sample_id.matches(r"^\d+$")),
    ))
    col_n_samples = int(col_id_summary.n_samples)
    col_n_numeric_person_id = int(col_id_summary.n_numeric_person_id)
    block5_status("column_id_probe", "pass", f"sample_source={sample_source}; n_samples={col_n_samples}; numeric_like_person_id={col_n_numeric_person_id}")
except Exception:
    col_probe_tb = traceback.format_exc()
    save_text(col_probe_tb, "14_hail_block5_column_probe_failure.txt")
    col_n_samples = 0
    col_n_numeric_person_id = 0
    block5_status("column_id_probe", "fail", col_probe_tb[:2000])

entries_table = mt_target.entries()
entries_fields = list(entries_table.row)

if sample_source in entries_table.row:
    entry_sample_source = sample_source
    entry_sample_expr = hl.str(entries_table[entry_sample_source])
else:
    fallback_sample_source = next((field for field in ["s", "sample_id", "person_id", "sample_name"] if field in entries_table.row), "")
    if fallback_sample_source:
        entry_sample_source = fallback_sample_source
        entry_sample_expr = hl.str(entries_table[entry_sample_source])
    else:
        entry_sample_source = "missing"
        entry_sample_expr = hl.str("")

entries = entries_table.select(
    sample_id=entry_sample_expr,
    target_rsid=entries_table.target_rsid,
    expected_vid=entries_table.expected_vid,
    q3_alt_dosage=entries_table.q3_alt_dosage,
)

block5_preflight = pd.DataFrame([{
    "sample_source": sample_source,
    "entry_sample_source": entry_sample_source,
    "col_key_fields": ";".join(map(str, col_key_fields)),
    "cols_fields": ";".join(map(str, cols_fields)),
    "entries_has_sample_source": sample_source in entries_table.row,
    "entries_has_target_rsid": "target_rsid" in entries_table.row,
    "entries_has_expected_vid": "expected_vid" in entries_table.row,
    "entries_has_q3_alt_dosage": "q3_alt_dosage" in entries_table.row,
    "n_samples_from_column_probe": col_n_samples,
    "n_numeric_person_id_from_column_probe": col_n_numeric_person_id,
    "pct_numeric_person_id_from_column_probe": pct(col_n_numeric_person_id, col_n_samples),
}])
save(block5_preflight, "14_hail_block5_preflight_summary.csv")
block5_status("block5_preflight", "pass", f"entry_sample_source={entry_sample_source}; entries_fields={len(entries_fields)}")
save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

# %% Cell 17
# Block 5B. Per-SNP entry vector smoke tests.
# Default is intentionally non-entry-level. Prior AoU runs showed that even entries.take(5)
# can shut down SparkContext in this runtime. Keep this block as a diagnostic gate.
RUN_ENTRY_SMOKE = os.environ.get("Q3_VDR_RUN_ENTRY_SMOKE", "false").strip().lower() in {"1", "true", "yes", "y"}
rsids_for_ld = [x["rsid"] for x in TARGET_SNPS]

vector_probe_df = pd.DataFrame(columns=["rsid", "sample_id", "expected_vid", "q3_alt_dosage"])

if not RUN_ENTRY_SMOKE:
    if "qc_df" in globals() and not qc_df.empty:
        keep_cols = [c for c in [
            "rsid", "n_total_cols", "n_called", "n_missing", "missing_pct",
            "genotype_0", "genotype_1", "genotype_2", "dosage_mean",
            "alt_allele_frequency", "maf", "genotype_source", "qc_method",
        ] if c in qc_df.columns]
        vector_summary_df = qc_df[keep_cols].copy()
        vector_summary_df["smoke_method"] = "row_level_qc_proxy_no_entry_action"
    else:
        vector_summary_df = pd.DataFrame(columns=[
            "rsid", "n_total_cols", "n_called", "n_missing", "missing_pct",
            "genotype_0", "genotype_1", "genotype_2", "dosage_mean",
            "alt_allele_frequency", "maf", "genotype_source", "qc_method", "smoke_method",
        ])
    save(vector_probe_df, "14_hail_block5_snp_vector_probe.csv")
    save(vector_summary_df, "14_hail_block5_snp_vector_summary.csv")
    block5_status(
        "snp_vector_smoke",
        "skip_disabled_by_default",
        "Q3_VDR_RUN_ENTRY_SMOKE=false; skipped entry-level take/aggregate because this AoU runtime previously shut down SparkContext on entries.take(5). Row-level QC proxy saved.",
    )
else:
    vector_probe_rows = []
    vector_summary_rows = []
    try:
        for rsid in rsids_for_ld:
            e = entries.filter(entries.target_rsid == rsid)
            probe_rows = e.take(5)
            for row in probe_rows:
                vector_probe_rows.append({
                    "rsid": rsid,
                    "sample_id": str(row.sample_id),
                    "expected_vid": row.expected_vid,
                    "q3_alt_dosage": None if row.q3_alt_dosage is None else int(row.q3_alt_dosage),
                })
            stats = e.aggregate(hl.struct(
                n_rows=hl.agg.count(),
                n_called=hl.agg.count_where(hl.is_defined(e.q3_alt_dosage)),
                n_missing=hl.agg.count_where(hl.is_missing(e.q3_alt_dosage)),
                n_dosage_0=hl.agg.count_where(e.q3_alt_dosage == 0),
                n_dosage_1=hl.agg.count_where(e.q3_alt_dosage == 1),
                n_dosage_2=hl.agg.count_where(e.q3_alt_dosage == 2),
                dosage_mean=hl.agg.mean(hl.float64(e.q3_alt_dosage)),
            ))
            vector_summary_rows.append({
                "rsid": rsid,
                "n_rows": int(stats.n_rows),
                "n_called": int(stats.n_called),
                "n_missing": int(stats.n_missing),
                "missing_pct": pct(int(stats.n_missing), int(stats.n_rows)),
                "genotype_0": int(stats.n_dosage_0),
                "genotype_1": int(stats.n_dosage_1),
                "genotype_2": int(stats.n_dosage_2),
                "dosage_mean": float(stats.dosage_mean) if stats.dosage_mean is not None else np.nan,
                "alt_allele_frequency": float(stats.dosage_mean / 2) if stats.dosage_mean is not None else np.nan,
                "smoke_method": "entry_level_take_and_aggregate",
            })
        vector_probe_df = pd.DataFrame(vector_probe_rows)
        vector_summary_df = pd.DataFrame(vector_summary_rows)
        if not vector_summary_df.empty:
            vector_summary_df["maf"] = vector_summary_df["alt_allele_frequency"].map(lambda x: min(x, 1 - x) if pd.notna(x) else np.nan)
        save(vector_probe_df, "14_hail_block5_snp_vector_probe.csv")
        save(vector_summary_df, "14_hail_block5_snp_vector_summary.csv")
        block5_status("snp_vector_smoke", "pass", f"snps={len(vector_summary_df)}")
    except Exception:
        vector_tb = traceback.format_exc()
        save_text(vector_tb, "14_hail_block5_snp_vector_failure.txt")
        vector_summary_df = pd.DataFrame(columns=["rsid", "n_rows", "n_called", "n_missing", "missing_pct", "genotype_0", "genotype_1", "genotype_2", "dosage_mean", "alt_allele_frequency", "maf", "smoke_method"])
        save(vector_summary_df, "14_hail_block5_snp_vector_summary.csv")
        block5_status("snp_vector_smoke", "fail", vector_tb[:2000])
        if any(token in vector_tb for token in ["SparkContext was shut down", "CoarseGrainedScheduler", "Py4JNetworkError", "RemoteDisconnected"]):
            block5_status("snp_vector_smoke_runtime", "fatal_runtime", "Spark/Hail backend appears dead after entry-level smoke test. Restart kernel before any additional Hail action.")

save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

# %% Cell 19
# Block 5C. Full pairwise LD by Hail's ld_matrix path.
# Avoid entries_table joins/sorts. This still scans entry data, but uses Hail's optimized LD routine.
RUN_HAIL_LD = os.environ.get("Q3_VDR_RUN_HAIL_LD", "true").strip().lower() in {"1", "true", "yes", "y"}
ld_rows = []

if not RUN_HAIL_LD:
    ld_df = pd.DataFrame(columns=["snp1", "snp2", "n_pair_called", "correlation", "r2", "d_prime", "ld_method", "pair_status"])
    block5_status("pairwise_ld_hail_ld_matrix", "skip_disabled_by_env", "Q3_VDR_RUN_HAIL_LD=false")
else:
    try:
        mt_ld = mt_target.add_row_index("q3_ld_idx")
        ld_rows_table = mt_ld.rows()
        ld_row_order_df = ld_rows_table.select(
            q3_ld_idx=ld_rows_table.q3_ld_idx,
            rsid=ld_rows_table.target_rsid,
            expected_vid=ld_rows_table.expected_vid,
            contig=ld_rows_table.locus.contig,
            position=ld_rows_table.locus.position,
            ref=ld_rows_table.alleles[0],
            alt=ld_rows_table.alleles[1],
        ).to_pandas()
        ld_row_order_df["q3_ld_idx"] = pd.to_numeric(ld_row_order_df["q3_ld_idx"], errors="coerce").astype(int)
        ld_row_order_df = ld_row_order_df.sort_values("q3_ld_idx").reset_index(drop=True)
        save(ld_row_order_df, "14_hail_block5_ld_row_order.csv")

        ld_result = hl.ld_matrix(mt_ld.q3_alt_dosage, mt_ld.locus, radius=1_000_000)
        ld_bm = ld_result[0] if isinstance(ld_result, tuple) else ld_result
        ld_np = ld_bm.to_numpy()

        for i in range(len(ld_row_order_df)):
            for j in range(i + 1, len(ld_row_order_df)):
                corr = float(ld_np[i, j]) if pd.notna(ld_np[i, j]) else np.nan
                if pd.isna(corr) and j < ld_np.shape[0] and i < ld_np.shape[1]:
                    corr = float(ld_np[j, i]) if pd.notna(ld_np[j, i]) else np.nan
                ld_rows.append({
                    "snp1": ld_row_order_df.loc[i, "rsid"],
                    "snp2": ld_row_order_df.loc[j, "rsid"],
                    "n_pair_called": np.nan,
                    "correlation": corr,
                    "r2": corr ** 2 if pd.notna(corr) else np.nan,
                    "d_prime": np.nan,
                    "ld_method": "Hail hl.ld_matrix dosage correlation; D-prime not estimated from unphased hard calls",
                    "pair_status": "pass" if pd.notna(corr) else "missing_corr",
                })
        ld_df = pd.DataFrame(ld_rows)
        block5_status("pairwise_ld_hail_ld_matrix", "pass", f"pairs={len(ld_df)}; nonmissing_r2={pd.to_numeric(ld_df.get('r2', pd.Series(dtype=float)), errors='coerce').notna().sum()}")
    except Exception:
        ld_tb = traceback.format_exc()
        save_text(ld_tb, "14_hail_block5_ld_failure.txt")
        ld_df = pd.DataFrame(ld_rows) if ld_rows else pd.DataFrame(columns=["snp1", "snp2", "n_pair_called", "correlation", "r2", "d_prime", "ld_method", "pair_status"])
        block5_status("pairwise_ld_hail_ld_matrix", "fail", ld_tb[:2000])
        if any(token in ld_tb for token in ["SparkContext was shut down", "CoarseGrainedScheduler", "Py4JNetworkError", "RemoteDisconnected"]):
            block5_status("pairwise_ld_runtime", "fatal_runtime", "Spark/Hail backend appears dead during ld_matrix. Restart kernel before any additional Hail action.")

save(ld_df, "14_tier1_vdr_pairwise_ld.csv")
save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

# %% Cell 21
# Block 5D. Optional person-level long/wide genotype export.
target_count_rows_for_export = int(target_count_rows) if "target_count_rows" in globals() and pd.notna(target_count_rows) else len(TARGET_SNPS)
target_count_cols_for_export = np.nan
if "target_count_cols" in globals() and pd.notna(target_count_cols):
    target_count_cols_for_export = int(target_count_cols)
else:
    qc_summary_path = OUTPUT_DIR / "14_tier1_vdr_genotype_qc_summary.csv"
    if qc_summary_path.exists():
        try:
            qc_cols_df = pd.read_csv(qc_summary_path, usecols=["n_total_cols"])
            qc_cols_values = pd.to_numeric(qc_cols_df["n_total_cols"], errors="coerce")
            target_count_cols_for_export = int(qc_cols_values.max()) if qc_cols_values.notna().any() else np.nan
        except Exception:
            target_count_cols_for_export = np.nan
entries_n = target_count_rows_for_export * target_count_cols_for_export if pd.notna(target_count_cols_for_export) else np.nan
block5_status("entry_export_size_check", "pass", f"target_rows_x_cols={entries_n}; run_entry_level_export={RUN_ENTRY_LEVEL_EXPORT}")

if not RUN_ENTRY_LEVEL_EXPORT:
    block5_status("entry_to_pandas", "skip_disabled_by_default", "Set Q3_VDR_RUN_ENTRY_LEVEL_EXPORT=true only if person-level long/wide genotype export is needed. Pairwise dosage r2 is computed in Block 5C by default.")
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    sample_summary = pd.DataFrame([{
        "sample_source": sample_source,
        "n_samples": col_n_samples,
        "n_sample_id_numeric_person_id": col_n_numeric_person_id,
        "pct_sample_id_numeric_person_id": pct(col_n_numeric_person_id, col_n_samples),
        "n_complete_four_snp_genotype": 0,
        "entry_export_status": "skipped_disabled_by_default_ld_attempted_in_hail",
        "entry_export_rows_if_enabled": entries_n,
    }])
    save(sample_summary, "14_tier1_vdr_sample_id_person_id_summary.csv")
elif pd.notna(entries_n) and entries_n > MAX_TARGET_ENTRY_ROWS_TO_PANDAS:
    block5_status("entry_to_pandas", "skip_too_many_rows", f"{entries_n} > {MAX_TARGET_ENTRY_ROWS_TO_PANDAS}")
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    sample_summary = pd.DataFrame([{
        "sample_source": sample_source,
        "n_samples": col_n_samples,
        "n_sample_id_numeric_person_id": col_n_numeric_person_id,
        "pct_sample_id_numeric_person_id": pct(col_n_numeric_person_id, col_n_samples),
        "n_complete_four_snp_genotype": 0,
        "entry_export_status": "skipped_too_many_rows_ld_attempted_in_hail",
        "entry_export_rows_if_enabled": entries_n,
    }])
    save(sample_summary, "14_tier1_vdr_sample_id_person_id_summary.csv")
else:
    try:
        entries_df = entries.to_pandas()
    except Exception as entry_export_exc:
        entry_export_tb = traceback.format_exc()
        save_text(entry_export_tb, "14_hail_block5_entry_export_failure.txt")
        block5_status("entry_to_pandas", "fail", entry_export_tb[:2000])
        save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
        if any(token in entry_export_tb for token in ["RemoteDisconnected", "Py4JNetworkError", "ConnectionError", "SparkContext was shut down"]):
            raise RuntimeError("Hail/Spark backend disconnected during Block 5D entry-level export. Keep Q3_VDR_RUN_ENTRY_LEVEL_EXPORT=false for this runtime, or rerun in a larger/stabler Spark/Hail app. Details saved to 14_hail_block5_entry_export_failure.txt.") from entry_export_exc
        raise
    entries_df["q3_alt_dosage"] = pd.to_numeric(entries_df["q3_alt_dosage"], errors="coerce")
    entries_df["person_id"] = pd.to_numeric(entries_df["sample_id"], errors="coerce").astype("Int64")
    save(entries_df, "14_tier1_vdr_genotype_long.csv")

    wide = entries_df.pivot_table(index=["sample_id", "person_id"], columns="target_rsid", values="q3_alt_dosage", aggfunc="first").reset_index()
    wide.columns.name = None
    if EXPORT_WIDE_GENOTYPE:
        save(wide, "14_tier1_vdr_genotype_wide.csv")

    rsids = [x["rsid"] for x in TARGET_SNPS if x["rsid"] in wide.columns]
    ld_rows_from_export = []
    for i, snp1 in enumerate(rsids):
        for snp2 in rsids[i + 1:]:
            pair = wide[[snp1, snp2]].dropna()
            corr = float(pair[snp1].corr(pair[snp2])) if len(pair) >= 10 else np.nan
            ld_rows_from_export.append({
                "snp1": snp1,
                "snp2": snp2,
                "n_pair_called": len(pair),
                "correlation": corr,
                "r2": corr ** 2 if pd.notna(corr) else np.nan,
                "d_prime": np.nan,
                "ld_method": "pandas dosage correlation from exported entries; D-prime not estimated from unphased hard calls",
                "pair_status": "pass",
            })
    save(pd.DataFrame(ld_rows_from_export), "14_tier1_vdr_pairwise_ld.csv")

    sample_summary = pd.DataFrame([{
        "sample_source": sample_source,
        "n_samples": len(wide),
        "n_sample_id_numeric_person_id": int(wide["person_id"].notna().sum()),
        "pct_sample_id_numeric_person_id": pct(int(wide["person_id"].notna().sum()), len(wide)),
        "n_complete_four_snp_genotype": int(wide[rsids].dropna().shape[0]) if rsids else 0,
        "entry_export_status": "exported_to_pandas",
        "entry_export_rows_if_enabled": entries_n,
    }])
    save(sample_summary, "14_tier1_vdr_sample_id_person_id_summary.csv")
    block5_status("entry_to_pandas", "pass", f"entries={len(entries_df)}; wide_samples={len(wide)}")
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

# %% Cell 23
# Block 5E. Analytic person-level genotype export for notebook 15.
# This default modeling export path avoids collecting the full 535k-column entry
# matrix. It reads 09 model-prep person IDs and exports genotype entries in
# manageable chunks, so each Hail action sends only a small literal set to Spark.
RUN_ANALYTIC_PERSON_EXPORT = os.environ.get("Q3_VDR_RUN_ANALYTIC_PERSON_EXPORT", "true").strip().lower() in {"1", "true", "yes", "y"}
ANALYTIC_EXPORT_RAISE_ON_FAIL = os.environ.get("Q3_VDR_ANALYTIC_EXPORT_RAISE_ON_FAIL", "false").strip().lower() in {"1", "true", "yes", "y"}
ANALYTIC_EXPORT_CHUNK_SIZE = int(os.environ.get("Q3_VDR_ANALYTIC_EXPORT_CHUNK_SIZE", "10000"))
ANALYTIC_EXPORT_MAX_CHUNKS = int(os.environ.get("Q3_VDR_ANALYTIC_EXPORT_MAX_CHUNKS", "0"))


def resolve_model_prep_dir_for_block5e():
    manual = os.environ.get("Q3_VDR_MODEL_OUTPUT_DIR", "").strip()
    candidates = []
    if manual:
        candidates.append(Path(manual).expanduser())
    for root in [
        DATA_DIR,
        Path("/home/jupyter/workspace") / "VDR",
        Path("/home/dataproc/workspace") / "VDR",
        Path.cwd(),
        Path.cwd().parent,
    ]:
        candidates.append(root / "Q3_VDR_v9_model_prep_outputs")
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists() and (candidate / "02_cvd_model_dataset_v9.csv").exists():
                return candidate
        except Exception:
            pass
    return candidates[0] if candidates else (DATA_DIR / "Q3_VDR_v9_model_prep_outputs")


def read_person_ids_for_block5e(path, label):
    row = {"label": label, "path": str(path), "exists": path.exists(), "rows": 0, "person_id_n": 0, "status": "missing"}
    if not path.exists():
        return [], row
    try:
        df = pd.read_csv(path, usecols=lambda c: c == "person_id", low_memory=False)
        row["rows"] = len(df)
        if "person_id" not in df.columns:
            row["status"] = "missing_person_id_column"
            return [], row
        ids = pd.to_numeric(df["person_id"], errors="coerce").dropna().astype("int64").astype(str).drop_duplicates().tolist()
        row["person_id_n"] = len(ids)
        row["status"] = "pass" if ids else "empty_person_id"
        return ids, row
    except Exception:
        row["status"] = "fail"
        row["detail"] = traceback.format_exc()[:2000]
        return [], row


def make_block5e_export_table(mt_for_export):
    entry_table = mt_for_export.entries()
    entry_fields_for_export = set(entry_table.row)
    sample_expr = hl.str(entry_table[sample_source]) if sample_source in entry_fields_for_export else hl.str("")
    export_table = entry_table.annotate(
        sample_id=sample_expr,
        person_id_string=sample_expr,
        rsid=entry_table.target_rsid,
        expected_vid=entry_table.expected_vid,
    ).key_by()
    return export_table.select(
        sample_id=export_table.sample_id,
        person_id_string=export_table.person_id_string,
        rsid=export_table.rsid,
        expected_vid=export_table.expected_vid,
        q3_alt_dosage=export_table.q3_alt_dosage,
    )


analytic_export_status_rows = []
chunk_status_rows = []
if not RUN_ANALYTIC_PERSON_EXPORT:
    block5_status("analytic_person_export", "skip_disabled_by_env", "Q3_VDR_RUN_ANALYTIC_PERSON_EXPORT=false")
    save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
else:
    try:
        model_prep_dir = resolve_model_prep_dir_for_block5e()
        cvd_ids, cvd_row = read_person_ids_for_block5e(model_prep_dir / "02_cvd_model_dataset_v9.csv", "CVD model-prep dataset")
        crp_ids, crp_row = read_person_ids_for_block5e(model_prep_dir / "03_crp_model_dataset_v9.csv", "CRP model-prep dataset")
        analytic_export_status_rows.extend([cvd_row, crp_row])
        analytic_ids = sorted(set(cvd_ids).union(crp_ids), key=lambda x: int(x))
        save(pd.DataFrame(analytic_export_status_rows), "14_hail_block5e_model_prep_person_id_status.csv")
        save(pd.DataFrame([{
            "model_prep_dir": str(model_prep_dir),
            "cvd_person_id_n": len(cvd_ids),
            "crp_person_id_n": len(crp_ids),
            "analytic_person_id_union_n": len(analytic_ids),
            "run_analytic_person_export": RUN_ANALYTIC_PERSON_EXPORT,
            "chunk_size": ANALYTIC_EXPORT_CHUNK_SIZE,
            "max_chunks": ANALYTIC_EXPORT_MAX_CHUNKS,
        }]), "14_hail_block5e_analytic_person_id_summary.csv")

        col_field_names_for_export = set(cols_fields) if "cols_fields" in globals() else set(mt_target.cols().row)
        if not analytic_ids:
            block5_status("analytic_person_export", "blocked_no_analytic_person_ids", f"model_prep_dir={model_prep_dir}")
            save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
        elif sample_source == "col_idx_unavailable" or sample_source not in col_field_names_for_export:
            block5_status("analytic_person_export", "blocked_no_joinable_sample_source", f"sample_source={sample_source}; col_fields={sorted(col_field_names_for_export)}")
            save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
        else:
            long_path = OUTPUT_DIR / "14_tier1_vdr_genotype_long.csv"
            if long_path.exists():
                if OVERWRITE_OUTPUTS:
                    long_path.unlink()
                else:
                    raise FileExistsError(f"{long_path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")

            total_chunks = int(math.ceil(len(analytic_ids) / ANALYTIC_EXPORT_CHUNK_SIZE)) if ANALYTIC_EXPORT_CHUNK_SIZE > 0 else 0
            chunks_to_run = min(total_chunks, ANALYTIC_EXPORT_MAX_CHUNKS) if ANALYTIC_EXPORT_MAX_CHUNKS > 0 else total_chunks
            block5_status("analytic_person_chunk_plan", "pass", f"analytic_ids={len(analytic_ids)}; chunk_size={ANALYTIC_EXPORT_CHUNK_SIZE}; chunks_to_run={chunks_to_run}; total_chunks={total_chunks}")
            save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

            for chunk_idx in range(chunks_to_run):
                start = chunk_idx * ANALYTIC_EXPORT_CHUNK_SIZE
                stop = min(start + ANALYTIC_EXPORT_CHUNK_SIZE, len(analytic_ids))
                chunk_ids = analytic_ids[start:stop]
                chunk_label = f"chunk_{chunk_idx + 1:04d}_of_{chunks_to_run:04d}"
                chunk_row = {"chunk": chunk_idx + 1, "start": start, "stop": stop, "n_requested_ids": len(chunk_ids), "status": "started", "rows": 0, "unique_person_ids": 0}
                chunk_status_rows.append(chunk_row.copy())
                save(pd.DataFrame(chunk_status_rows), "14_hail_block5e_chunk_status.csv")

                chunk_id_set = hl.literal(set(chunk_ids))
                mt_chunk = mt_target.filter_cols(chunk_id_set.contains(hl.str(mt_target[sample_source])))
                export_table = make_block5e_export_table(mt_chunk)
                chunk_df = export_table.to_pandas()
                if not chunk_df.empty:
                    chunk_df["q3_alt_dosage"] = pd.to_numeric(chunk_df["q3_alt_dosage"], errors="coerce")
                    chunk_df["person_id"] = pd.to_numeric(chunk_df["person_id_string"], errors="coerce").astype("Int64")
                    chunk_df = chunk_df[["sample_id", "person_id", "rsid", "expected_vid", "q3_alt_dosage"]].copy()
                    chunk_df = chunk_df.sort_values(["person_id", "rsid"], na_position="last").reset_index(drop=True)
                    chunk_df.to_csv(long_path, mode="a", header=not long_path.exists(), index=False)
                chunk_status_rows[-1].update({
                    "status": "pass",
                    "rows": int(len(chunk_df)),
                    "unique_person_ids": int(chunk_df["person_id"].nunique(dropna=True)) if not chunk_df.empty else 0,
                })
                save(pd.DataFrame(chunk_status_rows), "14_hail_block5e_chunk_status.csv")
                block5_status("analytic_person_export_chunk", "pass", f"{chunk_label}; rows={len(chunk_df)}; requested_ids={len(chunk_ids)}")
                save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")

            if not long_path.exists():
                block5_status("analytic_person_export", "blocked_empty_export", "No chunk produced genotype rows.")
                save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
            else:
                genotype_long = pd.read_csv(long_path, low_memory=False)
                genotype_long["q3_alt_dosage"] = pd.to_numeric(genotype_long["q3_alt_dosage"], errors="coerce")
                genotype_long["person_id"] = pd.to_numeric(genotype_long["person_id"], errors="coerce").astype("Int64")
                genotype_long = genotype_long.drop_duplicates(subset=["person_id", "rsid"], keep="first").copy()
                genotype_long = genotype_long.sort_values(["person_id", "rsid"], na_position="last").reset_index(drop=True)
                save(genotype_long, "14_tier1_vdr_genotype_long.csv")

                wide = genotype_long.pivot_table(index=["sample_id", "person_id"], columns="rsid", values="q3_alt_dosage", aggfunc="first").reset_index()
                wide.columns.name = None
                rsids = [x["rsid"] for x in TARGET_SNPS if x["rsid"] in wide.columns]
                wide = wide[["sample_id", "person_id"] + rsids].copy()
                save(wide, "14_tier1_vdr_genotype_wide.csv")

                complete_four = int(wide[rsids].dropna().shape[0]) if rsids else 0
                sample_summary = pd.DataFrame([{
                    "sample_source": sample_source,
                    "n_samples": len(wide),
                    "n_sample_id_numeric_person_id": int(wide["person_id"].notna().sum()),
                    "pct_sample_id_numeric_person_id": pct(int(wide["person_id"].notna().sum()), len(wide)),
                    "n_complete_four_snp_genotype": complete_four,
                    "entry_export_status": "analytic_person_chunked_export_to_pandas",
                    "entry_export_rows": len(genotype_long),
                    "analytic_person_id_union_n": len(analytic_ids),
                    "chunks_completed": chunks_to_run,
                    "total_chunks_planned": total_chunks,
                }])
                save(sample_summary, "14_tier1_vdr_sample_id_person_id_summary.csv")
                block5_status("analytic_person_export", "pass", f"long_rows={len(genotype_long)}; wide_rows={len(wide)}; complete_four={complete_four}; chunks={chunks_to_run}/{total_chunks}")
                save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
    except Exception as analytic_export_exc:
        tb = traceback.format_exc()
        save_text(tb, "14_hail_block5e_analytic_person_export_failure.txt")
        status = "fail_backend_stopped" if is_hail_backend_stopped_error(tb) else "fail"
        if chunk_status_rows:
            chunk_status_rows[-1]["status"] = status
            chunk_status_rows[-1]["detail"] = tb[:2000]
            save(pd.DataFrame(chunk_status_rows), "14_hail_block5e_chunk_status.csv")
        block5_status("analytic_person_export", status, tb[:2000])
        save(pd.DataFrame(status_rows), "14_hail_extraction_status.csv")
        if ANALYTIC_EXPORT_RAISE_ON_FAIL:
            raise

# %% Cell 25
import os
import traceback
from pathlib import Path

import pandas as pd

# Block 6 is intentionally local-only. It can be rerun after a kernel restart
# because it reads CSV/TXT outputs already written by earlier blocks.
try:
    OUTPUT_DIR
except NameError:
    def _infer_default_data_dir_block6():
        cwd = Path.cwd()
        candidates = [
            cwd,
            cwd / "data",
            cwd.parent / "data",
            Path("/home/jupyter/workspace") / "VDR",
            Path("/home/jupyter/workspace") / "VDR" / "data",
        ]
        for candidate in candidates:
            try:
                if candidate.exists() and os.access(candidate, os.W_OK):
                    return candidate
            except Exception:
                pass
        return cwd

    DATA_DIR = Path(os.environ.get("Q3_VDR_DATA_DIR", str(_infer_default_data_dir_block6()))).expanduser()
    OUTPUT_DIR = Path(os.environ.get("Q3_VDR_HAIL_OUTPUT_DIR", str(DATA_DIR / "Q3_VDR_tier1_hail_genotype_outputs_2026_08_07"))).expanduser()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    OVERWRITE_OUTPUTS
except NameError:
    OVERWRITE_OUTPUTS = os.environ.get("Q3_VDR_OVERWRITE_OUTPUTS", "true").strip().lower() in {"1", "true", "yes", "y"}


def _save_block6(df, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")
    df.to_csv(path, index=False)
    print(f"Saved {filename}: {len(df):,} rows")
    return path


def _save_text_block6(text, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false")
    path.write_text(text, encoding="utf-8")
    print(f"Saved {filename}: {len(text):,} characters")
    return path


def _read_csv_or_empty(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        _save_text_block6(traceback.format_exc(), f"14_hail_block6_read_failure_{path.name}.txt")
        return pd.DataFrame()


target_rsids = {"rs731236", "rs7975232", "rs1544410", "rs2228570"}
qc_path = OUTPUT_DIR / "14_tier1_vdr_genotype_qc_summary.csv"
ld_path = OUTPUT_DIR / "14_tier1_vdr_pairwise_ld.csv"
sample_path = OUTPUT_DIR / "14_tier1_vdr_sample_id_person_id_summary.csv"
status_path = OUTPUT_DIR / "14_hail_extraction_status.csv"

qc_df = _read_csv_or_empty(qc_path)
ld_df = _read_csv_or_empty(ld_path)
sample_summary = _read_csv_or_empty(sample_path)
status_df = _read_csv_or_empty(status_path)

extracted_rsids = set(qc_df.get("rsid", pd.Series(dtype=str)).astype(str)) if not qc_df.empty else set()
all_four = target_rsids.issubset(extracted_rsids)

missing_pct = pd.to_numeric(qc_df.get("missing_pct", pd.Series(dtype=float)), errors="coerce")
low_missing = bool((missing_pct < 5).all()) if not qc_df.empty and len(missing_pct) == len(qc_df) and missing_pct.notna().all() else False

person_numeric_pct = 0.0
if not sample_summary.empty and "pct_sample_id_numeric_person_id" in sample_summary:
    person_numeric_series = pd.to_numeric(sample_summary["pct_sample_id_numeric_person_id"], errors="coerce")
    if person_numeric_series.notna().any():
        person_numeric_pct = float(person_numeric_series.dropna().iloc[0])
person_id_joinable = bool(person_numeric_pct > 95)

ld_r2 = pd.to_numeric(ld_df.get("r2", pd.Series(dtype=float)), errors="coerce")
ld_pair_count = int(ld_r2.notna().sum()) if not ld_df.empty else 0
ld_has_r2 = bool(ld_pair_count > 0)

selected_source_detail = ""
genotype_source = "not available; see 14_hail_extraction_status.csv"
pairwise_status = "not available"
if not status_df.empty and {"step", "status", "detail"}.issubset(status_df.columns):
    read_rows = status_df.loc[(status_df["step"] == "read_matrix_table") & (status_df["status"] == "pass"), "detail"]
    if not read_rows.empty:
        selected_source_detail = str(read_rows.iloc[-1])
    genotype_rows = status_df.loc[status_df["step"] == "define_dosage", "detail"]
    if not genotype_rows.empty:
        genotype_source = str(genotype_rows.iloc[-1])
    ld_rows = status_df.loc[status_df["step"] == "pairwise_ld_hail_ld_matrix", "status"]
    if not ld_rows.empty:
        pairwise_status = str(ld_rows.iloc[-1])

if "=" in selected_source_detail:
    selected_hail_source, selected_hail_path = selected_source_detail.split("=", 1)
else:
    selected_hail_source = "see 14_hail_extraction_status.csv"
    selected_hail_path = selected_source_detail or "not available"

decision = pd.DataFrame([
    {"question": "Were all four Tier 1 VDR SNPs extracted?", "status": "yes" if all_four else "no_or_partial", "evidence": "14_tier1_vdr_genotype_qc_summary.csv"},
    {"question": "Is genotype missingness low enough for additive sensitivity?", "status": "yes" if low_missing else "needs_review", "evidence": "14_tier1_vdr_genotype_qc_summary.csv"},
    {"question": "Can sample IDs join to person_id?", "status": "yes" if person_id_joinable else "needs_review", "evidence": "14_tier1_vdr_sample_id_person_id_summary.csv"},
    {"question": "Can Q3 report pairwise LD?", "status": "yes_if_ld_reviewed" if ld_has_r2 else "no", "evidence": "14_tier1_vdr_pairwise_ld.csv"},
    {"question": "Can Q3 move beyond ALT-carrier wording?", "status": "only_for_genotype_model_not_functional_allele", "evidence": "Hail genotype extraction does not by itself map literature functional/risk alleles."},
    {"question": "Can Block 6 be rerun after Spark/Hail disconnects?", "status": "yes", "evidence": "Block 6 reads local CSV outputs only and does not call Hail."},
])
_save_block6(decision, "14_hail_genotype_qc_decision_matrix.csv")

note = f"""# Q3 VDR Hail Genotype Extraction Decision Note

Selected Hail source: `{selected_hail_source}`

Selected Hail path:

```text
{selected_hail_path}
```

Genotype field used: `{genotype_source}`

## Main Decision

- All four SNPs extracted: {'yes' if all_four else 'no_or_partial'}
- Low missingness (<5% for all extracted SNPs): {'yes' if low_missing else 'needs_review'}
- Sample ID joinable to numeric person_id: {'yes' if person_id_joinable else 'needs_review'}
- Pairwise LD output available: {'yes' if ld_has_r2 else 'no'}
- Pairwise LD rows with non-missing r2: {ld_pair_count}
- Latest pairwise LD Hail status: `{pairwise_status}`

## Runtime Note

Block 6 uses only local CSV/TXT outputs and performs no Hail/Spark actions. If old `14_hail_block5_*failure*.txt` files are present, interpret the current run by `14_hail_extraction_status.csv` and `14_tier1_vdr_pairwise_ld.csv`; stale failure logs can remain from earlier attempts.

## Q3 Interpretation Rule

If all four SNPs were extracted with low missingness and sample IDs map to person_id, Q3 can add genotype-based additive sensitivity and genotype QC reporting. The pairwise LD CSV can support a full-WGS dosage-correlation LD diagnostic. This still does not prove that AoU ALT alleles are literature functional/risk alleles; allele harmonization against FokI/BsmI/ApaI/TaqI nomenclature remains a separate literature/database step.
"""
_save_text_block6(note, "14_hail_genotype_extraction_decision_note.md")

block6_status = pd.DataFrame([{
    "block": "Block 6",
    "status": "pass",
    "all_four_snps_extracted": all_four,
    "low_missingness": low_missing,
    "person_id_joinable": person_id_joinable,
    "ld_has_nonmissing_r2": ld_has_r2,
    "ld_pair_count": ld_pair_count,
    "note": "Block 6 uses local CSV/pandas outputs only; no Hail action is required.",
}])
_save_block6(block6_status, "14_hail_block6_finalization_status.csv")

block6_status_row = pd.DataFrame([{"step": "block6_finalization", "status": "pass", "detail": f"ld_pair_count={ld_pair_count}; local_csv_only=true"}])
if status_df.empty or not {"step", "status", "detail"}.issubset(status_df.columns):
    updated_status = block6_status_row
else:
    updated_status = pd.concat([status_df.loc[status_df["step"] != "block6_finalization", ["step", "status", "detail"]], block6_status_row], ignore_index=True)
_save_block6(updated_status, "14_hail_extraction_status.csv")

try:
    manifest_filenames = [
        "14_hail_run_configuration.csv",
        "14_hail_extraction_status.csv",
        "14_tier1_vdr_target_snp_list.csv",
        "14_tier1_vdr_hail_target_rows.csv",
        "14_tier1_vdr_genotype_qc_summary.csv",
        "14_tier1_vdr_pairwise_ld.csv",
        "14_tier1_vdr_sample_id_person_id_summary.csv",
        "14_tier1_vdr_genotype_long.csv",
        "14_tier1_vdr_genotype_wide.csv",
        "14_hail_block5e_model_prep_person_id_status.csv",
        "14_hail_block5e_analytic_person_id_summary.csv",
        "14_hail_genotype_qc_decision_matrix.csv",
        "14_hail_genotype_extraction_decision_note.md",
        "14_hail_block6_finalization_status.csv",
    ]
    manifest_rows = []
    for filename in manifest_filenames:
        p = OUTPUT_DIR / filename
        if p.exists() and p.is_file():
            manifest_rows.append({"filename": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    manifest = pd.DataFrame(manifest_rows)
    _save_block6(manifest, "99_hail_genotype_output_manifest.csv")
except Exception:
    _save_text_block6(traceback.format_exc(), "14_hail_block6_manifest_failure.txt")

print(f"Done. Block 6 finalized without Hail actions. Output dir: {OUTPUT_DIR}")
