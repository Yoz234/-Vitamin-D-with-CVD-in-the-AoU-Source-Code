# Source companion of 19_q3_vdr_r_environment_detection_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# Cell 0. Detect Rscript and run a minimal R environment audit.
import csv
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = os.environ.get("Q3_VDR_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        cwd,
        Path("/home/dataproc/workspace") / "VDR",
        Path("/home/jupyter/workspace") / "VDR",
        cwd.parent,
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            pass
    return cwd

DATA_DIR = Path(os.environ.get("Q3_VDR_DATA_DIR", str(infer_default_data_dir()))).expanduser()
OUTPUT_DIR = Path(os.environ.get("Q3_VDR_R_ENV_OUTPUT_DIR", str(DATA_DIR / "Q3_VDR_r_environment_detection_outputs_2026_08_12"))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

status_path = OUTPUT_DIR / "19_r_environment_detection_status.csv"

def write_status(rows):
    with status_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "step", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

status_rows = []

def add_status(step, status, detail=""):
    row = {"run_id": RUN_ID, "step": step, "status": status, "detail": str(detail)[:4000]}
    status_rows.append(row)
    write_status(status_rows)
    print(f"[{status}] {step}: {detail}")

rscript_path = shutil.which("Rscript")
python_context = {
    "python_executable": sys.executable,
    "python_version": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "cwd": str(Path.cwd()),
    "output_dir": str(OUTPUT_DIR),
    "rscript_path": rscript_path or "",
}

with (OUTPUT_DIR / "19_python_jupyter_context.txt").open("w", encoding="utf-8") as f:
    for key, value in python_context.items():
        f.write(f"{key}={value}\n")

if not rscript_path:
    add_status("rscript_available", "fail", "Rscript not found on PATH. Run this in an AoU/Jupyter runtime with R available or set PATH to the intended Rscript.")
else:
    add_status("rscript_available", "pass", rscript_path)

    r_code = r'''
out_dir <- Sys.getenv("Q3_VDR_R_ENV_OUTPUT_DIR")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

key_pkgs <- c(
  "survival",
  "splines",
  "stats",
  "utils",
  "graphics",
  "grDevices",
  "dplyr",
  "tidyr",
  "readr",
  "data.table",
  "ggplot2",
  "broom"
)

pkg_rows <- lapply(key_pkgs, function(p) {
  available <- requireNamespace(p, quietly=TRUE)
  version <- if (available) as.character(utils::packageVersion(p)) else NA_character_
  lib_path <- if (available) dirname(system.file(package=p)) else NA_character_
  data.frame(package=p, available=available, version=version, lib_path=lib_path, stringsAsFactors=FALSE)
})
pkg_versions <- do.call(rbind, pkg_rows)

env_rows <- data.frame(
  field=c("R.version.string", "R.version$platform", "R.home", "getwd", ".libPaths"),
  value=c(R.version.string, R.version$platform, R.home(), getwd(), paste(.libPaths(), collapse="; ")),
  stringsAsFactors=FALSE
)

write.csv(env_rows, file.path(out_dir, "19_r_environment_core.csv"), row.names=FALSE)
write.csv(pkg_versions, file.path(out_dir, "19_r_key_package_versions.csv"), row.names=FALSE)

sink(file.path(out_dir, "19_r_sessionInfo.txt"))
cat("R_VERSION_STRING=", R.version.string, "\n", sep="")
cat("R_HOME=", R.home(), "\n", sep="")
cat("R_PLATFORM=", R.version$platform, "\n", sep="")
cat("LIB_PATHS=", paste(.libPaths(), collapse="; "), "\n\n", sep="")
print(sessionInfo())
sink()

cat("R_VERSION_STRING=", R.version.string, "\n", sep="")
cat("R_HOME=", R.home(), "\n", sep="")
cat("R_PLATFORM=", R.version$platform, "\n", sep="")
cat("\nKEY_PACKAGE_VERSIONS\n")
print(pkg_versions, row.names=FALSE)
'''

    env = os.environ.copy()
    env["Q3_VDR_R_ENV_OUTPUT_DIR"] = str(OUTPUT_DIR)

    result = subprocess.run(
        [rscript_path, "-"],
        input=r_code,
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )

    (OUTPUT_DIR / "19_r_environment_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (OUTPUT_DIR / "19_r_environment_stderr.txt").write_text(result.stderr, encoding="utf-8")

    if result.returncode == 0:
        add_status("r_environment_audit", "pass", "R environment and key package versions were recorded.")
    else:
        add_status("r_environment_audit", "fail", f"Rscript exited with code {result.returncode}. See stderr output.")

    print("\n--- R stdout ---")
    print(result.stdout)
    if result.stderr.strip():
        print("\n--- R stderr ---")
        print(result.stderr)

print(f"\nOutput directory: {OUTPUT_DIR}")

# %% Cell 3
# Cell 1. Build a short manuscript-ready software note from the recorded outputs.
import csv
from pathlib import Path

core_path = OUTPUT_DIR / "19_r_environment_core.csv"
pkg_path = OUTPUT_DIR / "19_r_key_package_versions.csv"
note_path = OUTPUT_DIR / "19_r_environment_manuscript_note.md"

core = {}
if core_path.exists():
    with core_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            core[row["field"]] = row["value"]

pkgs = {}
if pkg_path.exists():
    with pkg_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pkgs[row["package"]] = row

def pkg_version(name):
    row = pkgs.get(name, {})
    if str(row.get("available", "")).lower() == "true":
        return row.get("version", "NA")
    return "not available"

r_version = core.get("R.version.string", "R version not recorded")
survival_version = pkg_version("survival")

note = f"""# Q3 VDR R Environment Manuscript Note

Detected R environment: `{r_version}`

Key modeling package: `survival` version `{survival_version}`

Suggested Methods/Supplement wording:

> R was invoked from the Jupyter notebook workflow using `Rscript`. The executed R environment was `{r_version}`. Cox proportional hazards models were fitted using the `survival` package version {survival_version}. Package versions were recorded from the executed R subprocess using `sessionInfo()` and `packageVersion()`.

Full outputs:

- `19_r_environment_core.csv`
- `19_r_key_package_versions.csv`
- `19_r_sessionInfo.txt`
- `19_r_environment_stdout.txt`
- `19_r_environment_stderr.txt`
- `19_python_jupyter_context.txt`
"""

note_path.write_text(note, encoding="utf-8")
print(note)
print(f"Saved: {note_path}")
