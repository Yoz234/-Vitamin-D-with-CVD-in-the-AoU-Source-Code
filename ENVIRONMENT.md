# Environment and configuration

Use an authorized All of Us Researcher Workbench environment for actual participant analyses. Authenticate through that environment; no service-account file or token is distributed here.

The source includes Q1/Q2 references to release `C2024Q3R9` and Q3 v9 rebuild/genomics references including `C2025Q4R6`. Those reflect separate analysis contexts, not interchangeable versions. Project portions of private dataset identifiers have been replaced with `YOUR_GOOGLE_PROJECT`. Configure each notebook for the same authorized release and tables as the analysis being reproduced.

## Python

`requirements.txt` is an **unpinned dependency inventory**, not a recovered lockfile or a tested new environment. Use the original server environment when available and record versions on rerun. Hail/Spark require a compatible AoU genomics runtime and are managed separately; avoid blindly upgrading that runtime.

| Variable | Purpose |
| --- | --- |
| `WORKSPACE_CDR` | Authorized full BigQuery CDR dataset name; set before notebook setup |
| `GOOGLE_PROJECT` | Authorized billing project for BigQuery/Hail |
| `WORKSPACE_BUCKET` | Authorized workspace bucket; used for runtime outputs/staging |
| `CAPSTONE_DATA_DIR` | Existing Q1/Q2 analysis-ready input directory |
| `CAPSTONE_OUTPUT_DIR` | Main Q1/Q2 workflow output directory |
| `Q3_VDR_DATA_DIR` | Q3 server data root |
| `Q3_VDR_MODEL_OUTPUT_DIR` | Q3 model-preparation directory |
| `Q3_VDR_HAIL_OUTPUT_DIR` | Q3 Hail extraction/QC directory |
| `Q3_VDR_ADDITIVE_OUTPUT_DIR` | Q3 additive model directory |
| `Q3_VDR_CRP_ADDITIVE_DATASET` | Existing Q3 additive CRP model input CSV for current 16 |
| `Q3_VDR_CRP_SENS_OUTPUT_DIR` | Optional fresh output directory for current 16 |
| `Q3_VDR_CRP16_DIR` | Exact current-16 run directory consumed by current 18 |
| `Q3_VDR_FIGURE_OUTPUT_DIR` | Optional current-18 figure output directory |

Some standalone Q1 modules explicitly change their working directory. Their publication copy uses `/home/jupyter/capstone/data` as an example; edit those configuration lines to the authorized runtime directory. Setting `CAPSTONE_DATA_DIR` alone does not override an explicit `os.chdir()` in these retained blocks. The manuscript table notebook 03 has its own first-cell `WORKSPACE_NAME`/`MIGRATION_ROOT`/`DATA_DIR`/`OUT_DIR` configuration and does not read `CAPSTONE_DATA_DIR`. Its private workspace name was replaced with `capstone`. The Hail staging alternative now derives its default staging URI from `WORKSPACE_BUCKET` instead of a private temporary bucket.

Keep all generated data/results outside the source checkout. `.gitignore` is a secondary guard; the validator enforces an allowlist for this snapshot.

## R

The two R-kernel notebooks have `.R` companions. Install/use IRkernel for notebook execution and `survival` for Cox models. Python notebooks 15, 16, and the corrected 15 fallback may call `Rscript` and write embedded R scripts at runtime. The CRP30 source computes OLS t-based confidence intervals in R. R availability and execution must be checked on the actual server.

## Local source checks

From the repository root, run `python tools/validate_source_archive.py`. This uses the Python standard library and does not open participant files or contact BigQuery/Hail. With an existing local environment containing numpy, pandas, and matplotlib, `python Q3_VDR/maintenance/test_crp30_notebook_logic.py` runs the retained synthetic CRP30 tests. A passing synthetic check is not a successful AoU model run.
