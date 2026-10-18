# Vitamin D, CVD, inflammation and VDR: analysis code example

This instructor-facing example presents the analysis logic for three related questions in the All of Us Research Program:

| Question | Exposure / modifier | Outcome | Main method |
| --- | --- | --- | --- |
| Q1 | Baseline measured 25(OH)D | Incident myocardial infarction or stroke | Cox regression, subgroup interactions and dose-response sensitivity |
| Q2 | Measured 25(OH)D | Baseline/pre-index CRP | Correlation and adjusted log-log regression |
| Q3 | Four VDR SNPs coded as additive ALT dosage, interacting with measured vitamin D | CVD and the retained CRP extension | Cox/OLS interaction models, with carrier/PH sensitivity and genotype/LD checks |

The example was organized on **2026-10-18** from the existing analysis sources. No participant-level data or saved notebook outputs are distributed. The formulas, exposure coding and cohort-selection rules in the selected implementations are retained. The reorganized notebooks have been checked locally for Python syntax and dependencies between extracted blocks; they have not been rerun against AoU participant data after reorganization. Native/embedded R execution remains a server check.

## Folder logic

```text
teacher_code_example_2026-10-18/
├── README.md
├── requirements.txt
├── shared/                         Reusable Q1/Q2 configuration, cleaning and cohort functions
├── 01_data_preparation/            Covariates → clinical cohorts → genotypes → joined model inputs
├── 02_primary_analysis/            The three research-question analyses
├── 03_sensitivity_and_quality/     Dose response, measurement checks, carrier/PH and genetic QC
├── 04_tables_and_figures/          Presentation of existing model results
└── 05_pending_extensions/          CRP30 source awaiting AoU/R execution
```

Read the research-question notebooks first to understand the models, then trace their inputs back to preparation and their interpretation forward to quality checks. Folder numbers describe the stages; they do not require running all notebooks unconditionally. Q1/Q2 and Q3 have separate input contracts, CDR contexts and participant denominators.

## Suggested first reading

1. [Q1: vitamin D and incident CVD](02_primary_analysis/01_q1_vitamin_d_cvd.ipynb).
2. [Q2: vitamin D and CRP](02_primary_analysis/02_q2_vitamin_d_crp.ipynb).
3. [Q3: VDR interaction models](02_primary_analysis/03_q3_vdr_interactions.ipynb).
4. The corresponding sensitivity/quality notebook, followed by its table/figure notebook.

Each notebook starts with its question, required inputs, intended outputs and interpretation limits. The small [shared module](shared/q1q2_common.py) contains real reusable preparation/cleaning code; there are no duplicate Python/R companions of the notebooks.

## Execution paths

| Analysis path | Preparation | Model fitting | Checks and presentation |
| --- | --- | --- | --- |
| Q1 | Preparation 01 if covariates need rebuilding, then 02 | Primary analysis 01 | Sensitivity/quality 01 and 02; tables/figures 01 |
| Q2 | Preparation 02, then 03 if covariates need construction | Primary analysis 02 | Sensitivity/quality 02; model-specific plots remain next to the Q2 models |
| Q3 | Preparation 04 → 05 when clinical rebuild is needed; 06 for genotype extraction; 07 for the additive join | Primary analysis 03 | Sensitivity/quality 03 and 04; tables/figures 02 |

Existing validated upstream files can be used at the appropriate entry point. Do not rerun clinical rebuild, Hail extraction or covariate construction solely to inspect the model examples. Use a fresh kernel for each notebook; the carrier/PH notebook requires an **R** kernel and the other notebooks use Python. The Q3 interaction notebook invokes `Rscript` from Python.

The Q1/Q2 example starts from existing follow-up and covariate files; the complete original Q1 raw-CDR cohort builder is not supplied. Its primary categorical Cox analysis, companion BMI/trend/continuous checks and subgroups keep the maintained model specification. The final baseline table uses a broader recorded baseline denominator than the complete-case Cox analysis; its original cohort-size assertions and the forest plot's recorded eight-test correction are explicit in the notebook.

The Q3 additive entry uses the retained corrected R implementation with its existing minimum model N/events defaults of 50/10. The older alternative Python/R model route with different thresholds is omitted. Carrier coding remains separately labeled. ALT dosage alone does not establish a functional, protective or risk-allele direction; absent PC, relatedness or harmonization evidence remains a limitation.

## Runtime configuration

For Q1/Q2, set the following before importing the shared module:

```python
import os
os.environ["WORKSPACE_CDR"] = "YOUR_AUTHORIZED_PROJECT.YOUR_CDR_DATASET"
os.environ["CAPSTONE_DATA_DIR"] = "/your/authorized/input_directory"
os.environ["CAPSTONE_OUTPUT_DIR"] = "/your/authorized/new_output_directory"
# Optional if the notebook is launched outside the example folder:
os.environ["CAPSTONE_PROJECT_ROOT"] = "/your/path/to/teacher_code_example_2026-10-18"
```

The shared module writes the analysis base and derived tables to `CAPSTONE_OUTPUT_DIR/tables/` and figures to `CAPSTONE_OUTPUT_DIR/figures/`. The optional Q1 BMI/eGFR construction notebook writes the original named covariate files in `CAPSTONE_DATA_DIR`; skip it when the intended files already exist. Quality summaries go to the configured output directory's `quality_checks/` subfolder.

For Q3, configure the corresponding notebook's first code block. Its principal path variables are:

| Variable | Role |
| --- | --- |
| `WORKSPACE_CDR` / `GOOGLE_PROJECT` | Authorized Q3 CDR / billing project |
| `WORKSPACE_BUCKET` | Writable workspace staging bucket for extraction |
| `Q3_VDR_DATA_DIR` | Q3 runtime root |
| `Q3_VDR_REBUILD_OUTPUT_DIR` | Clinical rebuild outputs |
| `Q3_VDR_MODEL_OUTPUT_DIR` | Prepared clinical model inputs |
| `Q3_VDR_HAIL_OUTPUT_DIR` | Genotype extraction outputs |
| `Q3_VDR_ADDITIVE_OUTPUT_DIR` | Joined additive model inputs and results |
| `Q3_VDR_GENOTYPE_WIDE_CSV` | Exact genotype file for genetic QC when needed |
| `Q3_VDR_GENETIC_QC_CSV` / `Q3_VDR_ALLELE_HARMONIZATION_CSV` | Optional supplied evidence; availability is checked explicitly |
| `Q3_VDR_HAIL_DIR` / `Q3_VDR_MODEL15_DIR` / `Q3_VDR_STEP17_DIR` | Exact aggregate-result directories for CVD/LD/QC figures |

Q3 sources retain their established runtime output filenames so downstream inputs continue to match. Set paths explicitly when using a fresh output layout; the retained defaults are historical server folder names. Use the authorized CDR context appropriate to each analysis rather than substituting releases or sharing Q1/Q2 and Q3 denominators.

`requirements.txt` is an unpinned Python dependency inventory, not a recovered environment lockfile. Hail/Spark require a compatible genomics runtime. R requires `survival`; R notebooks additionally require IRkernel. Actual participant analyses run in an authorized AoU environment, with outputs kept outside this source folder.

## Pending extension

`05_pending_extensions/` contains two CRP30 notebooks. The retained source snapshot marks them as awaiting actual AoU/R execution. They are not included in the completed-analysis reading path and supply no newly verified estimates. Model notebook 01 reuses the existing Q3 additive CRP file; figure notebook 02 requires that exact new run directory. The Q2 nearest same-day/pre-index CRP analysis, its 30-/90-day timing sensitivities, the earlier Q3 CRP extension and this pending Q3 CRP30 source remain distinct.

The full local source archive retains historical recovery scripts, alternative implementations and packaging checks. Those internal materials are omitted here so the example follows the experiment's data-to-model-to-check-to-presentation logic.
