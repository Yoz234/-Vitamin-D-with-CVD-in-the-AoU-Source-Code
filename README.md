# Vitamin D, CVD, CRP, and VDR in All of Us — Source Code

Source-only snapshot assembled on **2026-10-17** for the capstone project on measured vitamin D, incident myocardial infarction/stroke, inflammation, and VDR genetic interactions in the All of Us Research Program.

The archive consolidates the maintained Q1/Q2 workflow, later manuscript table and figure revisions, supporting covariate/audit modules, and the Q3 genetic analysis sources. Each notebook has a readable `.py` or `.R` companion. Use either representation, not both in the same run.

## Contents

| Directory | Scope |
| --- | --- |
| `Q1_Q2/` | Q1 incident MI/stroke analysis; Q2 vitamin D–CRP regression; later manuscript revisions and supplementary tables |
| `Q1_Q2/support/` | Smoking/alcohol, BMI, eGFR, concept/unit audits, repeated-measure coverage, time-varying Cox, and subgroup modules |
| `Q3_VDR/upstream/` | Clinical rebuild, carrier repair/denominator checks, model preparation, and resource checks |
| `Q3_VDR/genetics/` | Carrier models and PH sensitivity, Hail extraction, additive ALT-dosage models, R recovery, and genetic QC |
| `Q3_VDR/crp30/` | Current CRP 0–30-day interaction analysis and corresponding figures |
| `Q3_VDR/legacy_figures/` | Prior combined figure generator retained for CVD/LD/QC provenance; its CRP panels are historical |
| `Q3_VDR/maintenance/` | CRP30 notebook generator and synthetic-data checks |
| `provenance/` | Original-file/cell mapping, SHA-256 checksums, and source validation report |

## Status and use

Read [RUN_ORDER.md](RUN_ORDER.md) before execution. This is an archival consolidation of the available final working sources, including conditional recovery routes and supplementary modules. It is not a newly validated single-click raw-data pipeline. Q1/Q2 require existing analysis-ready cohort files; the original complete Q1 raw-CDR-to-cohort construction is not present in the selected final sources. Dataset definitions, units, model formulas, and statistical choices are preserved.

The **2026-10-17 Q3 CRP30 notebooks await actual AoU server execution**. They have not supplied new participant counts, coefficients, or manuscript results. Earlier Q3 genetic sources and the new CRP30 source have separate roles. The source distinguishes additive ALT dosage from the earlier carrier sensitivity analyses; no Mendelian-randomization pipeline is presented here.

Run participant analyses only in an authorized AoU environment. This repository contains code and documentation, with all saved notebook outputs, execution counts, attachments, and private notebook metadata removed. Participant-level files, fitted results, images, and credentials are excluded. Generated runtime outputs remain outside this source archive.

Private cloud project identifiers and a private Workbench path were replaced in the publication copies. Explicit assignments to the original `WORKSPACE_CDR` were removed. Set your authorized environment and paths as described in [ENVIRONMENT.md](ENVIRONMENT.md). Original working files were left unchanged.

[SOURCE_SELECTION.md](SOURCE_SELECTION.md) explains precedence and exclusions. [VALIDATION.md](VALIDATION.md) records the validation scope and limits. No license grant is added by this archival snapshot.
