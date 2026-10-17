# Selection and provenance

The archive is based on source content and the latest local workflow notes as of 2026-10-17. Original filenames are not treated as reliable revision timestamps.

| Published source | Selection basis |
| --- | --- |
| `Q1_Q2/01_q1_q2_workflow` | Maintained `Q2/简版代码分块/0729_clean.ipynb`; shared preparation, Q1 models, Q2 CRP/covariate/extended models |
| `Q1_Q2/02_q1_final_manuscript_revisions` | Final independent cells 152, 155, 156, 157 (zero based) from `Q1终板/0604 Q1.ipynb`; later table/figure revisions live at the end of that historically named file |
| `Q1_Q2/03_q1_q2_manuscript_tables` | 2026-08-27 maintained manuscript table redraw notebook |
| `Q1_Q2/support` | Ten already curated simplified non-main modules; detailed duplicate exports omitted |
| Q3 upstream/genetics | Selected final supporting sources recovered from `temporarily_unused` in the 2026-10-17 archive; required to retain the provenance of the existing clinical/genetic analysis |
| Q3 current 16/18 | Active 2026-10-17 CRP30 source; server execution still pending |
| Q3 prior combined 18 | Preserves the existing CVD/LD/QC figure generator; old CRP content explicitly historical |

The selected Q1 revision blocks and the later table-redraw workflow are retained separately because they have different input contracts and tasks. They are not concatenated into a new statistical analysis or claimed to be equivalent. No numbers are taken from saved notebook outputs to construct a new result.

`provenance/source_manifest.json` maps each published notebook to the original relative path, original file SHA-256, selected original cell indices, role, and companion source. Those hashes refer to the unmodified local originals and allow provenance checks without publishing their outputs. `provenance/SHA256SUMS.txt` covers the published files, excluding itself and its generated validation report to avoid circular hashes. The validation report separately hashes that checksum inventory.

Publication transformations are restricted to removing saved outputs/execution counts/attachments/private metadata, adding archive-role notes, and replacing private infrastructure configuration. The Python companions replace Jupyter shell/line-magics with IPython calls; R notebooks retain R kernels and `.R` sources. The CRP30 generator's destination changes from the local Chinese folder name to repository `Q3_VDR/crp30`. It regenerates notebook logic, not archive annotations or companion files; do not run it over the archival snapshot without reviewing the diff.

Excluded material includes participant-level CSV/TSV/Parquet/pickle/genotype/model-input files; downloaded aggregate results and image/PDF exports; full output-bearing historical notebooks; duplicate detailed module exports; discarded/buggy/early CRP pipelines; exploration-only MR/GRS notebooks; literature, meeting notes, manuscripts, and document-formatting utilities. The archive does not assert that excluded old cohort-building templates implement the final cohort definition.

This snapshot does not add a new analysis design, reconcile different cohorts or releases, modify formulas, or declare pending analyses complete.
