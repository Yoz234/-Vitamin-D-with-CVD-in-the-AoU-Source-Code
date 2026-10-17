# Validation scope

The source archive was checked locally without opening participant-level datasets or running participant analyses.

- All 29 notebooks are valid JSON with nbformat 4.5 fields, language-appropriate kernels, empty code outputs, null execution counts, and no cell attachments/private metadata. This is a standard-library structural check, not an `nbformat` package schema-validation claim.
- All Python companions and source utilities were parsed and compiled without execution. The two native R notebooks retain R kernels and `.R` companions. R execution/syntax validation remains a server task.
- Companion code was compared against notebook code cell by cell. Published notebook sources were also compared locally to the exact selected original cells after the documented publication transformations; the originals were not modified.
- The source manifest records each original notebook SHA-256 and selected zero-based cell indices. A separate checksum inventory covers the publication files. The validator checks both exact membership and content hashes and records the checksum inventory's own hash in its report.
- An extension allowlist and source-pattern scan check the selected publication files for accidental data/binary files, credential/private-key patterns, email addresses, and obvious literal participant-ID predicates. Saved results, images, data files, and full historical notebooks were not copied into the archive. These checks do not constitute a formal disclosure certification.

`python tools/validate_source_archive.py` repeats the structural, syntax, companion, allowlist, and checksum checks using only the standard library. The machine-readable result is `provenance/validation_report.json`. After a deliberate reviewed source edit, `--refresh` regenerates the checksum inventory.

The retained CRP30 synthetic tests are provided in `Q3_VDR/maintenance/`; this publication pass does not claim a new successful execution of them. The earlier local work log records synthetic validation, but no genuine AoU/R model run of the 2026-10-17 CRP30 source has been returned. No counts, estimates, figures, or new scientific conclusions are validated by packaging this source. Runtime dependency installation, BigQuery queries, Hail/Spark access, R regressions, and full manuscript-result reproduction remain outside these local source checks.
