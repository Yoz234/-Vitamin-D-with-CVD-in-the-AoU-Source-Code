# Source companion of 08_v9_prep_sample_name_denominator_probe.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# ============================================================
# Block 0. Setup and output policy
# ============================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

FALLBACK_CDR = 'YOUR_GOOGLE_PROJECT.C2025Q4R6'
FALLBACK_PREP = 'YOUR_GOOGLE_PROJECT.prep_C2025Q4R6'
SERVER_OUTPUT_DIR_NAME = 'Q3_VDR_v9_rebuild_outputs'
OVERWRITE_OUTPUTS = True  # Same-name denominator/carrier output CSVs will be overwritten.

def first_env(*names):
    for name in names:
        value = os.environ.get(name, '').strip()
        if value:
            return value, name
    return '', ''

def infer_prep_from_cdr(cdr):
    if '.' not in cdr:
        return ''
    project, dataset = cdr.split('.', 1)
    if dataset.startswith('prep_'):
        return cdr
    return f'{project}.prep_{dataset}'

def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        Path('/home/jupyter/workspace/VDR'),
        Path('/home/dataproc/workspace/VDR'),
        Path('/home/jupyter/workspace/Q3_VDR'),
        cwd,
        cwd / 'data',
        Path('/home/jupyter/workspace/VDR/data'),
        Path('/home/dataproc/workspace/VDR/data'),
        Path('/home/jupyter/workspace/Q3_VDR/data'),
    ]
    for p in candidates:
        try:
            if p.exists() and os.access(p, os.W_OK):
                return p
        except Exception:
            pass
    return cwd

cdr_env, cdr_source = first_env('WORKSPACE_CDR', 'Q3_VDR_CDR', 'CDR')
CDR = cdr_env or FALLBACK_CDR
cdr_source = cdr_source or 'fallback_hardcoded'
prep_env, prep_source = first_env('WORKSPACE_PREP_CDR', 'Q3_VDR_PREP_CDR', 'PREP_CDR')
PREP_CDR = prep_env or infer_prep_from_cdr(CDR) or FALLBACK_PREP
prep_source = prep_source or ('derived_from_cdr' if prep_env == '' and infer_prep_from_cdr(CDR) else 'fallback_hardcoded')
if '.' not in CDR or '.' not in PREP_CDR:
    raise ValueError('CDR and PREP_CDR must be fully qualified BigQuery datasets like project.dataset. Set WORKSPACE_CDR/Q3_VDR_CDR and WORKSPACE_PREP_CDR/Q3_VDR_PREP_CDR if needed.')
DATA_DIR = Path(os.environ.get('Q3_VDR_DATA_DIR', str(infer_default_data_dir()))).expanduser()
OUTPUT_DIR = Path(os.environ.get('Q3_VDR_REBUILD_OUTPUT_DIR', str(DATA_DIR / SERVER_OUTPUT_DIR_NAME))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GOOGLE_PROJECT, google_project_source = first_env('GOOGLE_PROJECT', 'GOOGLE_CLOUD_PROJECT')
WORKSPACE_BUCKET = os.environ.get('WORKSPACE_BUCKET', '').strip()
client = bigquery.Client(project=GOOGLE_PROJECT or None)
query_status_rows = []

def save(df, name):
    path = OUTPUT_DIR / name
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f'{path} exists and OVERWRITE_OUTPUTS is False')
    df.to_csv(path, index=False)
    print(f'Saved {name}: {len(df):,} rows')
    return path

def q(sql, label):
    try:
        df = client.query(sql).to_dataframe()
        query_status_rows.append({'label': label, 'status': 'pass', 'detail': f'n_rows={len(df)}'})
        return df
    except Exception as exc:
        query_status_rows.append({'label': label, 'status': 'fail', 'detail': repr(exc)[:2000]})
        print(f'QUERY FAILED [{label}]: {repr(exc)[:700]}')
        return pd.DataFrame()

def bq_str(value):
    return "'" + str(value).replace("'", "\\'") + "'"

save(pd.DataFrame([{
    'cdr': CDR,
    'cdr_source': cdr_source,
    'prep_cdr': PREP_CDR,
    'prep_cdr_source': prep_source,
    'google_project': GOOGLE_PROJECT,
    'google_project_source': google_project_source,
    'workspace_bucket': WORKSPACE_BUCKET,
    'data_dir': str(DATA_DIR),
    'output_dir': str(OUTPUT_DIR),
    'overwrite_outputs': OVERWRITE_OUTPUTS,
    'purpose': 'probe prep sample_name to person_id denominator mapping',
}]), '00_sample_name_denominator_probe_run_configuration.csv')
print('Outputs will be written to:', OUTPUT_DIR)

# %% Cell 3
# ============================================================
# Block 1. Candidate VDR/pathway vid list
# ============================================================
candidate_rows = [
    {'tier': 'Tier 1', 'gene': 'VDR', 'rsid': 'rs1544410', 'label': 'BsmI', 'role': 'classic VDR', 'vid': '12-47846052-C-T'},
    {'tier': 'Tier 1', 'gene': 'VDR', 'rsid': 'rs2228570', 'label': 'FokI', 'role': 'classic VDR', 'vid': '12-47879112-A-G'},
    {'tier': 'Tier 1', 'gene': 'VDR', 'rsid': 'rs731236', 'label': 'TaqI', 'role': 'classic VDR', 'vid': '12-47844974-A-G'},
    {'tier': 'Tier 1', 'gene': 'VDR', 'rsid': 'rs7975232', 'label': 'ApaI', 'role': 'classic VDR', 'vid': '12-47845054-C-A'},
    {'tier': 'Tier 2', 'gene': 'VDR', 'rsid': 'rs11568820', 'label': 'Cdx2', 'role': 'expanded VDR', 'vid': '12-47908762-C-T'},
    {'tier': 'Tier 2', 'gene': 'VDR', 'rsid': 'rs12721377', 'label': 'VDR interaction candidate', 'role': 'expanded VDR', 'vid': '12-47901568-T-C'},
    {'tier': 'Tier 2', 'gene': 'VDR', 'rsid': 'rs2239179', 'label': 'VDR interaction candidate', 'role': 'expanded VDR', 'vid': '12-47863983-T-C'},
    {'tier': 'Tier 2', 'gene': 'VDR', 'rsid': 'rs4516035', 'label': 'promoter candidate', 'role': 'expanded VDR', 'vid': '12-47906043-T-C'},
    {'tier': 'Tier 2', 'gene': 'VDR', 'rsid': 'rs7968585', 'label': 'VDR interaction candidate', 'role': 'expanded VDR', 'vid': '12-47838310-C-T'},
    {'tier': 'Tier 3', 'gene': 'CYP2R1', 'rsid': 'rs10741657', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '11-14893332-A-G'},
    {'tier': 'Tier 3', 'gene': 'AMDHD1', 'rsid': 'rs10745742', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '12-95964751-C-T'},
    {'tier': 'Tier 3', 'gene': 'DHCR7/NADSYN1', 'rsid': 'rs12785878', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '11-71456403-G-T'},
    {'tier': 'Tier 3', 'gene': 'CYP2R1', 'rsid': 'rs12794714', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '11-14892029-G-A'},
    {'tier': 'Tier 3', 'gene': 'GC', 'rsid': 'rs2282679', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '4-71742666-T-G'},
    {'tier': 'Tier 3', 'gene': 'GC', 'rsid': 'rs4588', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '4-71752606-G-T'},
    {'tier': 'Tier 3', 'gene': 'CYP24A1', 'rsid': 'rs6013897', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '20-54125940-T-A'},
    {'tier': 'Tier 3', 'gene': 'GC', 'rsid': 'rs7041', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '4-71752617-A-C'},
    {'tier': 'Tier 3', 'gene': 'SEC23A', 'rsid': 'rs8018720', 'label': '25OHD GWAS/pathway', 'role': 'MR/GRS candidate', 'vid': '14-39086981-G-C'},
]
candidate_df = pd.DataFrame(candidate_rows)
candidate_structs = ',\n  '.join([
    'STRUCT(' + bq_str(r.tier) + ' AS tier, ' + bq_str(r.gene) + ' AS gene, ' + bq_str(r.rsid) + ' AS rsid, ' + bq_str(r.label) + ' AS label, ' + bq_str(r.role) + ' AS role, ' + bq_str(r.vid) + ' AS vid)'
    for r in candidate_df.itertuples()
])
save(candidate_df, '10_sample_name_probe_candidate_vid_list.csv')

# %% Cell 4
# ============================================================
# Block 2. sample_name examples and mapping audit
# ============================================================
examples = q(f'''
SELECT * FROM (SELECT 'prep_wgs_metadata' AS table_name, CAST(sample_name AS STRING) AS sample_name FROM `{PREP_CDR}.prep_wgs_metadata` LIMIT 25)
UNION ALL
SELECT * FROM (SELECT 'prep_microarray_metadata' AS table_name, CAST(sample_name AS STRING) AS sample_name FROM `{PREP_CDR}.prep_microarray_metadata` LIMIT 25)
UNION ALL
SELECT * FROM (SELECT 'prep_structural_variants_metadata' AS table_name, CAST(sample_name AS STRING) AS sample_name FROM `{PREP_CDR}.prep_structural_variants_metadata` LIMIT 25)
''', 'prep_sample_name_examples_fixed')
save(examples, '10_prep_sample_name_examples_v9.csv')

mapping_audit = q(f'''
WITH prep AS (
  SELECT 'prep_wgs_metadata' AS table_name, CAST(sample_name AS STRING) AS sample_name FROM `{PREP_CDR}.prep_wgs_metadata`
  UNION ALL
  SELECT 'prep_microarray_metadata' AS table_name, CAST(sample_name AS STRING) AS sample_name FROM `{PREP_CDR}.prep_microarray_metadata`
  UNION ALL
  SELECT 'prep_structural_variants_metadata' AS table_name, CAST(sample_name AS STRING) AS sample_name FROM `{PREP_CDR}.prep_structural_variants_metadata`
), parsed AS (
  SELECT
    table_name,
    sample_name,
    SAFE_CAST(sample_name AS INT64) AS person_id_from_sample_name
  FROM prep
)
SELECT
  table_name,
  COUNT(*) AS n_rows,
  COUNT(DISTINCT sample_name) AS n_distinct_sample_name,
  COUNTIF(person_id_from_sample_name IS NOT NULL) AS n_exact_cast_rows,
  COUNT(DISTINCT person_id_from_sample_name) AS n_distinct_exact_person_id,
  COUNT(DISTINCT IF(p.person_id IS NOT NULL, person_id_from_sample_name, NULL)) AS n_matches_person,
  COUNT(DISTINCT IF(csp.person_id IS NOT NULL, person_id_from_sample_name, NULL)) AS n_matches_cb_search_person
FROM parsed x
LEFT JOIN `{CDR}.person` p ON x.person_id_from_sample_name = p.person_id
LEFT JOIN `{CDR}.cb_search_person` csp ON x.person_id_from_sample_name = csp.person_id
GROUP BY table_name
ORDER BY table_name
''', 'prep_sample_name_person_id_mapping_audit')
save(mapping_audit, '10_prep_sample_name_person_id_mapping_audit_v9.csv')
print(mapping_audit.to_string(index=False))

# %% Cell 5
# ============================================================
# Block 3. Choose denominator mapping rule
# ============================================================
denominator_source = 'none'
person_expr = None
if not mapping_audit.empty:
    exact_matches = int(pd.to_numeric(mapping_audit.get('n_matches_person', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
    if exact_matches > 0:
        denominator_source = 'prep_sample_name_exact_cast_to_person_id'
        person_expr = 'SAFE_CAST(sample_name AS INT64)'

can_build_denominator = person_expr is not None
status = pd.DataFrame([{
    'denominator_source': denominator_source,
    'can_build_valid_noncarrier_denominator': can_build_denominator,
    'note': 'If false, current workspace still lacks a person-level genomics denominator for non-carriers.'
}])
save(status, '10_sample_name_denominator_decision_v9.csv')
print(status.to_string(index=False))

# %% Cell 6
# ============================================================
# Block 4. Rebuild carrier denominator if sample_name maps to person_id
# ============================================================
base_path = OUTPUT_DIR / '04_analysis_base_clean_v9.csv'
q2_path = OUTPUT_DIR / '05_q2_crp_pair_v9.csv'
base = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()
q2_pair = pd.read_csv(q2_path) if q2_path.exists() else pd.DataFrame()
if not base.empty:
    base['person_id'] = pd.to_numeric(base['person_id'], errors='coerce').astype('Int64')
if not q2_pair.empty:
    q2_pair['person_id'] = pd.to_numeric(q2_pair['person_id'], errors='coerce').astype('Int64')

direct_counts = q(f'''
WITH candidate AS (
  SELECT * FROM UNNEST([
  {candidate_structs}
  ])
)
SELECT
  c.tier, c.gene, c.rsid, c.label, c.role, c.vid,
  IFNULL((SELECT COUNT(1) FROM UNNEST(vtp.person_ids.list) AS pid), 0) AS n_alt_carriers_in_cb_variant_to_person
FROM candidate c
LEFT JOIN `{CDR}.cb_variant_to_person` vtp ON c.vid = vtp.vid
ORDER BY c.tier, c.gene, c.rsid
''', 'direct_candidate_carrier_counts_sample_name_probe')
save(direct_counts, '06_direct_candidate_alt_carrier_counts_v9.csv')

carrier_cols = [f'{r.rsid}_carrier' for r in candidate_df.itertuples()]
if not can_build_denominator:
    save(pd.DataFrame(columns=['metric', 'n']), '06_genomics_denominator_summary_v9.csv')
    save(pd.DataFrame(columns=['tier', 'gene', 'rsid', 'label', 'vid', 'all_genomics_carriers', 'analysis_base_genomics_n', 'analysis_base_carriers', 'analysis_base_events_in_carriers', 'q2_pair_genomics_n', 'q2_pair_carriers']), '06_vdr_candidate_carrier_counts_v9.csv')
    save(pd.DataFrame(columns=['tier', 'gene', 'rsid', 'vid', 'carrier_status', 'vitd_group', 'n', 'events_mi_or_stroke', 'median_vitd', 'bmi_available']), '07_vdr_x_vitd_interaction_cell_counts_v9.csv')
else:
    carrier_exprs = ',\n    '.join([f"MAX(IF(cl.rsid = '{r.rsid}', 1, 0)) AS {r.rsid}_carrier" for r in candidate_df.itertuples()])
    denominator_sql = f'''
    SELECT person_id,
      MAX(has_whole_genome_variant) AS has_whole_genome_variant,
      MAX(has_array_data) AS has_array_data,
      MAX(has_structural_variant_data) AS has_structural_variant_data
    FROM (
      SELECT {person_expr} AS person_id, 1 AS has_whole_genome_variant, 0 AS has_array_data, 0 AS has_structural_variant_data FROM `{PREP_CDR}.prep_wgs_metadata`
      UNION ALL
      SELECT {person_expr} AS person_id, 0 AS has_whole_genome_variant, 1 AS has_array_data, 0 AS has_structural_variant_data FROM `{PREP_CDR}.prep_microarray_metadata`
      UNION ALL
      SELECT {person_expr} AS person_id, 0 AS has_whole_genome_variant, 0 AS has_array_data, 1 AS has_structural_variant_data FROM `{PREP_CDR}.prep_structural_variants_metadata`
    )
    WHERE person_id IS NOT NULL
    GROUP BY person_id
    '''
    carrier_sql = f'''
    WITH candidate AS (
      SELECT * FROM UNNEST([
      {candidate_structs}
      ])
    ),
    genomic_denominator AS ({denominator_sql}),
    carrier_long AS (
      SELECT c.tier, c.gene, c.rsid, c.label, c.role, c.vid, pid.element AS person_id
      FROM candidate c
      JOIN `{CDR}.cb_variant_to_person` vtp ON c.vid = vtp.vid
      CROSS JOIN UNNEST(vtp.person_ids.list) AS pid
    ),
    carrier_wide AS (
      SELECT
        gd.person_id,
        gd.has_whole_genome_variant,
        gd.has_array_data,
        gd.has_structural_variant_data,
        {carrier_exprs}
      FROM genomic_denominator gd
      LEFT JOIN carrier_long cl USING (person_id)
      GROUP BY gd.person_id, gd.has_whole_genome_variant, gd.has_array_data, gd.has_structural_variant_data
    )
    SELECT * FROM carrier_wide
    '''
    carrier_wide = q(carrier_sql, 'sample_name_candidate_carrier_wide_genomics_denominator')
    if carrier_wide.empty:
        save(pd.DataFrame(columns=['metric', 'n']), '06_genomics_denominator_summary_v9.csv')
        save(pd.DataFrame(columns=['tier', 'gene', 'rsid', 'label', 'vid', 'all_genomics_carriers', 'analysis_base_genomics_n', 'analysis_base_carriers', 'analysis_base_events_in_carriers', 'q2_pair_genomics_n', 'q2_pair_carriers']), '06_vdr_candidate_carrier_counts_v9.csv')
        save(pd.DataFrame(columns=['tier', 'gene', 'rsid', 'vid', 'carrier_status', 'vitd_group', 'n', 'events_mi_or_stroke', 'median_vitd', 'bmi_available']), '07_vdr_x_vitd_interaction_cell_counts_v9.csv')
    else:
        carrier_wide['person_id'] = pd.to_numeric(carrier_wide['person_id'], errors='coerce').astype('Int64')
        for col in carrier_cols:
            carrier_wide[col] = pd.to_numeric(carrier_wide[col], errors='coerce').fillna(0).astype(int)
        base_gen = base.merge(carrier_wide, on='person_id', how='inner') if not base.empty else pd.DataFrame()
        q2_pair_gen = q2_pair.merge(carrier_wide, on='person_id', how='inner') if not q2_pair.empty else pd.DataFrame()
        save(base_gen, '06_analysis_base_v9_genomics_carrier.csv')
        if not q2_pair_gen.empty:
            save(q2_pair_gen, '06_q2_crp_pair_v9_genomics_carrier.csv')
        denom_summary = pd.DataFrame([
            {'metric': 'denominator_source_' + denominator_source, 'n': 1},
            {'metric': 'genomics_denominator_rows', 'n': len(carrier_wide)},
            {'metric': 'denominator_has_wgs', 'n': int(carrier_wide['has_whole_genome_variant'].sum())},
            {'metric': 'denominator_has_array', 'n': int(carrier_wide['has_array_data'].sum())},
            {'metric': 'analysis_base_rows_before_genomics_restriction', 'n': len(base)},
            {'metric': 'analysis_base_rows_with_genomics_denominator', 'n': len(base_gen)},
            {'metric': 'q2_pair_rows_before_genomics_restriction', 'n': len(q2_pair)},
            {'metric': 'q2_pair_rows_with_genomics_denominator', 'n': len(q2_pair_gen)},
        ])
        save(denom_summary, '06_genomics_denominator_summary_v9.csv')
        carrier_count_rows = []
        for r in candidate_df.itertuples():
            col = f'{r.rsid}_carrier'
            carrier_count_rows.append({
                'tier': r.tier, 'gene': r.gene, 'rsid': r.rsid, 'label': r.label, 'vid': r.vid,
                'all_genomics_carriers': int(carrier_wide[col].sum()),
                'analysis_base_genomics_n': len(base_gen),
                'analysis_base_carriers': int(base_gen[col].sum()) if not base_gen.empty else 0,
                'analysis_base_events_in_carriers': int(base_gen.loc[base_gen[col].eq(1), 'event_mi_or_stroke'].sum()) if (not base_gen.empty and 'event_mi_or_stroke' in base_gen) else 0,
                'q2_pair_genomics_n': len(q2_pair_gen),
                'q2_pair_carriers': int(q2_pair_gen[col].sum()) if not q2_pair_gen.empty else 0,
            })
        save(pd.DataFrame(carrier_count_rows), '06_vdr_candidate_carrier_counts_v9.csv')
        readiness_rows = []
        if not base_gen.empty and 'vitd_group' in base_gen.columns:
            for r in candidate_df.itertuples():
                col = f'{r.rsid}_carrier'
                for carrier_status, sub1 in base_gen.groupby(col, dropna=False):
                    for vitd_group, sub2 in sub1.groupby('vitd_group', dropna=False):
                        readiness_rows.append({
                            'tier': r.tier, 'gene': r.gene, 'rsid': r.rsid, 'vid': r.vid,
                            'carrier_status': int(carrier_status) if pd.notna(carrier_status) else np.nan,
                            'vitd_group': str(vitd_group), 'n': len(sub2),
                            'events_mi_or_stroke': int(sub2['event_mi_or_stroke'].sum()) if 'event_mi_or_stroke' in sub2 else 0,
                            'median_vitd': float(pd.to_numeric(sub2['vitd_main'], errors='coerce').median()) if 'vitd_main' in sub2 else np.nan,
                            'bmi_available': int(sub2['bmi_full'].notna().sum()) if 'bmi_full' in sub2 else 0,
                        })
        save(pd.DataFrame(readiness_rows), '07_vdr_x_vitd_interaction_cell_counts_v9.csv')

# %% Cell 7
# ============================================================
# Block 5. Dataset-local PC/relatedness search and final status
# ============================================================
pc_schema_hits = q(f'''
SELECT '{CDR}' AS bq_dataset, table_name, column_name, data_type
FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS`
WHERE LOWER(column_name) LIKE '%ancestry%'
   OR LOWER(column_name) LIKE 'pc%'
   OR LOWER(column_name) LIKE '%principal%component%'
   OR LOWER(column_name) LIKE '%related%'
   OR LOWER(table_name) LIKE '%kinship%'
   OR LOWER(table_name) LIKE '%related%'
UNION ALL
SELECT '{PREP_CDR}' AS bq_dataset, table_name, column_name, data_type
FROM `{PREP_CDR}.INFORMATION_SCHEMA.COLUMNS`
WHERE LOWER(column_name) LIKE '%ancestry%'
   OR LOWER(column_name) LIKE 'pc%'
   OR LOWER(column_name) LIKE '%principal%component%'
   OR LOWER(column_name) LIKE '%related%'
   OR LOWER(table_name) LIKE '%kinship%'
   OR LOWER(table_name) LIKE '%related%'
ORDER BY bq_dataset, table_name, column_name
''', 'dataset_local_pc_relatedness_schema_search')
save(pc_schema_hits, '08_pc_relatedness_schema_search_v9.csv')

mr_status = pd.DataFrame([
    {'requirement': 'Candidate ALT-carrier counts', 'status': 'available' if (not direct_counts.empty and pd.to_numeric(direct_counts.get('n_alt_carriers_in_cb_variant_to_person', pd.Series(dtype=float)), errors='coerce').sum() > 0) else 'missing_or_empty', 'evidence': '06_direct_candidate_alt_carrier_counts_v9.csv'},
    {'requirement': 'Valid non-carrier denominator', 'status': 'available' if can_build_denominator else 'not_available', 'evidence': '10_sample_name_denominator_decision_v9.csv; 10_prep_sample_name_person_id_mapping_audit_v9.csv'},
    {'requirement': 'Main VDR x 25(OH)D interaction readiness', 'status': 'available_if_07_nonempty' if can_build_denominator else 'not_feasible_without_denominator', 'evidence': '07_vdr_x_vitd_interaction_cell_counts_v9.csv'},
    {'requirement': 'Additive genotype dosage / hard calls', 'status': 'not_available_from_cb_variant_to_person', 'evidence': 'carrier table is dominant ALT carrier membership only'},
    {'requirement': 'Genetic ancestry PCs', 'status': 'needs_review' if not pc_schema_hits.empty else 'not_found_in_schema_search', 'evidence': '08_pc_relatedness_schema_search_v9.csv'},
    {'requirement': 'Formal MR readiness', 'status': 'not_currently_feasible', 'evidence': 'Need dosage/hard calls, harmonized GWAS weights, ancestry PCs, QC/relatedness'},
])
save(mr_status, '08_mr_readiness_status_v9.csv')
save(pd.DataFrame(query_status_rows), '99_sample_name_denominator_probe_query_status.csv')
manifest = pd.DataFrame([{'output_file': p.name, 'path': str(p), 'size_bytes': p.stat().st_size} for p in sorted(OUTPUT_DIR.glob('*.csv'))])
save(manifest, '99_output_manifest.csv')
print('DONE')
print('Upload these first: 10_prep_sample_name_examples_v9.csv, 10_prep_sample_name_person_id_mapping_audit_v9.csv, 10_sample_name_denominator_decision_v9.csv, 06_genomics_denominator_summary_v9.csv, 06_vdr_candidate_carrier_counts_v9.csv, 07_vdr_x_vitd_interaction_cell_counts_v9.csv, 08_mr_readiness_status_v9.csv, 99_sample_name_denominator_probe_query_status.csv')
