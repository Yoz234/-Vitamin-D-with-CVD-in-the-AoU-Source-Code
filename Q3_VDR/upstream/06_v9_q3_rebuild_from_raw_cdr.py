# Source companion of 06_v9_q3_rebuild_from_raw_cdr.ipynb
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
OVERWRITE_OUTPUTS = True  # Same-name CSV outputs in OUTPUT_DIR will be overwritten by this notebook.
INCLUDE_D2_AS_FALLBACK = False  # Keep False for primary 25(OH)D rebuild; D2-only is audited but not primary total 25(OH)D.
SAVE_FULL_GENOMICS_DENOMINATOR = False  # Set True only if you need the large all-genomics carrier table.

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
        print(f'QUERY FAILED [{label}]: {repr(exc)[:500]}')
        return pd.DataFrame()

def int_list(values):
    values = [int(v) for v in values]
    return '(' + ', '.join(map(str, sorted(set(values)))) + ')'

def bq_str(value):
    return "'" + str(value).replace("'", "\\'") + "'"

run_config = pd.DataFrame([{
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
    'include_d2_as_fallback': INCLUDE_D2_AS_FALLBACK,
    'save_full_genomics_denominator': SAVE_FULL_GENOMICS_DENOMINATOR,
}])
save(run_config, '00_run_configuration.csv')
print('Outputs will be written to:', OUTPUT_DIR)

# %% Cell 3
# ============================================================
# Block 1. Embedded concept sets and VDR candidate vid list
# ============================================================
VITD_TOTAL_OR_D3 = [40765040, 3020149]
VITD_D2_ONLY = [3049536]
VITD_AUDIT_EXTRA = [37077414, 40791366]
VITD_PRIMARY = VITD_TOTAL_OR_D3 + (VITD_D2_ONLY if INCLUDE_D2_AS_FALLBACK else [])

CRP_CONCEPTS = [3020460, 3010156, 4208414, 46272908]
CRP_UNIT_FACTOR = {8751: 1.0, 4122414: 1.0, 8840: 10.0, 4121396: 10.0}
BMI_CONCEPTS = [3038553, 4245997]

DIABETES = [443732, 376065, 43531564, 37016349, 4130162, 201826, 4304377, 40485020, 4200875, 4193704, 40482801, 201820]
HYPERTENSION = [312648, 4028741, 320128, 316866, 443771, 317895]
HYPERLIPIDEMIA = [4134862, 4029305, 432867, 4120314, 4030619, 438720, 437827]

MI_PRIMARY = [312327, 434376, 438170, 438447, 441579, 436706, 4270024, 4296653, 46270162, 46270163, 444406, 4329847]
OLD_MI_BASELINE = [314666]
CAD_CHD_BASELINE = [36712983, 764123, 4242670, 40481919, 36712982, 317576]
STROKE_ENDPOINT = [36684840, 4111710, 36716999, 43531607, 42539262, 42535148, 42535461, 42535149, 37395574, 37395575, 42535460, 42535459, 762344, 764721, 4099974, 4045734, 4153352, 4310996, 43531605, 4110196, 4046363, 4159140]
STROKE_PRIMARY = [4111710, 4111711, 443454, 4108356, 45772786, 4110190, 46273649, 46270031, 4110192, 45767658, 381316, 4153352, 4310996, 43531605]
INCIDENT_CVD = sorted(set(MI_PRIMARY + STROKE_PRIMARY + STROKE_ENDPOINT))
BASELINE_CVD_EXCLUSION = sorted(set(MI_PRIMARY + OLD_MI_BASELINE + CAD_CHD_BASELINE + STROKE_PRIMARY + STROKE_ENDPOINT))

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
save(candidate_df, '01_vdr_candidate_vid_list.csv')

concept_rows = []
for name, role, ids in [
    ('vitd_primary', 'exposure', VITD_PRIMARY), ('vitd_audit_extra', 'audit only', VITD_D2_ONLY + VITD_AUDIT_EXTRA),
    ('crp', 'inflammation outcome', CRP_CONCEPTS), ('bmi', 'covariate', BMI_CONCEPTS),
    ('diabetes', 'baseline covariate', DIABETES), ('hypertension', 'baseline covariate', HYPERTENSION), ('hyperlipidemia', 'baseline covariate', HYPERLIPIDEMIA),
    ('mi_primary', 'incident outcome', MI_PRIMARY), ('stroke_primary_endpoint', 'incident outcome', INCIDENT_CVD), ('baseline_cvd_exclusion', 'baseline exclusion', BASELINE_CVD_EXCLUSION),
]:
    for cid in sorted(set(ids)):
        concept_rows.append({'concept_set': name, 'role': role, 'concept_id': cid})
concept_catalog = pd.DataFrame(concept_rows)
save(concept_catalog, '01_embedded_concept_catalog.csv')

# %% Cell 4
# ============================================================
# Block 2. v9 raw concept/unit audits before rebuilding sample
# ============================================================
vitd_audit_ids = sorted(set(VITD_TOTAL_OR_D3 + VITD_D2_ONLY + VITD_AUDIT_EXTRA))
vitd_unit_audit = q(f'''
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
WHERE m.measurement_concept_id IN {int_list(vitd_audit_ids)}
GROUP BY concept_id, concept_name, unit_concept_id, unit_name
ORDER BY concept_id, n_rows DESC
''', 'vitd_unit_audit')
save(vitd_unit_audit, '02_vitd_unit_concept_audit_v9.csv')

crp_unit_audit = q(f'''
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
''', 'crp_unit_audit')
save(crp_unit_audit, '02_crp_unit_concept_audit_v9.csv')

target_condition_union = ' UNION ALL '.join([f'SELECT {bq_str(row.concept_set)} AS concept_set, {int(row.concept_id)} AS concept_id' for row in concept_catalog[concept_catalog['role'].isin(['baseline covariate', 'incident outcome', 'baseline exclusion'])].itertuples()])
condition_audit = q(f'''
WITH target AS ({target_condition_union})
SELECT
  t.concept_set,
  t.concept_id,
  c.concept_name,
  COUNT(co.condition_occurrence_id) AS n_rows,
  COUNT(DISTINCT co.person_id) AS n_persons,
  MIN(co.condition_start_date) AS min_date,
  MAX(co.condition_start_date) AS max_date
FROM target t
LEFT JOIN `{CDR}.concept` c ON t.concept_id = c.concept_id
LEFT JOIN `{CDR}.condition_occurrence` co ON t.concept_id = co.condition_concept_id
GROUP BY t.concept_set, t.concept_id, c.concept_name
ORDER BY t.concept_set, n_rows DESC
''', 'condition_concept_audit')
save(condition_audit, '02_condition_concept_count_audit_v9.csv')

# %% Cell 5
# ============================================================
# Block 3. Build v9 clinical analysis base from raw CDR
# ============================================================
base_sql = f'''
WITH vitd_valid AS (
  SELECT
    m.person_id,
    m.measurement_date AS index_date,
    m.measurement_datetime AS index_datetime,
    m.measurement_concept_id AS vitd_concept_id,
    c.concept_name AS vitd_concept_name,
    m.unit_concept_id AS vitd_unit_concept_id,
    cu.concept_name AS vitd_unit_name,
    CAST(m.value_as_number AS FLOAT64) AS vitd_ng_ml,
    CASE
      WHEN m.measurement_concept_id = 40765040 THEN 0
      WHEN m.measurement_concept_id = 3020149 THEN 1
      WHEN m.measurement_concept_id = 3049536 THEN 2
      ELSE 9
    END AS vitd_priority
  FROM `{CDR}.measurement` m
  LEFT JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
  LEFT JOIN `{CDR}.concept` cu ON m.unit_concept_id = cu.concept_id
  WHERE m.measurement_concept_id IN {int_list(VITD_PRIMARY)}
    AND m.measurement_date IS NOT NULL
    AND m.value_as_number IS NOT NULL
    AND CAST(m.value_as_number AS FLOAT64) BETWEEN 1 AND 150
  QUALIFY ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date, vitd_priority, m.measurement_datetime) = 1
),
demo AS (
  SELECT
    p.person_id,
    p.year_of_birth,
    p.month_of_birth,
    p.day_of_birth,
    gc.concept_name AS gender,
    sc.concept_name AS sex_at_birth,
    rc.concept_name AS race,
    ec.concept_name AS ethnicity,
    CASE
      WHEN LOWER(COALESCE(sc.concept_name, gc.concept_name, '')) LIKE '%female%' THEN 1
      WHEN LOWER(COALESCE(sc.concept_name, gc.concept_name, '')) LIKE '%male%' THEN 0
      ELSE NULL
    END AS female
  FROM `{CDR}.person` p
  LEFT JOIN `{CDR}.concept` gc ON p.gender_concept_id = gc.concept_id
  LEFT JOIN `{CDR}.concept` sc ON p.sex_at_birth_concept_id = sc.concept_id
  LEFT JOIN `{CDR}.concept` rc ON p.race_concept_id = rc.concept_id
  LEFT JOIN `{CDR}.concept` ec ON p.ethnicity_concept_id = ec.concept_id
),
index_demo AS (
  SELECT
    v.*,
    d.gender,
    d.sex_at_birth,
    d.race,
    d.ethnicity,
    d.female,
    DATE_DIFF(v.index_date, DATE(d.year_of_birth, COALESCE(NULLIF(d.month_of_birth, 0), 7), COALESCE(NULLIF(d.day_of_birth, 0), 15)), YEAR) AS age_at_index
  FROM vitd_valid v
  JOIN demo d USING (person_id)
),
bmi AS (
  SELECT
    i.person_id,
    CAST(m.value_as_number AS FLOAT64) AS bmi_full,
    m.measurement_date AS bmi_date,
    DATE_DIFF(i.index_date, m.measurement_date, DAY) AS bmi_days_before_index,
    m.measurement_concept_id AS bmi_concept_id
  FROM index_demo i
  JOIN `{CDR}.measurement` m ON i.person_id = m.person_id
  WHERE m.measurement_concept_id IN {int_list(BMI_CONCEPTS)}
    AND m.measurement_date IS NOT NULL
    AND m.value_as_number IS NOT NULL
    AND CAST(m.value_as_number AS FLOAT64) BETWEEN 12 AND 70
    AND DATE_DIFF(i.index_date, m.measurement_date, DAY) BETWEEN -30 AND 365
  QUALIFY ROW_NUMBER() OVER (PARTITION BY i.person_id ORDER BY ABS(DATE_DIFF(i.index_date, m.measurement_date, DAY)), IF(DATE_DIFF(i.index_date, m.measurement_date, DAY) >= 0, 0, 1)) = 1
),
condition_flags AS (
  SELECT
    i.person_id,
    COALESCE(MAX(IF(co.condition_concept_id IN {int_list(DIABETES)}, 1, 0)), 0) AS baseline_diabetes,
    COALESCE(MAX(IF(co.condition_concept_id IN {int_list(HYPERTENSION)}, 1, 0)), 0) AS baseline_hypertension,
    COALESCE(MAX(IF(co.condition_concept_id IN {int_list(HYPERLIPIDEMIA)}, 1, 0)), 0) AS baseline_hyperlipidemia,
    COALESCE(MAX(IF(co.condition_concept_id IN {int_list(BASELINE_CVD_EXCLUSION)}, 1, 0)), 0) AS baseline_cvd_exclusion
  FROM index_demo i
  LEFT JOIN `{CDR}.condition_occurrence` co
    ON i.person_id = co.person_id
   AND co.condition_start_date IS NOT NULL
   AND co.condition_start_date <= i.index_date
  GROUP BY i.person_id
),
obs AS (
  SELECT
    i.person_id,
    MIN(IF(op.observation_period_start_date <= i.index_date AND op.observation_period_end_date >= i.index_date, op.observation_period_start_date, NULL)) AS index_observation_start_date,
    MAX(IF(op.observation_period_end_date >= i.index_date, op.observation_period_end_date, NULL)) AS censor_date
  FROM index_demo i
  LEFT JOIN `{CDR}.observation_period` op ON i.person_id = op.person_id
  GROUP BY i.person_id
),
incident_events AS (
  SELECT
    i.person_id,
    MIN(co.condition_start_date) AS first_cvd_event_date,
    MIN(IF(co.condition_concept_id IN {int_list(MI_PRIMARY)}, co.condition_start_date, NULL)) AS first_mi_date,
    MIN(IF(co.condition_concept_id IN {int_list(STROKE_PRIMARY + STROKE_ENDPOINT)}, co.condition_start_date, NULL)) AS first_stroke_date
  FROM index_demo i
  LEFT JOIN obs o USING (person_id)
  JOIN `{CDR}.condition_occurrence` co
    ON i.person_id = co.person_id
   AND co.condition_start_date IS NOT NULL
   AND co.condition_start_date > i.index_date
   AND (o.censor_date IS NULL OR co.condition_start_date <= o.censor_date)
   AND co.condition_concept_id IN {int_list(INCIDENT_CVD)}
  GROUP BY i.person_id
)
SELECT
  i.person_id,
  i.index_date,
  i.vitd_ng_ml,
  i.vitd_ng_ml AS vitd_main,
  i.vitd_concept_id,
  i.vitd_concept_name,
  i.vitd_unit_concept_id,
  i.vitd_unit_name,
  i.age_at_index,
  i.gender,
  i.sex_at_birth,
  i.female,
  i.race,
  i.ethnicity,
  b.bmi_full,
  b.bmi_date,
  b.bmi_days_before_index,
  cf.baseline_diabetes,
  cf.baseline_hypertension,
  cf.baseline_hyperlipidemia,
  cf.baseline_cvd_exclusion,
  o.index_observation_start_date,
  o.censor_date,
  ev.first_cvd_event_date,
  ev.first_mi_date,
  ev.first_stroke_date,
  IF(ev.first_cvd_event_date IS NULL, 0, 1) AS event_mi_or_stroke,
  IF(ev.first_mi_date IS NULL, 0, 1) AS event_mi,
  IF(ev.first_stroke_date IS NULL, 0, 1) AS event_stroke,
  DATE_DIFF(COALESCE(ev.first_cvd_event_date, o.censor_date), i.index_date, DAY) AS follow_up_days
FROM index_demo i
LEFT JOIN bmi b USING (person_id)
LEFT JOIN condition_flags cf USING (person_id)
LEFT JOIN obs o USING (person_id)
LEFT JOIN incident_events ev USING (person_id)
WHERE i.age_at_index >= 18
'''

base_all = q(base_sql, 'clinical_base_all_adult_vitd')
if base_all.empty:
    save(pd.DataFrame(query_status_rows), '99_bigquery_query_status.csv')
    raise RuntimeError('Clinical base query returned no rows; inspect 99_bigquery_query_status.csv after this cell.')

date_cols = ['index_date', 'bmi_date', 'index_observation_start_date', 'censor_date', 'first_cvd_event_date', 'first_mi_date', 'first_stroke_date']
for col in date_cols:
    if col in base_all.columns:
        base_all[col] = pd.to_datetime(base_all[col], errors='coerce')
for col in ['vitd_ng_ml', 'vitd_main', 'age_at_index', 'bmi_full', 'follow_up_days']:
    base_all[col] = pd.to_numeric(base_all[col], errors='coerce')

base_all['winter'] = base_all['index_date'].dt.month.isin([11, 12, 1, 2, 3, 4]).astype('Int64')
base_all['vitd_deficient'] = (base_all['vitd_main'] < 20).astype('Int64')
base_all['vitd_severe_deficient'] = (base_all['vitd_main'] < 10).astype('Int64')
base_all['vitd_group'] = pd.cut(base_all['vitd_main'], bins=[-np.inf, 10, 20, 30, np.inf], labels=['<10', '10-<20', '20-<30', '>=30'])
save(base_all, '03_v9_all_adult_vitd_index_with_flags.csv')

base = base_all[(base_all['baseline_cvd_exclusion'].eq(0)) & (base_all['follow_up_days'] > 0)].copy()
base['follow_up_years'] = base['follow_up_days'] / 365.25
save(base, '04_analysis_base_clean_v9.csv')

clinical_summary_rows = [
    {'metric': 'adult_valid_25ohd_index_rows_before_cvd_exclusion', 'n': len(base_all)},
    {'metric': 'excluded_baseline_cvd', 'n': int(base_all['baseline_cvd_exclusion'].eq(1).sum())},
    {'metric': 'adult_valid_25ohd_cvd_free_positive_followup', 'n': len(base)},
    {'metric': 'incident_mi_or_stroke_events', 'n': int(base['event_mi_or_stroke'].sum())},
    {'metric': 'incident_mi_events', 'n': int(base['event_mi'].sum())},
    {'metric': 'incident_stroke_events', 'n': int(base['event_stroke'].sum())},
    {'metric': 'bmi_available_in_analysis_base', 'n': int(base['bmi_full'].notna().sum())},
]
save(pd.DataFrame(clinical_summary_rows), '04_analysis_base_summary_v9.csv')

# %% Cell 6
# ============================================================
# Block 4. Rebuild baseline/pre-index CRP pair using Q2 cleaning rules
# ============================================================
crp_raw = q(f'''
SELECT
  m.person_id,
  m.measurement_date AS crp_date,
  m.measurement_concept_id AS crp_concept_id,
  c.concept_name AS crp_concept_name,
  m.unit_concept_id AS unit_concept_id,
  cu.concept_name AS unit_name,
  CAST(m.value_as_number AS FLOAT64) AS crp_value
FROM `{CDR}.measurement` m
LEFT JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
LEFT JOIN `{CDR}.concept` cu ON m.unit_concept_id = cu.concept_id
WHERE m.measurement_concept_id IN {int_list(CRP_CONCEPTS)}
  AND m.measurement_date IS NOT NULL
  AND m.value_as_number IS NOT NULL
''', 'crp_raw_records')

if crp_raw.empty:
    crp_clean = pd.DataFrame(columns=['person_id', 'crp_date', 'crp_concept_id', 'unit_concept_id', 'crp_mgL', 'ambiguous_unit'])
    crp_baseline = pd.DataFrame(columns=['person_id', 'crp_mgL', 'days_before', 'ambiguous_unit', 'crp_date'])
    q2_pair = pd.DataFrame()
else:
    crp = crp_raw.copy()
    crp['crp_date'] = pd.to_datetime(crp['crp_date'], errors='coerce')
    crp['crp_value'] = pd.to_numeric(crp['crp_value'], errors='coerce')
    crp['unit_key'] = pd.to_numeric(crp['unit_concept_id'], errors='coerce').fillna(-1).astype(int)
    crp['factor'] = crp['unit_key'].map(CRP_UNIT_FACTOR)
    group_median = crp.groupby(['crp_concept_id', 'unit_key'])['crp_value'].transform('median')
    ambiguous = crp['unit_key'].isin([0, -1])
    crp.loc[ambiguous, 'factor'] = np.where(group_median[ambiguous] < 1.5, 10.0, 1.0)
    crp = crp.dropna(subset=['factor', 'crp_value']).copy()
    crp['crp_mgL'] = crp['crp_value'] * crp['factor']
    crp['ambiguous_unit'] = crp['unit_key'].isin([0, -1]).astype(int)
    crp_clean = crp[(crp['crp_mgL'] >= 0.1) & (crp['crp_mgL'] <= 300)].copy()
    save(crp_clean[['person_id', 'crp_date', 'crp_concept_id', 'unit_concept_id', 'unit_name', 'crp_value', 'crp_mgL', 'ambiguous_unit']], '05_crp_long_clean_v9.csv')
    matched = crp_clean.merge(base[['person_id', 'index_date', 'vitd_main']], on='person_id', how='inner')
    matched['days_before'] = (matched['index_date'] - matched['crp_date']).dt.days
    matched = matched[matched['days_before'] >= 0].copy()
    crp_baseline = matched.sort_values(['person_id', 'days_before']).groupby('person_id', as_index=False).first()
    crp_baseline = crp_baseline[['person_id', 'crp_mgL', 'days_before', 'ambiguous_unit', 'crp_date', 'crp_concept_id', 'unit_concept_id']]
    q2_pair = base.merge(crp_baseline, on='person_id', how='inner')
    q2_pair = q2_pair[(q2_pair['vitd_main'] > 0) & (q2_pair['crp_mgL'] > 0)].copy()
    q2_pair['log_vitd'] = np.log(q2_pair['vitd_main'])
    q2_pair['log_crp'] = np.log(q2_pair['crp_mgL'])

save(crp_baseline, '05_q2_crp_baseline_clean_v9.csv')
if not q2_pair.empty:
    save(q2_pair, '05_q2_crp_pair_v9.csv')
crp_summary = pd.DataFrame([
    {'metric': 'raw_crp_rows', 'n': len(crp_raw)},
    {'metric': 'clean_crp_rows', 'n': len(crp_clean)},
    {'metric': 'baseline_preindex_crp_persons_in_analysis_base', 'n': len(crp_baseline)},
    {'metric': 'q2_pair_rows', 'n': len(q2_pair)},
    {'metric': 'q2_pair_ambiguous_unit_rows', 'n': int(q2_pair['ambiguous_unit'].sum()) if not q2_pair.empty else 0},
])
save(crp_summary, '05_q2_crp_pair_summary_v9.csv')

# %% Cell 7
# ============================================================
# Block 5. Build v9 genomics denominator and candidate carrier layer
# ============================================================
candidate_structs = ',\n  '.join([
    'STRUCT(' + bq_str(r.tier) + ' AS tier, ' + bq_str(r.gene) + ' AS gene, ' + bq_str(r.rsid) + ' AS rsid, ' + bq_str(r.label) + ' AS label, ' + bq_str(r.role) + ' AS role, ' + bq_str(r.vid) + ' AS vid)'
    for r in candidate_df.itertuples()
])
carrier_exprs = ',\n    '.join([f"MAX(IF(cl.rsid = '{r.rsid}', 1, 0)) AS {r.rsid}_carrier" for r in candidate_df.itertuples()])

carrier_sql = f'''
WITH candidate AS (
  SELECT * FROM UNNEST([
  {candidate_structs}
  ])
),
genomic_denominator AS (
  SELECT
    person_id,
    COALESCE(has_whole_genome_variant, 0) AS has_whole_genome_variant,
    COALESCE(has_array_data, 0) AS has_array_data,
    COALESCE(has_lr_whole_genome_variant, 0) AS has_lr_whole_genome_variant,
    COALESCE(has_structural_variant_data, 0) AS has_structural_variant_data
  FROM `{CDR}.cb_search_person`
  WHERE COALESCE(has_whole_genome_variant, 0) = 1 OR COALESCE(has_array_data, 0) = 1
),
carrier_long AS (
  SELECT
    c.tier,
    c.gene,
    c.rsid,
    c.label,
    c.role,
    c.vid,
    pid.element AS person_id
  FROM candidate c
  JOIN `{CDR}.cb_variant_to_person` vtp ON c.vid = vtp.vid
  CROSS JOIN UNNEST(vtp.person_ids.list) AS pid
),
carrier_wide AS (
  SELECT
    gd.person_id,
    gd.has_whole_genome_variant,
    gd.has_array_data,
    gd.has_lr_whole_genome_variant,
    gd.has_structural_variant_data,
    {carrier_exprs}
  FROM genomic_denominator gd
  LEFT JOIN carrier_long cl USING (person_id)
  GROUP BY gd.person_id, gd.has_whole_genome_variant, gd.has_array_data, gd.has_lr_whole_genome_variant, gd.has_structural_variant_data
)
SELECT * FROM carrier_wide
'''

carrier_wide = q(carrier_sql, 'candidate_carrier_wide_genomics_denominator')
if carrier_wide.empty:
    print('Carrier wide table is empty. Inspect cb_variant_to_person schema and query status.')
else:
    carrier_wide['person_id'] = pd.to_numeric(carrier_wide['person_id'], errors='coerce').astype('Int64')
    carrier_cols = [c for c in carrier_wide.columns if c.endswith('_carrier')]
    for col in carrier_cols:
        carrier_wide[col] = pd.to_numeric(carrier_wide[col], errors='coerce').fillna(0).astype(int)
    if SAVE_FULL_GENOMICS_DENOMINATOR:
        save(carrier_wide, '06_vdr_carrier_layer_all_genomics_denominator_v9.csv')
    base_gen = base.merge(carrier_wide, on='person_id', how='inner')
    save(base_gen, '06_analysis_base_v9_genomics_carrier.csv')
    q2_pair_gen = q2_pair.merge(carrier_wide, on='person_id', how='inner') if 'q2_pair' in globals() and not q2_pair.empty else pd.DataFrame()
    if not q2_pair_gen.empty:
        save(q2_pair_gen, '06_q2_crp_pair_v9_genomics_carrier.csv')

    denom_summary = pd.DataFrame([
        {'metric': 'genomics_denominator_rows_wgs_or_array', 'n': len(carrier_wide)},
        {'metric': 'genomics_denominator_has_wgs', 'n': int(carrier_wide['has_whole_genome_variant'].sum())},
        {'metric': 'genomics_denominator_has_array', 'n': int(carrier_wide['has_array_data'].sum())},
        {'metric': 'analysis_base_rows_before_genomics_restriction', 'n': len(base)},
        {'metric': 'analysis_base_rows_with_wgs_or_array', 'n': len(base_gen)},
        {'metric': 'q2_pair_rows_before_genomics_restriction', 'n': len(q2_pair) if 'q2_pair' in globals() else 0},
        {'metric': 'q2_pair_rows_with_wgs_or_array', 'n': len(q2_pair_gen)},
    ])
    save(denom_summary, '06_genomics_denominator_summary_v9.csv')

    carrier_count_rows = []
    for r in candidate_df.itertuples():
        col = f'{r.rsid}_carrier'
        if col not in carrier_wide.columns:
            continue
        carrier_count_rows.append({
            'tier': r.tier, 'gene': r.gene, 'rsid': r.rsid, 'label': r.label, 'vid': r.vid,
            'all_genomics_carriers': int(carrier_wide[col].sum()),
            'analysis_base_genomics_n': len(base_gen),
            'analysis_base_carriers': int(base_gen[col].sum()) if not base_gen.empty else 0,
            'analysis_base_events_in_carriers': int(base_gen.loc[base_gen[col].eq(1), 'event_mi_or_stroke'].sum()) if not base_gen.empty else 0,
            'q2_pair_genomics_n': len(q2_pair_gen),
            'q2_pair_carriers': int(q2_pair_gen[col].sum()) if not q2_pair_gen.empty else 0,
        })
    carrier_counts = pd.DataFrame(carrier_count_rows)
    save(carrier_counts, '06_vdr_candidate_carrier_counts_v9.csv')

# %% Cell 8
# ============================================================
# Block 6. Interaction readiness, MR-readiness status, and manifest
# ============================================================
readiness_rows = []
if 'base_gen' in globals() and not base_gen.empty:
    for r in candidate_df.itertuples():
        col = f'{r.rsid}_carrier'
        if col not in base_gen.columns:
            continue
        for carrier_status, sub1 in base_gen.groupby(col, dropna=False):
            for vitd_group, sub2 in sub1.groupby('vitd_group', dropna=False):
                readiness_rows.append({
                    'tier': r.tier,
                    'gene': r.gene,
                    'rsid': r.rsid,
                    'vid': r.vid,
                    'carrier_status': int(carrier_status) if pd.notna(carrier_status) else np.nan,
                    'vitd_group': str(vitd_group),
                    'n': len(sub2),
                    'events_mi_or_stroke': int(sub2['event_mi_or_stroke'].sum()),
                    'median_vitd': float(sub2['vitd_main'].median()) if len(sub2) else np.nan,
                    'bmi_available': int(sub2['bmi_full'].notna().sum()),
                })
interaction_readiness = pd.DataFrame(readiness_rows)
save(interaction_readiness, '07_vdr_x_vitd_interaction_cell_counts_v9.csv')

pc_schema_hits = q(f'''
SELECT table_schema, table_name, column_name, data_type
FROM `{CDR.split('.')[0]}`.region-us.INFORMATION_SCHEMA.COLUMNS
WHERE table_schema IN ('{CDR.split('.')[1]}', '{PREP_CDR.split('.')[1]}')
  AND (
    LOWER(column_name) LIKE '%ancestry%'
    OR LOWER(column_name) LIKE 'pc%'
    OR LOWER(column_name) LIKE '%principal%component%'
    OR LOWER(column_name) LIKE '%related%'
    OR LOWER(table_name) LIKE '%kinship%'
    OR LOWER(table_name) LIKE '%related%'
  )
ORDER BY table_schema, table_name, column_name
''', 'pc_relatedness_schema_search')
save(pc_schema_hits, '08_pc_relatedness_schema_search_v9.csv')

weights_candidates = [DATA_DIR / 'vitd_gwas_instrument_weights.csv', DATA_DIR / '25ohd_gwas_instrument_weights.csv', DATA_DIR / 'mr_vitd_instrument_weights.csv']
weights_found = [str(p) for p in weights_candidates if p.exists()]
mr_rows = [
    {'requirement': 'VDR candidate carrier status', 'status': 'available_if_06_outputs_nonempty', 'evidence': '06_analysis_base_v9_genomics_carrier.csv; 06_vdr_candidate_carrier_counts_v9.csv'},
    {'requirement': 'Non-carrier denominator', 'status': 'available_for_wgs_or_array_subset', 'evidence': 'cb_search_person restricted to has_whole_genome_variant or has_array_data'},
    {'requirement': 'Additive genotype dosage / hard calls', 'status': 'not_available_from_carrier_table', 'evidence': 'cb_variant_to_person gives ALT-carrier membership, not 0/1/2 dosage'},
    {'requirement': 'Published 25(OH)D GWAS weights', 'status': 'found' if weights_found else 'missing', 'evidence': ';'.join(weights_found)},
    {'requirement': 'Genetic ancestry PCs', 'status': 'needs_review' if not pc_schema_hits.empty else 'not_found_in_bq_schema_search', 'evidence': '08_pc_relatedness_schema_search_v9.csv'},
    {'requirement': 'Relatedness / unrelated set', 'status': 'needs_review' if (not pc_schema_hits.empty and pc_schema_hits['table_name'].astype(str).str.lower().str.contains('related|kinship').any()) else 'not_found_in_bq_schema_search', 'evidence': '08_pc_relatedness_schema_search_v9.csv'},
    {'requirement': 'Formal MR readiness', 'status': 'not_currently_feasible', 'evidence': 'Missing dosage/hard-call route and GWAS weights/PCs unless supplied separately'},
]
save(pd.DataFrame(mr_rows), '08_mr_readiness_status_v9.csv')

flow_rows = []
def add_flow(step, n, note):
    flow_rows.append({'step': step, 'n': int(n) if pd.notna(n) else np.nan, 'note': note})
add_flow('Adult valid primary 25(OH)D index', len(base_all) if 'base_all' in globals() else 0, 'age >=18, vitamin D value 1-150 ng/mL-like')
add_flow('After baseline CVD exclusion and positive follow-up', len(base) if 'base' in globals() else 0, 'clinical Q3 CVD base')
add_flow('After WGS/array genomic denominator restriction', len(base_gen) if 'base_gen' in globals() else 0, 'main VDR x 25(OH)D candidate sample')
add_flow('Q2 CRP pair from clinical base', len(q2_pair) if 'q2_pair' in globals() else 0, 'pre-index CRP paired with vitamin D index')
add_flow('Q2 CRP pair after WGS/array restriction', len(q2_pair_gen) if 'q2_pair_gen' in globals() else 0, 'inflammation extension for VDR interaction')
save(pd.DataFrame(flow_rows), '09_sample_flow_v9.csv')

save(pd.DataFrame(query_status_rows), '99_bigquery_query_status.csv')
manifest = pd.DataFrame([{'output_file': p.name, 'path': str(p), 'size_bytes': p.stat().st_size} for p in sorted(OUTPUT_DIR.glob('*.csv'))])
save(manifest, '99_output_manifest.csv')
print('DONE')
print('All outputs are in:', OUTPUT_DIR)
print('Upload these first for review: 09_sample_flow_v9.csv, 06_genomics_denominator_summary_v9.csv, 06_vdr_candidate_carrier_counts_v9.csv, 07_vdr_x_vitd_interaction_cell_counts_v9.csv, 08_mr_readiness_status_v9.csv, 99_bigquery_query_status.csv')
