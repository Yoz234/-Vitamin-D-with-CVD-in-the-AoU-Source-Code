# Source companion of 05_egfr_ckd_evidence_audit.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# ============================================================
# 给老师看的证据：eGFR 来源核对
# 跑完把整个输出截图，附在邮件里
# ============================================================
print("="*100)
print("  EVIDENCE 1  |  OMOP concept underlying our kidney function covariate")
print("="*100)
print(q(f"""
SELECT c.concept_id, c.concept_name, c.vocabulary_id,
       c.concept_code AS loinc_code, c.domain_id,
       COUNT(DISTINCT m.person_id) AS n_people,
       COUNT(*) AS n_rows
FROM `{CDR}.concept` c
LEFT JOIN `{CDR}.measurement` m
       ON m.measurement_concept_id = c.concept_id
WHERE c.concept_id = 3016723
GROUP BY 1,2,3,4,5
""").to_string(index=False))

print()
print("="*100)
print("  EVIDENCE 2  |  Reported units and value distribution")
print("             |  Values are consistent with serum creatinine (adult range ~0.6-1.2 mg/dL)")
print("="*100)
print(q(f"""
SELECT COALESCE(u.concept_name, 'NULL')                       AS reported_unit,
       COUNT(*)                                               AS n_rows,
       APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(5)]    AS p5,
       APPROX_QUANTILES(m.value_as_number, 2)[OFFSET(1)]      AS median_value,
       APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(95)]   AS p95
FROM `{CDR}.measurement` m
LEFT JOIN `{CDR}.concept` u ON m.unit_concept_id = u.concept_id
WHERE m.measurement_concept_id = 3016723
  AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1
ORDER BY n_rows DESC
LIMIT 6
""").to_string(index=False))

print()
print("="*100)
print("  Note: eGFR is not stored in All of Us. It is derived from this creatinine")
print("        measurement using the race-free CKD-EPI 2021 equation (creatinine, age, sex),")
print("        and expressed in mL/min/1.73 m2.")
print("="*100)

# %% Cell 4
# 看肌酐在高值区到底怎么分布，再决定上界
d = q(f"""
SELECT 
  CASE 
    WHEN value_as_number < 0.2  THEN '1. below 0.2'
    WHEN value_as_number <= 1.5 THEN '2. 0.2 to 1.5   normal'
    WHEN value_as_number <= 5   THEN '3. 1.5 to 5     impaired'
    WHEN value_as_number <= 15  THEN '4. 5 to 15      severe / dialysis'
    WHEN value_as_number <= 20  THEN '5. 15 to 20     extreme'
    WHEN value_as_number <= 100 THEN '6. 20 to 100    implausible'
    ELSE                             '7. above 100    sentinel / error'
  END AS creatinine_band,
  COUNT(*) AS n_rows,
  COUNT(DISTINCT person_id) AS n_people
FROM `{CDR}.measurement`
WHERE measurement_concept_id = 3016723
  AND value_as_number IS NOT NULL AND value_as_number > 0
GROUP BY 1 ORDER BY 1
""")
d['pct_of_rows'] = (d['n_rows'] / d['n_rows'].sum() * 100).round(3)
print("="*78)
print("  Serum creatinine (concept 3016723): distribution by value band")
print("  Retained range for analysis: 0.2 to 20 mg/dL  (bands 2 to 5)")
print("="*78)
print(d.to_string(index=False))

# %% Cell 6
# ============================================================
# eGFR 层面的合理性检查
# 比起过滤肌酐，直接看算出来的 eGFR 更直接
# ============================================================
egfr_chk = q(f"""
WITH cr AS (
  SELECT m.person_id,
         m.value_as_number AS scr,
         ROW_NUMBER() OVER (PARTITION BY m.person_id
                            ORDER BY m.measurement_datetime) AS rn
  FROM `{CDR}.measurement` m
  WHERE m.measurement_concept_id = 3016723
    AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
)
SELECT
  CASE WHEN scr BETWEEN 0.2 AND 20 THEN 'kept  (0.2-20)'
       ELSE 'excluded' END AS filter_status,
  COUNT(*) AS n_people,
  ROUND(MIN(scr),2) AS min_scr,
  ROUND(APPROX_QUANTILES(scr,2)[OFFSET(1)],2) AS median_scr,
  ROUND(MAX(scr),2) AS max_scr
FROM cr WHERE rn = 1
GROUP BY 1
""")
print(egfr_chk.to_string(index=False))
print()
print("★ 如果 excluded 那行的人数只有几百，说明之前的 eGFR 结果基本不受影响")
print("★ 如果有几千人，需要重跑 eGFR 那部分确认 HR 1.74")

# %% Cell 8

import os
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')

import pandas as pd, numpy as np
from google.cloud import bigquery

client = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]

def q(sql):
    df = client.query(sql).to_dataframe()
    for c in df.columns:
        if str(df[c].dtype).startswith(('Int','Float','boolean')):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

print("ready, CDR =", CDR)
base = pd.read_csv('analysis_base_clean.csv')
base['person_id'] = pd.to_numeric(base['person_id'], errors='coerce').astype('Int64')

cr_people = q(f"""
SELECT DISTINCT person_id
FROM `{CDR}.measurement`
WHERE measurement_concept_id = 3016723
  AND value_as_number BETWEEN 0.2 AND 20
""")
cr_people['person_id'] = pd.to_numeric(cr_people['person_id'], errors='coerce').astype('Int64')

n_cohort = len(base)
n_with_cr = base['person_id'].isin(set(cr_people['person_id'])).sum()
print(f"cohort: {n_cohort:,}")
print(f"有肌酐: {n_with_cr:,}")
print(f"覆盖率: {n_with_cr/n_cohort*100:.1f}%")

# %% Cell 10
from google.cloud import bigquery
import os
import pandas as pd

client = bigquery.Client()
dataset = os.environ["WORKSPACE_CDR"]

CREATININE_CONCEPT = 3016723

query = f"""
SELECT
  COUNT(*) AS n_rows_any,
  COUNT(DISTINCT person_id) AS n_people_any_record,

  COUNTIF(value_as_number IS NOT NULL) AS n_rows_numeric,
  COUNT(DISTINCT IF(value_as_number IS NOT NULL, person_id, NULL)) AS n_people_numeric,

  COUNTIF(value_as_number BETWEEN 0.2 AND 20) AS n_rows_in_0_2_to_20,
  COUNT(DISTINCT IF(value_as_number BETWEEN 0.2 AND 20, person_id, NULL)) AS n_people_with_any_value_in_0_2_to_20,

  COUNTIF(value_as_number IS NOT NULL AND (value_as_number < 0.2 OR value_as_number > 20)) AS n_rows_outside_0_2_to_20,
  COUNT(DISTINCT IF(value_as_number IS NOT NULL AND (value_as_number < 0.2 OR value_as_number > 20), person_id, NULL)) AS n_people_with_any_value_outside_0_2_to_20,

  COUNTIF(value_as_number IS NULL) AS n_rows_missing_numeric,
  COUNT(DISTINCT IF(value_as_number IS NULL, person_id, NULL)) AS n_people_with_any_missing_numeric

FROM `{dataset}.measurement`
WHERE measurement_concept_id = {CREATININE_CONCEPT}
"""

denom = client.query(query).to_dataframe()
denom
