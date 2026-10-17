# Source companion of 08_vitd_concept_unit_audit.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')
from google.cloud import bigquery
client = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]

# ============================================================
# PANEL A: The concept set actually used as the Q1 exposure
# ============================================================
print("=" * 100)
print("PANEL A  |  CONCEPT SET USED FOR OUR Q1 EXPOSURE  (3 concepts, serum/plasma 25(OH)D only)")
print("=" * 100)
print(client.query(f"""
SELECT concept_id, concept_name, vocabulary_id, concept_code
FROM `{CDR}.concept`
WHERE concept_id IN (3020149, 40765040, 3049536)
""").to_dataframe().to_string(index=False))

# ============================================================
# PANEL B: Diagnostic query of ALL vitamin D concepts in All of Us
#          (for reference only; NOT our concept set)
# ============================================================
print("\n" + "=" * 100)
print("PANEL B  |  DIAGNOSTIC QUERY: ALL VITAMIN D RELATED CONCEPTS IN ALL OF US")
print("          FOR REFERENCE ONLY. THESE ARE NOT PART OF OUR STUDY EXPOSURE.")
print("          Ranked by number of participants.")
print("=" * 100)
print(client.query(f"""
SELECT m.measurement_concept_id AS concept_id,
       c.concept_name,
       COUNT(*) AS n_rows,
       COUNT(DISTINCT m.person_id) AS n_people
FROM `{CDR}.measurement` m
JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
WHERE LOWER(c.concept_name) LIKE '%vitamin d%'
   OR LOWER(c.concept_name) LIKE '%hydroxyvitamin%'
   OR LOWER(c.concept_name) LIKE '%calcitriol%'
   OR LOWER(c.concept_name) LIKE '%calcidiol%'
   OR LOWER(c.concept_name) LIKE '%25-oh%'
   OR LOWER(c.concept_name) LIKE '%1,25%'
   OR LOWER(c.concept_name) LIKE '%dihydroxy%'
GROUP BY 1, 2
ORDER BY n_people DESC
LIMIT 40
""").to_dataframe().to_string(index=False))

# %% Cell 4
import os
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')
import pandas as pd, numpy as np
from google.cloud import bigquery
client = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]

cov = pd.read_csv('q1_covariate_table_v2.csv')[['person_id','index_date','vitd_ng_ml']]
cov['person_id'] = pd.to_numeric(cov['person_id'], errors='coerce').astype('Int64')
cov['index_date'] = pd.to_datetime(cov['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)

raw = client.query(f"""
SELECT person_id, measurement_concept_id AS cid,
       DATE(measurement_datetime) AS day, value_as_number AS val
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3020149, 40765040, 3049536)
  AND value_as_number BETWEEN 0 AND 200
""").to_dataframe()
raw['person_id'] = pd.to_numeric(raw['person_id'], errors='coerce').astype('Int64')
raw['day'] = pd.to_datetime(raw['day'])

# 每个 concept 的取值分布 —— D2 应该明显偏低
print("=== 三个 concept 的取值分布 ===")
print(raw.groupby('cid')['val'].describe(percentiles=[.25,.5,.75]).round(1).to_string())
print("\n★ 如果 3049536 (D2) 的中位数只有个位数，就证实了它不能当 total 用")

# 关键：deficient 组的人，同一天有没有更高的 D3/total？
defi = cov[cov['vitd_ng_ml'] < 10][['person_id','index_date','vitd_ng_ml']]
print(f"\ndeficient 组 (<10 ng/mL) 人数: {len(defi)}")

m = raw.merge(defi, on='person_id')
m['gap'] = (m['day'] - m['index_date']).abs()
# 找到每人被选中那天的所有记录
best_day = m.loc[m.groupby('person_id')['gap'].idxmin()][['person_id','day']]
same_day = m.merge(best_day, on=['person_id','day'])
piv = same_day.pivot_table(index='person_id', columns='cid', values='val', aggfunc='max')
piv.columns = [{3020149:'d3', 40765040:'total', 3049536:'d2'}.get(c, c) for c in piv.columns]
for c in ['d3','total','d2']:
    if c not in piv.columns: piv[c] = np.nan

piv['true_best'] = piv[['total','d3']].max(axis=1)
contaminated = piv[(piv['true_best'] >= 10) & (piv['true_best'].notna())]
print("\n" + "="*60)
print(f"  deficient 组中，同一天存在 D3 或 total ≥ 10 的人: {len(contaminated)}")
print(f"  占 deficient 组的: {len(contaminated)/len(defi)*100:.1f}%")
print("="*60)
if len(contaminated) > 0:
    print("\n这些人的真实值分布:")
    print(contaminated['true_best'].describe().round(1).to_string())
    print("\n★ 这个数字如果上千，HR 1.81 必须重跑。")

# %% Cell 6
# ============================================================
# 维生素 D 单位审计
# ============================================================
VITD_25OH = [3020149, 40765040, 3049536]
VITD_125  = [40765038, 40765039, 4219068, 3014173]

vitd_units = q(f"""
SELECT m.measurement_concept_id           AS cid,
       c.concept_name                     AS test_name,
       m.unit_concept_id                  AS unit_id,
       COALESCE(u.concept_name, 'NULL')   AS unit_name,
       COUNT(*)                           AS n_rows,
       COUNT(DISTINCT m.person_id)        AS n_people,
       APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(5)]  AS p5,
       APPROX_QUANTILES(m.value_as_number, 2)[OFFSET(1)]    AS median_val,
       APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(95)] AS p95
FROM `{CDR}.measurement` m
JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
LEFT JOIN `{CDR}.concept` u ON m.unit_concept_id = u.concept_id
WHERE m.measurement_concept_id IN ({','.join(map(str, VITD_25OH + VITD_125))})
  AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1,2,3,4
ORDER BY cid, n_rows DESC
""")

print(vitd_units.to_string(index=False))

# %% Cell 8
# ============================================================
# Vitamin D concept set — evidence for the report / meeting
# 跑完截图: 3 张表 + 1 张图, 全英文
# ============================================================
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

VITD = {3020149:'25(OH)D3', 40765040:'Total (D3+D2)', 3049536:'25(OH)D2'}
PRIORITY = {40765040:0, 3020149:1, 3049536:2}
WCM_RED, WCM_NAVY = '#B31B1B', '#1f3a5f'

# ---------- TABLE 1: unit labels are unreliable ----------
u = q(f"""
SELECT m.measurement_concept_id AS cid,
       COALESCE(un.concept_name,'NULL') AS unit_label,
       COUNT(*) AS n_rows,
       APPROX_QUANTILES(m.value_as_number,2)[OFFSET(1)]   AS median_val,
       APPROX_QUANTILES(m.value_as_number,100)[OFFSET(95)] AS p95
FROM `{CDR}.measurement` m
LEFT JOIN `{CDR}.concept` un ON m.unit_concept_id = un.concept_id
WHERE m.measurement_concept_id IN (3020149, 40765040, 3049536)
  AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1,2
HAVING n_rows >= 50
ORDER BY cid, n_rows DESC
""")
u['concept'] = u['cid'].map(VITD)
print("="*92)
print("  TABLE 1  |  Unit labels differ but values do not — labels are unreliable")
print("           |  If pg/mL were taken literally it would convert to ~0.03 ng/mL (false 'deficient')")
print("="*92)
print(u[['concept','unit_label','n_rows','median_val','p95']].to_string(index=False))

# ---------- TABLE 2: all consistent with ng/mL ----------
s = q(f"""
SELECT measurement_concept_id AS cid,
       COUNT(*) AS n_rows,
       APPROX_QUANTILES(value_as_number,100)[OFFSET(5)]  AS p5,
       APPROX_QUANTILES(value_as_number,2)[OFFSET(1)]    AS median_val,
       APPROX_QUANTILES(value_as_number,100)[OFFSET(95)] AS p95
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3020149, 40765040, 3049536)
  AND value_as_number BETWEEN 1 AND 150
GROUP BY 1 ORDER BY cid
""")
s['concept'] = s['cid'].map(VITD)
print("\n"+"="*92)
print("  TABLE 2  |  Value distribution by concept (all consistent with ng/mL, no nmol/L mixed in)")
print("           |  ng/mL: median ~25-35, P95 ~60-80   |   nmol/L would be ~2.5x higher")
print("="*92)
print(s[['concept','n_rows','p5','median_val','p95']].to_string(index=False))
print("\n  Note: 25(OH)D2 median sits at 5.0 with P5=P25=5.0 -> detection-limit floor effect,")
print("        so D2 is used only as a tiebreaker, never as a standalone exposure.")
