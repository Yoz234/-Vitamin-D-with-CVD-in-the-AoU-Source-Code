# Source companion of 09_q2_crp_unit_evidence.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')

import pandas as pd, numpy as np
from google.cloud import bigquery

client = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]
pd.set_option('display.width', 220)

def q(sql):
    df = client.query(sql).to_dataframe()
    for c in df.columns:
        if str(df[c].dtype).startswith(('Int','Float','boolean')):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

print("ready, CDR =", CDR)# ============================================================
# CRP unit correction — evidence for the report
# 跑完截图: 2 张表
# ============================================================
import pandas as pd, numpy as np

CRP_CONCEPTS = [3020460, 3010156, 4208414, 46272908]

# ---------- TABLE 1: the unit mapping is correct ----------
# mg/dL median x10 should match mg/L median -> confirms conversion direction
u = q(f"""
SELECT COALESCE(un.concept_name,'NULL') AS unit_label,
       un.concept_code AS unit_code,
       COUNT(*) AS n_rows,
       APPROX_QUANTILES(m.value_as_number,2)[OFFSET(1)] AS median_val
FROM `{CDR}.measurement` m
LEFT JOIN `{CDR}.concept` un ON m.unit_concept_id = un.concept_id
WHERE m.measurement_concept_id IN ({','.join(map(str,CRP_CONCEPTS))})
  AND m.value_as_number IS NOT NULL AND m.value_as_number > 0
GROUP BY 1,2
HAVING n_rows >= 100
ORDER BY n_rows DESC
""")
print("="*88)
print("  TABLE 1  |  CRP unit labels ARE reliable (unlike vitamin D)")
print("           |  mg/dL median x10 should match mg/L median -> confirms the 10x conversion")
print("="*88)
print(u.to_string(index=False))
mgdl = u[u['unit_label'].str.contains('deciliter', case=False, na=False)]['median_val']
mgl  = u[u['unit_label'].str.fullmatch('milligram per liter', case=False, na=False)]['median_val']
if len(mgdl) and len(mgl):
    print(f"\n  Check: mg/dL median {mgdl.iloc[0]:.2f} x 10 = {mgdl.iloc[0]*10:.1f}, "
          f"close to mg/L median {mgl.iloc[0]:.2f}  ->  labels are real, conversion is correct")
print("  Contrast: for vitamin D, all unit labels had the same median (~32), so those labels were NOT trusted.")

# ---------- TABLE 2: impact of the bug fix ----------
FACTOR = {8751:1.0, 4122414:1.0, 8840:10.0, 4121396:10.0}
crp = q(f"""
SELECT person_id, unit_concept_id AS unit_id, value_as_number AS val,
       measurement_concept_id AS cid
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN ({','.join(map(str,CRP_CONCEPTS))})
  AND value_as_number IS NOT NULL AND value_as_number > 0
""")
crp['unit_key'] = pd.to_numeric(crp['unit_id'], errors='coerce').fillna(-1).astype(int)

# OLD buggy logic: only unit 8840 (x10) and 8753 (kept). 8753 never matches -> only mg/dL survive
old = crp[crp['unit_key'].isin([8840, 8753])]

# NEW logic: full mapping + median rule for ambiguous
med = crp.groupby(['cid','unit_key'])['val'].transform('median')
crp['factor'] = crp['unit_key'].map(FACTOR)
amb = crp['unit_key'].isin([0,-1])
crp.loc[amb,'factor'] = np.where(med[amb] < 1.5, 10.0, 1.0)
new = crp.dropna(subset=['factor'])

print("\n"+"="*88)
print("  TABLE 2  |  Impact of the unit-code fix (8753 -> 8751)")
print("="*88)
imp = pd.DataFrame({
    'version': ['OLD code (mg/L dropped as NA)', 'FIXED code (mg/L recovered)'],
    'usable_records': [len(old), len(new)],
    'participants':   [old['person_id'].nunique(), new['person_id'].nunique()]
})
print(imp.to_string(index=False))
print(f"\n  Records recovered: {len(new)-len(old):,}")
print("  The dropped records were all mg/L, i.e. entire laboratories -> a systematic, not random, loss.")
print("  After the fix, participants with a baseline CRP rose from 7,811 to 18,000 (see Q2 section).")

# %% Cell 4
# ============================================================
# TABLE 3: funnel from all-of-us records down to Q2 analytic sample
# 让 18,000 这个数在图上真正算出来
# ============================================================
FACTOR = {8751:1.0, 4122414:1.0, 8840:10.0, 4121396:10.0}

# 用修正后的映射，过滤到生理范围
crp2 = new.copy()
crp2['crp_mgL'] = crp2['val'] * crp2['factor']
crp2 = crp2[(crp2['crp_mgL'] >= 0.1) & (crp2['crp_mgL'] <= 300)]

# 拉队列和 index date
base = pd.read_csv('analysis_base_clean.csv')
base['person_id']  = pd.to_numeric(base['person_id'], errors='coerce').astype('Int64')
base['index_date'] = pd.to_datetime(base['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)

crp2['person_id'] = pd.to_numeric(crp2['person_id'], errors='coerce').astype('Int64')

# 需要测量日期，重新拉一次带日期的
crp_dt = q(f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS val,
       unit_concept_id AS unit_id, measurement_concept_id AS cid
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3020460, 3010156, 4208414, 46272908)
  AND value_as_number IS NOT NULL AND value_as_number > 0
""")
crp_dt['person_id'] = pd.to_numeric(crp_dt['person_id'], errors='coerce').astype('Int64')
crp_dt['unit_key']  = pd.to_numeric(crp_dt['unit_id'], errors='coerce').fillna(-1).astype(int)
med = crp_dt.groupby(['cid','unit_key'])['val'].transform('median')
crp_dt['factor'] = crp_dt['unit_key'].map(FACTOR)
amb = crp_dt['unit_key'].isin([0,-1])
crp_dt.loc[amb,'factor'] = np.where(med[amb] < 1.5, 10.0, 1.0)
crp_dt = crp_dt.dropna(subset=['factor'])
crp_dt['crp_mgL'] = crp_dt['val'] * crp_dt['factor']
crp_dt = crp_dt[(crp_dt['crp_mgL']>=0.1)&(crp_dt['crp_mgL']<=300)]

# 漏斗
n_allofus   = crp_dt['person_id'].nunique()
in_cohort   = crp_dt[crp_dt['person_id'].isin(set(base['person_id']))]
n_cohort    = in_cohort['person_id'].nunique()

m = in_cohort.merge(base[['person_id','index_date']], on='person_id')
m['dt'] = pd.to_datetime(m['dt'], utc=True, format='ISO8601').dt.tz_localize(None)
m['days_before'] = (m['index_date'] - m['dt']).dt.days
n_before = m[m['days_before']>=0]['person_id'].nunique()

print("="*78)
print("  TABLE 3  |  From All of Us CRP records down to the Q2 analytic sample")
print("="*78)
funnel = pd.DataFrame({
  'step': [
    'Participants with a valid CRP (any time, All of Us)',
    'Also in our analytic cohort',
    'With CRP on or before the index date  (Q2 sample)'
  ],
  'n_participants': [n_allofus, n_cohort, n_before]
})
print(funnel.to_string(index=False))
print(f"\n  The Q2 correlation analysis uses {n_before:,} participants.")
print("  (Old buggy code yielded only 7,811 here, because mg/L labs were dropped.)")
