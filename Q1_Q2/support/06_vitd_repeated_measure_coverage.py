# Source companion of 06_vitd_repeated_measure_coverage.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
os.chdir('/home/jupyter/capstone/data')
import pandas as pd, numpy as np
from google.cloud import bigquery
client = bigquery.Client()
CDR = "YOUR_GOOGLE_PROJECT.C2024Q3R9"

cov = pd.read_csv('q1_covariate_table_v2.csv')[['person_id','index_date']]
cov['person_id'] = pd.to_numeric(cov['person_id'], errors='coerce').astype('Int64')
cov['index_date'] = pd.to_datetime(cov['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)
N = cov['person_id'].nunique()
print(f"分析队列人数 N = {N}")

# 维D concept 改成此 CDR 里真正有数据的：D3 + (D3+D2总) + D2
vitd = client.query(f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS val
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3020149, 40765040, 3049536)
  AND value_as_number BETWEEN 1 AND 150
""").to_dataframe()
print("拉到维D记录数:", len(vitd), " 不同person:", vitd['person_id'].nunique())

vitd['person_id'] = pd.to_numeric(vitd['person_id'], errors='coerce').astype('Int64')
vitd['val'] = pd.to_numeric(vitd['val'], errors='coerce')
vitd = vitd.dropna(subset=['val','person_id'])
vitd['dt'] = pd.to_datetime(vitd['dt'], utc=True, format='ISO8601').dt.tz_localize(None)

cohort_ids = set(cov['person_id'].dropna())
vitd = vitd[vitd['person_id'].isin(cohort_ids)]
print("匹配到队列内:", vitd['person_id'].nunique(), "人")

# 同一个人同一天可能同时测 D3 和 D2 → 按人+天去重，算“测量次数=不同日期数”
vitd['day'] = vitd['dt'].dt.date
uniq = vitd.drop_duplicates(['person_id','day'])
cnt = uniq.groupby('person_id').size()
n_any=cnt.shape[0]; n_ge2=int((cnt>=2).sum()); n_ge3=int((cnt>=3).sum())
print("\n===== 重复测量覆盖率（按不同测量日期计）=====")
print(f"至少 1 次维D:  {n_any:>7}  ({n_any/N*100:.1f}% of N)")
print(f"≥2 次(不同日): {n_ge2:>7}  ({n_ge2/N*100:.1f}% of N, {n_ge2/n_any*100:.1f}% of 有维D者)")
print(f"≥3 次:         {n_ge3:>7}  ({n_ge3/N*100:.1f}% of N)")
print(f"每人测量日数  中位={cnt.median():.0f}  均值={cnt.mean():.2f}  最多={cnt.max()}")

multi = uniq[uniq['person_id'].isin(cnt[cnt>=2].index)]
span = multi.groupby('person_id')['dt'].agg(lambda s:(s.max()-s.min()).days/365.25).dropna()
if len(span):
    print(f"\n≥2次者 首末间隔(年): 中位={span.median():.1f} 均值={span.mean():.1f} 跨度≥2年={int((span>=2).sum())} ({(span>=2).mean()*100:.1f}%)")

pct = n_ge2/N*100
print("\n"+"="*55)
if pct < 10:   print(f"  结论: ≥2次占 {pct:.1f}%，数据【不支持】时变分析。")
elif pct < 25: print(f"  结论: ≥2次占 {pct:.1f}%，偏少，时变【有限】。")
else:          print(f"  结论: ≥2次占 {pct:.1f}%，【可以考虑】时变分析。")
print("="*55)

# %% Cell 4
# ===== 配置（kernel 重启后必跑）=====
import os
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')

import pandas as pd, numpy as np
from google.cloud import bigquery
client = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]
print("✅ CDR =", CDR)
print("✅ 目录 =", os.getcwd())

# ===== 查重复维D测量的人数 =====
cov = pd.read_csv('q1_covariate_table_v2.csv')[['person_id']]
cohort_ids = set(cov['person_id'])

vd = client.query(f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS val
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3020149, 40765040, 3049536)
  AND value_as_number BETWEEN 1 AND 150
""").to_dataframe()

vd = vd[vd['person_id'].isin(cohort_ids)]
cnt = vd.groupby('person_id').size()

print("\n" + "="*50)
print("队列总人数:", len(cohort_ids))
print("有 >=1 次维D测量:", len(cnt))
print("有 >=2 次维D测量:", (cnt >= 2).sum(), f"({(cnt>=2).sum()/len(cohort_ids)*100:.1f}%)")
print("有 >=3 次:", (cnt >= 3).sum())
print(f"\n每人测量次数: median = {cnt.median():.0f}, mean = {cnt.mean():.1f}")
print("="*50)
