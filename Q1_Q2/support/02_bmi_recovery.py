# Source companion of 02_bmi_recovery.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
from google.cloud import bigquery
client = bigquery.Client()

# 身高、体重在 measurement 表里的常见 concept
# 先看有哪些、各多少人、单位是什么
explore_sql = f"""
SELECT m.measurement_concept_id, c.concept_name,
       mu.concept_name AS unit_name,
       COUNT(*) AS n_rows, COUNT(DISTINCT m.person_id) AS n_people
FROM `{os.environ["WORKSPACE_CDR"]}.measurement` m
LEFT JOIN `{os.environ["WORKSPACE_CDR"]}.concept` c
  ON m.measurement_concept_id = c.concept_id
LEFT JOIN `{os.environ["WORKSPACE_CDR"]}.concept` mu
  ON m.unit_concept_id = mu.concept_id
WHERE LOWER(c.concept_name) LIKE '%body height%'
   OR LOWER(c.concept_name) LIKE '%body weight%'
   OR LOWER(c.concept_name) LIKE '%body mass index%'
GROUP BY 1,2,3
ORDER BY n_people DESC
LIMIT 40
"""
print(client.query(explore_sql).to_dataframe().to_string())

# %% Cell 4
import os
import pandas as pd
from google.cloud import bigquery
client = bigquery.Client()

cov = pd.read_csv('q1_covariate_table_v2.csv')
missing_ids = cov.loc[cov['bmi_value'].isna(), 'person_id'].tolist()
print("缺 BMI 的人:", len(missing_ids))

# 全量拉身高(cm)和体重(kg)，本地再筛队列（避开 UNNEST 大列表的坑）
ht_sql = f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS height_cm
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3036277          -- Body height
  AND unit_concept_id = 8582                     -- centimeter
  AND value_as_number BETWEEN 120 AND 220        -- 合理身高范围(cm)
"""
wt_sql = f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS weight_kg
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3025315          -- Body weight
  AND unit_concept_id = 9529                      -- kilogram
  AND value_as_number BETWEEN 30 AND 300         -- 合理体重范围(kg)
"""
ht = client.query(ht_sql).to_dataframe()
wt = client.query(wt_sql).to_dataframe()
print("有合格身高记录的人(全AoU):", ht['person_id'].nunique())
print("有合格体重记录的人(全AoU):", wt['person_id'].nunique())

# 在【缺 BMI 的人】里，有多少同时有身高 + 体重
miss = set(missing_ids)
ht_ids = set(ht['person_id']) & miss
wt_ids = set(wt['person_id']) & miss
both   = ht_ids & wt_ids
print("\n=== 缺 BMI 的", len(miss), "人里 ===")
print("有身高的:", len(ht_ids))
print("有体重的:", len(wt_ids))
print("同时有身高+体重(可手算BMI):", len(both),
      f"({len(both)/len(miss)*100:.1f}%)")

# %% Cell 6
import pandas as pd
import numpy as np

# index date（每人一个）
idx = cov[['person_id','index_date']].copy()
idx['index_date'] = pd.to_datetime(idx['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)

def nearest_to_index(df, valcol):
    """每人取离 index date 最近的一次测量"""
    d = df.merge(idx, on='person_id', how='inner').copy()
    d['dt'] = pd.to_datetime(d['dt'], utc=True, format='ISO8601').dt.tz_localize(None)
    d['gap'] = (d['dt'] - d['index_date']).abs()
    d = d.sort_values('gap').groupby('person_id', as_index=False).first()
    return d[['person_id', valcol]]

# 只对【缺 BMI 的人】算
miss_ids = set(cov.loc[cov['bmi_value'].isna(), 'person_id'])
ht_m = nearest_to_index(ht[ht['person_id'].isin(miss_ids)], 'height_cm')
wt_m = nearest_to_index(wt[wt['person_id'].isin(miss_ids)], 'weight_kg')

# 合并身高体重，算 BMI
calc = ht_m.merge(wt_m, on='person_id', how='inner')
calc['bmi_calc'] = calc['weight_kg'] / (calc['height_cm']/100)**2

# 合理性过滤：BMI 10–70 之间才保留
before = len(calc)
calc = calc[(calc['bmi_calc'] >= 10) & (calc['bmi_calc'] <= 70)].copy()
print(f"算出 BMI 的人: {before}，合理范围内保留: {len(calc)}")
print(f"补充 BMI 的分布:\n{calc['bmi_calc'].describe().round(1)}")

calc[['person_id','bmi_calc']].to_csv('q1_bmi_recovered.csv', index=False)
print("\nsaved q1_bmi_recovered.csv")

n_orig = cov['bmi_value'].notna().sum()
n_total = len(cov)
n_new = len(calc)
print(f"\n原有 BMI: {n_orig} ({n_orig/n_total*100:.1f}%)")
print(f"补充后总计: {n_orig+n_new} ({(n_orig+n_new)/n_total*100:.1f}%)")
print(f"仍缺失: {n_total-n_orig-n_new} ({(n_total-n_orig-n_new)/n_total*100:.1f}%)")

# %% Cell 8
import pandas as pd

# 1) 合并：原有 BMI 优先，缺的用补算的 BMI 填上
cov2 = cov.copy()
rec = calc[['person_id','bmi_calc']].rename(columns={'bmi_calc':'bmi_recovered'})
cov2 = cov2.merge(rec, on='person_id', how='left')

# bmi_full = 有原值用原值，没有就用补算值
cov2['bmi_full'] = cov2['bmi_value'].fillna(cov2['bmi_recovered'])
cov2['bmi_source'] = cov2.apply(
    lambda r: 'original' if pd.notna(r['bmi_value'])
    else ('recovered' if pd.notna(r['bmi_recovered']) else 'still_missing'), axis=1)

print("BMI 来源分布:")
print(cov2['bmi_source'].value_counts())
print(f"\nbmi_full 覆盖率: {cov2['bmi_full'].notna().mean()*100:.1f}%")

# 存一份带补充 BMI 的协变量表，后面建模/插补都用它
cov2.to_csv('q1_covariate_table_v3_bmi.csv', index=False)
print("saved q1_covariate_table_v3_bmi.csv")
