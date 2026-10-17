# Source companion of 01_smoking_alcohol_exploration.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# 覆盖率：队列里有多少人有“任何吸烟记录”、有多少有“任何饮酒记录”
cov_sql = f"""
WITH cohort AS (SELECT person_id FROM UNNEST(@ids) AS person_id)
SELECT
  COUNT(DISTINCT c.person_id) AS cohort_n,
  COUNT(DISTINCT CASE WHEN o.observation_concept_id IN (4282779,43054909) THEN o.person_id END) AS has_smoking,
  COUNT(DISTINCT CASE WHEN o.observation_concept_id IN (45766930,45772705) THEN o.person_id END) AS has_alcohol
FROM cohort c
LEFT JOIN `{os.environ["WORKSPACE_CDR"]}.observation` o
  ON c.person_id = o.person_id
 AND o.observation_concept_id IN (45766930,45772705,4282779,43054909)
"""
job2 = client.query(cov_sql, job_config=bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter("ids","INT64",ids)]))
print(job2.to_dataframe().to_string())

# %% Cell 4
import os, pandas as pd
from google.cloud import bigquery
client = bigquery.Client()

# 1) 全量拉 smoking status（不限队列，含日期）—— 数据量不大，能直接拉
smk_all_sql = f"""
SELECT o.person_id, o.observation_datetime,
  CASE
    WHEN o.value_as_concept_id IN (4144272,45879404,45883458,4222303,903653,903651) THEN 'never'
    WHEN o.value_as_concept_id IN (4310250) THEN 'former'
    WHEN o.value_as_concept_id IN (4276526,42709996,45881517,45884037,37395605,45881518,
          903657,45878118,762498,45884038,762499,903659,4298794,4034855,4219336,903654,4188539) THEN 'current'
    ELSE NULL END AS smk_status
FROM `{os.environ["WORKSPACE_CDR"]}.observation` o
WHERE o.observation_concept_id = 43054909
  AND o.value_as_concept_id IS NOT NULL
"""
smk_all = client.query(smk_all_sql).to_dataframe()
print("全 AoU smoking 记录行数:", len(smk_all))
print("有非空状态的记录:", smk_all['smk_status'].notna().sum())

# 2) 拉每人 index date（首次 25(OH)D）—— 也全量拉
idx_sql = f"""
SELECT person_id, MIN(measurement_datetime) AS index_date
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id IN (40791366, 37077414)
  AND value_as_number BETWEEN 1 AND 150
GROUP BY person_id
"""
idx = client.query(idx_sql).to_dataframe()
print("有 index date 的人:", len(idx))

# 3) 本地筛队列 + 取离 index 最近一次
cov = pd.read_csv('q1_covariate_table_v2.csv')[['person_id']]
s = (smk_all.dropna(subset=['smk_status'])
        .merge(cov, on='person_id')          # 只留你队列里的人
        .merge(idx, on='person_id'))         # 带上 index date
s['gap'] = (pd.to_datetime(s['observation_datetime']).dt.tz_localize(None)
            - pd.to_datetime(s['index_date']).dt.tz_localize(None)).abs()
s = s.sort_values('gap').groupby('person_id', as_index=False).first()  # 每人最近一次

print("\n队列内有可用吸烟状态的人数:", len(s))
print(s['smk_status'].value_counts())
s[['person_id','smk_status']].to_csv('q1_smoking_status.csv', index=False)
print("saved q1_smoking_status.csv")

# %% Cell 6
import pandas as pd

cov = pd.read_csv('q1_covariate_table_v2.csv')
idx = cov[['person_id','index_date']].copy()

s = (smk_all.dropna(subset=['smk_status'])
        .merge(idx, on='person_id')).copy()

# 两个日期都去掉时区再相减
obs = pd.to_datetime(s['observation_datetime'], utc=True).dt.tz_localize(None)
ix  = pd.to_datetime(s['index_date'], utc=True).dt.tz_localize(None)
s['gap'] = (obs - ix).abs()

s = s.sort_values('gap').groupby('person_id', as_index=False).first()

print("队列内有可用吸烟状态的人数:", len(s))
print(s['smk_status'].value_counts())
s[['person_id','smk_status']].to_csv('q1_smoking_status.csv', index=False)
print("saved q1_smoking_status.csv")
