# Source companion of 04_egfr_ckd_construction.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os, pandas as pd
from google.cloud import bigquery
client = bigquery.Client()

# 放宽匹配：creatinine 只要带 creatinine，eGFR 用多种写法
explore_sql = f"""
SELECT m.measurement_concept_id, c.concept_name,
       COUNT(*) AS n_rows, COUNT(DISTINCT m.person_id) AS n_people
FROM `{os.environ["WORKSPACE_CDR"]}.measurement` m
LEFT JOIN `{os.environ["WORKSPACE_CDR"]}.concept` c
  ON m.measurement_concept_id = c.concept_id
WHERE LOWER(c.concept_name) LIKE '%creatinine%'
   OR LOWER(c.concept_name) LIKE '%glomerular%'
   OR LOWER(c.concept_name) LIKE '%gfr%'
GROUP BY 1,2
ORDER BY n_people DESC
LIMIT 40
"""
df = client.query(explore_sql).to_dataframe()
print("=== creatinine / GFR 相关 concept（全 AoU，按人数排序）===")
print(df.to_string())

# %% Cell 4
import os, pandas as pd, numpy as np
from google.cloud import bigquery
client = bigquery.Client()

cov = pd.read_csv('q1_covariate_table_v2.csv')[['person_id','index_date']]
cov['index_date'] = pd.to_datetime(cov['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)
cohort_ids = set(cov['person_id'])
N = len(cohort_ids)
print("队列总人数:", N)

# ===== 1) 血肌酐（Serum/Plasma, concept 3016723）：全量拉再本地筛 =====
cr_sql = f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS creat
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3016723           -- Creatinine, Serum/Plasma
  AND value_as_number BETWEEN 0.2 AND 20         -- 合理范围 mg/dL，剔异常
"""
cr = client.query(cr_sql).to_dataframe()
print("全 AoU 有合格血肌酐的人:", cr['person_id'].nunique())

# 只留队列内的人，带上 index date
cr = cr[cr['person_id'].isin(cohort_ids)].merge(cov, on='person_id', how='inner')
cr['dt'] = pd.to_datetime(cr['dt'], utc=True, format='ISO8601').dt.tz_localize(None)
cr['days_before_index'] = (cr['index_date'] - cr['dt']).dt.days   # 正=index前

# ===== 2) 覆盖率：队列里有多少人「任意时间」/「index前」/「index前1年内」有肌酐 =====
any_cr   = cr['person_id'].nunique()
before   = cr[cr['days_before_index'] >= 0]['person_id'].nunique()
within1y = cr[(cr['days_before_index'] >= 0) & (cr['days_before_index'] <= 365)]['person_id'].nunique()

print("\n=== 队列内血肌酐覆盖率 ===")
print(f"  任意时间有肌酐:     {any_cr:6d}  ({any_cr/N*100:.1f}%)")
print(f"  index 之前有肌酐:   {before:6d}  ({before/N*100:.1f}%)")
print(f"  index 前 1 年内:    {within1y:6d}  ({within1y/N*100:.1f}%)")

# ===== 3) CKD 诊断（condition 表）作对照 =====
ckd_ids_sql = f"""
SELECT DISTINCT co.person_id
FROM `{os.environ["WORKSPACE_CDR"]}.condition_occurrence` co
JOIN `{os.environ["WORKSPACE_CDR"]}.concept` c
  ON co.condition_concept_id = c.concept_id
WHERE LOWER(c.concept_name) LIKE '%chronic kidney%'
   OR LOWER(c.concept_name) LIKE '%end stage renal%'
   OR LOWER(c.concept_name) LIKE '%chronic renal%'
"""
ckd = client.query(ckd_ids_sql).to_dataframe()
ckd_in_cohort = len(set(ckd['person_id']) & cohort_ids)
print(f"\n=== 队列内有 CKD/慢性肾病诊断的人 ===")
print(f"  {ckd_in_cohort}  ({ckd_in_cohort/N*100:.1f}%)")

# %% Cell 6
import os, pandas as pd, numpy as np
from google.cloud import bigquery
from lifelines import CoxPHFitter
client = bigquery.Client()

# ========== 1) 读队列 + index date + 协变量 ==========
cov = pd.read_csv('q1_covariate_table_v2.csv')
cov['index_date'] = pd.to_datetime(cov['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)
cohort_ids = set(cov['person_id'])

# ========== 2) 全量拉血肌酐，本地筛队列，取 index 前最近一次 ==========
cr_sql = f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS creat
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3016723           -- Creatinine, Serum/Plasma (mg/dL)
  AND value_as_number BETWEEN 0.2 AND 20
"""
cr = client.query(cr_sql).to_dataframe()
cr = cr[cr['person_id'].isin(cohort_ids)].merge(cov[['person_id','index_date']], on='person_id')
cr['dt'] = pd.to_datetime(cr['dt'], utc=True, format='ISO8601').dt.tz_localize(None)
cr['days_before_index'] = (cr['index_date'] - cr['dt']).dt.days   # 正=index前

# 优先 index 前（含当天）；每人取离 index 最近的一次
before = cr[cr['days_before_index'] >= 0].copy()
before = before.sort_values('days_before_index').groupby('person_id', as_index=False).first()
print("有 baseline（index前）肌酐的人:", len(before), f"({len(before)/len(cohort_ids)*100:.1f}%)")

# ========== 3) CKD-EPI 2021 公式算 eGFR（不含种族，只需 肌酐+年龄+性别）==========
# eGFR = 142 × min(Scr/κ,1)^α × max(Scr/κ,1)^(-1.200) × 0.9938^age × (1.012 if female)
#   女: κ=0.7, α=-0.241 ; 男: κ=0.9, α=-0.302
b = before.merge(cov[['person_id','age_at_index','gender']], on='person_id')
b = b[b['gender'].isin(['Female','Male'])].copy()

def ckd_epi_2021(scr, age, female):
    k     = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    r = scr / k
    egfr = (142 * (min(r,1)**alpha) * (max(r,1)**-1.200)
            * (0.9938**age) * (1.012 if female else 1.0))
    return egfr

b['female_flag'] = (b['gender']=='Female')
b['egfr'] = b.apply(lambda r: ckd_epi_2021(r['creat'], r['age_at_index'], r['female_flag']), axis=1)
b['baseline_ckd'] = (b['egfr'] < 60).astype(int)   # 临床 CKD 切点

print("\n=== baseline eGFR 分布 ===")
print(b['egfr'].describe().round(1))
print(f"\nbaseline CKD (eGFR<60) 人数: {b['baseline_ckd'].sum()} "
      f"({b['baseline_ckd'].mean()*100:.1f}% of 有肌酐者)")

# 存一份，下次直接用
b[['person_id','egfr','baseline_ckd']].to_csv('q1_egfr_baseline.csv', index=False)
print("saved q1_egfr_baseline.csv")

# ========== 4) 加进 Cox 主模型重跑 ==========
fu = pd.read_csv('q1_followup_person_level_v2.csv')
d = fu[['person_id','follow_up_years','event_mi_or_stroke']].merge(cov, on='person_id', how='inner')
d = d.merge(b[['person_id','egfr','baseline_ckd']], on='person_id', how='left')   # 没肌酐的人 = NaN
d = d[d['gender'].isin(['Female','Male'])].copy()
d = d[d['follow_up_years'] > 0].copy()

d['vitd_deficient']    = (d['vitd_ng_ml'] < 10).astype(int)
d['vitd_insufficient'] = ((d['vitd_ng_ml'] >= 10) & (d['vitd_ng_ml'] < 20)).astype(int)
d['vitd_adequate']     = ((d['vitd_ng_ml'] >= 20) & (d['vitd_ng_ml'] < 30)).astype(int)
d['female']        = (d['gender'] == 'Female').astype(int)
d['race_black']    = (d['race_eth_simplified'] == 'Black or African American').astype(int)
d['race_hispanic'] = (d['race_eth_simplified'] == 'Hispanic or Latino').astype(int)
d['race_asian']    = (d['race_eth_simplified'] == 'Asian').astype(int)
d['race_other']    = (d['race_eth_simplified'].isin(['Other/Multiracial','Unknown/Other'])).astype(int)

base = ['follow_up_years','event_mi_or_stroke',
        'vitd_deficient','vitd_insufficient','vitd_adequate',
        'age_at_index','female','race_black','race_hispanic','race_asian','race_other',
        'baseline_diabetes','baseline_hypertension','baseline_hyperlipidemia']

def show(name, cph):
    print(f"\n### {name} | n={int(cph._n_examples)} events={int(cph.event_observed.sum())}")
    for v in ['vitd_deficient','vitd_insufficient','vitd_adequate']:
        r = cph.summary.loc[v]
        print(f"  {v}: HR {r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}), p={r['p']:.3g}")

# 模型 A：原主模型（参照，不含肾功能）
mA = d[base].dropna()
cA = CoxPHFitter().fit(mA,'follow_up_years','event_mi_or_stroke')
show('A: 原主模型（无肾功能）', cA)

# 模型 D1：+ 连续 eGFR（complete-case，只用有肌酐的人）
mD1 = d[base + ['egfr']].dropna()
cD1 = CoxPHFitter().fit(mD1,'follow_up_years','event_mi_or_stroke')
show('D1: + eGFR 连续', cD1)
r = cD1.summary.loc['egfr']
print(f"  >>> eGFR 自己: HR {r['exp(coef)']:.3f} (每+1单位), p={r['p']:.3g}")

# 模型 D2：+ baseline_ckd 二分类（complete-case）
mD2 = d[base + ['baseline_ckd']].dropna()
cD2 = CoxPHFitter().fit(mD2,'follow_up_years','event_mi_or_stroke')
show('D2: + baseline CKD (eGFR<60)', cD2)
r = cD2.summary.loc['baseline_ckd']
print(f"  >>> baseline_ckd 自己: HR {r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}), p={r['p']:.3g}")
