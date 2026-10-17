# Source companion of 03_bmi_audit_sensitivity.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os, pandas as pd, numpy as np
from google.cloud import bigquery
client = bigquery.Client()

# ---- 1) 读队列 + index date，并标出"恢复BMI"的那批人 ----
cov = pd.read_csv('q1_covariate_table_v3_bmi.csv')   # 含 bmi_source
idx = cov[['person_id','index_date']].copy()
idx['index_date'] = pd.to_datetime(idx['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)

recovered_ids = cov.loc[cov['bmi_source']=='recovered', 'person_id'].tolist()
print("需要审计的 recovered 人数:", len(recovered_ids), "(应为 36,312)")

# ---- 2) 全量拉身高/体重(含日期)，本地再筛 ----
ht_sql = f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS height_cm
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3036277        -- Body height
  AND unit_concept_id = 8582                   -- centimeter
  AND value_as_number BETWEEN 120 AND 220
"""
wt_sql = f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS weight_kg
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3025315        -- Body weight
  AND unit_concept_id = 9529                    -- kilogram
  AND value_as_number BETWEEN 30 AND 300
"""
ht = client.query(ht_sql).to_dataframe()
wt = client.query(wt_sql).to_dataframe()

# ---- 3) 只保留 recovered 那批人 ----
rec = set(recovered_ids)
ht = ht[ht['person_id'].isin(rec)].copy()
wt = wt[wt['person_id'].isin(rec)].copy()

# ---- 4) 关键修正:这次"不"限制 measurement<=index_date,
#         因为我们就是要检查到底有没有 index 之后测的(负值) ----
def nearest(df, valcol):
    d = df.merge(idx, on='person_id', how='inner')
    d['dt'] = pd.to_datetime(d['dt'], utc=True, format='ISO8601').dt.tz_localize(None)
    # 距离 index 的天数(正=index之前测,负=index之后测)
    d['days_before_index'] = (d['index_date'] - d['dt']).dt.days
    # 每人取"绝对距离最近"的一条(和你原来 nearest-to-index 的逻辑一致)
    d['absgap'] = d['days_before_index'].abs()
    d = d.sort_values('absgap').groupby('person_id', as_index=False).first()
    return d[['person_id','days_before_index', valcol]]

ht_n = nearest(ht, 'height_cm')
wt_n = nearest(wt, 'weight_kg')

# ---- 5) 身高体重合并,每人一个"恢复BMI的测量时间" ----
# BMI 用的是身高+体重,取两者里离 index 更远的那个作为该人的"测量时间"(保守)
m = ht_n.merge(wt_n, on='person_id', suffixes=('_ht','_wt'))
m['bmi_meas_days_before_index'] = m[['days_before_index_ht','days_before_index_wt']].min(axis=1)
# min = 更晚的测量(更接近index或在index之后),保守地暴露时间倒错风险

print("有完整身高+体重时间的 recovered 人数:", len(m))

# %% Cell 4
# ---- 6) 审计:三个关键检查 ----
d = m['bmi_meas_days_before_index']

print("\n===== 恢复 BMI 测量时间审计 =====")
print(f"可审计人数: {len(d)}")
print(f"\n距离 index 天数 (正=index前, 负=index后):")
print(d.describe().round(1))

n_neg   = (d < 0).sum()
n_zero  = (d == 0).sum()
n_pos   = (d > 0).sum()
print(f"\n负值(index之后测,有时间倒错): {n_neg}  ({n_neg/len(d)*100:.1f}%)")
print(f"零  (index当天)            : {n_zero}")
print(f"正值(index之前测)          : {n_pos}")

within_365 = (d.abs() <= 365).mean()*100
within_730 = (d.abs() <= 730).mean()*100
before_365 = ((d >= 0) & (d <= 365)).mean()*100
print(f"\n|距离| ≤ 365 天: {within_365:.1f}%")
print(f"|距离| ≤ 730 天: {within_730:.1f}%")
print(f"index 之前且 ≤365 天 (理想 baseline): {before_365:.1f}%")

# ---- 7) 存档 ----
out = m[['person_id','bmi_meas_days_before_index']]
out.to_csv('q1_recovered_bmi_timing_audit.csv', index=False)
b = os.environ.get("WORKSPACE_BUCKET")
if b:
    os.system(f"gsutil cp q1_recovered_bmi_timing_audit.csv {b}/data/")
print("\nsaved q1_recovered_bmi_timing_audit.csv")

# %% Cell 6
# ============================================================
# Model C-clean：只用"时间干净"的恢复 BMI 重做多重插补
# 时间干净 = index 之前 且 一年内 (0 <= days_before_index <= 365)
# ============================================================
import pandas as pd, numpy as np
from lifelines import CoxPHFitter, NelsonAalenFitter
import statsmodels.api as sm

# ---- 0) 从刚跑完的审计结果 m 里,挑出时间干净的 recovered 人 ----
clean_ids = set(
    m.loc[(m['bmi_meas_days_before_index'] >= 0) &
          (m['bmi_meas_days_before_index'] <= 365), 'person_id']
)
print("时间干净的 recovered BMI 人数:", len(clean_ids), f"({len(clean_ids)/len(m)*100:.1f}% of recovered)")

# ---- 1) 读数据,造 bmi_clean 列 ----
fu   = pd.read_csv('q1_followup_person_level_v2.csv')
cov2 = pd.read_csv('q1_covariate_table_v3_bmi.csv')   # 含 bmi_full, bmi_source

# bmi_clean: 原始BMI全留; 恢复BMI只留时间干净的; 其余=NaN(交给插补)
def make_clean(r):
    if r['bmi_source'] == 'original':
        return r['bmi_full']
    elif r['bmi_source'] == 'recovered':
        return r['bmi_full'] if r['person_id'] in clean_ids else np.nan
    else:
        return np.nan
cov2['bmi_clean'] = cov2.apply(make_clean, axis=1)
print("bmi_clean 覆盖率:", f"{cov2['bmi_clean'].notna().mean()*100:.1f}% (原 bmi_full 是 98.8%)")

# ---- 2) 合并 + 和主模型一致的过滤 ----
d = fu[['person_id','follow_up_years','event_mi_or_stroke']].merge(cov2, on='person_id', how='inner')
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

# ---- 3) 多重插补(和你原 Model C 完全一样的设置,只把 bmi_full 换成 bmi_clean)----
dat = d[base + ['bmi_clean']].copy().dropna(subset=base)
miss = dat['bmi_clean'].isna()
print("插补总人数:", len(dat), " BMI缺失(含被剔除的脏恢复值):", miss.sum())

naf = NelsonAalenFitter().fit(dat['follow_up_years'], dat['event_mi_or_stroke'])
H = naf.cumulative_hazard_.iloc[:,0]
dat['na_H0'] = dat['follow_up_years'].map(lambda t: H.loc[:t].iloc[-1] if (H.index <= t).any() else 0.0)

predictors = ['age_at_index','female','race_black','race_hispanic','race_asian','race_other',
              'baseline_diabetes','baseline_hypertension','baseline_hyperlipidemia',
              'vitd_deficient','vitd_insufficient','vitd_adequate',
              'event_mi_or_stroke','na_H0']

M = 20
np.random.seed(42)
obs = dat[dat['bmi_clean'].notna()]
X_obs = sm.add_constant(obs[predictors])
ols = sm.OLS(obs['bmi_clean'], X_obs).fit()
resid_sd = np.sqrt(ols.scale)
X_miss = sm.add_constant(dat.loc[miss, predictors], has_constant='add')
pred_mean = ols.predict(X_miss)

hr_store  = {v: [] for v in ['vitd_deficient','vitd_insufficient','vitd_adequate']}
var_store = {v: [] for v in hr_store}
for it in range(M):
    dd = dat.copy()
    imp = np.clip(pred_mean + np.random.normal(0, resid_sd, len(pred_mean)), 10, 70)
    dd.loc[miss, 'bmi_clean'] = imp.values
    cph = CoxPHFitter().fit(dd[base+['bmi_clean']], 'follow_up_years', 'event_mi_or_stroke')
    for v in hr_store:
        hr_store[v].append(cph.summary.loc[v,'coef'])
        var_store[v].append(cph.summary.loc[v,'se(coef)']**2)

print(f"\n### C-clean: 只用时间干净恢复BMI + 多重插补 (M={M}, n={len(dat)})")
for v in hr_store:
    qbar=np.mean(hr_store[v]); ubar=np.mean(var_store[v]); b=np.var(hr_store[v],ddof=1)
    T=ubar+(1+1/M)*b; se=np.sqrt(T)
    print(f"  {v}: HR {np.exp(qbar):.2f} ({np.exp(qbar-1.96*se):.2f}-{np.exp(qbar+1.96*se):.2f})")

# %% Cell 8
import pandas as pd
pd.DataFrame({
    'model': ['A_primary_noBMI','C_allrecovered_MI','C_clean_baselineBMI_MI'],
    'n': [89104, 89104, 89104],
    'deficient_HR': ['1.81 (1.58-2.08)','1.82 (1.59-2.09)','1.86 (1.62-2.14)'],
    'insufficient_HR': ['1.42 (1.31-1.54)','1.43 (1.31-1.55)','1.45 (1.33-1.58)'],
    'adequate_HR': ['1.06 (0.99-1.15)','1.07 (0.99-1.15)','1.08 (1.00-1.16)'],
}).to_csv('q1_bmi_sensitivity_summary.csv', index=False)
print("saved q1_bmi_sensitivity_summary.csv")
import os
os.system(f"gsutil cp q1_bmi_sensitivity_summary.csv {os.environ['WORKSPACE_BUCKET']}/data/")
