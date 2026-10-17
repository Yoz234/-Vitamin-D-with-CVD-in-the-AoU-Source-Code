# Source companion of 07_q1_timevarying_vitd_cox.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import sys
get_ipython().system('{sys.executable} -m pip install lifelines --quiet')
import os
os.chdir('/home/jupyter/capstone/data')
import pandas as pd, numpy as np
from google.cloud import bigquery
from lifelines import CoxTimeVaryingFitter
client = bigquery.Client()
CDR = "YOUR_GOOGLE_PROJECT.C2024Q3R9"

# ========== 1) 人级基础表：和 Q1 主模型完全同一套（n≈89,104）==========
fu  = pd.read_csv('q1_followup_person_level_v2.csv')[['person_id','index_date','follow_up_years','event_mi_or_stroke']]
cov = pd.read_csv('q1_covariate_table_v2.csv')
base = fu.merge(cov, on='person_id', how='inner', suffixes=('','_c'))
base = base[base['gender'].isin(['Female','Male'])].copy()
base = base[base['follow_up_years'] > 0].copy()
base['index_date'] = pd.to_datetime(base['index_date'], utc=True, format='ISO8601').dt.tz_localize(None)
base['person_id'] = pd.to_numeric(base['person_id'], errors='coerce').astype('Int64')
# 固定协变量（和 Q1 一字不差）
base['female']        = (base['gender']=='Female').astype(int)
base['race_black']    = (base['race_eth_simplified']=='Black or African American').astype(int)
base['race_hispanic'] = (base['race_eth_simplified']=='Hispanic or Latino').astype(int)
base['race_asian']    = (base['race_eth_simplified']=='Asian').astype(int)
base['race_other']    = (base['race_eth_simplified'].isin(['Other/Multiracial','Unknown/Other'])).astype(int)
FIX = ['age_at_index','female','race_black','race_hispanic','race_asian','race_other',
       'baseline_diabetes','baseline_hypertension','baseline_hyperlipidemia']
base = base.dropna(subset=['vitd_ng_ml','follow_up_years','event_mi_or_stroke']+FIX)
base = base.rename(columns={'vitd_ng_ml':'vitd0','follow_up_years':'T','event_mi_or_stroke':'event'})
print(f"基础人群 n={len(base)}  事件={int(base['event'].sum())}  (应≈89,104 / 4,173)")
ids = set(base['person_id'].dropna())

# ========== 2) 拉所有重复维D测量（当前 CDR 正确 concept），按人-天取均值 ==========
vitd = client.query(f"""
SELECT person_id, measurement_datetime AS dt, value_as_number AS val
FROM `{CDR}.measurement`
WHERE measurement_concept_id IN (3020149, 40765040, 3049536)
  AND value_as_number BETWEEN 1 AND 150
""").to_dataframe()
vitd['person_id'] = pd.to_numeric(vitd['person_id'], errors='coerce').astype('Int64')
vitd['val'] = pd.to_numeric(vitd['val'], errors='coerce')
vitd = vitd.dropna(subset=['person_id','val'])
vitd = vitd[vitd['person_id'].isin(ids)]
vitd['dt'] = pd.to_datetime(vitd['dt'], utc=True, format='ISO8601').dt.tz_localize(None)
vitd['day'] = vitd['dt'].dt.floor('D')
vitd = vitd.groupby(['person_id','day'], as_index=False)['val'].mean()   # 同一天多次→均值
# 转成“距 index 的年数”
vitd = vitd.merge(base[['person_id','index_date','T']], on='person_id', how='inner')
vitd['t'] = (vitd['day'] - vitd['index_date']).dt.days / 365.25
# 只保留 index 之后、随访结束之前的重复测量
rep = vitd[(vitd['t'] > 0) & (vitd['t'] < vitd['T'])][['person_id','t','val']].copy()
print(f"用于更新的重复测量数: {len(rep)}  涉及 {rep['person_id'].nunique()} 人")

# ========== 3) 搭长表 (start, stop] ==========
# 基线行：start=0，vitd=基线值
b0 = base[['person_id','vitd0']].rename(columns={'vitd0':'vitd'}); b0['start']=0.0
# 重复行：start=t，vitd=当次值
r1 = rep.rename(columns={'t':'start','val':'vitd'})[['person_id','start','vitd']]
long = pd.concat([b0, r1], ignore_index=True)
long = long.merge(base[['person_id','T','event']+FIX], on='person_id', how='left')
long = long[long['start'] < long['T']].sort_values(['person_id','start']).reset_index(drop=True)
# stop = 同人下一段的 start，最后一段=T
long['stop'] = long.groupby('person_id')['start'].shift(-1)
long['stop'] = long['stop'].fillna(long['T'])
long['stop'] = np.minimum(long['stop'], long['T'])
long = long[long['stop'] > long['start']].copy()          # 去掉零长区间
# 事件只落在每人最后一段
long['is_last'] = ~long['person_id'].duplicated(keep='last')
long['ev'] = np.where(long['is_last'] & (long['event']==1), 1, 0)
# 时变维D dummy（参照 = sufficient ≥30）
long['vitd_deficient']    = (long['vitd']<10).astype(int)
long['vitd_insufficient'] = ((long['vitd']>=10)&(long['vitd']<20)).astype(int)
long['vitd_adequate']     = ((long['vitd']>=20)&(long['vitd']<30)).astype(int)
print(f"长表: {len(long)} 行, {long['person_id'].nunique()} 人, 事件 {int(long['ev'].sum())}")
print(f"平均每人区间数: {len(long)/long['person_id'].nunique():.2f}")

# ========== 4) 拟合时变 Cox ==========
TV = ['vitd_deficient','vitd_insufficient','vitd_adequate'] + FIX
fit_df = long[['person_id','start','stop','ev']+TV].dropna().reset_index(drop=True)
ctv = CoxTimeVaryingFitter(penalizer=0.0)
ctv.fit(fit_df, id_col='person_id', event_col='ev', start_col='start', stop_col='stop', show_progress=True)

# ========== 5) 结果 + 和 Q1 baseline 对比 ==========
s = ctv.summary
def row(name, q1):
    hr=s.loc[name,'exp(coef)']; lo=s.loc[name,'exp(coef) lower 95%']; hi=s.loc[name,'exp(coef) upper 95%']; p=s.loc[name,'p']
    print(f"  {name:20} 时变HR {hr:.2f} ({lo:.2f}-{hi:.2f}) p={p:.2e}   | Q1 baseline={q1}")
print("\n" + "="*66)
print("  时变维D → 心梗/中风  (exposure 随每次复测更新)")
print("="*66)
row('vitd_deficient',   '1.81')
row('vitd_insufficient','1.42')
row('vitd_adequate',    '1.06')
print("="*66)
s.to_csv('q1_timevarying_cox.csv')
print("saved q1_timevarying_cox.csv")
