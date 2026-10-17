# Source companion of 10_q1_subgroup_bonferroni_update.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
# WORKSPACE_CDR must be supplied by the authorized Workbench environment.
os.chdir('/home/jupyter/capstone/data')
import pandas as pd, numpy as np
from google.cloud import bigquery
client = bigquery.Client(); CDR = os.environ["WORKSPACE_CDR"]
def q(sql):
    df = client.query(sql).to_dataframe()
    for c in df.columns:
        if str(df[c].dtype).startswith(('Int','Float','boolean')):
            df[c]=pd.to_numeric(df[c],errors='coerce')
    return df
print("ready")# ============================================================
# Q1 亚组分析 + 交互检验 + 森林图
# 交互检验 penalizer=0（旧代码 0.01 使 p 值不可靠）
# 各亚组协变量排除分层变量本身（对齐 Xiao Fig 3）
# ============================================================
from lifelines import CoxPHFitter
from scipy import stats as st
import matplotlib.pyplot as plt
WCM_RED, WCM_NAVY = '#B31B1B', '#1f3a5f'

d = pd.read_csv('analysis_base_clean.csv')
d['person_id']=pd.to_numeric(d['person_id'],errors='coerce').astype('Int64')
d['vitd_neg10']=-pd.to_numeric(d['vitd_main'],errors='coerce')/10.0
RACE=['race_black','race_hispanic','race_asian','race_other']
FULL=['age_at_index','female']+RACE+['baseline_diabetes','baseline_hypertension','baseline_hyperlipidemia']
FULL=[c for c in FULL if c in d.columns]
d=d.dropna(subset=['follow_up_years','event_mi_or_stroke','vitd_neg10']+FULL)

# 分层变量
d['g_age']=np.where(d['age_at_index']>=65,'Age >=65','Age <65')
d['g_sex']=np.where(d['female']==1,'Female','Male')
d['g_dm']=np.where(d['baseline_diabetes']==1,'Diabetes: Yes','Diabetes: No')
d['g_htn']=np.where(d['baseline_hypertension']==1,'Hypertension: Yes','Hypertension: No')
d['g_hld']=np.where(d['baseline_hyperlipidemia']==1,'Hyperlipidemia: Yes','Hyperlipidemia: No')
d['g_race']=d['race_eth_simplified']

bmi=pd.read_csv('q1_covariate_table_v3_bmi.csv')[['person_id','bmi_full']]
bmi['person_id']=pd.to_numeric(bmi['person_id'],errors='coerce').astype('Int64')
d=d.merge(bmi,on='person_id',how='left')
d['bmi_full']=pd.to_numeric(d['bmi_full'],errors='coerce')
d['g_bmi']=pd.cut(d['bmi_full'],[0,25,30,200],labels=['BMI <25','BMI 25-30','BMI >=30']).astype(object)

# 收入
qc=q(f"""SELECT observation_concept_id AS cid, COUNT(DISTINCT person_id) n
         FROM `{CDR}.observation` o JOIN `{CDR}.concept` c ON o.observation_concept_id=c.concept_id
         WHERE LOWER(c.concept_name) LIKE '%income%' GROUP BY 1 ORDER BY n DESC LIMIT 1""")
if len(qc):
    inc=q(f"""SELECT o.person_id, c.concept_name ans FROM `{CDR}.observation` o
              JOIN `{CDR}.concept` c ON o.value_as_concept_id=c.concept_id
              WHERE o.observation_concept_id={int(qc.iloc[0]['cid'])}""")
    inc['person_id']=pd.to_numeric(inc['person_id'],errors='coerce').astype('Int64')
    def band(x):
        x=str(x).lower()
        if any(k in x for k in['less 10','10k 25','25k 35','less than 10']):return 'Income <35k'
        if any(k in x for k in['35k 50','50k 75','75k 100']):return 'Income 35-100k'
        if any(k in x for k in['100k 150','150k 200','more 200']):return 'Income >=100k'
        return None
    inc['g_inc']=inc['ans'].apply(band)
    inc=inc.dropna(subset=['g_inc']).drop_duplicates('person_id')
    d=d.merge(inc[['person_id','g_inc']],on='person_id',how='left')

EXCL={'g_age':['age_at_index'],'g_sex':['female'],'g_race':RACE,
      'g_dm':['baseline_diabetes'],'g_htn':['baseline_hypertension'],
      'g_hld':['baseline_hyperlipidemia'],'g_bmi':[],'g_inc':[]}
covs=lambda col:[c for c in FULL if c not in EXCL[col]]

def fit(df,adj):
    m=df[['follow_up_years','event_mi_or_stroke','vitd_neg10']+adj].dropna()
    m=m.loc[:,m.nunique()>1]
    c=CoxPHFitter().fit(m,'follow_up_years','event_mi_or_stroke')
    r=c.summary.loc['vitd_neg10']
    return r['exp(coef)'],r['exp(coef) lower 95%'],r['exp(coef) upper 95%'],r['p']

def interaction(df,col):
    dd=df.dropna(subset=[col]).copy()
    dum=pd.get_dummies(dd[col].astype(str),prefix='g',drop_first=True).astype(float)
    dd=pd.concat([dd.reset_index(drop=True),dum.reset_index(drop=True)],axis=1)
    dc=list(dum.columns);ix=[]
    for c_ in dc: dd['ix_'+c_]=dd['vitd_neg10']*dd[c_]; ix.append('ix_'+c_)
    adj=covs(col); base=['follow_up_years','event_mi_or_stroke','vitd_neg10']+adj+dc
    red=dd[base].dropna(); ful=dd[base+ix].dropna()
    cr=CoxPHFitter(penalizer=0.0).fit(red,'follow_up_years','event_mi_or_stroke')
    cf=CoxPHFitter(penalizer=0.0).fit(ful,'follow_up_years','event_mi_or_stroke')
    return st.chi2.sf(2*(cf.log_likelihood_-cr.log_likelihood_),len(ix))

hr_o,lo_o,hi_o,p_o=fit(d,FULL)
print(f"Overall HR per 10 ng/mL lower = {hr_o:.2f} ({lo_o:.2f}-{hi_o:.2f})\n")

SPECS=[('g_age','Age'),('g_sex','Sex'),('g_race','Race/ethnicity'),('g_bmi','BMI'),
       ('g_dm','Diabetes'),('g_htn','Hypertension'),('g_hld','Hyperlipidemia')]
if 'g_inc' in d.columns: SPECS.append(('g_inc','Income'))

rows,inter=[],{}
for col,lab in SPECS:
    for g in sorted(d[col].dropna().unique()):
        sub=d[d[col]==g]; ev=int(sub['event_mi_or_stroke'].sum())
        if len(sub)<300 or ev<25: continue
        hr,lo,hi,p=fit(sub,covs(col))
        rows.append({'subgroup':lab,'level':str(g),'n':len(sub),'events':ev,'hr':hr,'lo':lo,'hi':hi})
    inter[lab]=interaction(d,col)

res=pd.DataFrame(rows)
res.to_csv('q1_subgroups_all.csv',index=False)
pd.DataFrame([{'subgroup':k,'p_interaction':v} for k,v in inter.items()]).to_csv('q1_interaction_tests.csv',index=False)
print("=== Interaction p-values ===")
for k,v in inter.items(): print(f"  {k:16} p = {v:.3f}")

# ---------- 森林图 ----------
ORDER=['Age','Sex','Race/ethnicity','BMI','Diabetes','Hypertension','Hyperlipidemia','Income']
ORDER=[s for s in ORDER if s in set(res['subgroup'])]
plot=[('Overall',None,int(d['event_mi_or_stroke'].sum()),len(d),hr_o,lo_o,hi_o,True)]
for sg in ORDER:
    plot.append((sg,inter.get(sg),None,None,None,None,None,False))
    for _,r in res[res['subgroup']==sg].iterrows():
        plot.append(('   '+r['level'],None,int(r['events']),int(r['n']),r['hr'],r['lo'],r['hi'],False))

fig,ax=plt.subplots(figsize=(11,max(7,0.34*len(plot))))
y=np.arange(len(plot))[::-1]
for yy,(lab,pi,ev,n_,hr,lo,hi,isov) in zip(y,plot):
    if hr is None:
        ax.text(-0.62,yy,lab,transform=ax.get_yaxis_transform(),fontweight='bold',fontsize=9.5,va='center')
        if pi is not None:
            ax.text(1.34,yy,f'{pi:.3f}',transform=ax.get_yaxis_transform(),fontsize=9,va='center',ha='center',
                    fontweight='bold' if pi<0.05 else 'normal',color=WCM_RED if pi<0.05 else 'black')
        continue
    hic=min(hi,1.58)
    ax.plot([lo,hic],[yy,yy],color=WCM_NAVY,lw=1.7)
    if hi>1.58: ax.annotate('',xy=(1.6,yy),xytext=(hic,yy),arrowprops=dict(arrowstyle='->',color=WCM_NAVY,lw=1.7))
    ax.plot(hr,yy,'s' if isov else 'o',color=WCM_RED,ms=9 if isov else 6,zorder=3)
    ax.text(-0.62,yy,lab,transform=ax.get_yaxis_transform(),fontsize=9,va='center',fontweight='bold' if isov else 'normal')
    ax.text(-0.16,yy,f'{n_:,}',transform=ax.get_yaxis_transform(),fontsize=8.5,va='center',ha='right')
    ax.text(-0.04,yy,f'{ev:,}',transform=ax.get_yaxis_transform(),fontsize=8.5,va='center',ha='right')
    ax.text(1.04,yy,f'{hr:.2f} ({lo:.2f}-{hi:.2f})',transform=ax.get_yaxis_transform(),fontsize=8.5,va='center')
ax.axvline(1.0,color='grey',ls='--',lw=1)
ax.set_yticks([]); ax.set_ylim(-1,len(plot)); ax.set_xscale('log'); ax.set_xlim(0.92,1.6)
ax.set_xticks([0.95,1.0,1.1,1.2,1.4]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'{v:.2f}'))
ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.tick_params(axis='x',which='minor',length=0)
ax.set_xlabel('Hazard ratio per 10 ng/mL lower 25(OH)D (95% CI)',fontsize=11)
top=len(plot)-0.2
for xx,t,ha in [(-0.16,'N','right'),(-0.04,'Events','right'),(1.04,'HR (95% CI)','left'),(1.34,'P-int','center')]:
    ax.text(xx,top,t,transform=ax.get_yaxis_transform(),fontsize=9,ha=ha,fontweight='bold')
ax.set_title(f'Subgroup analysis: vitamin D and incident MI or stroke\n'
             f'n = {len(d):,}, {int(d["event_mi_or_stroke"].sum()):,} events',fontsize=12,pad=20)
ax.spines[['top','right','left']].set_visible(False)
plt.subplots_adjust(left=0.36,right=0.80)
plt.savefig('q1_forest_subgroups.png',dpi=170,bbox_inches='tight')
plt.show()
print("\nsaved q1_forest_subgroups.png")

# %% Cell 4
# Bonferroni 校正
alpha = 0.05
k = len(inter)
thresh = alpha / k
print(f"Bonferroni threshold: 0.05 / {k} = {thresh:.5f}\n")
for name, p in sorted(inter.items(), key=lambda x: x[1]):
    mark = "PASS" if p < thresh else "fail"
    print(f"  {name:16} p = {p:.4f}   {mark}")
