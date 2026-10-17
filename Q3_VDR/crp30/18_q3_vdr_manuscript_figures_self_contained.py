# Source companion of 18_q3_vdr_manuscript_figures_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
import os
import sys
import json
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if sys.platform != 'linux' or not os.environ.get('WORKSPACE_CDR'):
    raise RuntimeError('Run figure generation on the AoU Researcher Workbench server.')
source = os.environ.get('Q3_VDR_CRP16_DIR', '').strip()
if not source:
    raise ValueError('Set Q3_VDR_CRP16_DIR to the exact new notebook 16 output directory.')
CRP16_DIR = Path(source)
config = pd.read_csv(CRP16_DIR / '16_crp_sensitivity_run_configuration.csv')
status = pd.read_csv(CRP16_DIR / '16_crp_sensitivity_status.csv')
run_id = str(config.iloc[0].run_id)
if len(config) != 1 or len(status) != 1 or str(status.iloc[0].run_id) != run_id:
    raise ValueError('Invalid/mixed-run configuration.')
primary_complete = str(status.iloc[0].primary_complete).lower() == 'true'
if not primary_complete:
    raise ValueError('Notebook 16 primary models are incomplete.')
source_manifest = pd.read_csv(CRP16_DIR / '99_crp_sensitivity_output_manifest.csv')
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read_result(name):
    p = CRP16_DIR / name
    m = source_manifest.loc[source_manifest.filename.eq(name)]
    if len(m) != 1 or sha256(p) != m.iloc[0].sha256:
        raise ValueError('Source manifest hash mismatch: ' + name)
    d = pd.read_csv(p)
    if 'run_id' in d and not d.run_id.astype(str).eq(run_id).all():
        raise ValueError('Mixed-run data: ' + name)
    return d
results = read_result('16_crp_sensitivity_model_comparison.csv')
dist = read_result('16_crp_distribution_extreme_value_summary.csv')
timing = read_result('16_crp_timing_relative_to_vitd_summary.csv')
fit_status = read_result('16_crp_sensitivity_model_fit_status.csv')
primary = results.loc[results.model_dataset.eq('crp_30day_primary')]
RSIDS = ['rs2228570','rs1544410','rs7975232','rs731236']
if set(primary.rsid) != set(RSIDS) or len(primary) != 4:
    raise ValueError('Expected all four 30-day primary SNPs.')
OUTPUT_DIR = Path(os.environ.get('Q3_VDR_FIGURE_OUTPUT_DIR', str(CRP16_DIR / 'figures_crp30')))
if OUTPUT_DIR.exists():
    raise FileExistsError('Use a fresh figure output directory: ' + str(OUTPUT_DIR))
OUTPUT_DIR.mkdir(parents=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
WRITTEN = []
figure_rows = []
LABELS = {'crp_30day_primary':'30-day primary', 'crp_30day_exclude_gt10':'30-day: CRP <=10',
 'crp_30day_exclude_ambiguous':'30-day: no ambiguous units','crp_30day_exclude_both':'30-day: both exclusions',
 'all_crp_model_base':'Unrestricted pre-index','exclude_crp_gt10':'Unrestricted: CRP <=10',
 'exclude_ambiguous_unit':'Unrestricted: no ambiguous units',
 'exclude_crp_gt10_and_ambiguous_unit':'Unrestricted: both exclusions',
 'same_day_or_365d_before':'0-365 days before','same_day_or_5y_before':'0-1825 days before'}
COLORS = ['#20746e','#bd5b29','#506cb3','#8d4280']
def save_figure(fig, stem, caption):
    for suffix in ['png','pdf']:
        p = OUTPUT_DIR / (stem + '.' + suffix)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        WRITTEN.append(p)
    figure_rows.append(dict(run_id=run_id,figure_stem=stem,caption=caption,source_dir=str(CRP16_DIR)))
    plt.show()
    plt.close(fig)
def forest(order, stem, title):
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(order)*1.35)))
    for j, rs in enumerate(RSIDS):
        d = results.loc[results.model_dataset.isin(order) & results.rsid.eq(rs)].set_index('model_dataset').reindex(order)
        y = np.arange(len(order)) + (j-1.5)*0.18
        good = d[['coef','ci_lower_coef','ci_upper_coef']].notna().all(axis=1).to_numpy()
        ax.errorbar(d.coef.to_numpy()[good], y[good], xerr=np.vstack([
            (d.coef-d.ci_lower_coef).to_numpy()[good], (d.ci_upper_coef-d.coef).to_numpy()[good]]),
            fmt='o', capsize=2, color=COLORS[j], label=rs)
    ax.axvline(0, color='#444444', linestyle='--', linewidth=1)
    ax.set_yticks(np.arange(len(order)), [LABELS[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel('Interaction beta: change in log-log slope per additional ALT allele (95% CI)')
    ax.set_title(title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.12), ncol=2, frameon=False)
    ax.margins(x=0.12)
    failed = fit_status.loc[fit_status.model_dataset.isin(order) & fit_status.status.ne('pass')]
    caption = title + '. 30-day means index day or previous 30 calendar days. Beta and t-based CI from classical OLS; zero is the null.'
    if len(failed):
        caption += ' Unavailable models: ' + '; '.join(failed.model_dataset + '/' + failed.rsid + ':' + failed.status)
    fig.tight_layout()
    save_figure(fig, stem, caption)

# %% Cell 3
forest(['crp_30day_primary','crp_30day_exclude_gt10','crp_30day_exclude_ambiguous','crp_30day_exclude_both'],
       'fig10_crp_30day_primary_and_exclusions', 'Q3 CRP extension: 30-day primary and exclusion sensitivities')
forest(['crp_30day_primary','all_crp_model_base','same_day_or_365d_before','same_day_or_5y_before',
        'exclude_crp_gt10','exclude_ambiguous_unit','exclude_crp_gt10_and_ambiguous_unit'],
       'figS_crp_legacy_window_sensitivity', 'Q3 CRP extension: retained wider-window sensitivities')
row = dist.loc[dist.model_dataset.eq('crp_30day_primary')].iloc[0]
t = timing.loc[timing.model_dataset.eq('crp_30day_primary') & timing.timing_group.isin(['same_day','1_30d_before'])]
t = t.set_index('timing_group').reindex(['same_day','1_30d_before'])
counts = [int(row.high_only),int(row.ambiguous_only),
          int(row.both_flags),int(row.neither_flag),int(row.unknown_unit_flag)]
if sum(counts) != int(row.n) or min(counts) < 0:
    raise ValueError('Exclusion flag categories do not partition the primary base.')
fig, axes = plt.subplots(1,2,figsize=(11,4.7))
axes[0].bar(['Same day','1-30 days before'],t.n,color=['#20746e','#506cb3'])
axes[0].set_title('30-day CRP timing'); axes[0].set_ylabel('Participants')
axes[1].bar(['High CRP\nonly','Ambiguous\nonly','Both','Neither','Unknown\nflag'],counts,
            color=['#bd5b29','#506cb3','#8d4280','#20746e','#888888'])
axes[1].set_title('Mutually exclusive flags'); axes[1].set_ylabel('Participants')
fig.tight_layout()
save_figure(fig,'figS_crp_30day_timing_and_flags',
            f'30-day base N={int(row.n)}. High CRP means >10 mg/L; overlapping flags counted once. Separate from model complete-case N.')
table_path = OUTPUT_DIR / '18_table6_crp_interactions_prioritized.csv'
results.sort_values(['priority','rsid']).to_csv(table_path,index=False)
WRITTEN.append(table_path)
manifest_path = OUTPUT_DIR / '18_q3_crp_figure_manifest.csv'
pd.DataFrame(figure_rows).to_csv(manifest_path,index=False)
WRITTEN.append(manifest_path)
caption_path = OUTPUT_DIR / '18_q3_crp_figure_captions.md'
caption_path.write_text('\n\n'.join(f"## {x['figure_stem']}\n\n{x['caption']}" for x in figure_rows),encoding='utf-8')
WRITTEN.append(caption_path)
status_path = OUTPUT_DIR / '18_q3_crp_figure_run_status.csv'
pd.DataFrame([dict(run_id=run_id,status='pass',source_dir=str(CRP16_DIR),figure_count=len(figure_rows),
                   source_status=status.iloc[0].status,primary_complete=True)]).to_csv(status_path,index=False)
WRITTEN.append(status_path)
checksums = OUTPUT_DIR / '18_q3_crp_artifact_checksums.csv'
pd.DataFrame([dict(filename=p.name,sha256=sha256(p),size_bytes=p.stat().st_size) for p in WRITTEN]).to_csv(checksums,index=False)
bundle = OUTPUT_DIR / '18_q3_crp30_figures_return_bundle.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in WRITTEN+[checksums]:
        z.write(p,arcname=p.name)
print('Return figure bundle:',bundle)
