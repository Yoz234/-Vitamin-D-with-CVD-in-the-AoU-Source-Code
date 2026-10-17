"""Package and statically validate server-only notebooks; never read AoU data."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'crp30'

CONFIG = r'''
import os
import sys
import json
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

if sys.platform != 'linux' or not os.environ.get('WORKSPACE_CDR'):
    raise RuntimeError('Run this notebook on the AoU Researcher Workbench server with WORKSPACE_CDR set.')
RUN_ID = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
DATA_DIR = Path(os.environ.get('Q3_VDR_DATA_DIR', '/home/dataproc/workspace/VDR'))
CRP_PATH = Path(os.environ.get('Q3_VDR_CRP_ADDITIVE_DATASET', str(
    DATA_DIR / 'Q3_VDR_tier1_additive_genotype_outputs_2026_08_08' / '05_crp_additive_genotype_model_dataset.csv')))
OUTPUT_DIR = Path(os.environ.get('Q3_VDR_CRP_SENS_OUTPUT_DIR', str(DATA_DIR / ('Q3_VDR_crp30_' + RUN_ID))))
if not CRP_PATH.is_file():
    raise FileNotFoundError(f'Set Q3_VDR_CRP_ADDITIVE_DATASET to the Q3 additive input: {CRP_PATH}')
if OUTPUT_DIR.exists():
    raise FileExistsError('Use a fresh output directory; do not mix runs: ' + str(OUTPUT_DIR))
OUTPUT_DIR.mkdir(parents=True)
RSIDS = ['rs731236', 'rs7975232', 'rs1544410', 'rs2228570']
WRITTEN = []
def write_csv(df, name):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    WRITTEN.append(path)
    return path
def write_text(value, name):
    path = OUTPUT_DIR / name
    path.write_text(value, encoding='utf-8')
    WRITTEN.append(path)
    return path
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
config = dict(run_id=RUN_ID, input_path=str(CRP_PATH), input_sha256=digest(CRP_PATH),
              workspace_cdr=os.environ['WORKSPACE_CDR'], window='calendar days before vitamin D index, inclusive 0..30',
              primary_family='four_crp30_snp_interactions', primary_family_size=4,
              formula='log_crp ~ log_vitd + ALT_dosage + ALT_dosage_x_log_vitd + age_at_index + female + winter + bmi_full',
              python=sys.version, numpy=np.__version__, pandas=pd.__version__, output_dir=str(OUTPUT_DIR),
              revision='30-day extension added 2026-10-17; wider windows retained as sensitivity')
write_csv(pd.DataFrame([config]), '16_crp_sensitivity_run_configuration.csv')
print('Server output:', OUTPUT_DIR)
'''

PREP = r'''
crp = pd.read_csv(CRP_PATH, dtype={'person_id': 'string'})
base_columns = ['person_id', 'index_date', 'crp_date', 'days_before', 'crp_mgL',
                'log_crp', 'log_vitd', 'age_at_index', 'female', 'winter', 'bmi_full', 'ambiguous_unit']
required = base_columns + [f'{rs}_alt_dosage' for rs in RSIDS]
missing = sorted(set(required) - set(crp.columns))
if missing:
    raise ValueError('Missing required fields: ' + ', '.join(missing))
if crp.person_id.isna().any() or crp.person_id.duplicated().any():
    raise ValueError('Input must have one nonmissing person_id per row.')
for col in required:
    if col not in ['person_id', 'index_date', 'crp_date']:
        original = crp[col]
        crp[col] = pd.to_numeric(original, errors='coerce')
        if (original.notna() & crp[col].isna()).any():
            raise ValueError('Non-numeric supplied value in ' + col)
for col in ['index_date', 'crp_date']:
    # Calendar dates, not elapsed 24-hour blocks.
    crp[col] = pd.to_datetime(crp[col].astype('string').str.slice(0, 10), format='%Y-%m-%d', errors='coerce')
delta = (crp.index_date - crp.crp_date).dt.days
if delta.isna().any() or not np.array_equal(delta.to_numpy(), crp.days_before.to_numpy()):
    raise ValueError('Dates missing or days_before differs from calendar-date subtraction.')
if (delta < 0).any():
    raise ValueError('Historical Q3 input should contain only same-day/pre-index CRP.')
for col, choices in [('female', [0, 1]), ('winter', [0, 1]), ('ambiguous_unit', [0, 1])]:
    if not crp[col].dropna().isin(choices).all():
        raise ValueError('Invalid binary coding in ' + col)
for rs in RSIDS:
    col = rs + '_alt_dosage'
    if not crp[col].dropna().isin([0, 1, 2]).all():
        raise ValueError('Dosage must be 0/1/2 or missing: ' + col)
    crp[rs + '_x_log_vitd_additive'] = crp[col] * crp.log_vitd
for col in ['log_crp', 'log_vitd', 'age_at_index', 'female', 'winter', 'bmi_full']:
    if np.isinf(crp[col].to_numpy(dtype=float)).any():
        raise ValueError('Infinite model value in ' + col)
if (crp.crp_mgL.isna() | ~np.isfinite(crp.crp_mgL) | (crp.crp_mgL <= 0)).any():
    raise ValueError('CRP must be positive and finite.')
if not np.allclose(np.log(crp.crp_mgL), crp.log_crp, rtol=0, atol=1e-8, equal_nan=False):
    raise ValueError('Saved log_crp differs from natural log of CRP.')
if 'vitd_main' in crp:
    vitd = pd.to_numeric(crp.vitd_main, errors='coerce')
    if (vitd.isna() | (vitd <= 0) | ~np.isfinite(vitd)).any() or not np.allclose(np.log(vitd), crp.log_vitd, rtol=0, atol=1e-8):
        raise ValueError('Invalid vitamin D or inconsistent natural log.')

# One specification table is consumed by Python and R.
rows = [
    ('crp_30day_primary', 30, 0, 0, 'primary_extension'),
    ('crp_30day_exclude_gt10', 30, 1, 0, 'within_30day_sensitivity'),
    ('crp_30day_exclude_ambiguous', 30, 0, 1, 'within_30day_sensitivity'),
    ('crp_30day_exclude_both', 30, 1, 1, 'within_30day_sensitivity'),
    ('all_crp_model_base', None, 0, 0, 'legacy_window_sensitivity'),
    ('exclude_crp_gt10', None, 1, 0, 'legacy_exclusion_sensitivity'),
    ('exclude_ambiguous_unit', None, 0, 1, 'legacy_exclusion_sensitivity'),
    ('exclude_crp_gt10_and_ambiguous_unit', None, 1, 1, 'legacy_exclusion_sensitivity'),
    ('same_day_or_365d_before', 365, 0, 0, 'legacy_window_sensitivity'),
    ('same_day_or_5y_before', 1825, 0, 0, 'legacy_window_sensitivity'),
]
spec = pd.DataFrame(rows, columns=['model_dataset', 'max_days', 'exclude_gt10', 'exclude_ambiguous', 'analysis_role'])
spec['priority'] = np.arange(1, len(spec) + 1)
spec['run_id'] = RUN_ID
def select_mask(data, row):
    mask = data.days_before.ge(0)
    if pd.notna(row.max_days):
        mask &= data.days_before.le(row.max_days)
    if row.exclude_gt10:
        mask &= data.crp_mgL.le(10)
    if row.exclude_ambiguous:
        mask &= data.ambiguous_unit.eq(0)
    return mask.fillna(False)
MASKS = {row.model_dataset: select_mask(crp, row) for row in spec.itertuples(index=False)}
spec['python_n'] = [int(MASKS[x].sum()) for x in spec.model_dataset]
SPEC_PATH = write_csv(spec, '16_crp_dataset_specifications.csv')
audits, timing, flow = [], [], []
for row in spec.itertuples(index=False):
    d = crp.loc[MASKS[row.model_dataset]]
    high, amb = d.crp_mgL.gt(10), d.ambiguous_unit.eq(1)
    audits.append(dict(run_id=RUN_ID, model_dataset=row.model_dataset, priority=row.priority,
        analysis_role=row.analysis_role, n=len(d), n_removed=len(crp)-len(d),
        crp_gt10_remaining=int(high.sum()), ambiguous_unit_remaining=int(amb.sum()),
        both_flags=int((high & amb).sum()), high_only=int((high & d.ambiguous_unit.eq(0)).sum()),
        ambiguous_only=int((~high & amb).sum()), neither_flag=int((~high & d.ambiguous_unit.eq(0)).sum()),
        unknown_unit_flag=int(d.ambiguous_unit.isna().sum()),
        median_crp_mgL=d.crp_mgL.median(), q25_crp_mgL=d.crp_mgL.quantile(.25),
        q75_crp_mgL=d.crp_mgL.quantile(.75), median_days_before=d.days_before.median(),
        min_days_before=d.days_before.min(), max_days_before=d.days_before.max()))
    for label, mask in [('same_day', d.days_before.eq(0)), ('1_30d_before', d.days_before.between(1,30)),
                        ('31_365d_before', d.days_before.between(31,365)), ('1_5y_before', d.days_before.between(366,1825)),
                        ('gt5y_before', d.days_before.gt(1825))]:
        timing.append(dict(run_id=RUN_ID, model_dataset=row.model_dataset, timing_group=label,
                           n=int(mask.sum()), denominator=len(d), percent=100*mask.sum()/len(d) if len(d) else np.nan))
    for rs in RSIDS:
        dose = rs + '_alt_dosage'
        columns = ['log_crp','log_vitd',dose,rs+'_x_log_vitd_additive','age_at_index','female','winter','bmi_full']
        cc = np.isfinite(d[columns].to_numpy(dtype=float)).all(axis=1)
        flow.append(dict(run_id=RUN_ID, model_dataset=row.model_dataset, rsid=rs, input_n=len(crp),
                         window_subset_n=len(d), dosage_called_n=int(d[dose].notna().sum()),
                         complete_case_n=int(cc.sum()), missing_covariate_or_genotype_n=int((~cc).sum())))
write_csv(pd.DataFrame(audits), '16_crp_sensitivity_dataset_counts.csv')
write_csv(pd.DataFrame(audits), '16_crp_distribution_extreme_value_summary.csv')
write_csv(pd.DataFrame(timing), '16_crp_timing_relative_to_vitd_summary.csv')
write_csv(pd.DataFrame(flow), '16_crp_sample_flow_all_datasets.csv')
write_csv(pd.DataFrame(flow).query("model_dataset == 'crp_30day_primary'"), '16_crp_30day_sample_flow.csv')
# Server-only model input. This participant-level file is never included in the return ZIP.
SERVER_INPUT = OUTPUT_DIR / '_server_only_crp_model_input.csv'
crp[required + [rs+'_x_log_vitd_additive' for rs in RSIDS]].to_csv(SERVER_INPUT, index=False)
'''

R_CODE = r'''
out <- Sys.getenv('Q3_CRP_OUT')
d <- read.csv(Sys.getenv('Q3_CRP_INPUT'), stringsAsFactors=FALSE, check.names=FALSE)
spec <- read.csv(Sys.getenv('Q3_CRP_SPEC'), stringsAsFactors=FALSE)
rsids <- strsplit(Sys.getenv('Q3_CRP_RSIDS'), ';', fixed=TRUE)[[1]]
run <- Sys.getenv('Q3_CRP_RUN')
statuses <- list(); terms <- list(); genotypes <- list()
for (i in seq_len(nrow(spec))) {
  s <- spec[i, ]
  mask <- is.finite(d$days_before) & d$days_before >= 0
  if (!is.na(s$max_days)) mask <- mask & d$days_before <= s$max_days
  if (s$exclude_gt10 == 1) mask <- mask & d$crp_mgL <= 10
  if (s$exclude_ambiguous == 1) mask <- mask & !is.na(d$ambiguous_unit) & d$ambiguous_unit == 0
  mask[is.na(mask)] <- FALSE
  subset <- d[mask, , drop=FALSE]
  if (nrow(subset) != s$python_n) stop('Python/R subset count mismatch: ', s$model_dataset)
  if (!is.na(s$max_days) && nrow(subset) && any(subset$days_before > s$max_days)) stop('Window violation')
  for (rs in rsids) {
    dose <- paste0(rs, '_alt_dosage'); ix <- paste0(rs, '_x_log_vitd_additive')
    subset[[ix]] <- subset[[dose]] * subset$log_vitd
    needed <- c('log_crp', 'log_vitd', dose, ix, 'age_at_index', 'female', 'winter', 'bmi_full')
    good <- complete.cases(subset[, needed, drop=FALSE]) & apply(subset[, needed, drop=FALSE], 1, function(x) all(is.finite(x)))
    dd <- subset[good, , drop=FALSE]
    state <- 'pass'; detail <- ''; fit <- NULL
    formula_text <- paste('log_crp ~', paste(c('log_vitd', dose, ix, 'age_at_index','female','winter','bmi_full'), collapse=' + '))
    if (nrow(dd) < 50) { state <- 'too_few_rows'; detail <- 'Less than 50 complete cases; operational guard only' }
    if (state == 'pass' && (length(unique(dd[[dose]])) < 2 || length(unique(dd$log_vitd)) < 2)) {
      state <- 'no_variation'; detail <- 'No dosage or exposure variation'
    }
    if (state == 'pass') {
      fit <- tryCatch(lm(as.formula(formula_text), data=dd), error=function(e) e)
      if (inherits(fit, 'error')) { state <- 'fit_error'; detail <- conditionMessage(fit); fit <- NULL }
    }
    if (state == 'pass') {
      if (fit$rank < ncol(model.matrix(fit)) || df.residual(fit) <= 0) {
        state <- 'rank_deficient'; detail <- 'Design matrix not full rank or no residual df'
      } else if (!(ix %in% rownames(summary(fit)$coefficients))) {
        state <- 'interaction_not_estimable'; detail <- 'Interaction absent from coefficient table'
      } else if (!all(is.finite(summary(fit)$coefficients[ix, ]))) {
        state <- 'invalid_estimate'; detail <- 'Non-finite interaction coefficient, SE or P'
      }
    }
    for (g in 0:2) genotypes[[length(genotypes)+1]] <- data.frame(run_id=run, model_dataset=s$model_dataset, rsid=rs,
      alt_dosage=g, subset_n=sum(subset[[dose]] == g, na.rm=TRUE), complete_case_n=sum(dd[[dose]] == g, na.rm=TRUE))
    statuses[[length(statuses)+1]] <- data.frame(run_id=run, model_dataset=s$model_dataset, priority=s$priority,
      analysis_role=s$analysis_role, rsid=rs, status=state, detail=detail, n=nrow(dd),
      subset_n=nrow(subset), python_subset_n=s$python_n, formula=formula_text, backend='R stats::lm')
    if (state == 'pass') {
      co <- summary(fit)$coefficients
      ci <- confint(fit, level=0.95)
      for (term in setdiff(rownames(co), '(Intercept)')) {
        b <- co[term, 'Estimate']; lo <- ci[term,1]; hi <- ci[term,2]
        terms[[length(terms)+1]] <- data.frame(run_id=run, model_dataset=s$model_dataset, priority=s$priority,
          analysis_role=s$analysis_role, rsid=rs, term=term, interaction=ix, dosage=dose, n=nrow(dd),
          coef=b, se=co[term,'Std. Error'], z_or_t=co[term,'t value'], p=co[term,'Pr(>|t|)'],
          ci_lower_coef=lo, ci_upper_coef=hi, exp_coef=exp(b), exp_ci_lower=exp(lo), exp_ci_upper=exp(hi),
          percent_change=(exp(b)-1)*100, doubling_ratio=exp(b*log(2)),
          doubling_ci_lower=exp(lo*log(2)), doubling_ci_upper=exp(hi*log(2)), residual_df=df.residual(fit),
          backend='R stats::lm')
      }
    }
  }
}
write.csv(do.call(rbind,statuses), file.path(out,'16_crp_sensitivity_model_fit_status.csv'), row.names=FALSE, na='')
template <- data.frame(run_id=character(),model_dataset=character(),priority=integer(),analysis_role=character(),rsid=character(),
 term=character(),interaction=character(),dosage=character(),n=integer(),coef=numeric(),se=numeric(),z_or_t=numeric(),p=numeric(),
 ci_lower_coef=numeric(),ci_upper_coef=numeric(),exp_coef=numeric(),exp_ci_lower=numeric(),exp_ci_upper=numeric(),
 percent_change=numeric(),doubling_ratio=numeric(),doubling_ci_lower=numeric(),doubling_ci_upper=numeric(),residual_df=numeric(),backend=character())
write.csv(if(length(terms)) do.call(rbind,terms) else template, file.path(out,'16_crp_sensitivity_all_term_results.csv'), row.names=FALSE, na='')
write.csv(do.call(rbind,genotypes), file.path(out,'16_crp_genotype_counts_by_dataset.csv'), row.names=FALSE, na='')
writeLines(c(capture.output(sessionInfo()), 'CI method: t-based confint.lm; classical OLS SE.'), file.path(out,'16_r_sessionInfo.txt'))
'''

RUN_R = r'''
rscript = os.environ.get('Q3_VDR_RSCRIPT') or shutil.which('Rscript')
if not rscript:
    raise RuntimeError('Rscript missing on server. Set Q3_VDR_RSCRIPT to its executable path.')
write_text(R_CODE, '16_crp_model_code.R')
env = os.environ.copy()
env.update(Q3_CRP_OUT=str(OUTPUT_DIR), Q3_CRP_INPUT=str(SERVER_INPUT), Q3_CRP_SPEC=str(SPEC_PATH),
           Q3_CRP_RSIDS=';'.join(RSIDS), Q3_CRP_RUN=RUN_ID)
result = subprocess.run([rscript, '-'], input=R_CODE, text=True, capture_output=True, env=env, timeout=7200)
write_text(result.stdout, '16_crp_sensitivity_r_stdout.txt')
write_text(result.stderr, '16_crp_sensitivity_r_stderr.txt')
write_csv(pd.DataFrame([dict(run_id=RUN_ID, returncode=result.returncode, rscript=rscript)]), '16_crp_sensitivity_r_run_status.csv')
if result.returncode:
    raise RuntimeError('R fitting failed; inspect 16_crp_sensitivity_r_stderr.txt before proceeding.')
for name in ['16_crp_sensitivity_model_fit_status.csv','16_crp_sensitivity_all_term_results.csv',
             '16_crp_genotype_counts_by_dataset.csv','16_r_sessionInfo.txt']:
    WRITTEN.append(OUTPUT_DIR / name)
'''

FINALIZE = r'''
fit_status = pd.read_csv(OUTPUT_DIR / '16_crp_sensitivity_model_fit_status.csv')
terms = pd.read_csv(OUTPUT_DIR / '16_crp_sensitivity_all_term_results.csv')
if len(fit_status) != 40 or fit_status.duplicated(['model_dataset','rsid']).any():
    raise ValueError('Expected exactly 40 unique dataset/SNP status rows.')
flow = pd.DataFrame(flow)
check = fit_status.merge(flow, on=['run_id','model_dataset','rsid'], validate='one_to_one')
if not (check.n == check.complete_case_n).all():
    raise ValueError('Python/R complete-case counts differ.')
interaction = terms.loc[terms.term == terms.interaction].copy()
primary_mask = interaction.model_dataset.eq('crp_30day_primary')
interaction['p_bonferroni_4'] = np.where(primary_mask, np.minimum(4 * interaction.p, 1), np.nan)
interaction['multiple_testing_family'] = np.where(primary_mask, 'four_crp30_snp_interactions', 'exploratory_nominal_sensitivity')
reference = interaction.loc[primary_mask, ['rsid','coef','p']].rename(columns={'coef':'base_coef','p':'base_p'})
interaction = interaction.merge(reference, on='rsid', how='left', validate='many_to_one')
interaction['reference_dataset'] = 'crp_30day_primary'
interaction['abs_coef_delta_vs_base'] = abs(interaction.coef - interaction.base_coef)
interaction = interaction.sort_values(['priority','rsid'])
primary = interaction.loc[interaction.model_dataset.eq('crp_30day_primary')].copy()
write_csv(interaction, '16_crp_sensitivity_model_comparison.csv')
write_csv(primary, '16_crp_30day_primary_interactions.csv')
write_csv(interaction.loc[interaction.priority.between(2,4)], '16_crp_30day_exclusion_interactions.csv')
write_csv(interaction.loc[interaction.priority.gt(4)], '16_crp_legacy_window_interactions.csv')
write_csv(interaction, '16_crp_table6_prioritized.csv')
write_csv(fit_status.groupby(['model_dataset','status'], sort=False).size().reset_index(name='count'), '16_crp_sensitivity_fit_status_counts.csv')
primary_complete = set(primary.rsid) == set(RSIDS) and len(primary) == 4
all_pass = fit_status.status.eq('pass').all()
run_status = dict(run_id=RUN_ID, status='pass' if primary_complete and all_pass else 'partial',
                  primary_complete=bool(primary_complete), expected_models=40, status_rows=len(fit_status),
                  successful_models=int(fit_status.status.eq('pass').sum()), primary_window='0..30 inclusive',
                  comparison_reference='crp_30day_primary')
write_csv(pd.DataFrame([run_status]), '16_crp_sensitivity_status.csv')
write_csv(pd.DataFrame([dict(run_id=RUN_ID,question='CRP mediation established',answer='no'),
                        dict(run_id=RUN_ID,question='Four primary models complete',answer=str(primary_complete)),
                        dict(run_id=RUN_ID,question='Old windows retained as sensitivity',answer='yes')]),
          '16_crp_sensitivity_decision_matrix.csv')
lines = ['# Q3 CRP extension: 30-day prioritized results', '', f'Run: {RUN_ID}',
         '', 'Primary window: index day or preceding 30 calendar days. All 10 datasets use the same Q3 server input.',
         'Primary family: four SNP interaction tests; Bonferroni P = min(4 * raw P, 1).',
         'Exclusion analyses and legacy windows are exploratory sensitivities. Cross-window differences also reflect participant selection.',
         'Classical OLS SE; t-based 95% CI. Beta is the difference in log-log slope per extra AoU ALT allele.',
         'For a doubling of 25(OH)D, the corresponding ratio of CRP fold associations is exp(beta * ln(2)).',
         'These analyses do not establish mediation, risk-allele direction or a causal pathway.',
         '', '## Primary estimates']
for row in primary.itertuples(index=False):
    lines.append(f'- {row.rsid}: N={row.n}; beta={row.coef:.5f}; CI [{row.ci_lower_coef:.5f}, {row.ci_upper_coef:.5f}]; raw P={row.p:.5g}; Bonferroni P={row.p_bonferroni_4:.5g}.')
lines.extend(['', '## Model status', fit_status[['model_dataset','rsid','status','n']].to_string(index=False),
              '', 'Only aggregate summaries, code and figure artifacts belong in the return package. Do not return _server_only_crp_model_input.csv.'])
write_text('\n'.join(lines), '16_crp_exclusion_decision_note.md')
manifest = pd.DataFrame([dict(run_id=RUN_ID,filename=p.name,size_bytes=p.stat().st_size,sha256=digest(p)) for p in WRITTEN])
manifest.to_csv(OUTPUT_DIR / '99_crp_sensitivity_output_manifest.csv', index=False)
bundle = OUTPUT_DIR / '16_q3_crp30_return_bundle.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in WRITTEN + [OUTPUT_DIR / '99_crp_sensitivity_output_manifest.csv']:
        z.write(p, arcname=p.name)
print('Return aggregate bundle:', bundle)
print(pd.DataFrame([run_status]).to_string(index=False))
print('For notebook 18 set Q3_VDR_CRP16_DIR =', OUTPUT_DIR)
if not primary_complete:
    raise RuntimeError('Primary result incomplete. Diagnostic bundle written; resolve model statuses before paper update.')
'''

FIG_CONFIG = r'''
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
'''

FIG_PLOTS = r'''
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
'''

def cell(kind, source):
    value = {'cell_type': kind, 'metadata': {}, 'source': source.strip().splitlines(keepends=True)}
    if value['source']:
        value['source'][-1] = value['source'][-1].rstrip('\n')
    if kind == 'code':
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value

def build():
    DEST.mkdir(exist_ok=True)
    notebooks = {
      '16_q3_vdr_crp_sensitivity_step4_self_contained.ipynb': [
        cell('markdown', '# Q3 CRP Extension: 30-Day Primary, Retained Window Sensitivities\n\nRun only on AoU server. Set Q3_VDR_CRP_ADDITIVE_DATASET if needed; input is the existing Q3 additive CRP file. Ten datasets, four SNPs each. No Hail rebuild. Participant-level server input is excluded from return bundles. Run all cells in order.'),
        cell('code',CONFIG),cell('code',PREP),cell('code','R_CODE = '+repr(R_CODE)),cell('code',RUN_R),cell('code',FINALIZE)],
      '18_q3_vdr_manuscript_figures_self_contained.ipynb': [
        cell('markdown','# Q3 CRP Figures: 30-Day Results First\n\nRun only on AoU server after new notebook 16. Set Q3_VDR_CRP16_DIR to that exact run directory. Produces the Figure 10 source, retained-window supplementary plot, timing/flags audit and prioritized Table 6 CSV. Existing CVD/LD/genotype figures are retained separately and not regenerated.'),
        cell('code',FIG_CONFIG),cell('code',FIG_PLOTS)]}
    for name, cells in notebooks.items():
        path = DEST / name
        if path.exists():
            raise FileExistsError('Archive old notebook before building: '+str(path))
        notebook = dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'q3_crp_revision':'2026-10-17'},nbformat=4,nbformat_minor=4)
        path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        reloaded=json.loads(path.read_text(encoding='utf-8'))
        for entry in reloaded['cells']:
            if entry['cell_type']=='code': ast.parse(''.join(entry['source']))
        print('Packaged and AST validated:',path.name)

if __name__ == '__main__':
    build()
