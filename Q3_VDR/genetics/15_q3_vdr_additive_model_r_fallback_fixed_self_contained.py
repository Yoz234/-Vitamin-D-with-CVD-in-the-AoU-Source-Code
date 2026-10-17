# Source companion of 15_q3_vdr_additive_model_r_fallback_fixed_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# Cell 0. Configuration and input checks.
import os
import shutil
import subprocess
import traceback
from pathlib import Path

import pandas as pd

RUN_ID = os.environ.get('Q3_VDR_RUN_ID', pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ'))

def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        cwd,
        Path('/home/dataproc/workspace') / 'VDR',
        Path('/home/jupyter/workspace') / 'VDR',
        cwd.parent,
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            pass
    return cwd

DATA_DIR = Path(os.environ.get('Q3_VDR_DATA_DIR', str(infer_default_data_dir()))).expanduser()
OUTPUT_DIR = Path(os.environ.get('Q3_VDR_ADDITIVE_OUTPUT_DIR', str(DATA_DIR / 'Q3_VDR_tier1_additive_genotype_outputs_2026_08_08'))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIER1_RSIDS = ['rs731236', 'rs7975232', 'rs1544410', 'rs2228570']
CVD_PATH = OUTPUT_DIR / '04_cvd_additive_genotype_model_dataset.csv'
CRP_PATH = OUTPUT_DIR / '05_crp_additive_genotype_model_dataset.csv'

status_rows = []

def save(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f'Saved {filename}: {len(df):,} rows')
    return path

def save_text(text, filename):
    path = OUTPUT_DIR / filename
    path.write_text(str(text), encoding='utf-8')
    print(f'Saved {filename}: {len(str(text)):,} chars')
    return path

def add_status(step, status, detail=''):
    row = {'run_id': RUN_ID, 'step': step, 'status': status, 'detail': str(detail)[:4000]}
    status_rows.append(row)
    save(pd.DataFrame(status_rows), '15b_additive_r_fallback_fixed_status.csv')
    return row

input_status = pd.DataFrame([
    {'run_id': RUN_ID, 'label': 'CVD additive model dataset', 'path': str(CVD_PATH), 'exists': CVD_PATH.exists(), 'size_bytes': CVD_PATH.stat().st_size if CVD_PATH.exists() else 0},
    {'run_id': RUN_ID, 'label': 'CRP additive model dataset', 'path': str(CRP_PATH), 'exists': CRP_PATH.exists(), 'size_bytes': CRP_PATH.stat().st_size if CRP_PATH.exists() else 0},
])
save(input_status, '15b_additive_r_fallback_fixed_input_status.csv')
if not CVD_PATH.exists() or not CRP_PATH.exists():
    raise FileNotFoundError('Required 04/05 additive model datasets are missing in OUTPUT_DIR. Run notebook 15 through genotype join first.')

rscript_path = shutil.which('Rscript')
save(pd.DataFrame([{'run_id': RUN_ID, 'rscript_path': rscript_path or '', 'output_dir': str(OUTPUT_DIR)}]), '15b_additive_r_fallback_fixed_run_configuration.csv')
if not rscript_path:
    raise RuntimeError('Rscript not found. Use an AoU runtime with R available or install/enable R.')
add_status('configuration', 'pass', f'Rscript={rscript_path}; output_dir={OUTPUT_DIR}')

# %% Cell 3
# Cell 1. Run fixed R model fitting.
r_code = r'''
run_id <- Sys.getenv('Q3_VDR_RUN_ID')
out_dir <- Sys.getenv('Q3_VDR_R_FALLBACK_OUTPUT_DIR')
rsids <- strsplit(Sys.getenv('Q3_VDR_R_FALLBACK_TIER1'), ';', fixed=TRUE)[[1]]
min_n <- as.integer(Sys.getenv('Q3_VDR_R_MIN_N'))
if (is.na(min_n) || min_n < 1) min_n <- 50
min_events <- as.integer(Sys.getenv('Q3_VDR_R_MIN_EVENTS'))
if (is.na(min_events) || min_events < 1) min_events <- 10

read_opt <- function(name) {
  path <- file.path(out_dir, name)
  if (!file.exists(path)) return(data.frame())
  read.csv(path, stringsAsFactors=FALSE, check.names=FALSE)
}
cvd <- read_opt('04_cvd_additive_genotype_model_dataset.csv')
crp <- read_opt('05_crp_additive_genotype_model_dataset.csv')
status_rows <- list()
term_rows <- list()

add_status <- function(fam, role, model_name, rsid, status, detail, n=0, events=NA, backend='R') {
  status_rows[[length(status_rows)+1]] <<- data.frame(
    run_id=run_id, analysis_family=fam, analysis_role=role, model_name=model_name,
    rsid=rsid, status=status, detail=detail, n=n, events=events, backend=backend,
    stringsAsFactors=FALSE
  )
}

append_term <- function(row) term_rows[[length(term_rows)+1]] <<- row

bind_rows_fill <- function(rows) {
  if (!length(rows)) return(data.frame())
  all_names <- unique(unlist(lapply(rows, names)))
  aligned <- lapply(rows, function(x) {
    missing <- setdiff(all_names, names(x))
    for (m in missing) x[[m]] <- NA
    x[, all_names, drop=FALSE]
  })
  do.call(rbind, aligned)
}

complete_data <- function(d, needed, positive_followup=FALSE) {
  miss <- setdiff(needed, names(d))
  if (length(miss) > 0) return(list(data=data.frame(), status='missing_terms', detail=paste(miss, collapse=';')))
  x <- d[complete.cases(d[, needed, drop=FALSE]), , drop=FALSE]
  if (positive_followup) x <- x[as.numeric(x$follow_up_years) > 0, , drop=FALSE]
  if (nrow(x) == 0) return(list(data=x, status='no_complete_cases', detail=''))
  list(data=x, status='ready', detail='')
}

fit_cox <- function(d0, role, model_name, rsid, exposure, dosage, interaction, terms) {
  if (nrow(d0) == 0) { add_status('cvd_cox', role, model_name, rsid, 'dataset_missing', 'cvd'); return() }
  if (!requireNamespace('survival', quietly=TRUE)) { add_status('cvd_cox', role, model_name, rsid, 'package_missing', 'survival'); return() }
  cc <- complete_data(d0, c('follow_up_years', 'event_mi_or_stroke', terms), TRUE)
  ev <- if (nrow(cc$data) > 0) sum(as.numeric(cc$data$event_mi_or_stroke), na.rm=TRUE) else 0
  if (cc$status != 'ready' || nrow(cc$data) < min_n || ev < min_events) {
    add_status('cvd_cox', role, model_name, rsid, ifelse(cc$status == 'ready', 'too_few_rows_or_events', cc$status), cc$detail, nrow(cc$data), ev, 'R survival::coxph')
    return()
  }
  f <- as.formula(paste('survival::Surv(follow_up_years, event_mi_or_stroke) ~', paste(terms, collapse=' + ')))
  fit <- tryCatch(survival::coxph(f, data=cc$data, ties='efron', x=FALSE, model=FALSE), error=function(e) e)
  if (inherits(fit, 'error')) { add_status('cvd_cox', role, model_name, rsid, 'fit_error', conditionMessage(fit), nrow(cc$data), ev, 'R survival::coxph'); return() }
  co <- summary(fit)$coefficients
  for (term in rownames(co)) {
    coef <- as.numeric(co[term, 'coef'])
    se <- as.numeric(co[term, 'se(coef)'])
    lo <- coef - 1.96 * se
    hi <- coef + 1.96 * se
    append_term(data.frame(
      run_id=run_id, term=term, coef=coef, se=se, z_or_t=as.numeric(co[term, 'z']), p=as.numeric(co[term, 'Pr(>|z|)']),
      ci_lower_coef=lo, ci_upper_coef=hi, hr=exp(coef), hr_ci_lower=exp(lo), hr_ci_upper=exp(hi),
      exp_coef=NA_real_, percent_change=NA_real_, hr_ratio_or_exp_coef=exp(coef),
      analysis_family='cvd_cox', analysis_role=role, model_name=model_name, rsid=rsid, n=nrow(cc$data), events=ev,
      dosage=dosage, interaction=interaction, backend='R survival::coxph', stringsAsFactors=FALSE
    ))
  }
  add_status('cvd_cox', role, model_name, rsid, 'pass', '', nrow(cc$data), ev, 'R survival::coxph')
}

fit_lm <- function(d0, role, model_name, rsid, exposure, dosage, interaction, terms) {
  if (nrow(d0) == 0) { add_status('crp_ols', role, model_name, rsid, 'dataset_missing', 'crp'); return() }
  cc <- complete_data(d0, c('log_crp', terms), FALSE)
  if (cc$status != 'ready' || nrow(cc$data) < min_n) { add_status('crp_ols', role, model_name, rsid, ifelse(cc$status == 'ready', 'too_few_rows', cc$status), cc$detail, nrow(cc$data), NA, 'R stats::lm'); return() }
  f <- as.formula(paste('log_crp ~', paste(terms, collapse=' + ')))
  fit <- tryCatch(stats::lm(f, data=cc$data), error=function(e) e)
  if (inherits(fit, 'error')) { add_status('crp_ols', role, model_name, rsid, 'fit_error', conditionMessage(fit), nrow(cc$data), NA, 'R stats::lm'); return() }
  co <- summary(fit)$coefficients
  for (term in setdiff(rownames(co), '(Intercept)')) {
    coef <- as.numeric(co[term, 'Estimate'])
    se <- as.numeric(co[term, 'Std. Error'])
    lo <- coef - 1.96 * se
    hi <- coef + 1.96 * se
    append_term(data.frame(
      run_id=run_id, term=term, coef=coef, se=se, z_or_t=as.numeric(co[term, 't value']), p=as.numeric(co[term, 'Pr(>|t|)']),
      ci_lower_coef=lo, ci_upper_coef=hi, hr=NA_real_, hr_ci_lower=NA_real_, hr_ci_upper=NA_real_,
      exp_coef=exp(coef), percent_change=(exp(coef)-1)*100, hr_ratio_or_exp_coef=exp(coef),
      analysis_family='crp_ols', analysis_role=role, model_name=model_name, rsid=rsid, n=nrow(cc$data), events=NA_real_,
      dosage=dosage, interaction=interaction, backend='R stats::lm', stringsAsFactors=FALSE
    ))
  }
  add_status('crp_ols', role, model_name, rsid, 'pass', '', nrow(cc$data), NA, 'R stats::lm')
}

for (rsid in rsids) {
  dosage <- paste0(rsid, '_alt_dosage')
  ix_bin <- paste0(rsid, '_x_vitd_lt20_additive')
  ix_log <- paste0(rsid, '_x_log_vitd_additive')
  fit_cox(cvd, 'primary_additive', 'cvd_binary_additive_core_with_bmi', rsid, 'vitd_lt20', dosage, ix_bin, c('vitd_lt20', dosage, ix_bin, 'age_at_index', 'female', 'winter', 'bmi_full'))
  fit_cox(cvd, 'missingness_sensitivity', 'cvd_binary_additive_core_no_bmi', rsid, 'vitd_lt20', dosage, ix_bin, c('vitd_lt20', dosage, ix_bin, 'age_at_index', 'female', 'winter'))
  fit_cox(cvd, 'continuous_vitd_sensitivity', 'cvd_logvitd_additive_core_with_bmi', rsid, 'log_vitd', dosage, ix_log, c('log_vitd', dosage, ix_log, 'age_at_index', 'female', 'winter', 'bmi_full'))
  fit_lm(crp, 'crp_extension', 'crp_loglog_additive_core_with_bmi', rsid, 'log_vitd', dosage, ix_log, c('log_vitd', dosage, ix_log, 'age_at_index', 'female', 'winter', 'bmi_full'))
}

status_df <- bind_rows_fill(status_rows)
term_df <- bind_rows_fill(term_rows)
primary_df <- if (nrow(term_df)) term_df[term_df$analysis_family == 'cvd_cox' & term_df$analysis_role == 'primary_additive' & term_df$term == term_df$interaction, , drop=FALSE] else data.frame()
if (nrow(primary_df)) {
  primary_df$p_bonferroni_4 <- pmin(as.numeric(primary_df$p) * length(rsids), 1)
  primary_df$multiple_testing_family <- 'four Tier 1 VDR additive ALT-dosage primary interaction tests'
}

write.csv(status_df, file.path(out_dir, '12_additive_model_fit_status.csv'), row.names=FALSE, na='')
write.csv(term_df, file.path(out_dir, '13_additive_all_model_term_results.csv'), row.names=FALSE, na='')
write.csv(primary_df, file.path(out_dir, '14_additive_primary_interaction_results.csv'), row.names=FALSE, na='')
write.csv(data.frame(run_id=run_id, status_rows=nrow(status_df), term_rows=nrow(term_df), primary_rows=nrow(primary_df), stringsAsFactors=FALSE), file.path(out_dir, '15b_additive_r_fallback_fixed_summary.csv'), row.names=FALSE, na='')
'''

env = os.environ.copy()
env['Q3_VDR_RUN_ID'] = RUN_ID
env['Q3_VDR_R_FALLBACK_OUTPUT_DIR'] = str(OUTPUT_DIR)
env['Q3_VDR_R_FALLBACK_TIER1'] = ';'.join(TIER1_RSIDS)
env.setdefault('Q3_VDR_R_MIN_N', '50')
env.setdefault('Q3_VDR_R_MIN_EVENTS', '10')

r_run = subprocess.run([rscript_path, '-'], input=r_code, text=True, capture_output=True, env=env, timeout=7200)
save_text(r_run.stdout, '15b_additive_r_fallback_fixed_stdout.txt')
save_text(r_run.stderr, '15b_additive_r_fallback_fixed_stderr.txt')
save(pd.DataFrame([{'run_id': RUN_ID, 'r_returncode': r_run.returncode, 'rscript_path': rscript_path}]), '15b_additive_r_fallback_fixed_run_status.csv')
if r_run.returncode != 0:
    add_status('r_fallback_fixed', 'fail', r_run.stderr[:4000])
    raise RuntimeError('Fixed R fallback failed. See 15b_additive_r_fallback_fixed_stderr.txt')

add_status('r_fallback_fixed', 'pass', 'R model outputs written')
fit_status = pd.read_csv(OUTPUT_DIR / '12_additive_model_fit_status.csv')
term_results = pd.read_csv(OUTPUT_DIR / '13_additive_all_model_term_results.csv')
primary_ix = pd.read_csv(OUTPUT_DIR / '14_additive_primary_interaction_results.csv')
print('Fit status rows:', len(fit_status))
print('Term result rows:', len(term_results))
print('Primary interaction rows:', len(primary_ix))

# %% Cell 4
# Cell 2. Local post-run decision matrix and manifest.
fit_status_path = OUTPUT_DIR / '12_additive_model_fit_status.csv'
term_results_path = OUTPUT_DIR / '13_additive_all_model_term_results.csv'
primary_path = OUTPUT_DIR / '14_additive_primary_interaction_results.csv'

fit_status = pd.read_csv(fit_status_path) if fit_status_path.exists() else pd.DataFrame()
term_results = pd.read_csv(term_results_path) if term_results_path.exists() else pd.DataFrame()
primary_ix = pd.read_csv(primary_path) if primary_path.exists() else pd.DataFrame()
primary_pass_n = int(((fit_status.get('analysis_role', pd.Series(dtype=str)) == 'primary_additive') & (fit_status.get('status', pd.Series(dtype=str)) == 'pass')).sum()) if not fit_status.empty else 0

decision = pd.DataFrame([
    {'run_id': RUN_ID, 'question': 'Did fixed R fallback complete?', 'status': 'yes' if not fit_status.empty and not term_results.empty else 'no', 'evidence': '12_additive_model_fit_status.csv; 13_additive_all_model_term_results.csv'},
    {'run_id': RUN_ID, 'question': 'Did all four primary additive CVD Cox models fit?', 'status': 'yes' if primary_pass_n == len(TIER1_RSIDS) else 'no_or_partial', 'evidence': '12_additive_model_fit_status.csv'},
    {'run_id': RUN_ID, 'question': 'Are primary additive interaction rows available?', 'status': 'yes' if len(primary_ix) > 0 else 'no', 'evidence': '14_additive_primary_interaction_results.csv'},
    {'run_id': RUN_ID, 'question': 'Can ALT dosage be interpreted as functional/risk allele dosage?', 'status': 'no', 'evidence': 'Allele harmonization remains outside this notebook; ALT dosage is a coding label only.'},
])
save(decision, '15b_additive_r_fallback_fixed_decision_matrix.csv')

note = f'''# Q3 VDR Additive Genotype R Fallback Fixed Note

Run ID: `{RUN_ID}`

This rerun uses existing additive model datasets in `{OUTPUT_DIR}` and does not call Hail or rebuild genotype joins.

## Fix

The original R fallback failed because Cox and linear-model term result rows had different columns but were combined with base `do.call(rbind, term_rows)`. This fixed version aligns missing columns before binding rows.

## Status

- Fit status rows: {len(fit_status)}
- Term result rows: {len(term_results)}
- Primary additive Cox models passed: {primary_pass_n}/{len(TIER1_RSIDS)}
- Primary interaction rows: {len(primary_ix)}

Interpret ALT dosage as `per additional AoU ALT allele`; do not call it functional/risk dosage without separate allele harmonization.
'''
save_text(note, '16_additive_modeling_interpretation_note.md')

manifest_files = [
    '15b_additive_r_fallback_fixed_input_status.csv',
    '15b_additive_r_fallback_fixed_run_configuration.csv',
    '15b_additive_r_fallback_fixed_status.csv',
    '15b_additive_r_fallback_fixed_stdout.txt',
    '15b_additive_r_fallback_fixed_stderr.txt',
    '15b_additive_r_fallback_fixed_run_status.csv',
    '15b_additive_r_fallback_fixed_summary.csv',
    '12_additive_model_fit_status.csv',
    '13_additive_all_model_term_results.csv',
    '14_additive_primary_interaction_results.csv',
    '15b_additive_r_fallback_fixed_decision_matrix.csv',
    '16_additive_modeling_interpretation_note.md',
]
manifest = []
for filename in manifest_files:
    path = OUTPUT_DIR / filename
    if path.exists():
        manifest.append({'run_id': RUN_ID, 'filename': filename, 'path': str(path), 'size_bytes': path.stat().st_size})
save(pd.DataFrame(manifest), '99_additive_r_fallback_fixed_manifest.csv')
print('Done. Primary additive Cox models passed:', primary_pass_n, '/', len(TIER1_RSIDS))
