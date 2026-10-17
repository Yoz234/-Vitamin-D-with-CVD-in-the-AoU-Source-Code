# Source companion of 03_q1_q2_manuscript_tables.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
from pathlib import Path
import math
import re
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

try:
    from lifelines import CoxPHFitter
except Exception:
    CoxPHFitter = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

WORKSPACE_NAME = 'capstone'
RUN_ROOT = Path.cwd().resolve()

candidate_roots = []
for base in [RUN_ROOT] + list(RUN_ROOT.parents):
    if base.name == WORKSPACE_NAME:
        candidate_roots.append(base)
    candidate_roots.append(base / 'workspace' / WORKSPACE_NAME)
candidate_roots.append(Path('workspace') / WORKSPACE_NAME)

seen = set()
candidate_roots = [p for p in candidate_roots if not (str(p) in seen or seen.add(str(p)))]
MIGRATION_ROOT = next((p for p in candidate_roots if (p / 'data').exists()), candidate_roots[0])
DATA_DIR = MIGRATION_ROOT / 'data'
OUT_DIR = MIGRATION_ROOT / 'paperaddition' / 'q1_q2_manuscript_tables_latest'
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    warnings.warn(f'DATA_DIR does not exist yet: {DATA_DIR}')

DATA_FILES_INDEX = sorted(DATA_DIR.rglob('*.csv')) if DATA_DIR.exists() else []

def find_data_file(filename, preferred_parts=()):
    matches = [p for p in DATA_FILES_INDEX if p.name == filename]
    if not matches:
        return DATA_DIR / filename
    preferred = [part.lower() for part in preferred_parts]
    def score(path):
        s = str(path).lower()
        return sum(part in s for part in preferred)
    return sorted(matches, key=lambda p: (-score(p), len(str(p)), str(p)))[0]

Q1_FILES = {
    'q1_baseline': find_data_file('q1_covariate_table_v3_bmi.csv', ['q1']),
    'q1_cox': find_data_file('analysis_base_clean.csv', ['q1']),
    'q1_followup': find_data_file('q1_competing_risk_data.csv', ['q1']),
    'q1_vitd_long': find_data_file('vitd_long_clean.csv', ['q1']),
    'q1_vitd_baseline_versions': find_data_file('vitd_baseline_versions.csv', ['q1']),
    'q1_vitd_unit_audit': find_data_file('audit_vitd_units.csv', ['q1']),
    'q1_crp_unit_audit': find_data_file('audit_crp_units.csv', ['q1', 'crp']),
    'q1_crp_unit_map': find_data_file('audit_crp_unit_map.csv', ['q1', 'crp']),
    'q1_primary_cox_table': find_data_file('final_table2_primary_cox.csv', ['q1']),
    'q1_finegray': find_data_file('q1_finegray_shr_rcheck.csv', ['q1']),
    'q1_bmi_sensitivity': find_data_file('q1_bmi_sensitivity_summary.csv', ['q1']),
    'q1_bmi_audit': find_data_file('q1_bmi_audit.csv', ['q1', 'bmi']),
    'q1_bmi_timing_audit': find_data_file('q1_bmi_timing_audit_rcheck.csv', ['q1', 'bmi']),
    'q1_recovered_bmi_timing_audit': find_data_file('q1_recovered_bmi_timing_audit.csv', ['q1', 'bmi']),
    'q1_egfr_baseline': find_data_file('q1_egfr_baseline.csv', ['q1', 'egfr']),
    'q1_concept_sensitivity': find_data_file('q1_conceptset_sensitivity.csv', ['q1']),
    'q1_formal_concepts': find_data_file('formal_concept_sets.csv', ['q1']),
    'q1_finalized_concepts': find_data_file('finalized_concept_sets.csv', ['q1']),
    'q1_subgroups': find_data_file('q1_subgroups_all.csv', ['q1']),
    'q1_interactions': find_data_file('q1_interaction_tests.csv', ['q1']),
}

Q2_FILES = {
    'q2_loglog': find_data_file('q2_loglog_results.csv', ['q2', 'results']),
    'q2_models': find_data_file('q2_extended_model_loglog.csv', ['q2', 'results']),
    'q2_full_coefficients': find_data_file('q2_extended_model_full_coefficients.csv', ['q2', 'results']),
    'q2_dataset': find_data_file('q2_extended_model_dataset.csv', ['q2', 'results']),
    'q2_coverage': find_data_file('q2_covariate_coverage.csv', ['q2', 'results']),
    'q2_egfr_baseline': find_data_file('q2_egfr_baseline.csv', ['q2', 'egfr']),
}

Q2_MODEL_LABELS = {
    'A_core_current': 'Model A: age, sex, winter season, BMI',
    'B_extended_race_diabetes_SES': 'Model B: Model A + race/ethnicity, diabetes, SES',
    'C_extended_plus_vitd_supplement': 'Model C: Model B + recorded vitamin D supplement use',
}

print('Migration root:', MIGRATION_ROOT)
print('Data directory:', DATA_DIR)
print('Indexed CSV files:', len(DATA_FILES_INDEX))
if DATA_FILES_INDEX:
    print('\nCSV files indexed under DATA_DIR:')
    for path in DATA_FILES_INDEX:
        try:
            display_path = path.relative_to(DATA_DIR)
        except ValueError:
            display_path = path
        print(f'  {display_path}')
print('Output directory:', OUT_DIR)
print('\nResolved Q1 files:')
for key, path in Q1_FILES.items():
    print(f'  {key}: {path}')
print('\nResolved Q2 files:')
for key, path in Q2_FILES.items():
    print(f'  {key}: {path}')

# %% Cell 3
TABLES = {}
MANIFEST = []

def read_csv_required(path, **kwargs):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(path, **kwargs)

def read_csv_optional(path, **kwargs):
    path = Path(path)
    if not path.exists():
        print(f'Optional file not found: {path}')
        return None
    return pd.read_csv(path, **kwargs)

def fmt_int(x):
    if pd.isna(x):
        return ''
    return f'{int(round(float(x))):,}'

def fmt_num(x, digits=1):
    if pd.isna(x):
        return ''
    return f'{float(x):.{digits}f}'

def fmt_p(x):
    if pd.isna(x):
        return ''
    x = float(x)
    if x < 0.001:
        return '<0.001'
    return f'{x:.3f}'

def fmt_percent(x, digits=1):
    if pd.isna(x):
        return ''
    return f'{100 * float(x):.{digits}f}'

def n_pct(n, denom):
    if denom == 0 or pd.isna(denom):
        return ''
    return f'{int(n):,} ({100 * n / denom:.1f})'

def median_iqr(values, digits=1):
    x = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if x.empty:
        return ''
    q1, med, q3 = x.quantile([0.25, 0.50, 0.75])
    return f'{med:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})'

def range_cell(values, digits=1):
    x = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if x.empty:
        return ''
    return f'{x.min():.{digits}f}-{x.max():.{digits}f}'

def missing_cell(series):
    s = pd.Series(series)
    return n_pct(int(s.isna().sum()), len(s))

def clean_sheet_name(name):
    name = re.sub(r'[\\/*?:\[\]]', '_', name)
    return name[:31]

def df_to_markdown(df):
    cols = list(df.columns)
    lines = []
    lines.append('| ' + ' | '.join(map(str, cols)) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, row in df.iterrows():
        vals = [str(row[c]).replace('\n', '<br>').replace('|', '\\|') for c in cols]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'

def save_table(name, df, title=None, png_preview=True):
    df = df.copy()
    TABLES[name] = df
    csv_path = OUT_DIR / f'{name}.csv'
    md_path = OUT_DIR / f'{name}.md'
    html_path = OUT_DIR / f'{name}.html'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    md_title = f'# {title or name}\n\n' if title else ''
    md_path.write_text(md_title + df_to_markdown(df), encoding='utf-8')
    html_title = f'<h3>{title}</h3>' if title else ''
    html_path.write_text(html_title + df.to_html(index=False, escape=False), encoding='utf-8')
    png_path = None
    if png_preview and plt is not None and len(df) <= 80:
        png_path = OUT_DIR / f'{name}.png'
        fig_w = min(max(8, 1.5 * len(df.columns)), 18)
        fig_h = min(max(2.5, 0.35 * len(df) + 1.5), 30)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=11, loc='left', pad=8)
        tbl = ax.table(cellText=df.astype(str).values, colLabels=df.columns, loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.25)
        fig.tight_layout()
        fig.savefig(png_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
    MANIFEST.append({'name': name, 'rows': len(df), 'columns': len(df.columns), 'csv': str(csv_path), 'markdown': str(md_path), 'html': str(html_path), 'png_preview': str(png_path) if png_path else ''})
    print(f'Saved {name}: {len(df)} rows')
    return df

def write_excel_workbook(filename='q1_q2_manuscript_tables_latest.xlsx'):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        print('openpyxl is not installed; skipping Excel workbook export. CSV/Markdown/HTML outputs remain available.')
        return None
    xlsx_path = OUT_DIR / filename
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        used = set()
        for name, df in TABLES.items():
            sheet = clean_sheet_name(name)
            base = sheet
            i = 1
            while sheet in used:
                suffix = f'_{i}'
                sheet = (base[:31-len(suffix)] + suffix)
                i += 1
            used.add(sheet)
            df.to_excel(writer, sheet_name=sheet, index=False)
    print('Saved Excel workbook:', xlsx_path)
    return xlsx_path


def write_markdown_workbook(filename='q1_q2_manuscript_tables_latest.md'):
    md_path = OUT_DIR / filename
    lines = [
        '# Q1/Q2 Manuscript Tables - Latest Server Outputs',
        '',
        'This file aggregates the table outputs generated by this notebook in manuscript order. Q3/VDR tables are excluded.',
        '',
        f'Output directory: `{OUT_DIR}`',
        '',
        '## Table Index',
        '',
        '| Order | Table key | Rows | Columns | CSV | Individual MD |',
        '| ---: | --- | ---: | ---: | --- | --- |',
    ]
    for i, item in enumerate(MANIFEST, 1):
        lines.append(f"| {i} | `{item['name']}` | {item['rows']} | {item['columns']} | `{Path(item['csv']).name}` | `{Path(item['markdown']).name}` |")
    lines.append('')
    for i, (name, df) in enumerate(TABLES.items(), 1):
        lines.append(f'## {i}. {name}')
        lines.append('')
        match = next((m for m in MANIFEST if m['name'] == name), None)
        if match:
            lines.append(f"Source files: `{Path(match['csv']).name}`, `{Path(match['markdown']).name}`, `{Path(match['html']).name}`")
            lines.append('')
        lines.append(df_to_markdown(df))
        lines.append('')
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print('Saved Markdown workbook:', md_path)
    return md_path

# %% Cell 4
q1_baseline = read_csv_required(Q1_FILES['q1_baseline'])
q1_cox = read_csv_required(Q1_FILES['q1_cox'])
q1_followup = read_csv_required(Q1_FILES['q1_followup'])
q1_vitd_long = read_csv_required(Q1_FILES['q1_vitd_long'])

q2_dataset = read_csv_required(Q2_FILES['q2_dataset'])
q2_loglog = read_csv_required(Q2_FILES['q2_loglog'])
q2_models = read_csv_required(Q2_FILES['q2_models'])
q2_full = read_csv_optional(Q2_FILES['q2_full_coefficients'])
q2_coverage = read_csv_optional(Q2_FILES['q2_coverage'])

validation_rows = []
for label, df, key in [
    ('Q1 baseline covariates', q1_baseline, 'person_id'),
    ('Q1 Cox modeling cohort', q1_cox, 'person_id'),
    ('Q1 follow-up/competing-risk cohort', q1_followup, 'person_id'),
    ('Q1 clean long 25(OH)D records', q1_vitd_long, None),
    ('Q2 paired/model dataset', q2_dataset, 'person_id'),
]:
    duplicate_keys = ''
    if key is not None and key in df.columns:
        duplicate_keys = int(df[key].duplicated().sum())
    validation_rows.append({'source': label, 'rows': len(df), 'columns': len(df.columns), 'duplicate_person_id': duplicate_keys})

source_validation = pd.DataFrame(validation_rows)
save_table('00_source_validation_q1_q2', source_validation, 'Source validation for Q1/Q2 table redraw', png_preview=False)

resolved_rows = []
for family, mapping in [('Q1', Q1_FILES), ('Q2', Q2_FILES)]:
    for key, path in mapping.items():
        path = Path(path)
        resolved_rows.append({
            'Family': family,
            'Key': key,
            'Expected filename': path.name,
            'Resolved path': str(path),
            'Exists': path.exists(),
            'Required for core tables': key not in {'q1_primary_cox_table', 'q1_subgroups', 'q1_interactions', 'q2_full_coefficients'},
        })
resolved_inputs = pd.DataFrame(resolved_rows)
save_table('00_resolved_input_files_q1_q2', resolved_inputs, 'Resolved input files under server data folder', png_preview=False)

# %% Cell 5
# Comment-driven table and supplemental-analysis scope
scope_rows = [
    {'Comment ID': 'C108-C110', 'Manuscript location': 'Study design / Fig. 1', 'Notebook action': 'Prepare subcohort/timeline source checks only', 'Extra analysis?': 'No new model; figure-support counts only', 'Decision basis': 'Teacher requested timeline/subcohorts; C109 assigned to teammate for formal figure redraw.'},
    {'Comment ID': 'C115', 'Manuscript location': '25(OH)D measurement', 'Notebook action': 'Build concept-source/event/person-year audit table', 'Extra analysis?': 'Yes, descriptive aggregation from existing Q1 person-level outputs', 'Decision basis': 'Chinese plan asks for concept_id/name, clean records/persons, selected baseline persons, events, PY.'},
    {'Comment ID': 'C116', 'Manuscript location': '25(OH)D harmonization', 'Notebook action': 'Build unit/concept harmonization tables', 'Extra analysis?': 'Yes, descriptive harmonization aggregation only', 'Decision basis': 'Teacher requested n per harmonization issue; user marked data now available.'},
    {'Comment ID': 'C120', 'Manuscript location': 'Baseline 25(OH)D selection', 'Notebook action': 'Build repeated-measure and same-day audit table', 'Extra analysis?': 'Yes, descriptive audit from clean long 25(OH)D records', 'Decision basis': 'Teacher asked same-day and range; user decided supplement table/paragraph.'},
    {'Comment ID': 'C123', 'Manuscript location': 'Outcome concept set', 'Notebook action': 'Build full OMOP concept reproducibility table', 'Extra analysis?': 'No model; complete concept-set table', 'Decision basis': 'Teacher requested concept ID/name/detail for reproducibility.'},
    {'Comment ID': 'C124/C153', 'Manuscript location': 'Follow-up / Results', 'Notebook action': 'Build follow-up and event-timing table', 'Extra analysis?': 'Yes, descriptive event-time aggregation', 'Decision basis': 'Teacher asked time from index to event, follow-up, event rate; user planned final notebook-derived numbers.'},
    {'Comment ID': 'C126/C129', 'Manuscript location': 'CRP measurement', 'Notebook action': 'Build CRP unit/timing tables and 30-/90-day timing-window sensitivity', 'Extra analysis?': 'Yes, Q2 timing-restricted log-log refits when possible', 'Decision basis': 'Teacher asked CRP details and possible 30-/90-day analyses; user noted final code/meeting discrepancy must be handled.'},
    {'Comment ID': 'C139/C141', 'Manuscript location': 'BMI covariate', 'Notebook action': 'Build BMI range/source/timing audit tables', 'Extra analysis?': 'Yes, descriptive BMI audit only', 'Decision basis': 'Teacher asked low BMI plausibility and timing relative to index.'},
    {'Comment ID': 'C144', 'Manuscript location': 'eGFR covariate', 'Notebook action': 'Build eGFR availability/timing status table if files exist', 'Extra analysis?': 'Descriptive only; no new outcome model', 'Decision basis': 'Teacher asked timing/definition; user noted single eGFR cannot establish CKD persistence.'},
    {'Comment ID': 'C158/C159/C162', 'Manuscript location': 'Table 1 / baseline Results', 'Notebook action': 'Build Table 1 and missingness supplement', 'Extra analysis?': 'Yes, descriptive summaries and event rates by category', 'Decision basis': 'Teacher requested real table, missingness, event rate, follow-up; supplement acceptable.'},
    {'Comment ID': 'C163-C165', 'Manuscript location': 'Table 2 / Cox Results', 'Notebook action': 'Build primary Cox exposure table; full model supplement available for discussion', 'Extra analysis?': 'No new analysis; refit only if pre-exported primary Cox table absent', 'Decision basis': 'Teacher requested real table and asked whether full model should be included; user has not finalized inclusion.'},
    {'Comment ID': 'C166', 'Manuscript location': 'Fig. 5 subgroup Results', 'Notebook action': 'Build subgroup n/events/PY table and Bonferroni-threshold footnote support', 'Extra analysis?': 'Refit subgroup Cox/interactions only if exported subgroup files absent', 'Decision basis': 'Teacher requested subgroup n/events and Bonferroni threshold; subgroup analysis already in manuscript.'},
    {'Comment ID': 'C167', 'Manuscript location': 'Sensitivity Table 3', 'Notebook action': 'Build MI/stroke component event-count table only', 'Extra analysis?': 'No MI-only/stroke-only Cox unless later approved', 'Decision basis': 'User decided primary composite remains; component-specific modeling requires teacher/journal confirmation.'},
]
scope_table = pd.DataFrame(scope_rows)
save_table('00_comment_driven_table_analysis_scope', scope_table, 'Comment-driven table and supplemental-analysis scope', png_preview=False)

# %% Cell 6
vitd_unit_audit = read_csv_required(Q1_FILES['q1_vitd_unit_audit'])
included_vitd_cids = {3020149, 40765040, 3049536}
vitd_long_included = q1_vitd_long[q1_vitd_long['cid'].isin(included_vitd_cids)].copy()

concept_label = {
    3020149: '25(OH)D3',
    40765040: 'Total 25(OH)D from D3 plus D2',
    3049536: '25(OH)D2',
}
concept_counts = (
    vitd_long_included.groupby('cid')
    .agg(Records=('vitd_ng_ml', 'size'), Persons=('person_id', 'nunique'))
    .reset_index()
)
concept_counts['Concept'] = concept_counts['cid'].map(concept_label)
concept_counts = concept_counts[['Concept', 'cid', 'Records', 'Persons']].rename(columns={'cid': 'OMOP concept ID'})
concept_counts['Records'] = concept_counts['Records'].map(fmt_int)
concept_counts['Persons'] = concept_counts['Persons'].map(fmt_int)
final_row = pd.DataFrame([{'Concept': 'Final cleaned records', 'OMOP concept ID': 'NA', 'Records': fmt_int(len(vitd_long_included)), 'Persons': fmt_int(vitd_long_included['person_id'].nunique())}])
concept_counts = pd.concat([concept_counts, final_row], ignore_index=True)
save_table('q1_supp_vitd_clean_concept_counts', concept_counts, 'Supplementary table. Final clean 25(OH)D concept counts')

def classify_unit_issue(unit_name):
    u = str(unit_name).lower()
    if 'nanogram per milliliter' in u:
        return 'Recorded ng/mL'
    if 'microgram per liter' in u:
        return 'Microgram per L equivalent'
    if 'picogram' in u or 'pg/ml' in u:
        return 'pg/picogram labels with ng/mL-scale distributions'
    if u in ['nan', 'null', 'no matching concept', 'no value', 'none', '']:
        return 'Missing/no matching/no value unit labels'
    return 'Other rare unit labels reviewed during harmonization'

unit_detail = vitd_unit_audit[vitd_unit_audit['cid'].isin(included_vitd_cids)].copy()
unit_detail['Harmonization issue'] = unit_detail['unit_name'].map(classify_unit_issue)
unit_detail['Concept'] = unit_detail['cid'].map(concept_label)
unit_detail = unit_detail.rename(columns={'cid': 'OMOP concept ID', 'test_name': 'Concept name', 'unit_id': 'Unit concept ID', 'unit_name': 'Unit label', 'n_rows': 'Records', 'n_people': 'Persons', 'p5': 'P5', 'median_val': 'Median', 'p95': 'P95'})
unit_detail = unit_detail[['Concept', 'OMOP concept ID', 'Concept name', 'Unit concept ID', 'Unit label', 'Harmonization issue', 'Records', 'Persons', 'P5', 'Median', 'P95']]
save_table('q1_supp_vitd_unit_harmonization_detail', unit_detail, 'Supplementary table. 25(OH)D unit harmonization detail', png_preview=False)

unit_summary = (
    unit_detail.groupby('Harmonization issue', dropna=False)
    .agg(Records=('Records', 'sum'), Person_unit_count=('Persons', 'sum'), Unit_labels=('Unit label', lambda x: '; '.join(sorted(set(map(str, x))))))
    .reset_index()
)
unit_summary['Records'] = unit_summary['Records'].map(fmt_int)
unit_summary['Person-unit count'] = unit_summary['Person_unit_count'].map(fmt_int)
unit_summary = unit_summary.drop(columns=['Person_unit_count'])
save_table('q1_supp_vitd_unit_harmonization_summary', unit_summary, 'Supplementary table. 25(OH)D unit harmonization summary')

# %% Cell 7
# C115: 25(OH)D concept-source, baseline selection, events, and person-years audit
vitd_versions = read_csv_optional(Q1_FILES['q1_vitd_baseline_versions'])
concept_name_map = {
    3020149: '25(OH)D3',
    40765040: 'Total 25(OH)D from D3 plus D2',
    3049536: '25(OH)D2',
}

clean_concept = q1_vitd_long.copy()
clean_concept['cid_numeric'] = pd.to_numeric(clean_concept['cid'], errors='coerce')
clean_summary = (
    clean_concept[clean_concept['cid_numeric'].isin(concept_name_map)]
    .groupby('cid_numeric')
    .agg(clean_records=('vitd_ng_ml', 'size'), clean_persons=('person_id', 'nunique'))
    .reset_index()
    .rename(columns={'cid_numeric': 'OMOP concept ID'})
)
clean_summary['Concept'] = clean_summary['OMOP concept ID'].map(concept_name_map)

if vitd_versions is not None:
    source_col = 'src_main' if 'src_main' in vitd_versions.columns else 'src_old' if 'src_old' in vitd_versions.columns else None
else:
    source_col = None

if source_col is not None:
    src = vitd_versions[['person_id', source_col]].copy()
    src[source_col] = pd.to_numeric(src[source_col], errors='coerce')

    primary_source = (
        q1_baseline[['person_id', 'vitd_ng_ml']]
        .merge(src, on='person_id', how='left')
        .merge(q1_followup[['person_id', 'event_mi_or_stroke', 'follow_up_years']], on='person_id', how='left')
    )
    cox_source = (
        q1_cox[['person_id', 'vitd_main', 'event_mi_or_stroke', 'follow_up_years']]
        .merge(src, on='person_id', how='left')
    )

    def summarize_source(df, cohort_label):
        rows = []
        for cid, sub in df.dropna(subset=[source_col]).groupby(source_col):
            cid_int = int(cid)
            py = pd.to_numeric(sub.get('follow_up_years'), errors='coerce').sum() if 'follow_up_years' in sub.columns else np.nan
            events = pd.to_numeric(sub.get('event_mi_or_stroke'), errors='coerce').fillna(0).sum() if 'event_mi_or_stroke' in sub.columns else np.nan
            rows.append({
                'Cohort': cohort_label,
                'OMOP concept ID': cid_int,
                'Concept': concept_name_map.get(cid_int, str(cid_int)),
                'Baseline-selected persons': int(sub['person_id'].nunique()),
                'Events': int(events) if pd.notna(events) else np.nan,
                'Person-years': py,
            })
        return pd.DataFrame(rows)

    selected_summary = pd.concat([
        summarize_source(primary_source, 'Primary analytic cohort'),
        summarize_source(cox_source, 'Cox modeling cohort'),
    ], ignore_index=True)
    concept_event = selected_summary.merge(clean_summary, on=['OMOP concept ID', 'Concept'], how='outer')
    concept_event['Clean records'] = concept_event['clean_records'].map(fmt_int)
    concept_event['Clean persons'] = concept_event['clean_persons'].map(fmt_int)
    concept_event['Baseline-selected persons'] = concept_event['Baseline-selected persons'].map(fmt_int)
    concept_event['Events'] = concept_event['Events'].map(fmt_int)
    concept_event['Person-years'] = concept_event['Person-years'].map(fmt_int)
    concept_event = concept_event[['Cohort', 'Concept', 'OMOP concept ID', 'Clean records', 'Clean persons', 'Baseline-selected persons', 'Events', 'Person-years']]
else:
    concept_event = pd.DataFrame([{
        'Cohort': 'Not generated',
        'Concept': '',
        'OMOP concept ID': '',
        'Clean records': '',
        'Clean persons': '',
        'Baseline-selected persons': '',
        'Events': '',
        'Person-years': '',
        'Reason': 'vitd_baseline_versions.csv with src_old/src_main was not found under DATA_DIR.',
    }])

save_table('q1_supp_vitd_concept_source_events_c115', concept_event, 'Supplementary table. 25(OH)D concept-source, baseline selection, events, and person-years', png_preview=False)

# %% Cell 8
def repeated_measure_audit(cohort_person_ids, cohort_label, baseline_df):
    ids = pd.Series(cohort_person_ids).dropna().unique()
    long = q1_vitd_long[q1_vitd_long['person_id'].isin(ids)].copy()
    baseline = baseline_df[baseline_df['person_id'].isin(ids)].copy()
    baseline['index_day'] = pd.to_datetime(baseline['index_date'], errors='coerce').dt.date
    long['day_as_date'] = pd.to_datetime(long['day'], errors='coerce').dt.date

    rec_per_person = long.groupby('person_id').size()
    day_counts = long.groupby(['person_id', 'day_as_date']).size().reset_index(name='records_on_day')
    any_same_day_persons = day_counts.loc[day_counts['records_on_day'] > 1, 'person_id'].nunique()
    baseline_day = long.merge(baseline[['person_id', 'index_day']], left_on=['person_id', 'day_as_date'], right_on=['person_id', 'index_day'], how='inner')
    baseline_day_counts = baseline_day.groupby('person_id').size()
    baseline_multi_persons = int((baseline_day_counts > 1).sum())
    baseline_multi_records = int(baseline_day_counts.loc[baseline_day_counts > 1].sum()) if baseline_multi_persons > 0 else 0

    return pd.DataFrame([
        {'Cohort': cohort_label, 'Metric': 'Participants', 'Value': fmt_int(len(ids))},
        {'Cohort': cohort_label, 'Metric': 'Eligible clean 25(OH)D records', 'Value': fmt_int(len(long))},
        {'Cohort': cohort_label, 'Metric': 'Records per participant, median (IQR)', 'Value': median_iqr(rec_per_person, 0)},
        {'Cohort': cohort_label, 'Metric': 'Records per participant, maximum', 'Value': fmt_int(rec_per_person.max())},
        {'Cohort': cohort_label, 'Metric': 'Participants with >1 eligible record', 'Value': fmt_int((rec_per_person > 1).sum())},
        {'Cohort': cohort_label, 'Metric': 'Participants with any same-day multiple eligible records', 'Value': fmt_int(any_same_day_persons)},
        {'Cohort': cohort_label, 'Metric': 'Participants with multiple eligible records on baseline date', 'Value': fmt_int(baseline_multi_persons)},
        {'Cohort': cohort_label, 'Metric': 'Baseline-date records among those participants', 'Value': fmt_int(baseline_multi_records)},
        {'Cohort': cohort_label, 'Metric': 'Baseline 25(OH)D range, ng/mL', 'Value': range_cell(baseline['vitd_ng_ml'] if 'vitd_ng_ml' in baseline.columns else baseline['vitd_main'], 1)},
        {'Cohort': cohort_label, 'Metric': 'Baseline 25(OH)D median (IQR), ng/mL', 'Value': median_iqr(baseline['vitd_ng_ml'] if 'vitd_ng_ml' in baseline.columns else baseline['vitd_main'], 1)},
    ])

repeat_primary = repeated_measure_audit(q1_baseline['person_id'], 'Primary analytic cohort', q1_baseline)
cox_baseline_proxy = q1_baseline[q1_baseline['person_id'].isin(q1_cox['person_id'])].copy()
repeat_cox = repeated_measure_audit(q1_cox['person_id'], 'Cox modeling cohort', cox_baseline_proxy)
repeat_audit = pd.concat([repeat_primary, repeat_cox], ignore_index=True)
save_table('q1_supp_vitd_repeated_measure_audit', repeat_audit, 'Supplementary table. Repeated 25(OH)D measurement audit')

# %% Cell 9
concept_source = read_csv_optional(Q1_FILES['q1_formal_concepts'])
if concept_source is None:
    concept_source = read_csv_optional(Q1_FILES['q1_finalized_concepts'])

if concept_source is not None:
    concept_table = concept_source.copy()
    rename_map = {
        'condition_concept_id': 'OMOP concept ID',
        'concept_id': 'OMOP concept ID',
        'concept_name': 'Concept name',
        'category': 'Concept set category',
        'n_records': 'Records',
        'n_people': 'Persons',
    }
    concept_table = concept_table.rename(columns={k: v for k, v in rename_map.items() if k in concept_table.columns})
    preferred = [c for c in ['Concept set category', 'OMOP concept ID', 'Concept name', 'Records', 'Persons'] if c in concept_table.columns]
    concept_table = concept_table[preferred + [c for c in concept_table.columns if c not in preferred]]
    save_table('q1_supp_omop_concept_set_reproducibility', concept_table, 'Supplementary Table 1. OMOP concept sets for baseline exclusion and incident MI/stroke ascertainment', png_preview=False)

# %% Cell 10
# C124/C153: follow-up and event-timing audit
SECONDS_PER_DAY = 24 * 60 * 60
DAYS_PER_YEAR = 365.25

def event_timing_table(cohort_ids, label):
    df = q1_followup[q1_followup['person_id'].isin(pd.Series(cohort_ids).dropna().unique())].copy()
    py = pd.to_numeric(df['follow_up_years'], errors='coerce').sum()
    events = pd.to_numeric(df['event_mi_or_stroke'], errors='coerce').fillna(0).astype(int)
    event_df = df.loc[events == 1].copy()
    event_years = pd.to_numeric(event_df['follow_up_years'], errors='coerce')
    event_days = pd.to_numeric(event_df['follow_up_days'], errors='coerce') if 'follow_up_days' in event_df.columns else event_years * DAYS_PER_YEAR
    return pd.DataFrame([{
        'Cohort': label,
        'N': fmt_int(len(df)),
        'Events': fmt_int(events.sum()),
        'Person-years': fmt_int(py),
        'Event rate per 1,000 PY': fmt_num(events.sum() / py * 1000, 2) if py > 0 else '',
        'Follow-up, years, median (IQR)': median_iqr(df['follow_up_years'], 2),
        'Follow-up, years, range': range_cell(df['follow_up_years'], 2),
        'Time from index to event, years, median (IQR)': median_iqr(event_years, 2),
        'Time from index to event, years, range': range_cell(event_years, 3),
        'Events within 30 days': fmt_int((event_days <= 30).sum()),
        'Events within 90 days': fmt_int((event_days <= 90).sum()),
        'Events within 1 year': fmt_int((event_days <= 365).sum()),
    }])

event_timing = pd.concat([
    event_timing_table(q1_followup['person_id'], 'Primary analytic/follow-up cohort'),
    event_timing_table(q1_cox['person_id'], 'Cox modeling cohort'),
], ignore_index=True)
save_table('q1_supp_followup_event_timing_c124_c153', event_timing, 'Supplementary table. Follow-up and event timing audit')

# %% Cell 11
# C126/C129: CRP unit, timing distribution, and 30-/90-day timing sensitivity
crp_unit_audit = read_csv_optional(Q1_FILES['q1_crp_unit_audit'])
crp_unit_map = read_csv_optional(Q1_FILES['q1_crp_unit_map'])

if crp_unit_audit is not None:
    save_table('q2_supp_crp_unit_audit_detail_c126', crp_unit_audit, 'Supplementary table. CRP unit audit detail', png_preview=False)
if crp_unit_map is not None:
    save_table('q2_supp_crp_unit_mapping_c126', crp_unit_map, 'Supplementary table. CRP unit mapping decisions', png_preview=False)

q2_timing = q2_dataset.copy()
q2_timing['days_before_numeric'] = pd.to_numeric(q2_timing.get('days_before'), errors='coerce')
q2_timing['crp_numeric'] = pd.to_numeric(q2_timing.get('crp_mgL'), errors='coerce')
q2_timing['ambiguous_numeric'] = pd.to_numeric(q2_timing.get('ambiguous'), errors='coerce') if 'ambiguous' in q2_timing.columns else np.nan

timing_bins = [-np.inf, -1, 0, 30, 90, 365, 1825, np.inf]
timing_labels = ['Post-index/negative timing flag', 'Same day', '1-30 days before', '31-90 days before', '91-365 days before', '1-5 years before', '>5 years before']
q2_timing['CRP timing window'] = pd.cut(q2_timing['days_before_numeric'], bins=timing_bins, labels=timing_labels, right=True)

timing_rows = []
for label, sub in q2_timing.groupby('CRP timing window', observed=False):
    if len(sub) == 0:
        continue
    timing_rows.append({
        'CRP timing window': str(label),
        'N': fmt_int(len(sub)),
        'Percent of Q2 dataset': fmt_num(len(sub) / len(q2_timing) * 100, 1),
        'CRP, mg/L, median (IQR)': median_iqr(sub['crp_numeric'], 2),
        'CRP >10 mg/L, n (%)': n_pct(int((sub['crp_numeric'] > 10).sum()), len(sub)),
        'Ambiguous-unit flag, n (%)': n_pct(int((sub['ambiguous_numeric'] == 1).sum()), len(sub)) if 'ambiguous' in q2_timing.columns else '',
    })

crp_overall = pd.DataFrame([{
    'Metric': 'Q2 CRP dataset N',
    'Value': fmt_int(len(q2_timing)),
}, {
    'Metric': 'CRP, mg/L, median (IQR)',
    'Value': median_iqr(q2_timing['crp_numeric'], 2),
}, {
    'Metric': 'CRP, mg/L, range',
    'Value': range_cell(q2_timing['crp_numeric'], 2),
}, {
    'Metric': 'CRP >10 mg/L, n (%)',
    'Value': n_pct(int((q2_timing['crp_numeric'] > 10).sum()), len(q2_timing)),
}, {
    'Metric': 'Ambiguous-unit flag, n (%)',
    'Value': n_pct(int((q2_timing['ambiguous_numeric'] == 1).sum()), len(q2_timing)) if 'ambiguous' in q2_timing.columns else '',
}, {
    'Metric': 'Days before index/target, median (IQR)',
    'Value': median_iqr(q2_timing['days_before_numeric'], 1),
}, {
    'Metric': 'Days before index/target, range',
    'Value': range_cell(q2_timing['days_before_numeric'], 0),
}])
save_table('q2_supp_crp_overall_audit_c126_c129', crp_overall, 'Supplementary table. Q2 CRP overall audit')
save_table('q2_supp_crp_timing_distribution_c129', pd.DataFrame(timing_rows), 'Supplementary table. Q2 CRP timing distribution')

def fit_ols_formula(data, formula):
    if smf is None:
        raise RuntimeError('statsmodels is not available')
    fit = smf.ols(formula=formula, data=data).fit()
    ci = fit.conf_int()
    return fit, ci

def q2_window_sensitivity_rows():
    if 'days_before_numeric' not in q2_timing.columns:
        return pd.DataFrame([{'Window': 'Not generated', 'Model': '', 'N': '', 'Beta for log 25(OH)D': '', '95% CI': '', 'P value': '', 'R-squared': '', 'Reason': 'days_before column not available.'}])
    windows = [
        ('Unrestricted nearest pre-index', q2_timing['days_before_numeric'].notna()),
        ('CRP within 30 days before index/target', q2_timing['days_before_numeric'].between(0, 30, inclusive='both')),
        ('CRP within 90 days before index/target', q2_timing['days_before_numeric'].between(0, 90, inclusive='both')),
        ('CRP within 365 days before index/target', q2_timing['days_before_numeric'].between(0, 365, inclusive='both')),
    ]
    rows = []
    for window_label, mask in windows:
        sub = q2_timing.loc[mask].copy()
        if len(sub) == 0:
            rows.append({'Window': window_label, 'Model': 'All', 'N': '0', 'Beta for log 25(OH)D': '', '95% CI': '', 'P value': '', 'R-squared': '', 'Reason': 'No records in window.'})
            continue
        if stats is not None:
            corr_data = sub[['log_vitd', 'log_crp']].dropna()
            if len(corr_data) >= 3:
                pearson = stats.pearsonr(corr_data['log_vitd'], corr_data['log_crp'])
                spearman = stats.spearmanr(sub['vitd_main'], sub['crp_mgL'], nan_policy='omit')
                rows.append({'Window': window_label, 'Model': 'Correlation summary', 'N': fmt_int(len(corr_data)), 'Beta for log 25(OH)D': '', '95% CI': '', 'P value': '', 'R-squared': '', 'Pearson log-log r': fmt_num(pearson.statistic, 3), 'Spearman raw rho': fmt_num(spearman.statistic, 3), 'Reason': ''})
        formulas = [('Unadjusted', 'log_crp ~ log_vitd', ['log_crp', 'log_vitd'])]
        for _, spec in q2_models.iterrows():
            model_name = Q2_MODEL_LABELS.get(spec.get('model', ''), spec.get('model', ''))
            fields = [f.strip() for f in str(spec.get('complete_case_fields', '')).split(',') if f.strip() in sub.columns]
            formulas.append((model_name, spec.get('formula', ''), fields))
        for model_name, formula, fields in formulas:
            if not formula:
                continue
            data = sub.dropna(subset=fields).copy() if fields else sub.copy()
            try:
                fit, ci = fit_ols_formula(data, formula)
                beta = float(fit.params['log_vitd'])
                rows.append({
                    'Window': window_label,
                    'Model': model_name,
                    'N': fmt_int(fit.nobs),
                    'Beta for log 25(OH)D': fmt_num(beta, 3),
                    '95% CI': f"{float(ci.loc['log_vitd', 0]):.3f} to {float(ci.loc['log_vitd', 1]):.3f}",
                    'P value': fmt_p(fit.pvalues['log_vitd']),
                    'R-squared': fmt_num(fit.rsquared, 3),
                    'Pearson log-log r': '',
                    'Spearman raw rho': '',
                    'Reason': '',
                })
            except Exception as exc:
                rows.append({'Window': window_label, 'Model': model_name, 'N': fmt_int(len(data)), 'Beta for log 25(OH)D': '', '95% CI': '', 'P value': '', 'R-squared': '', 'Pearson log-log r': '', 'Spearman raw rho': '', 'Reason': str(exc)})
    return pd.DataFrame(rows)

q2_timing_sensitivity = q2_window_sensitivity_rows()
save_table('q2_supp_crp_30_90_timing_sensitivity_c129', q2_timing_sensitivity, 'Supplementary table. Q2 CRP timing-window sensitivity', png_preview=False)

# %% Cell 12
# C139/C141/C144: BMI range/timing and eGFR availability/timing audit

def bmi_range_rows(df, cohort_label, value_col='bmi_full', source_col=None):
    if value_col not in df.columns:
        return []
    temp = df.copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
    bands = [
        ('<12 kg/m2', temp[value_col] < 12),
        ('12 to <18.5 kg/m2', temp[value_col].between(12, 18.5, inclusive='left')),
        ('18.5 to <25 kg/m2', temp[value_col].between(18.5, 25, inclusive='left')),
        ('25 to <30 kg/m2', temp[value_col].between(25, 30, inclusive='left')),
        ('>=30 kg/m2', temp[value_col] >= 30),
        ('Missing', temp[value_col].isna()),
    ]
    rows = [{
        'Cohort': cohort_label,
        'BMI category/audit metric': 'BMI, kg/m2, median (IQR)',
        'N': fmt_int(temp[value_col].notna().sum()),
        'Percent': '',
        'BMI summary': median_iqr(temp[value_col], 1),
        'BMI range': range_cell(temp[value_col], 2),
        'Source detail': source_col or '',
    }]
    for label, mask in bands:
        rows.append({
            'Cohort': cohort_label,
            'BMI category/audit metric': label,
            'N': fmt_int(mask.sum()),
            'Percent': fmt_num(mask.mean() * 100, 1),
            'BMI summary': '',
            'BMI range': '',
            'Source detail': source_col or '',
        })
    if source_col and source_col in temp.columns:
        for source, sub in temp.groupby(source_col, dropna=False):
            rows.append({
                'Cohort': cohort_label,
                'BMI category/audit metric': f'BMI source: {source}',
                'N': fmt_int(len(sub)),
                'Percent': fmt_num(len(sub) / len(temp) * 100, 1),
                'BMI summary': median_iqr(sub[value_col], 1),
                'BMI range': range_cell(sub[value_col], 2),
                'Source detail': source_col,
            })
    return rows

bmi_rows = []
bmi_rows.extend(bmi_range_rows(q1_baseline, 'Q1 primary analytic cohort', 'bmi_full', 'bmi_source' if 'bmi_source' in q1_baseline.columns else None))
bmi_rows.extend(bmi_range_rows(q1_baseline[q1_baseline['person_id'].isin(q1_cox['person_id'])], 'Q1 Cox modeling cohort', 'bmi_full', 'bmi_source' if 'bmi_source' in q1_baseline.columns else None))
bmi_rows.extend(bmi_range_rows(q2_dataset, 'Q2 CRP model dataset', 'bmi_full', None))
save_table('q1_q2_supp_bmi_range_source_audit_c139', pd.DataFrame(bmi_rows), 'Supplementary table. BMI range and source audit')

bmi_timing_rows = []
q1_bmi_timing = read_csv_optional(Q1_FILES['q1_bmi_timing_audit'])
if q1_bmi_timing is not None:
    for _, r in q1_bmi_timing.iterrows():
        bmi_timing_rows.append({'Source': 'Q1 recorded BMI timing audit', 'Metric': r.get('metric', ''), 'Value': r.get('value', '')})

q1_recovered_bmi_timing = read_csv_optional(Q1_FILES['q1_recovered_bmi_timing_audit'])
if q1_recovered_bmi_timing is not None:
    col = 'bmi_meas_days_before_index' if 'bmi_meas_days_before_index' in q1_recovered_bmi_timing.columns else q1_recovered_bmi_timing.columns[-1]
    x = pd.to_numeric(q1_recovered_bmi_timing[col], errors='coerce')
    bmi_timing_rows.extend([
        {'Source': 'Q1 recovered BMI timing audit', 'Metric': 'Rows with recovered BMI timing', 'Value': fmt_int(x.notna().sum())},
        {'Source': 'Q1 recovered BMI timing audit', 'Metric': 'Post-index recovered BMI timing, n (%)', 'Value': n_pct(int((x < 0).sum()), int(x.notna().sum()))},
        {'Source': 'Q1 recovered BMI timing audit', 'Metric': 'Same-day recovered BMI timing, n (%)', 'Value': n_pct(int((x == 0).sum()), int(x.notna().sum()))},
        {'Source': 'Q1 recovered BMI timing audit', 'Metric': 'Pre-index recovered BMI within 365 days, n (%)', 'Value': n_pct(int(x.between(0, 365, inclusive='both').sum()), int(x.notna().sum()))},
        {'Source': 'Q1 recovered BMI timing audit', 'Metric': 'Recovered BMI timing days before index, median (IQR)', 'Value': median_iqr(x, 0)},
        {'Source': 'Q1 recovered BMI timing audit', 'Metric': 'Recovered BMI timing days before index, range', 'Value': range_cell(x, 0)},
    ])

if not bmi_timing_rows:
    bmi_timing_rows.append({'Source': 'BMI timing audit', 'Metric': 'Status', 'Value': 'No BMI timing audit CSV found under DATA_DIR.'})
save_table('q1_supp_bmi_timing_audit_c141', pd.DataFrame(bmi_timing_rows), 'Supplementary table. BMI timing audit')

def summarize_egfr_file(path, label):
    df = read_csv_optional(path)
    if df is None:
        return pd.DataFrame([{'Source': label, 'Status': 'No eGFR detail file found', 'N': '', 'eGFR <60 n (%)': '', 'eGFR median (IQR)': '', 'eGFR range': '', 'Timing summary': ''}])
    egfr_candidates = [c for c in df.columns if c.lower() in {'egfr', 'baseline_egfr', 'egfr_ckd_epi_2021', 'egfr_2021'} or 'egfr' in c.lower()]
    egfr_col = egfr_candidates[0] if egfr_candidates else None
    timing_candidates = [c for c in df.columns if 'day' in c.lower() or 'before' in c.lower() or 'timing' in c.lower()]
    timing_col = timing_candidates[0] if timing_candidates else None
    if egfr_col is None:
        return pd.DataFrame([{'Source': label, 'Status': 'eGFR file found but no eGFR-like column detected', 'N': fmt_int(len(df)), 'eGFR <60 n (%)': '', 'eGFR median (IQR)': '', 'eGFR range': '', 'Timing summary': f'Columns: {", ".join(df.columns)}'}])
    egfr = pd.to_numeric(df[egfr_col], errors='coerce')
    timing_summary = ''
    if timing_col is not None:
        timing = pd.to_numeric(df[timing_col], errors='coerce')
        timing_summary = f'{timing_col}: median (IQR) {median_iqr(timing, 0)}; range {range_cell(timing, 0)}'
    return pd.DataFrame([{
        'Source': label,
        'Status': f'Read {Path(path).name}; eGFR column {egfr_col}',
        'N': fmt_int(len(df)),
        'eGFR <60 n (%)': n_pct(int((egfr < 60).sum()), int(egfr.notna().sum())),
        'eGFR median (IQR)': median_iqr(egfr, 1),
        'eGFR range': range_cell(egfr, 1),
        'Timing summary': timing_summary,
    }])

egfr_table = pd.concat([
    summarize_egfr_file(Q1_FILES['q1_egfr_baseline'], 'Q1 eGFR baseline file'),
    summarize_egfr_file(Q2_FILES['q2_egfr_baseline'], 'Q2 eGFR baseline file'),
], ignore_index=True)
if q2_coverage is not None and 'covariate' in q2_coverage.columns:
    kidney_rows = q2_coverage[q2_coverage['covariate'].astype(str).str.contains('Kidney|eGFR', case=False, na=False)].copy()
    if not kidney_rows.empty:
        kidney_rows.insert(0, 'Source', 'Q2 covariate coverage file')
        save_table('q2_supp_egfr_coverage_c144', kidney_rows, 'Supplementary table. Q2 eGFR coverage status', png_preview=False)
save_table('q1_q2_supp_egfr_availability_timing_c144', egfr_table, 'Supplementary table. eGFR availability and timing audit')

# %% Cell 13
VITD_LABELS = ['Very low (<10 ng/mL)', 'Low (10 to <20 ng/mL)', 'Intermediate (20 to <30 ng/mL)', 'Higher/reference (>=30 ng/mL)']

def add_vitd_category(df, value_col):
    out = df.copy()
    out['vitd_category'] = pd.cut(
        pd.to_numeric(out[value_col], errors='coerce'),
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=VITD_LABELS,
        right=False,
    )
    return out

def p_value_continuous(df, value_col, group_col='vitd_category'):
    if stats is None:
        return ''
    groups = [pd.to_numeric(df.loc[df[group_col] == g, value_col], errors='coerce').dropna() for g in VITD_LABELS]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return ''
    try:
        return fmt_p(stats.kruskal(*groups).pvalue)
    except Exception:
        return ''

def p_value_categorical(df, value_col, group_col='vitd_category'):
    if stats is None:
        return ''
    temp = df[[value_col, group_col]].dropna()
    if temp[value_col].nunique() < 2 or temp[group_col].nunique() < 2:
        return ''
    try:
        table = pd.crosstab(temp[value_col], temp[group_col])
        return fmt_p(stats.chi2_contingency(table).pvalue)
    except Exception:
        return ''

def summarize_continuous_row(df, label, value_col, digits=1):
    row = {'Characteristic': label, 'Overall': median_iqr(df[value_col], digits), 'Missing, n (%)': missing_cell(df[value_col]), 'P value': p_value_continuous(df, value_col)}
    for g in VITD_LABELS:
        row[g] = median_iqr(df.loc[df['vitd_category'] == g, value_col], digits)
    return row

def summarize_binary_row(df, label, value_col, positive=1):
    valid = df[value_col].notna()
    row = {'Characteristic': label, 'Overall': n_pct(int((df[value_col] == positive).sum()), int(valid.sum())), 'Missing, n (%)': missing_cell(df[value_col]), 'P value': p_value_categorical(df, value_col)}
    for g in VITD_LABELS:
        sub = df.loc[df['vitd_category'] == g]
        denom = int(sub[value_col].notna().sum())
        row[g] = n_pct(int((sub[value_col] == positive).sum()), denom)
    return row

def summarize_level_row(df, label, value_col, level):
    valid = df[value_col].notna()
    row = {'Characteristic': label, 'Overall': n_pct(int((df[value_col] == level).sum()), int(valid.sum())), 'Missing, n (%)': missing_cell(df[value_col]), 'P value': p_value_categorical(df, value_col)}
    for g in VITD_LABELS:
        sub = df.loc[df['vitd_category'] == g]
        denom = int(sub[value_col].notna().sum())
        row[g] = n_pct(int((sub[value_col] == level).sum()), denom)
    return row

def table1_event_rate_row(df):
    row = {'Characteristic': 'Incident MI or stroke rate, per 1,000 person-years', 'Missing, n (%)': '', 'P value': ''}
    py = pd.to_numeric(df['follow_up_years'], errors='coerce').sum()
    events = pd.to_numeric(df['event_mi_or_stroke'], errors='coerce').fillna(0).sum()
    row['Overall'] = fmt_num(events / py * 1000, 1) if py > 0 else ''
    for g in VITD_LABELS:
        sub = df.loc[df['vitd_category'] == g]
        py_g = pd.to_numeric(sub['follow_up_years'], errors='coerce').sum()
        ev_g = pd.to_numeric(sub['event_mi_or_stroke'], errors='coerce').fillna(0).sum()
        row[g] = fmt_num(ev_g / py_g * 1000, 1) if py_g > 0 else ''
    return row

q1_table1 = add_vitd_category(q1_baseline, 'vitd_ng_ml').merge(
    q1_followup[['person_id', 'follow_up_years', 'event_mi_or_stroke', 'first_mi_date', 'first_stroke_date', 'first_event_date']],
    on='person_id',
    how='left',
    validate='one_to_one',
)
q1_table1['female_binary'] = np.where(q1_table1['gender'].eq('Female'), 1, np.where(q1_table1['gender'].eq('Male'), 0, np.nan))
q1_table1['bmi_category'] = pd.cut(pd.to_numeric(q1_table1['bmi_full'], errors='coerce'), bins=[-np.inf, 25, 30, np.inf], labels=['<25 kg/m2', '25 to <30 kg/m2', '>=30 kg/m2'], right=False)

rows = []
rows.append({'Characteristic': 'N', 'Overall': fmt_int(len(q1_table1)), 'Missing, n (%)': '', 'P value': '', **{g: fmt_int((q1_table1['vitd_category'] == g).sum()) for g in VITD_LABELS}})
rows.append(summarize_continuous_row(q1_table1, '25(OH)D, ng/mL, median (IQR)', 'vitd_ng_ml', 1))
rows.append(summarize_continuous_row(q1_table1, 'Age, years, median (IQR)', 'age_at_index', 1))
rows.append(summarize_binary_row(q1_table1, 'Female, n (%)', 'female_binary', 1))
for level in ['White (Non-Hispanic)', 'Black', 'Hispanic', 'Asian', 'Other/Unknown']:
    if level in set(q1_table1['race_eth_simplified'].dropna()):
        rows.append(summarize_level_row(q1_table1, f'Race/ethnicity: {level}, n (%)', 'race_eth_simplified', level))
rows.append(summarize_continuous_row(q1_table1, 'BMI, kg/m2, median (IQR)', 'bmi_full', 1))
for level in ['<25 kg/m2', '25 to <30 kg/m2', '>=30 kg/m2']:
    rows.append(summarize_level_row(q1_table1, f'BMI category: {level}, n (%)', 'bmi_category', level))
rows.append(summarize_binary_row(q1_table1, 'Baseline diabetes, n (%)', 'baseline_diabetes', 1))
rows.append(summarize_binary_row(q1_table1, 'Baseline hypertension, n (%)', 'baseline_hypertension', 1))
rows.append(summarize_binary_row(q1_table1, 'Baseline hyperlipidemia, n (%)', 'baseline_hyperlipidemia', 1))
rows.append(summarize_continuous_row(q1_table1, 'Follow-up, years, median (IQR)', 'follow_up_years', 1))
rows.append({'Characteristic': 'Person-years', 'Overall': fmt_int(pd.to_numeric(q1_table1['follow_up_years'], errors='coerce').sum()), 'Missing, n (%)': missing_cell(q1_table1['follow_up_years']), 'P value': '', **{g: fmt_int(pd.to_numeric(q1_table1.loc[q1_table1['vitd_category'] == g, 'follow_up_years'], errors='coerce').sum()) for g in VITD_LABELS}})
rows.append(summarize_binary_row(q1_table1, 'Incident MI or stroke, n (%)', 'event_mi_or_stroke', 1))
rows.append(table1_event_rate_row(q1_table1))

table1 = pd.DataFrame(rows)
table1 = table1[['Characteristic', 'Overall'] + VITD_LABELS + ['Missing, n (%)', 'P value']]
save_table('q1_table1_primary_cohort_baseline_followup', table1, 'Table 1. Baseline characteristics, follow-up, and event rate by 25(OH)D category')

# %% Cell 14
missing_vars = [
    ('25(OH)D', 'vitd_ng_ml'),
    ('Age', 'age_at_index'),
    ('Gender', 'gender'),
    ('Race/ethnicity', 'race_eth_simplified'),
    ('BMI', 'bmi_full'),
    ('Baseline diabetes', 'baseline_diabetes'),
    ('Baseline hypertension', 'baseline_hypertension'),
    ('Baseline hyperlipidemia', 'baseline_hyperlipidemia'),
    ('Follow-up years', 'follow_up_years'),
    ('Incident MI or stroke', 'event_mi_or_stroke'),
]
miss_rows = []
for label, col in missing_vars:
    s = q1_table1[col]
    miss_rows.append({'Variable': label, 'Column': col, 'Nonmissing n': fmt_int(s.notna().sum()), 'Missing n': fmt_int(s.isna().sum()), 'Missing %': fmt_num(s.isna().mean() * 100, 1)})
q1_missing = pd.DataFrame(miss_rows)
save_table('q1_supp_table1_missingness_inventory', q1_missing, 'Supplementary table. Q1 Table 1 missingness inventory')

# %% Cell 15
primary_cox = read_csv_optional(Q1_FILES['q1_primary_cox_table'])

def relabel_cox_term(term):
    replacements = {
        'Vitamin D: Deficient (<10)': '25(OH)D: very low (<10 ng/mL)',
        'Vitamin D: Insufficient (10-<20)': '25(OH)D: low (10 to <20 ng/mL)',
        'Vitamin D: Adequate (20-<30)': '25(OH)D: intermediate (20 to <30 ng/mL)',
        'vitd_very_low': '25(OH)D: very low (<10 ng/mL)',
        'vitd_low': '25(OH)D: low (10 to <20 ng/mL)',
        'vitd_intermediate': '25(OH)D: intermediate (20 to <30 ng/mL)',
        'age_at_index': 'Age (per year)',
        'female': 'Female (vs Male)',
        'race_black': 'Black (vs White)',
        'race_hispanic': 'Hispanic (vs White)',
        'race_asian': 'Asian (vs White)',
        'race_other': 'Other/unknown race or ethnicity (vs White)',
        'baseline_diabetes': 'Baseline diabetes',
        'baseline_hypertension': 'Baseline hypertension',
        'baseline_hyperlipidemia': 'Baseline hyperlipidemia',
    }
    return replacements.get(term, term)

def build_primary_cox_table_from_data():
    if CoxPHFitter is None:
        return None, 'lifelines is not available; primary Cox table cannot be fitted from person-level data.'
    d = q1_cox.copy()
    for col in ['follow_up_years', 'event_mi_or_stroke', 'vitd_main', 'age_at_index', 'female', 'race_black', 'race_hispanic', 'race_asian', 'race_other', 'baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia']:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')
    d['vitd_very_low'] = (d['vitd_main'] < 10).astype(float)
    d['vitd_low'] = ((d['vitd_main'] >= 10) & (d['vitd_main'] < 20)).astype(float)
    d['vitd_intermediate'] = ((d['vitd_main'] >= 20) & (d['vitd_main'] < 30)).astype(float)
    terms = ['vitd_very_low', 'vitd_low', 'vitd_intermediate', 'age_at_index', 'female', 'race_black', 'race_hispanic', 'race_asian', 'race_other', 'baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia']
    model_cols = ['follow_up_years', 'event_mi_or_stroke'] + terms
    m = d[model_cols].dropna().copy()
    active_terms = [c for c in terms if m[c].nunique(dropna=True) > 1]
    m = m[['follow_up_years', 'event_mi_or_stroke'] + active_terms]
    if len(m) == 0 or m['event_mi_or_stroke'].sum() == 0:
        return None, 'No usable rows/events after complete-case filtering for primary Cox model.'
    fit_status = 'Fitted from analysis_base_clean.csv because final_table2_primary_cox.csv was not found.'
    try:
        cph = CoxPHFitter().fit(m, duration_col='follow_up_years', event_col='event_mi_or_stroke')
    except Exception as exc:
        try:
            cph = CoxPHFitter(penalizer=0.01).fit(m, duration_col='follow_up_years', event_col='event_mi_or_stroke')
            fit_status += f' Unpenalized fit failed; penalizer=0.01 used. Original error: {exc}'
        except Exception as exc2:
            return None, f'Primary Cox fallback fit failed: {exc2}'
    rows = []
    for term in active_terms:
        if term not in cph.summary.index:
            continue
        r = cph.summary.loc[term]
        rows.append({
            'Covariate': relabel_cox_term(term),
            'HR (95% CI)': f"{float(r['exp(coef)']):.2f} ({float(r['exp(coef) lower 95%']):.2f}-{float(r['exp(coef) upper 95%']):.2f})",
            'P value': fmt_p(r['p']),
        })
    return pd.DataFrame(rows), fit_status

if primary_cox is not None:
    cox_full = primary_cox.copy()
    cox_full['Covariate'] = cox_full['Covariate'].map(relabel_cox_term)
    cox_full = cox_full.rename(columns={'p': 'P value'})
    cox_source_status = 'Read from final_table2_primary_cox.csv under server data folder.'
else:
    cox_full, cox_source_status = build_primary_cox_table_from_data()

save_table('q1_primary_cox_table_source_status', pd.DataFrame([{'Table': 'q1_table2_primary_cox_exposure_rows / q1_supp_table2_full_primary_cox_model', 'Status': cox_source_status, 'Resolved path checked': str(Q1_FILES['q1_primary_cox_table'])}]), 'Q1 primary Cox table source status', png_preview=False)

if cox_full is not None and not cox_full.empty:
    if 'Model cohort' not in cox_full.columns:
        cox_full.insert(0, 'Model cohort', f"Cox modeling cohort, n={len(q1_cox):,}")
    exposure_mask = cox_full['Covariate'].str.startswith('25(OH)D:', na=False)
    cox_exposure = cox_full.loc[exposure_mask, ['Covariate', 'HR (95% CI)', 'P value']].copy()
    cox_exposure.loc[len(cox_exposure)] = ['Reference: higher/reference (>=30 ng/mL)', '1.00 (reference)', '']
    save_table('q1_table2_primary_cox_exposure_rows', cox_exposure, 'Table 2. Primary adjusted Cox model exposure rows')
    save_table('q1_supp_table2_full_primary_cox_model', cox_full, 'Supplementary table. Full primary Cox model coefficients')
else:
    save_table('q1_table2_primary_cox_not_generated', pd.DataFrame([{'Table': 'Q1 primary Cox table', 'Status': 'Not generated', 'Reason': cox_source_status}]), 'Q1 primary Cox table status', png_preview=False)

# %% Cell 16
RACE_TERMS = ['race_black', 'race_hispanic', 'race_asian', 'race_other']
FULL_COX_TERMS = ['age_at_index', 'female'] + RACE_TERMS + ['baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia']
EXCLUDE_FOR_SUBGROUP = {
    'age_group': ['age_at_index'],
    'sex_group': ['female'],
    'race_group': RACE_TERMS,
    'diabetes_group': ['baseline_diabetes'],
    'hypertension_group': ['baseline_hypertension'],
    'hyperlipidemia_group': ['baseline_hyperlipidemia'],
    'bmi_group': [],
    'income_group': [],
}

def find_income_file_for_q1():
    possible_cols = {'income_band', 'g_inc', 'household_income', 'income_group'}
    candidates = []
    for path in DATA_FILES_INDEX:
        name = path.name.lower()
        path_text = str(path).lower()
        if 'q2_extended_model_dataset' in name or '/q2' in path_text or '\\q2' in path_text:
            continue
        if 'income' not in name and 'survey' not in name and 'subgroup' not in name:
            continue
        try:
            cols = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if 'person_id' in cols and cols.intersection(possible_cols):
            candidates.append((path, sorted(cols.intersection(possible_cols))[0]))
    return candidates[0] if candidates else (None, None)

def income_band_from_text(x):
    x = str(x).lower()
    if any(k in x for k in ['income <35k', '<35', 'less 10', '10k 25', '25k 35', 'less than 10']):
        return 'Income <35k'
    if any(k in x for k in ['income 35-100k', '35-100', '35k 50', '50k 75', '75k 100']):
        return 'Income 35-100k'
    if any(k in x for k in ['income >=100k', '>=100', '100k 150', '150k 200', 'more 200']):
        return 'Income >=100k'
    return None

def prepare_subgroup_dataset():
    cox = q1_cox.merge(q1_baseline[['person_id', 'bmi_full']], on='person_id', how='left', validate='one_to_one')
    numeric_cols = ['follow_up_years', 'event_mi_or_stroke', 'vitd_main', 'age_at_index', 'female', 'bmi_full'] + RACE_TERMS + ['baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia']
    for col in numeric_cols:
        if col in cox.columns:
            cox[col] = pd.to_numeric(cox[col], errors='coerce')
    cox['vitd_neg10'] = -cox['vitd_main'] / 10.0
    cox['age_group'] = np.where(cox['age_at_index'] >= 65, 'Age >=65', 'Age <65')
    cox['sex_group'] = np.where(cox['female'] == 1, 'Female', 'Male')
    cox['race_group'] = cox['race_eth_simplified']
    cox['bmi_group'] = pd.cut(cox['bmi_full'], bins=[0, 25, 30, 200], labels=['BMI <25', 'BMI 25-30', 'BMI >=30'], right=False).astype(object)
    cox['diabetes_group'] = np.where(cox['baseline_diabetes'] == 1, 'Yes', 'No')
    cox['hypertension_group'] = np.where(cox['baseline_hypertension'] == 1, 'Yes', 'No')
    cox['hyperlipidemia_group'] = np.where(cox['baseline_hyperlipidemia'] == 1, 'Yes', 'No')
    income_path, income_col = find_income_file_for_q1()
    income_status = 'No Q1/person-level income file found under DATA_DIR; household-income subgroup not recomputed.'
    if income_path is not None:
        inc = pd.read_csv(income_path, usecols=['person_id', income_col]).copy()
        inc = inc.dropna(subset=[income_col]).drop_duplicates('person_id')
        inc['income_group'] = inc[income_col].map(income_band_from_text)
        inc = inc.dropna(subset=['income_group'])
        if not inc.empty:
            cox = cox.merge(inc[['person_id', 'income_group']], on='person_id', how='left')
            income_status = f'Household-income subgroup derived from {income_path.relative_to(DATA_DIR)} column {income_col}.'
    return cox, income_status

def subgroup_specs(cox):
    specs = [
        ('age_group', 'Age'),
        ('sex_group', 'Sex'),
        ('race_group', 'Race/ethnicity'),
        ('bmi_group', 'BMI'),
        ('diabetes_group', 'Diabetes'),
        ('hypertension_group', 'Hypertension'),
        ('hyperlipidemia_group', 'Hyperlipidemia'),
    ]
    if 'income_group' in cox.columns and cox['income_group'].notna().any():
        specs.append(('income_group', 'Household income'))
    return specs

def subgroup_count_rows(cox, specs):
    rows = []
    for col, subgroup in specs:
        for level, sub in cox.dropna(subset=[col]).groupby(col, observed=True):
            py = pd.to_numeric(sub['follow_up_years'], errors='coerce').sum()
            events = pd.to_numeric(sub['event_mi_or_stroke'], errors='coerce').fillna(0).sum()
            rows.append({'Subgroup': subgroup, 'Level': str(level), 'N': fmt_int(len(sub)), 'Events': fmt_int(events), 'Person-years': fmt_int(py), 'Event rate per 1,000 PY': fmt_num(events / py * 1000, 1) if py > 0 else ''})
    return pd.DataFrame(rows)

def subgroup_covariates(group_col):
    excluded = set(EXCLUDE_FOR_SUBGROUP.get(group_col, []))
    return [c for c in FULL_COX_TERMS if c not in excluded]

def fit_subgroup_hr(df, adj):
    cols = ['follow_up_years', 'event_mi_or_stroke', 'vitd_neg10'] + adj
    m = df[cols].dropna().copy()
    active_adj = [c for c in adj if m[c].nunique(dropna=True) > 1]
    m = m[['follow_up_years', 'event_mi_or_stroke', 'vitd_neg10'] + active_adj]
    if len(m) < 300 or m['event_mi_or_stroke'].sum() < 25:
        raise ValueError('Too few rows or events for stable subgroup Cox fitting.')
    try:
        cph = CoxPHFitter().fit(m, 'follow_up_years', 'event_mi_or_stroke')
    except Exception:
        cph = CoxPHFitter(penalizer=0.01).fit(m, 'follow_up_years', 'event_mi_or_stroke')
    r = cph.summary.loc['vitd_neg10']
    return float(r['exp(coef)']), float(r['exp(coef) lower 95%']), float(r['exp(coef) upper 95%']), float(r['p'])

def fit_interaction_test(df, group_col):
    if stats is None:
        raise RuntimeError('scipy is not available for likelihood-ratio P values.')
    dd = df.dropna(subset=[group_col]).copy()
    dummies = pd.get_dummies(dd[group_col].astype(str), prefix='g', drop_first=True).astype(float)
    if dummies.shape[1] == 0:
        raise ValueError('Only one subgroup level available.')
    dd = pd.concat([dd.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    dummy_cols = list(dummies.columns)
    interaction_cols = []
    for dummy_col in dummy_cols:
        ix = 'ix_' + dummy_col
        dd[ix] = dd['vitd_neg10'] * dd[dummy_col]
        interaction_cols.append(ix)
    adj = subgroup_covariates(group_col)
    base_cols = ['follow_up_years', 'event_mi_or_stroke', 'vitd_neg10'] + adj + dummy_cols
    full_cols = base_cols + interaction_cols
    m = dd[full_cols].dropna().copy()
    active_base = ['follow_up_years', 'event_mi_or_stroke', 'vitd_neg10'] + [c for c in adj + dummy_cols if m[c].nunique(dropna=True) > 1]
    active_ix = [c for c in interaction_cols if m[c].nunique(dropna=True) > 1]
    if not active_ix:
        raise ValueError('No estimable interaction columns.')
    m_reduced = m[active_base]
    m_full = m[active_base + active_ix]
    c_reduced = CoxPHFitter().fit(m_reduced, 'follow_up_years', 'event_mi_or_stroke')
    c_full = CoxPHFitter().fit(m_full, 'follow_up_years', 'event_mi_or_stroke')
    lr = 2 * (c_full.log_likelihood_ - c_reduced.log_likelihood_)
    df_lr = len(active_ix)
    return float(stats.chi2.sf(lr, df_lr)), float(lr), int(df_lr)

def recompute_subgroup_models(cox, specs):
    if CoxPHFitter is None:
        return None, None, 'lifelines is not available; subgroup Cox models were not recomputed.'
    rows, inter_rows, status_rows = [], [], []
    base = cox.dropna(subset=['follow_up_years', 'event_mi_or_stroke', 'vitd_neg10'] + FULL_COX_TERMS).copy()
    for col, subgroup in specs:
        adj = subgroup_covariates(col)
        for level, sub in base.dropna(subset=[col]).groupby(col, observed=True):
            events = int(pd.to_numeric(sub['event_mi_or_stroke'], errors='coerce').fillna(0).sum())
            try:
                hr, lo, hi, p = fit_subgroup_hr(sub, adj)
                rows.append({'subgroup': subgroup, 'level': str(level), 'n': len(sub), 'events': events, 'hr': hr, 'lo': lo, 'hi': hi, 'p': p})
            except Exception as exc:
                status_rows.append({'Subgroup': subgroup, 'Level': str(level), 'Status': 'HR not fitted', 'Reason': str(exc)})
        try:
            pi, lr, df_lr = fit_interaction_test(base, col)
            inter_rows.append({'subgroup': subgroup, 'p_interaction': pi, 'lr_chisq': lr, 'df': df_lr})
        except Exception as exc:
            inter_rows.append({'subgroup': subgroup, 'p_interaction': np.nan, 'lr_chisq': np.nan, 'df': np.nan})
            status_rows.append({'Subgroup': subgroup, 'Level': 'Overall interaction', 'Status': 'Interaction not fitted', 'Reason': str(exc)})
    status = pd.DataFrame(status_rows) if status_rows else pd.DataFrame([{'Subgroup': 'All available', 'Level': '', 'Status': 'Recomputed', 'Reason': 'Subgroup Cox HRs and interaction tests were fitted from analysis_base_clean.csv.'}])
    return pd.DataFrame(rows), pd.DataFrame(inter_rows), status

subgroup_data, income_status = prepare_subgroup_dataset()
specs = subgroup_specs(subgroup_data)
subgroup_counts = subgroup_count_rows(subgroup_data, specs)

subgroup_hr = read_csv_optional(Q1_FILES['q1_subgroups'])
interaction_tests = read_csv_optional(Q1_FILES['q1_interactions'])
subgroup_status_rows = [{'Subgroup': 'Household income', 'Level': '', 'Status': 'Input check', 'Reason': income_status}]

if subgroup_hr is None or interaction_tests is None:
    recomputed_hr, recomputed_it, recompute_status = recompute_subgroup_models(subgroup_data, specs)
    subgroup_status_rows.extend(recompute_status.to_dict('records'))
    if subgroup_hr is None and recomputed_hr is not None and not recomputed_hr.empty:
        subgroup_hr = recomputed_hr
        save_table('q1_subgroups_all_recomputed_if_needed', subgroup_hr, 'Recomputed Q1 subgroup Cox HRs', png_preview=False)
    if interaction_tests is None and recomputed_it is not None and not recomputed_it.empty:
        interaction_tests = recomputed_it
        save_table('q1_interaction_tests_recomputed_if_needed', interaction_tests, 'Recomputed Q1 subgroup interaction tests', png_preview=False)

if subgroup_hr is not None:
    temp = subgroup_hr.copy()
    temp['subgroup'] = temp['subgroup'].replace({'Income': 'Household income'})
    rename = {'subgroup': 'Subgroup', 'level': 'Level', 'n': 'N from subgroup model', 'events': 'Events from subgroup model', 'hr': 'HR', 'lo': 'CI low', 'hi': 'CI high'}
    temp = temp.rename(columns={k: v for k, v in rename.items() if k in temp.columns})
    if {'HR', 'CI low', 'CI high'}.issubset(temp.columns):
        temp['HR (95% CI)'] = temp.apply(lambda r: f"{float(r['HR']):.2f} ({float(r['CI low']):.2f}-{float(r['CI high']):.2f})" if pd.notna(r['HR']) else '', axis=1)
    keep_cols = [c for c in ['Subgroup', 'Level', 'N from subgroup model', 'Events from subgroup model', 'HR (95% CI)', 'p'] if c in temp.columns]
    subgroup_counts = subgroup_counts.merge(temp[keep_cols], on=['Subgroup', 'Level'], how='outer')
if interaction_tests is not None:
    it = interaction_tests.rename(columns={'subgroup': 'Subgroup', 'p_interaction': 'Interaction P value'})
    it['Subgroup'] = it['Subgroup'].replace({'Income': 'Household income'})
    it['Interaction P value formatted'] = it['Interaction P value'].map(fmt_p)
    it['Bonferroni threshold'] = 0.00625
    subgroup_counts = subgroup_counts.merge(it[[c for c in ['Subgroup', 'Interaction P value', 'Interaction P value formatted', 'Bonferroni threshold'] if c in it.columns]], on='Subgroup', how='left')
else:
    subgroup_counts['Interaction P value'] = ''
    subgroup_counts['Interaction P value formatted'] = ''
    subgroup_counts['Bonferroni threshold'] = '0.00625; interaction P values not available because subgroup model fitting was not possible'

save_table('q1_subgroup_analysis_source_status', pd.DataFrame(subgroup_status_rows), 'Q1 subgroup analysis source and recomputation status', png_preview=False)
save_table('q1_supp_subgroup_n_events_support', subgroup_counts, 'Supplementary table. Subgroup n/events support for Fig. 5', png_preview=False)

# %% Cell 17
sens_rows = []

bmi_sens = read_csv_optional(Q1_FILES['q1_bmi_sensitivity'])
if bmi_sens is not None:
    for _, r in bmi_sens.iterrows():
        sens_rows.append({'Analysis family': 'BMI sensitivity', 'Analysis': r.get('model', ''), 'N': fmt_int(r.get('n', np.nan)), 'Very low / deficient HR': r.get('deficient_HR', ''), 'Low / insufficient HR': r.get('insufficient_HR', ''), 'Intermediate / adequate HR': r.get('adequate_HR', ''), 'P value / note': ''})

concept_sens = read_csv_optional(Q1_FILES['q1_concept_sensitivity'])
if concept_sens is not None:
    for _, r in concept_sens.iterrows():
        sens_rows.append({'Analysis family': 'Exposure concept-set sensitivity', 'Analysis': r.get('version', ''), 'N': fmt_int(r.get('n', np.nan)), 'Very low / deficient HR': r.get('deficient', ''), 'Low / insufficient HR': r.get('insufficient', ''), 'Intermediate / adequate HR': r.get('adequate', ''), 'P value / note': r.get('defic_p', '')})

finegray = read_csv_optional(Q1_FILES['q1_finegray'])
if finegray is not None:
    fg_map = {'vitd_deficient': 'Very low / deficient', 'vitd_insufficient': 'Low / insufficient', 'vitd_adequate': 'Intermediate / adequate'}
    for _, r in finegray[finegray['covariate'].isin(fg_map)].iterrows():
        sens_rows.append({'Analysis family': 'Competing-risk Fine-Gray', 'Analysis': fg_map.get(r['covariate'], r['covariate']), 'N': '', 'Very low / deficient HR': f"{r['sHR']:.2f} ({r['lower95']:.2f}-{r['upper95']:.2f})" if r['covariate'] == 'vitd_deficient' else '', 'Low / insufficient HR': f"{r['sHR']:.2f} ({r['lower95']:.2f}-{r['upper95']:.2f})" if r['covariate'] == 'vitd_insufficient' else '', 'Intermediate / adequate HR': f"{r['sHR']:.2f} ({r['lower95']:.2f}-{r['upper95']:.2f})" if r['covariate'] == 'vitd_adequate' else '', 'P value / note': fmt_p(r['p'])})

q1_sensitivity = pd.DataFrame(sens_rows)
if not q1_sensitivity.empty:
    save_table('q1_supp_sensitivity_analyses', q1_sensitivity, 'Supplementary table. Q1 sensitivity analyses')

component_rows = []
for label, df in [('Primary analytic cohort', q1_followup), ('Cox modeling cohort', q1_followup[q1_followup['person_id'].isin(q1_cox['person_id'])])]:
    tmp = df.copy()
    tmp['has_mi'] = tmp['first_mi_date'].notna()
    tmp['has_stroke'] = tmp['first_stroke_date'].notna()
    tmp['has_both'] = tmp['has_mi'] & tmp['has_stroke']
    tmp['first_mi_dt'] = pd.to_datetime(tmp['first_mi_date'], errors='coerce')
    tmp['first_stroke_dt'] = pd.to_datetime(tmp['first_stroke_date'], errors='coerce')
    first_mi = ((tmp['first_mi_dt'].notna()) & ((tmp['first_stroke_dt'].isna()) | (tmp['first_mi_dt'] < tmp['first_stroke_dt']))).sum()
    first_stroke = ((tmp['first_stroke_dt'].notna()) & ((tmp['first_mi_dt'].isna()) | (tmp['first_stroke_dt'] < tmp['first_mi_dt']))).sum()
    same_day = ((tmp['first_mi_dt'].notna()) & (tmp['first_mi_dt'] == tmp['first_stroke_dt'])).sum()
    component_rows.append({'Cohort': label, 'N': fmt_int(len(tmp)), 'Composite MI or stroke events': fmt_int(tmp['event_mi_or_stroke'].sum()), 'Any MI': fmt_int(tmp['has_mi'].sum()), 'Any stroke': fmt_int(tmp['has_stroke'].sum()), 'Both MI and stroke recorded': fmt_int(tmp['has_both'].sum()), 'First event MI': fmt_int(first_mi), 'First event stroke': fmt_int(first_stroke), 'Same-day MI and stroke': fmt_int(same_day)})

component_table = pd.DataFrame(component_rows)
save_table('q1_supp_component_event_counts', component_table, 'Supplementary table. MI and stroke component event counts')

# %% Cell 18
q2_rows = []
if len(q2_loglog) > 0:
    r = q2_loglog.iloc[0]
    slope = float(r['loglog_slope'])
    q2_rows.append({'Model': 'Unadjusted log-log association', 'N': fmt_int(r['n']), 'Beta for log 25(OH)D': fmt_num(slope, 3), '95% CI': f"{float(r['loglog_ci_low']):.3f} to {float(r['loglog_ci_high']):.3f}", 'P value': fmt_p(r['loglog_p']), 'R-squared': '', 'Approximate CRP change for doubling 25(OH)D, %': fmt_num((2 ** slope - 1) * 100, 1)})

model_label = Q2_MODEL_LABELS
for _, r in q2_models.iterrows():
    slope = float(r['slope'])
    q2_rows.append({'Model': Q2_MODEL_LABELS.get(r['model'], r['model']), 'N': fmt_int(r['n']), 'Beta for log 25(OH)D': fmt_num(slope, 3), '95% CI': f"{float(r['ci_low']):.3f} to {float(r['ci_high']):.3f}", 'P value': fmt_p(r['p_value']), 'R-squared': fmt_num(r['r2'], 3), 'Approximate CRP change for doubling 25(OH)D, %': fmt_num((2 ** slope - 1) * 100, 1)})

q2_main = pd.DataFrame(q2_rows)
save_table('q2_table_loglog_crp_models', q2_main, 'Q2 table. Association between log 25(OH)D and log CRP')

def q2_term_role(term):
    if term == 'Intercept':
        return 'intercept'
    if term == 'log_vitd':
        return 'exposure'
    if term in {'age_at_index', 'female', 'winter', 'bmi_full'}:
        return 'core_covariate'
    if term in {'race_black', 'race_hispanic', 'race_asian', 'race_other', 'baseline_diabetes'}:
        return 'extended_covariate'
    if term.startswith('C(income_band)') or term.startswith('C(education_level)') or term.startswith('C(insurance_status)'):
        return 'SES_covariate'
    if term == 'vitd_supplement_use':
        return 'supplement_covariate'
    return 'model_term'

def recompute_q2_full_coefficients():
    if smf is None:
        return None, pd.DataFrame([{'Model': '', 'Status': 'Not generated', 'Reason': 'statsmodels is not available for OLS refitting.'}])
    rows, status_rows = [], []
    for _, spec in q2_models.iterrows():
        model_name = spec.get('model', '')
        formula = spec.get('formula', '')
        fields = [f.strip() for f in str(spec.get('complete_case_fields', '')).split(',') if f.strip()]
        fields = [f for f in fields if f in q2_dataset.columns]
        if not formula:
            status_rows.append({'Model': model_name, 'Status': 'Not generated', 'Reason': 'Missing formula in q2_extended_model_loglog.csv.'})
            continue
        data = q2_dataset.dropna(subset=fields).copy() if fields else q2_dataset.copy()
        try:
            fit = smf.ols(formula=formula, data=data).fit()
            ci = fit.conf_int(alpha=0.05)
            for term in fit.params.index:
                rows.append({
                    'model': model_name,
                    'status': 'fit',
                    'n': int(fit.nobs),
                    'formula': formula,
                    'term': term,
                    'term_role': q2_term_role(term),
                    'estimate': float(fit.params[term]),
                    'std_error': float(fit.bse[term]),
                    'ci_low': float(ci.loc[term, 0]),
                    'ci_high': float(ci.loc[term, 1]),
                    'p_value': float(fit.pvalues[term]),
                    'r2': float(fit.rsquared),
                })
            status_rows.append({'Model': model_name, 'Status': 'Recomputed', 'Reason': 'Fitted from q2_extended_model_dataset.csv using formula from q2_extended_model_loglog.csv.'})
        except Exception as exc:
            status_rows.append({'Model': model_name, 'Status': 'Not generated', 'Reason': str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(status_rows)

q2_full_status = pd.DataFrame([{'Model': 'All', 'Status': 'Read from CSV', 'Reason': 'Read from q2_extended_model_full_coefficients.csv under server data folder.'}]) if q2_full is not None else None
if q2_full is None:
    q2_full, q2_full_status = recompute_q2_full_coefficients()

save_table('q2_full_coefficients_source_status', q2_full_status, 'Q2 full coefficients source and recomputation status', png_preview=False)

if q2_full is not None and not q2_full.empty:
    q2_full_fmt = q2_full.copy()
    for col in ['estimate', 'std_error', 'ci_low', 'ci_high', 'p_value', 'r2']:
        if col in q2_full_fmt.columns:
            q2_full_fmt[col] = q2_full_fmt[col].map(lambda x: fmt_p(x) if col == 'p_value' else fmt_num(x, 3))
    save_table('q2_supp_full_coefficients', q2_full_fmt, 'Supplementary table. Q2 full model coefficients', png_preview=False)
else:
    q2_full_missing = pd.DataFrame([{
        'Table': 'q2_supp_full_coefficients',
        'Status': 'Not generated',
        'Reason': 'q2_extended_model_full_coefficients.csv was not found and OLS refitting was not possible.',
        'Resolved path checked': str(Q2_FILES['q2_full_coefficients']),
    }])
    save_table('q2_supp_full_coefficients_missing_file_note', q2_full_missing, 'Q2 full coefficients file status', png_preview=False)

# %% Cell 19
complete_fields = []
for fields in q2_models['complete_case_fields'].dropna():
    complete_fields.extend([f.strip() for f in str(fields).split(',')])
complete_fields = sorted(set([f for f in complete_fields if f in q2_dataset.columns]))

q2_miss_rows = []
for col in complete_fields:
    s = q2_dataset[col]
    q2_miss_rows.append({'Variable': col, 'Nonmissing n': fmt_int(s.notna().sum()), 'Missing n': fmt_int(s.isna().sum()), 'Missing %': fmt_num(s.isna().mean() * 100, 1)})
q2_missing = pd.DataFrame(q2_miss_rows)
save_table('q2_supp_model_covariate_missingness', q2_missing, 'Supplementary table. Q2 model covariate missingness')

sample_rows = [{'Step': 'Paired Q2 sample', 'N': fmt_int(len(q2_dataset)), 'Loss from previous step': '', 'Main explanation': 'Eligible paired 25(OH)D and CRP records'}]
prev = len(q2_dataset)
for _, r in q2_models.iterrows():
    n = int(r['n'])
    sample_rows.append({'Step': Q2_MODEL_LABELS.get(r['model'], r['model']), 'N': fmt_int(n), 'Loss from previous step': fmt_int(prev - n), 'Main explanation': r['complete_case_fields']})
    prev = n
q2_sample_flow = pd.DataFrame(sample_rows)
save_table('q2_supp_model_sample_flow', q2_sample_flow, 'Supplementary table. Q2 model sample flow')

q2_quart = q2_dataset.copy()
q2_quart['25(OH)D quartile'] = pd.qcut(pd.to_numeric(q2_quart['vitd_main'], errors='coerce'), q=4, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4 highest'], duplicates='drop')
quart_rows = []
for q, sub in q2_quart.dropna(subset=['25(OH)D quartile']).groupby('25(OH)D quartile', observed=True):
    quart_rows.append({'25(OH)D quartile': str(q), 'N': fmt_int(len(sub)), '25(OH)D range, ng/mL': range_cell(sub['vitd_main'], 1), 'CRP, mg/L, median (IQR)': median_iqr(sub['crp_mgL'], 2), 'log CRP, median (IQR)': median_iqr(sub['log_crp'], 2)})
q2_quartile_table = pd.DataFrame(quart_rows)
save_table('q2_supp_crp_distribution_by_vitd_quartile', q2_quartile_table, 'Supplementary table. CRP distribution by 25(OH)D quartile')

# %% Cell 20
markdown_workbook_path = write_markdown_workbook()
excel_path = write_excel_workbook()
manifest = pd.DataFrame(MANIFEST)
manifest['markdown_workbook'] = str(markdown_workbook_path)
manifest['excel_workbook'] = str(excel_path) if excel_path is not None else ''
manifest.to_csv(OUT_DIR / 'manifest_q1_q2_table_outputs.csv', index=False, encoding='utf-8-sig')
manifest
