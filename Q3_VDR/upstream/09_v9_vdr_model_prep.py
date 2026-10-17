# Source companion of 09_v9_vdr_model_prep.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# ============================================================
# Block 0. Setup and output policy
# ============================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd

REBUILD_OUTPUT_DIR_NAME = 'Q3_VDR_v9_rebuild_outputs'
MODEL_OUTPUT_DIR_NAME = 'Q3_VDR_v9_model_prep_outputs'
REBUILD_MARKER = '06_analysis_base_v9_genomics_carrier.csv'
MODEL_MARKER = '02_cvd_model_dataset_v9.csv'
OVERWRITE_OUTPUTS = True  # Same-name CSV outputs in MODEL_OUTPUT_DIR will be overwritten.

PRIMARY_TIER1_SNPS = ['rs1544410', 'rs2228570', 'rs731236', 'rs7975232']
EXPANDED_TIER2_SNPS = ['rs11568820', 'rs2239179', 'rs4516035', 'rs7968585']
EXPLORATORY_SNPS = ['rs12721377']

def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        Path('/home/jupyter/workspace/VDR'),
        Path('/home/dataproc/workspace/VDR'),
        Path('/home/jupyter/workspace/Q3_VDR'),
        cwd,
        cwd / 'data',
        Path('/home/jupyter/workspace/VDR/data'),
        Path('/home/dataproc/workspace/VDR/data'),
        Path('/home/jupyter/workspace/Q3_VDR/data'),
    ]
    for p in candidates:
        try:
            if p.exists() and os.access(p, os.W_OK):
                return p
        except Exception:
            pass
    return cwd

def unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        try:
            key = str(Path(path).expanduser().resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(Path(path).expanduser())
    return out

def resolve_existing_dir(env_name, default_dir_name, marker, aliases=()):
    manual = os.environ.get(env_name, '').strip()
    if manual:
        return Path(manual).expanduser()
    roots = unique_paths([
        DATA_DIR,
        Path('/home/jupyter/workspace/VDR'),
        Path('/home/dataproc/workspace/VDR'),
        Path('/home/jupyter/workspace/Q3_VDR'),
        Path.cwd(),
        Path.cwd().parent,
    ])
    for root in roots:
        for dirname in [default_dir_name, *aliases]:
            candidate = root / dirname
            try:
                if candidate.exists() and (candidate / marker).exists():
                    return candidate
            except Exception:
                pass
    for root in roots:
        try:
            if root.exists():
                hits = sorted(root.rglob(marker))
                if hits:
                    return hits[0].parent
        except Exception:
            pass
    return DATA_DIR / default_dir_name

DATA_DIR = Path(os.environ.get('Q3_VDR_DATA_DIR', str(infer_default_data_dir()))).expanduser()
REBUILD_DIR = resolve_existing_dir('Q3_VDR_REBUILD_OUTPUT_DIR', REBUILD_OUTPUT_DIR_NAME, REBUILD_MARKER)
default_model_root = REBUILD_DIR.parent if REBUILD_DIR.exists() else DATA_DIR
MODEL_DIR = Path(os.environ.get('Q3_VDR_MODEL_OUTPUT_DIR', str(default_model_root / MODEL_OUTPUT_DIR_NAME))).expanduser()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save(df, name):
    path = MODEL_DIR / name
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f'{path} exists and OVERWRITE_OUTPUTS is False')
    df.to_csv(path, index=False)
    print(f'Saved {name}: {len(df):,} rows')
    return path

def read_csv_required(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing required input for {label}: {path}')
    df = pd.read_csv(path, low_memory=False)
    print(f'Loaded {label}: {len(df):,} rows from {path}')
    return df

def read_csv_optional(path, label):
    path = Path(path)
    if not path.exists():
        print(f'Optional input missing for {label}: {path}')
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    print(f'Loaded {label}: {len(df):,} rows from {path}')
    return df

def to_num(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def to_date(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

run_config = pd.DataFrame([{
    'data_dir': str(DATA_DIR),
    'rebuild_dir': str(REBUILD_DIR),
    'rebuild_marker': REBUILD_MARKER,
    'rebuild_marker_exists': (REBUILD_DIR / REBUILD_MARKER).exists(),
    'model_dir': str(MODEL_DIR),
    'model_marker': MODEL_MARKER,
    'overwrite_outputs': OVERWRITE_OUTPUTS,
    'primary_tier1_snps': ';'.join(PRIMARY_TIER1_SNPS),
    'expanded_tier2_snps': ';'.join(EXPANDED_TIER2_SNPS),
    'exploratory_snps': ';'.join(EXPLORATORY_SNPS),
}])
save(run_config, '00_model_prep_run_configuration.csv')
print('Reading from:', REBUILD_DIR)
print('Writing to:', MODEL_DIR)

# %% Cell 3
# ============================================================
# Block 1. Load v9 rebuild outputs and candidate plan
# ============================================================
cvd_raw = read_csv_required(REBUILD_DIR / '06_analysis_base_v9_genomics_carrier.csv', 'CVD genomics carrier base')
crp_raw = read_csv_optional(REBUILD_DIR / '06_q2_crp_pair_v9_genomics_carrier.csv', 'CRP genomics carrier pair')
carrier_counts = read_csv_required(REBUILD_DIR / '06_vdr_candidate_carrier_counts_v9.csv', 'candidate carrier counts')
candidate_list = read_csv_optional(REBUILD_DIR / '01_vdr_candidate_vid_list.csv', 'candidate vid list')
if candidate_list.empty:
    candidate_list = read_csv_optional(REBUILD_DIR / '10_sample_name_probe_candidate_vid_list.csv', 'candidate vid list from probe')

candidate_meta = carrier_counts[['tier', 'gene', 'rsid', 'label', 'vid']].drop_duplicates().copy()
candidate_meta['carrier_col'] = candidate_meta['rsid'] + '_carrier'
candidate_meta['analysis_role'] = 'parked_pathway_or_mr_inspired'
candidate_meta.loc[candidate_meta['rsid'].isin(PRIMARY_TIER1_SNPS), 'analysis_role'] = 'primary_tier1_vdr'
candidate_meta.loc[candidate_meta['rsid'].isin(EXPANDED_TIER2_SNPS), 'analysis_role'] = 'expanded_tier2_vdr'
candidate_meta.loc[candidate_meta['rsid'].isin(EXPLORATORY_SNPS), 'analysis_role'] = 'exploratory_sparse_vdr'
candidate_meta['include_in_primary_cvd_prep'] = candidate_meta['analysis_role'].isin(['primary_tier1_vdr', 'expanded_tier2_vdr', 'exploratory_sparse_vdr'])
candidate_meta['include_in_primary_report'] = candidate_meta['analysis_role'].isin(['primary_tier1_vdr', 'expanded_tier2_vdr'])
candidate_meta['notes'] = np.select(
    [
        candidate_meta['analysis_role'].eq('primary_tier1_vdr'),
        candidate_meta['analysis_role'].eq('expanded_tier2_vdr'),
        candidate_meta['analysis_role'].eq('exploratory_sparse_vdr'),
    ],
    [
        'Primary classic VDR carrier-status interaction candidate.',
        'Expanded VDR-locus candidate; include after Tier 1 or as planned expansion.',
        'Keep exploratory because sparse 4-category vitamin D cells remain small.',
    ],
    default='Pathway/GWAS locus; keep parked for future GRS/MR-inspired work, not main VDR receptor layer.'
)
save(candidate_meta, '01_candidate_analysis_plan_v9.csv')

selected_snps = candidate_meta.loc[candidate_meta['include_in_primary_cvd_prep'], 'rsid'].tolist()
selected_carrier_cols = [f'{rsid}_carrier' for rsid in selected_snps]
all_carrier_cols = [c for c in cvd_raw.columns if c.startswith('rs') and c.endswith('_carrier')]
missing_selected = [c for c in selected_carrier_cols if c not in cvd_raw.columns]
if missing_selected:
    raise ValueError(f'Missing selected carrier columns in CVD base: {missing_selected}')
print('Selected Q3 VDR carrier columns:', selected_carrier_cols)

# %% Cell 4
# ============================================================
# Block 2. Clean shared variables and derive interaction terms
# ============================================================
def simplify_race_ethnicity(row):
    race = str(row.get('race', '')).strip().lower()
    ethnicity = str(row.get('ethnicity', '')).strip().lower()
    unknown_tokens = ['skip', 'prefer not', 'none indicated', 'no matching concept']
    # Important: "Not Hispanic or Latino" contains the words Hispanic and Latino.
    # Classify explicit Hispanic/Latino first only after excluding the negative label.
    if ('hispanic or latino' in ethnicity or ethnicity == 'hispanic') and 'not hispanic' not in ethnicity:
        return 'Hispanic/Latino'
    if 'black' in race or 'african american' in race:
        return 'Black'
    if 'asian' in race:
        return 'Asian'
    if 'white' in race:
        return 'White'
    if 'more than one' in race or 'multiple' in race:
        return 'Multiple'
    if 'american indian' in race or 'alaska native' in race:
        return 'American Indian/Alaska Native'
    if 'middle eastern' in race or 'north african' in race:
        return 'Middle Eastern/North African'
    if 'native hawaiian' in race or 'pacific islander' in race:
        return 'Native Hawaiian/Pacific Islander'
    if race in ['', 'nan'] or any(tok in race for tok in unknown_tokens):
        return 'Unknown/Not reported'
    if 'none of these' in race:
        return 'Other'
    return 'Other'

def clean_model_base(df, selected_cols):
    d = df.copy()
    d = to_num(d, [
        'person_id', 'vitd_main', 'vitd_ng_ml', 'age_at_index', 'female', 'bmi_full',
        'baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia',
        'baseline_cvd_exclusion', 'event_mi_or_stroke', 'event_mi', 'event_stroke',
        'follow_up_days', 'follow_up_years', 'winter', 'has_whole_genome_variant',
        'has_array_data', 'has_structural_variant_data', 'crp_mgL', 'log_crp',
        'log_vitd', 'days_before', 'ambiguous_unit'
    ] + all_carrier_cols)
    d = to_date(d, ['index_date', 'censor_date', 'first_cvd_event_date', 'first_mi_date', 'first_stroke_date', 'crp_date'])
    if 'vitd_main' not in d.columns and 'vitd_ng_ml' in d.columns:
        d['vitd_main'] = d['vitd_ng_ml']
    d['vitd_main'] = pd.to_numeric(d['vitd_main'], errors='coerce')
    d['log_vitd'] = np.where(d['vitd_main'] > 0, np.log(d['vitd_main']), np.nan)
    d['vitd_per10'] = d['vitd_main'] / 10.0
    d['vitd_neg10'] = -d['vitd_main'] / 10.0
    d['vitd_lt20'] = np.where(d['vitd_main'].notna(), (d['vitd_main'] < 20).astype(int), np.nan)
    d['vitd_lt10'] = np.where(d['vitd_main'].notna(), (d['vitd_main'] < 10).astype(int), np.nan)
    d['vitd_group4'] = pd.cut(d['vitd_main'], bins=[-np.inf, 10, 20, 30, np.inf], labels=['<10', '10-<20', '20-<30', '>=30'])
    d['race_eth_group'] = d.apply(simplify_race_ethnicity, axis=1)
    if 'female' in d.columns:
        d['female'] = d['female'].where(d['female'].isin([0, 1]), np.nan)
    for col in selected_cols:
        d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0).astype(int)
        rsid = col.replace('_carrier', '')
        d[f'{rsid}_x_vitd_lt20'] = d[col] * d['vitd_lt20']
        d[f'{rsid}_x_log_vitd'] = d[col] * d['log_vitd']
        d[f'{rsid}_x_vitd_per10'] = d[col] * d['vitd_per10']
    return d

cvd = clean_model_base(cvd_raw, selected_carrier_cols)
crp = clean_model_base(crp_raw, selected_carrier_cols) if not crp_raw.empty else pd.DataFrame()

cvd_required = ['person_id', 'follow_up_years', 'event_mi_or_stroke', 'vitd_main', 'log_vitd', 'vitd_lt20', 'age_at_index']
cvd_model_base = cvd.dropna(subset=[c for c in cvd_required if c in cvd.columns]).copy()
cvd_model_base = cvd_model_base[cvd_model_base['follow_up_years'] > 0].copy()

crp_model_base = pd.DataFrame()
if not crp.empty:
    crp_required = ['person_id', 'log_crp', 'log_vitd', 'vitd_main', 'vitd_lt20', 'age_at_index']
    crp_model_base = crp.dropna(subset=[c for c in crp_required if c in crp.columns]).copy()
    crp_model_base = crp_model_base[(crp_model_base['vitd_main'] > 0) & (crp_model_base['crp_mgL'] > 0)].copy()

print('CVD model-prep base rows:', len(cvd_model_base))
print('CRP model-prep base rows:', len(crp_model_base))

# %% Cell 5
# ============================================================
# Block 3. Save model-prep datasets
# ============================================================
base_keep = [
    'person_id', 'index_date', 'vitd_main', 'vitd_ng_ml', 'log_vitd', 'vitd_per10', 'vitd_neg10',
    'vitd_lt20', 'vitd_lt10', 'vitd_group4', 'age_at_index', 'female', 'gender', 'sex_at_birth',
    'race', 'ethnicity', 'race_eth_group', 'winter', 'bmi_full', 'baseline_diabetes',
    'baseline_hypertension', 'baseline_hyperlipidemia', 'has_whole_genome_variant', 'has_array_data',
    'has_structural_variant_data'
]
cvd_keep = base_keep + [
    'follow_up_days', 'follow_up_years', 'censor_date', 'first_cvd_event_date', 'first_mi_date',
    'first_stroke_date', 'event_mi_or_stroke', 'event_mi', 'event_stroke'
]
crp_keep = base_keep + ['crp_mgL', 'log_crp', 'crp_date', 'days_before', 'ambiguous_unit', 'crp_concept_id']
interaction_cols = []
for rsid in selected_snps:
    interaction_cols.extend([f'{rsid}_carrier', f'{rsid}_x_vitd_lt20', f'{rsid}_x_log_vitd', f'{rsid}_x_vitd_per10'])

cvd_out_cols = [c for c in cvd_keep + interaction_cols if c in cvd_model_base.columns]
crp_out_cols = [c for c in crp_keep + interaction_cols if c in crp_model_base.columns]

save(cvd_model_base[cvd_out_cols], '02_cvd_model_dataset_v9.csv')
if not crp_model_base.empty:
    save(crp_model_base[crp_out_cols], '03_crp_model_dataset_v9.csv')
else:
    save(pd.DataFrame(columns=crp_out_cols), '03_crp_model_dataset_v9.csv')

# %% Cell 6
# ============================================================
# Block 4. Dataset summaries and covariate coverage
# ============================================================
def n_pct(n, denom):
    return round(100 * n / denom, 2) if denom else np.nan

summary_rows = []
summary_rows.append({'dataset': 'cvd', 'metric': 'rows', 'n': len(cvd_model_base), 'percent': 100.0})
summary_rows.append({'dataset': 'cvd', 'metric': 'events_mi_or_stroke', 'n': int(cvd_model_base['event_mi_or_stroke'].sum()), 'percent': n_pct(int(cvd_model_base['event_mi_or_stroke'].sum()), len(cvd_model_base))})
summary_rows.append({'dataset': 'cvd', 'metric': 'vitd_lt20', 'n': int(cvd_model_base['vitd_lt20'].sum()), 'percent': n_pct(int(cvd_model_base['vitd_lt20'].sum()), len(cvd_model_base))})
summary_rows.append({'dataset': 'cvd', 'metric': 'bmi_available', 'n': int(cvd_model_base['bmi_full'].notna().sum()), 'percent': n_pct(int(cvd_model_base['bmi_full'].notna().sum()), len(cvd_model_base))})
if not crp_model_base.empty:
    summary_rows.append({'dataset': 'crp', 'metric': 'rows', 'n': len(crp_model_base), 'percent': 100.0})
    summary_rows.append({'dataset': 'crp', 'metric': 'crp_gt10', 'n': int((crp_model_base['crp_mgL'] > 10).sum()), 'percent': n_pct(int((crp_model_base['crp_mgL'] > 10).sum()), len(crp_model_base))})
    summary_rows.append({'dataset': 'crp', 'metric': 'ambiguous_unit', 'n': int(crp_model_base['ambiguous_unit'].fillna(0).sum()), 'percent': n_pct(int(crp_model_base['ambiguous_unit'].fillna(0).sum()), len(crp_model_base))})
save(pd.DataFrame(summary_rows), '04_model_dataset_summary_v9.csv')

coverage_fields = [
    'vitd_main', 'log_vitd', 'vitd_lt20', 'age_at_index', 'female', 'winter', 'bmi_full',
    'race_eth_group', 'baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia',
    'follow_up_years', 'event_mi_or_stroke', 'crp_mgL', 'log_crp'
]
coverage_rows = []
for dataset_name, d in [('cvd', cvd_model_base), ('crp', crp_model_base)]:
    if d.empty:
        continue
    for field in coverage_fields:
        if field in d.columns:
            nonmissing = int(d[field].notna().sum())
            coverage_rows.append({'dataset': dataset_name, 'field': field, 'nonmissing_n': nonmissing, 'total_n': len(d), 'nonmissing_percent': n_pct(nonmissing, len(d))})
save(pd.DataFrame(coverage_rows), '05_covariate_coverage_v9.csv')

race_dist = []
for dataset_name, d in [('cvd', cvd_model_base), ('crp', crp_model_base)]:
    if d.empty or 'race_eth_group' not in d.columns:
        continue
    vc = d['race_eth_group'].fillna('Unknown/Not reported').value_counts(dropna=False)
    for level, n in vc.items():
        race_dist.append({'dataset': dataset_name, 'field': 'race_eth_group', 'level': level, 'n': int(n), 'percent': n_pct(int(n), len(d))})
race_dist_df = pd.DataFrame(race_dist)
save(race_dist_df, '06_race_eth_distribution_v9.csv')
race_qc_rows = []
if not race_dist_df.empty:
    for dataset_name, sub in race_dist_df.groupby('dataset'):
        top = sub.sort_values('percent', ascending=False).iloc[0]
        race_qc_rows.append({
            'dataset': dataset_name,
            'top_level': top['level'],
            'top_percent': top['percent'],
            'n_levels': sub['level'].nunique(),
            'qc_note': 'review if top_percent > 90; expected only for genuinely homogeneous subsets'
        })
save(pd.DataFrame(race_qc_rows), '06_race_eth_distribution_qc_v9.csv')

# %% Cell 7
# ============================================================
# Block 5. SNP-level model readiness summary
# ============================================================
def summarize_binary_cells(d, carrier_col, outcome_col=None):
    rows = []
    if d.empty or carrier_col not in d.columns:
        return pd.DataFrame()
    for (carrier, vitd_lt20), sub in d.groupby([carrier_col, 'vitd_lt20'], dropna=False):
        row = {'carrier_status': carrier, 'vitd_lt20': vitd_lt20, 'n': len(sub)}
        if outcome_col and outcome_col in sub.columns:
            row['events'] = int(pd.to_numeric(sub[outcome_col], errors='coerce').fillna(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_cat4_cells(d, carrier_col, outcome_col=None):
    rows = []
    if d.empty or carrier_col not in d.columns:
        return pd.DataFrame()
    for (carrier, group), sub in d.groupby([carrier_col, 'vitd_group4'], dropna=False):
        row = {'carrier_status': carrier, 'vitd_group4': str(group), 'n': len(sub)}
        if outcome_col and outcome_col in sub.columns:
            row['events'] = int(pd.to_numeric(sub[outcome_col], errors='coerce').fillna(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)

readiness_rows = []
cell_rows = []
for r in candidate_meta.itertuples():
    rsid = r.rsid
    carrier_col = f'{rsid}_carrier'
    if carrier_col not in cvd_model_base.columns:
        continue
    d = cvd_model_base.copy()
    carrier_n = int(d[carrier_col].sum())
    noncarrier_n = int((d[carrier_col] == 0).sum())
    carrier_events = int(d.loc[d[carrier_col].eq(1), 'event_mi_or_stroke'].sum())
    noncarrier_events = int(d.loc[d[carrier_col].eq(0), 'event_mi_or_stroke'].sum())
    bin_cells = summarize_binary_cells(d, carrier_col, 'event_mi_or_stroke')
    cat_cells = summarize_cat4_cells(d, carrier_col, 'event_mi_or_stroke')
    for _, row in bin_cells.iterrows():
        cell_rows.append({'dataset': 'cvd', 'cell_type': 'vitd_lt20_binary', 'rsid': rsid, **row.to_dict()})
    min_bin_n = int(bin_cells['n'].min()) if not bin_cells.empty else 0
    min_bin_events = int(bin_cells['events'].min()) if 'events' in bin_cells.columns and not bin_cells.empty else 0
    min_cat_n = int(cat_cells['n'].min()) if not cat_cells.empty else 0
    min_cat_events = int(cat_cells['events'].min()) if 'events' in cat_cells.columns and not cat_cells.empty else 0
    crp_n = crp_carrier_n = crp_noncarrier_n = min_crp_bin_n = 0
    if not crp_model_base.empty and carrier_col in crp_model_base.columns:
        crp_n = len(crp_model_base)
        crp_carrier_n = int(crp_model_base[carrier_col].sum())
        crp_noncarrier_n = int((crp_model_base[carrier_col] == 0).sum())
        crp_bin_cells = summarize_binary_cells(crp_model_base, carrier_col, None)
        for _, row in crp_bin_cells.iterrows():
            cell_rows.append({'dataset': 'crp', 'cell_type': 'vitd_lt20_binary', 'rsid': rsid, **row.to_dict()})
        min_crp_bin_n = int(crp_bin_cells['n'].min()) if not crp_bin_cells.empty else 0
    readiness_rows.append({
        'tier': r.tier,
        'gene': r.gene,
        'rsid': rsid,
        'label': r.label,
        'analysis_role': r.analysis_role,
        'cvd_n': len(d),
        'cvd_carriers': carrier_n,
        'cvd_noncarriers': noncarrier_n,
        'cvd_events_total': int(d['event_mi_or_stroke'].sum()),
        'cvd_events_carriers': carrier_events,
        'cvd_events_noncarriers': noncarrier_events,
        'cvd_min_binary_cell_n': min_bin_n,
        'cvd_min_binary_cell_events': min_bin_events,
        'cvd_min_vitd_cat4_cell_n': min_cat_n,
        'cvd_min_vitd_cat4_cell_events': min_cat_events,
        'crp_n': crp_n,
        'crp_carriers': crp_carrier_n,
        'crp_noncarriers': crp_noncarrier_n,
        'crp_min_binary_cell_n': min_crp_bin_n,
        'binary_interaction_ready': bool(min_bin_n >= 100 and min_bin_events >= 20),
        'vitd_cat4_sensitivity_ready': bool(min_cat_n >= 500 and min_cat_events >= 50),
        'recommended_reporting': 'main' if r.analysis_role in ['primary_tier1_vdr', 'expanded_tier2_vdr'] else ('exploratory' if r.analysis_role == 'exploratory_sparse_vdr' else 'parked'),
    })

readiness = pd.DataFrame(readiness_rows)
save(readiness, '07_snp_model_readiness_v9.csv')
save(pd.DataFrame(cell_rows), '08_interaction_binary_cell_counts_v9.csv')

complete_case_rows = []
for r in candidate_meta[candidate_meta['include_in_primary_cvd_prep']].itertuples():
    rsid = r.rsid
    carrier_col = f'{rsid}_carrier'
    cvd_binary_terms = ['follow_up_years', 'event_mi_or_stroke', 'vitd_lt20', carrier_col, f'{rsid}_x_vitd_lt20', 'age_at_index', 'female', 'winter', 'bmi_full']
    cvd_cont_terms = ['follow_up_years', 'event_mi_or_stroke', 'log_vitd', carrier_col, f'{rsid}_x_log_vitd', 'age_at_index', 'female', 'winter', 'bmi_full']
    cvd_ext_terms = cvd_binary_terms + ['race_eth_group', 'baseline_diabetes', 'baseline_hypertension', 'baseline_hyperlipidemia']
    crp_terms = ['log_crp', 'log_vitd', carrier_col, f'{rsid}_x_log_vitd', 'age_at_index', 'female', 'winter', 'bmi_full']
    for model_name, dataset_name, d, terms, event_col in [
        ('cvd_binary_core_with_bmi', 'cvd', cvd_model_base, cvd_binary_terms, 'event_mi_or_stroke'),
        ('cvd_continuous_core_with_bmi', 'cvd', cvd_model_base, cvd_cont_terms, 'event_mi_or_stroke'),
        ('cvd_binary_extended_with_bmi', 'cvd', cvd_model_base, cvd_ext_terms, 'event_mi_or_stroke'),
        ('crp_loglog_core_with_bmi', 'crp', crp_model_base, crp_terms, None),
    ]:
        if d.empty:
            complete_case_rows.append({'rsid': rsid, 'analysis_role': r.analysis_role, 'dataset': dataset_name, 'model_name': model_name, 'complete_case_n': 0, 'events': 0})
            continue
        terms_present = [c for c in terms if c in d.columns]
        m = d.dropna(subset=terms_present).copy()
        events = int(pd.to_numeric(m[event_col], errors='coerce').fillna(0).sum()) if event_col and event_col in m.columns else np.nan
        complete_case_rows.append({'rsid': rsid, 'analysis_role': r.analysis_role, 'dataset': dataset_name, 'model_name': model_name, 'complete_case_n': len(m), 'events': events})
save(pd.DataFrame(complete_case_rows), '11_model_complete_case_counts_v9.csv')

# %% Cell 8
# ============================================================
# Block 6. Modeling formula plan, assumptions, and manifest
# ============================================================
formula_rows = [
    {
        'analysis': 'CVD Cox binary vitamin D interaction core',
        'dataset': '02_cvd_model_dataset_v9.csv',
        'outcome': 'follow_up_years + event_mi_or_stroke',
        'exposure': 'vitd_lt20',
        'modifier': '<rsid>_carrier',
        'formula_template': 'Surv(follow_up_years, event_mi_or_stroke) ~ vitd_lt20 + carrier + vitd_lt20:carrier + age_at_index + female + winter + bmi_full',
        'role': 'primary model-prep target; final covariates still require explicit decision',
    },
    {
        'analysis': 'CVD Cox continuous vitamin D interaction sensitivity',
        'dataset': '02_cvd_model_dataset_v9.csv',
        'outcome': 'follow_up_years + event_mi_or_stroke',
        'exposure': 'log_vitd or vitd_per10',
        'modifier': '<rsid>_carrier',
        'formula_template': 'Surv(follow_up_years, event_mi_or_stroke) ~ log_vitd + carrier + log_vitd:carrier + age_at_index + female + winter + bmi_full',
        'role': 'continuous exposure sensitivity; do not interpret as MR',
    },
    {
        'analysis': 'CRP log-log interaction extension',
        'dataset': '03_crp_model_dataset_v9.csv',
        'outcome': 'log_crp',
        'exposure': 'log_vitd',
        'modifier': '<rsid>_carrier',
        'formula_template': 'log_crp ~ log_vitd + carrier + log_vitd:carrier + age_at_index + female + winter + bmi_full',
        'role': 'inflammation association extension; not mediation proof',
    },
    {
        'analysis': 'Extended covariate sensitivity',
        'dataset': '02_cvd_model_dataset_v9.csv or 03_crp_model_dataset_v9.csv',
        'outcome': 'same as corresponding main layer',
        'exposure': 'same as corresponding main layer',
        'modifier': '<rsid>_carrier',
        'formula_template': 'core terms + C(race_eth_group) + baseline_diabetes + baseline_hypertension + baseline_hyperlipidemia',
        'role': 'sensitivity only unless final team decision; race/ethnicity is not a substitute for genetic PCs',
    },
]
save(pd.DataFrame(formula_rows), '09_model_formula_plan_v9.csv')

assumptions = pd.DataFrame([
    {'item': 'genetic coding', 'status': 'carrier_status_only', 'note': 'cb_variant_to_person defines ALT-carrier membership; no additive 0/1/2 dosage.'},
    {'item': 'genomics denominator', 'status': 'available', 'note': 'prep_*_metadata sample_name exactly casts to person_id and defines WGS/array denominator.'},
    {'item': 'non-carrier definition', 'status': 'available_within_genomics_denominator', 'note': 'Non-carriers are denominator participants not listed as ALT carriers for the selected vid.'},
    {'item': 'population stratification', 'status': 'not_fully_solved', 'note': 'No genetic ancestry PCs found; race/ethnicity can be used only as a sensitivity/proxy, not a replacement for PCs.'},
    {'item': 'MR readiness', 'status': 'not_currently_feasible', 'note': 'Still missing dosage/hard calls, harmonized GWAS weights, ancestry PCs, QC, and relatedness.'},
])
save(assumptions, '10_model_prep_assumptions_v9.csv')

manifest = pd.DataFrame([{'output_file': p.name, 'path': str(p), 'size_bytes': p.stat().st_size} for p in sorted(MODEL_DIR.glob('*.csv'))])
save(manifest, '99_model_prep_manifest.csv')
print('DONE')
print('Model-prep outputs are in:', MODEL_DIR)
print('Upload these summary files first: 04_model_dataset_summary_v9.csv, 05_covariate_coverage_v9.csv, 06_race_eth_distribution_v9.csv, 06_race_eth_distribution_qc_v9.csv, 07_snp_model_readiness_v9.csv, 09_model_formula_plan_v9.csv, 10_model_prep_assumptions_v9.csv, 11_model_complete_case_counts_v9.csv')
