# Source companion of 17_q3_vdr_step5_remaining_validation_checks_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 2
# Cell 0. Configuration.
import itertools
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

try:
    from scipy.stats import chi2
except Exception:
    chi2 = None

RUN_ID = os.environ.get('Q3_VDR_RUN_ID', pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ'))
TIER1_SNPS = [
    {'rsid': 'rs731236', 'label': 'TaqI', 'contig': 'chr12', 'position': 47844974, 'ref': 'A', 'alt': 'G'},
    {'rsid': 'rs7975232', 'label': 'ApaI', 'contig': 'chr12', 'position': 47845054, 'ref': 'C', 'alt': 'A'},
    {'rsid': 'rs1544410', 'label': 'BsmI', 'contig': 'chr12', 'position': 47846052, 'ref': 'C', 'alt': 'T'},
    {'rsid': 'rs2228570', 'label': 'FokI', 'contig': 'chr12', 'position': 47879112, 'ref': 'A', 'alt': 'G'},
]
TIER1_RSIDS = [x['rsid'] for x in TIER1_SNPS]

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
OUTPUT_DIR = Path(os.environ.get('Q3_VDR_STEP5_VALIDATION_OUTPUT_DIR', str(DATA_DIR / 'Q3_VDR_step5_remaining_validation_outputs_2026_08_09'))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def split_project_dataset(value):
    value = str(value).strip().strip('`')
    if '.' in value:
        return value.split('.', 1)
    return '', value

CDR = os.environ.get('WORKSPACE_CDR', 'YOUR_GOOGLE_PROJECT.C2025Q4R6').strip()
CDR_PROJECT, CDR_DATASET = split_project_dataset(CDR)
PREP_CDR = os.environ.get('WORKSPACE_PREP_CDR', f'{CDR_PROJECT}.prep_{CDR_DATASET}' if CDR_PROJECT and not CDR_DATASET.startswith('prep_') else '').strip()

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

def find_first_existing(names, dirs):
    for directory in dirs:
        directory = Path(directory)
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None

candidate_dirs = [
    DATA_DIR,
    DATA_DIR / 'Q3_VDR_tier1_hail_genotype_outputs_2026_08_07',
    DATA_DIR / 'Q3_VDR_tier1_additive_genotype_outputs_2026_08_08',
    DATA_DIR / 'Q3_VDR_step5_remaining_validation_outputs_2026_08_09',
]
env_genotype = os.environ.get('Q3_VDR_GENOTYPE_WIDE_CSV', '').strip()
GENOTYPE_WIDE_PATH = Path(env_genotype).expanduser() if env_genotype else find_first_existing(['14_tier1_vdr_genotype_wide.csv'], candidate_dirs)
env_qc = os.environ.get('Q3_VDR_GENETIC_QC_CSV', '').strip()
GENETIC_QC_PATH = Path(env_qc).expanduser() if env_qc else None
env_harmonization = os.environ.get('Q3_VDR_ALLELE_HARMONIZATION_CSV', '').strip()
ALLELE_HARMONIZATION_PATH = Path(env_harmonization).expanduser() if env_harmonization else None

status_rows = []
def add_status(step, status, detail=''):
    status_rows.append({'run_id': RUN_ID, 'step': step, 'status': status, 'detail': str(detail)[:4000]})
    save(pd.DataFrame(status_rows), '17_step5_validation_status.csv')

config = pd.DataFrame([{
    'run_id': RUN_ID,
    'data_dir': str(DATA_DIR),
    'output_dir': str(OUTPUT_DIR),
    'cdr': CDR,
    'prep_cdr': PREP_CDR,
    'genotype_wide_path': str(GENOTYPE_WIDE_PATH) if GENOTYPE_WIDE_PATH else '',
    'genetic_qc_path': str(GENETIC_QC_PATH) if GENETIC_QC_PATH else '',
    'allele_harmonization_path': str(ALLELE_HARMONIZATION_PATH) if ALLELE_HARMONIZATION_PATH else '',
}])
save(config, '17_step5_validation_run_configuration.csv')

# %% Cell 3
# Cell 1. Gate plan: what must be solved before stronger wording is allowed.
gate_plan = pd.DataFrame([
    {
        'gate': 'allele_harmonization',
        'required_evidence': 'AoU REF/ALT mapped to dbSNP/Ensembl GRCh38 alleles and literature FokI/BsmI/ApaI/TaqI allele nomenclature; effect/function/risk allele explicitly identified or marked unresolved.',
        'detected_by_this_notebook': 'Writes AoU selected REF/ALT records and external search checklist; optionally reads Q3_VDR_ALLELE_HARMONIZATION_CSV if supplied.',
        'information_to_search': 'dbSNP rs pages, Ensembl variant pages, reference/alternate strand, VDR transcript context, FokI F/f or T/C mapping, BsmI/ApaI/TaqI restriction-fragment nomenclature, CAD/vitamin-D-response effect allele direction from papers.',
        'allowed_if_pass': 'SNP-specific directional wording only for SNPs with confirmed mapping.',
        'wording_if_not_pass': 'Use AoU ALT allele or additive ALT dosage as coding labels only.',
    },
    {
        'gate': 'genetic_pcs',
        'required_evidence': 'Top genetic ancestry PCs are found and joinable to person_id/sample_id in the analytic cohort.',
        'detected_by_this_notebook': 'Searches BigQuery schemas, environment variables, and optional QC file for PC-like columns.',
        'information_to_search': 'AoU genomic QC documentation; WGS/array sample QC tables; files/tables containing PC1-PC10, principal components, inferred genetic ancestry, sample_id/person_id join keys.',
        'allowed_if_pass': 'PC-adjusted genetic models can be described or run as preferred genetic adjustment.',
        'wording_if_not_pass': 'Race/ethnicity remains an imperfect proxy sensitivity; population stratification remains a limitation.',
    },
    {
        'gate': 'relatedness_unrelated_set',
        'required_evidence': 'Relatedness/kinship fields or maximal unrelated set are found and joinable to analytic person_id/sample_id.',
        'detected_by_this_notebook': 'Searches BigQuery schemas, environment variables, and optional QC file for relatedness/unrelated-set columns.',
        'information_to_search': 'AoU relatedness/kinship outputs, maximal unrelated set flags, sample QC files, kinship threshold definitions, person_id/sample_id join keys.',
        'allowed_if_pass': 'Unrelated-only sensitivity or relatedness-aware reporting can be added.',
        'wording_if_not_pass': 'Relatedness remains an unresolved limitation.',
    },
    {
        'gate': 'hwe_by_unrelated_ancestry',
        'required_evidence': 'Genotype counts and HWE tests are run within unrelated participants, ideally stratified by genetic ancestry group.',
        'detected_by_this_notebook': 'Computes pooled descriptive genotype QC from genotype wide; computes ancestry/unrelated HWE only if a QC file with ancestry and unrelated flag is supplied/found.',
        'information_to_search': 'Ancestry group assignment, unrelated-set flag, genotype call dosages, sample exclusion/QC flags, minimum group sizes.',
        'allowed_if_pass': 'HWE can be reported as ancestry/unrelated QC.',
        'wording_if_not_pass': 'Report pooled genotype counts/MAF/missingness only; do not claim ancestry/unrelated HWE passed.',
    },
    {
        'gate': 'dprime_haplotype',
        'required_evidence': 'Pairwise D-prime or phased haplotype estimates for the VDR 3-prime markers, preferably by ancestry if LD differs.',
        'detected_by_this_notebook': 'Computes genotype-correlation r2 and pairwise unphased EM D-prime from the four-SNP genotype matrix; records phased-haplotype status separately.',
        'information_to_search': 'Phased VCF/PGEN/haplotype resource, PLINK LD outputs, haplotype software availability, rs1544410-rs7975232-rs731236 LD/haplotype literature.',
        'allowed_if_pass': 'D-prime or haplotype-aware wording can be added if estimates are valid and reported.',
        'wording_if_not_pass': 'Use LD-caution wording only; do not claim phased haplotypes were tested.',
    },
])
save(gate_plan, '17_step5_unresolved_issue_gate_plan.csv')
add_status('gate_plan', 'pass', 'Five unresolved Step 5 gates recorded')

# %% Cell 4
# Cell 2. Search AoU BigQuery schemas, environment variables, and GCS hints for PC/relatedness/QC resources.
def bq_df(sql, label):
    if bigquery is None:
        return pd.DataFrame(), {'run_id': RUN_ID, 'label': label, 'status': 'skip_no_bigquery_import', 'detail': ''}
    try:
        client = bigquery.Client()
        df = client.query(sql).to_dataframe()
        return df, {'run_id': RUN_ID, 'label': label, 'status': 'pass', 'detail': f'rows={len(df)}'}
    except Exception as exc:
        return pd.DataFrame(), {'run_id': RUN_ID, 'label': label, 'status': 'fail', 'detail': repr(exc)[:4000]}

datasets_to_search = []
for value in [CDR, PREP_CDR, os.environ.get('Q3_VDR_EXTRA_BQ_DATASETS', '')]:
    for item in str(value).split(';'):
        item = item.strip()
        if item and '.' in item and item not in datasets_to_search:
            datasets_to_search.append(item)

column_terms = ['pc', 'principal', 'component', 'ancestry', 'genetic_ancestry', 'related', 'relationship', 'kinship', 'unrelated', 'sample_qc', 'variant_qc', 'call_rate', 'missingness', 'hard_filter', 'filter']
table_terms = ['pc', 'ancestry', 'related', 'relationship', 'kinship', 'unrelated', 'qc', 'sample', 'genomic', 'wgs', 'array']
query_status = []
column_hits = []
table_hits = []
for dataset in datasets_to_search:
    project, ds = split_project_dataset(dataset)
    if not project or not ds:
        continue
    col_like = ' OR '.join([f"LOWER(column_name) LIKE '%{term.lower()}%'" for term in column_terms] + [f"LOWER(table_name) LIKE '%{term.lower()}%'" for term in table_terms])
    table_like = ' OR '.join([f"LOWER(table_name) LIKE '%{term.lower()}%'" for term in table_terms])
    col_sql = f'''
    SELECT '{dataset}' AS dataset, table_name, column_name, data_type
    FROM `{project}.{ds}.INFORMATION_SCHEMA.COLUMNS`
    WHERE {col_like}
    ORDER BY table_name, column_name
    '''
    tab_sql = f'''
    SELECT '{dataset}' AS dataset, table_name, table_type
    FROM `{project}.{ds}.INFORMATION_SCHEMA.TABLES`
    WHERE {table_like}
    ORDER BY table_name
    '''
    df, st = bq_df(col_sql, f'{dataset}_pc_relatedness_qc_columns')
    query_status.append(st)
    if not df.empty:
        column_hits.append(df)
    df, st = bq_df(tab_sql, f'{dataset}_pc_relatedness_qc_tables')
    query_status.append(st)
    if not df.empty:
        table_hits.append(df)

columns_df = pd.concat(column_hits, ignore_index=True) if column_hits else pd.DataFrame(columns=['dataset', 'table_name', 'column_name', 'data_type'])
tables_df = pd.concat(table_hits, ignore_index=True) if table_hits else pd.DataFrame(columns=['dataset', 'table_name', 'table_type'])
save(pd.DataFrame(query_status), '17_bigquery_pc_relatedness_qc_search_status.csv')
save(columns_df, '17_bigquery_pc_relatedness_qc_column_hits.csv')
save(tables_df, '17_bigquery_pc_relatedness_qc_table_hits.csv')

env_keywords = ['PC', 'PRINCIPAL', 'ANCESTRY', 'RELATED', 'RELATIONSHIP', 'KINSHIP', 'UNRELATED', 'QC', 'SAMPLE', 'GENOMIC', 'WGS', 'ARRAY', 'HAIL', 'VCF', 'PLINK', 'PGEN']
env_rows = []
for key, value in sorted(os.environ.items()):
    upper = key.upper() + '=' + str(value).upper()
    if any(k in upper for k in env_keywords):
        env_rows.append({'run_id': RUN_ID, 'env_name': key, 'env_value': str(value)[:1000]})
env_df = pd.DataFrame(env_rows, columns=['run_id', 'env_name', 'env_value'])
save(env_df, '17_environment_pc_relatedness_qc_hints.csv')

gsutil_path = shutil.which('gsutil')
gcs_rows = []
if gsutil_path and not env_df.empty:
    candidate_paths = []
    for value in env_df['env_value'].dropna().astype(str):
        for token in re.split(r'[\s,;]+', value):
            token = token.strip()
            if token.startswith('gs://') and token not in candidate_paths:
                candidate_paths.append(token)
    for uri in candidate_paths[:30]:
        try:
            proc = subprocess.run([gsutil_path, 'ls', uri], text=True, capture_output=True, timeout=60)
            out = (proc.stdout or '')[:5000]
            err = (proc.stderr or '')[:1000]
            hit_text = '\n'.join([line for line in out.splitlines() if re.search(r'pc|ancestry|related|kinship|unrelated|qc|sample', line, re.I)])
            gcs_rows.append({'run_id': RUN_ID, 'uri': uri, 'status': 'pass' if proc.returncode == 0 else 'fail', 'returncode': proc.returncode, 'matching_lines': hit_text[:4000], 'stderr': err})
        except Exception as exc:
            gcs_rows.append({'run_id': RUN_ID, 'uri': uri, 'status': 'fail', 'returncode': -1, 'matching_lines': '', 'stderr': repr(exc)[:1000]})
else:
    gcs_rows.append({'run_id': RUN_ID, 'uri': '', 'status': 'skip', 'returncode': np.nan, 'matching_lines': '', 'stderr': 'gsutil unavailable or no env hints'})
save(pd.DataFrame(gcs_rows), '17_gcs_pc_relatedness_qc_hint_listing.csv')
add_status('resource_search', 'pass', f'bq_columns={len(columns_df)}; bq_tables={len(tables_df)}; env_hints={len(env_df)}')

# %% Cell 5
# Cell 3. Allele harmonization readiness package.
aou_records = pd.DataFrame([
    {
        'run_id': RUN_ID,
        'rsid': x['rsid'],
        'label': x['label'],
        'aou_contig': x['contig'],
        'aou_grch38_position': x['position'],
        'aou_ref': x['ref'],
        'aou_alt': x['alt'],
        'aou_coding': 'ALT dosage = GT.n_alt_alleles() from AoU v9 WGS split Hail MT',
        'directional_status': 'not_harmonized_to_literature_effect_or_function_allele',
    }
    for x in TIER1_SNPS
])
save(aou_records, '17_tier1_vdr_aou_ref_alt_records_for_harmonization.csv')

search_tasks = pd.DataFrame([
    {
        'rsid': 'rs2228570',
        'label': 'FokI',
        'must_find': 'Map AoU A/G GRCh38 allele to F/f or T/C literature nomenclature and to long/short VDR isoform, with strand and transcript noted.',
        'suggested_sources': 'dbSNP rs2228570; Ensembl variant page; Arai 1997; Gross 1998; Jurutka 2000; VDR polymorphism reviews.',
        'writing_if_not_found': 'FokI remains a direct functional candidate, but AoU ALT cannot be named functional/protective/risk allele.',
    },
    {
        'rsid': 'rs1544410',
        'label': 'BsmI',
        'must_find': 'Map AoU C/T to B/b nomenclature and any reported CAD/vitamin-D-response effect allele; document multiple records if present.',
        'suggested_sources': 'dbSNP rs1544410; Ensembl; Uitterlinden 2004; Fang 2005; CAD meta-analysis; supplementation response review.',
        'writing_if_not_found': 'BsmI should be written as a 3-prime VDR regional marker; ALT direction unresolved.',
    },
    {
        'rsid': 'rs7975232',
        'label': 'ApaI',
        'must_find': 'Map AoU C/A to A/a nomenclature and reported effect/risk allele direction if any.',
        'suggested_sources': 'dbSNP rs7975232; Ensembl; Uitterlinden 2004; Fang 2005; CAD meta-analysis; supplementation response review.',
        'writing_if_not_found': 'ApaI should be written as a 3-prime VDR regional marker; ALT direction unresolved.',
    },
    {
        'rsid': 'rs731236',
        'label': 'TaqI',
        'must_find': 'Map AoU A/G to T/t nomenclature and reported effect/risk allele direction if any.',
        'suggested_sources': 'dbSNP rs731236; Ensembl; Uitterlinden 2004; Fang 2005; Verbeek 1997; CAD meta-analysis.',
        'writing_if_not_found': 'TaqI should be written as a synonymous 3-prime VDR regional marker; ALT direction unresolved.',
    },
])
save(search_tasks, '17_allele_harmonization_external_search_tasks.csv')

harm_status = []
if ALLELE_HARMONIZATION_PATH and ALLELE_HARMONIZATION_PATH.exists():
    h = pd.read_csv(ALLELE_HARMONIZATION_PATH)
    for rsid in TIER1_RSIDS:
        sub = h[h.astype(str).apply(lambda row: row.str.contains(rsid, case=False, regex=False).any(), axis=1)]
        status = 'provided_review_required' if not sub.empty else 'missing_for_rsid'
        harm_status.append({'run_id': RUN_ID, 'rsid': rsid, 'status': status, 'evidence': str(ALLELE_HARMONIZATION_PATH)})
else:
    for rsid in TIER1_RSIDS:
        harm_status.append({'run_id': RUN_ID, 'rsid': rsid, 'status': 'not_provided', 'evidence': 'Set Q3_VDR_ALLELE_HARMONIZATION_CSV after manual dbSNP/Ensembl/literature harmonization.'})
save(pd.DataFrame(harm_status), '17_allele_harmonization_gate_status.csv')
add_status('allele_harmonization_package', 'pass', 'AoU REF/ALT records and external search tasks written')

# %% Cell 6
# Cell 4. Load genotype wide and optional genetic QC/ancestry/unrelated file.
genotype_status = []
geno = pd.DataFrame()
if GENOTYPE_WIDE_PATH and GENOTYPE_WIDE_PATH.exists():
    geno = pd.read_csv(GENOTYPE_WIDE_PATH)
    genotype_status.append({'run_id': RUN_ID, 'input': 'genotype_wide', 'status': 'loaded', 'path': str(GENOTYPE_WIDE_PATH), 'rows': len(geno), 'columns': len(geno.columns)})
else:
    genotype_status.append({'run_id': RUN_ID, 'input': 'genotype_wide', 'status': 'missing', 'path': str(GENOTYPE_WIDE_PATH) if GENOTYPE_WIDE_PATH else '', 'rows': 0, 'columns': 0})

if not geno.empty:
    if 'person_id' not in geno.columns and 'sample_id' in geno.columns:
        geno['person_id'] = pd.to_numeric(geno['sample_id'], errors='coerce')
    for rsid in TIER1_RSIDS:
        if rsid in geno.columns and f'{rsid}_alt_dosage' not in geno.columns:
            geno[f'{rsid}_alt_dosage'] = pd.to_numeric(geno[rsid], errors='coerce')
        elif f'{rsid}_alt_dosage' in geno.columns:
            geno[f'{rsid}_alt_dosage'] = pd.to_numeric(geno[f'{rsid}_alt_dosage'], errors='coerce')
    keep = ['person_id', 'sample_id'] + [c for c in [f'{rsid}_alt_dosage' for rsid in TIER1_RSIDS] if c in geno.columns]
    geno = geno[[c for c in keep if c in geno.columns]].copy()
save(pd.DataFrame(genotype_status), '17_genotype_wide_load_status.csv')

qc = pd.DataFrame()
qc_status = []
if GENETIC_QC_PATH and GENETIC_QC_PATH.exists():
    qc = pd.read_csv(GENETIC_QC_PATH)
    qc_status.append({'run_id': RUN_ID, 'input': 'genetic_qc', 'status': 'loaded', 'path': str(GENETIC_QC_PATH), 'rows': len(qc), 'columns': len(qc.columns)})
else:
    qc_status.append({'run_id': RUN_ID, 'input': 'genetic_qc', 'status': 'not_provided', 'path': str(GENETIC_QC_PATH) if GENETIC_QC_PATH else '', 'rows': 0, 'columns': 0})

joined = geno.copy()
join_status = 'no_qc_join'
if not geno.empty and not qc.empty:
    join_key = None
    for key in ['person_id', 'sample_id']:
        if key in geno.columns and key in qc.columns:
            join_key = key
            break
    if join_key:
        joined = geno.merge(qc, on=join_key, how='left', validate='many_to_one')
        join_status = f'joined_on_{join_key}'
    else:
        join_status = 'qc_loaded_no_common_person_or_sample_key'
qc_status.append({'run_id': RUN_ID, 'input': 'genotype_qc_join', 'status': join_status, 'path': '', 'rows': len(joined), 'columns': len(joined.columns)})
save(pd.DataFrame(qc_status), '17_genetic_qc_load_join_status.csv')

def find_cols(patterns, cols):
    hits = []
    for col in cols:
        low = col.lower()
        if any(re.search(p, low) for p in patterns):
            hits.append(col)
    return hits

pc_cols = find_cols([r'^pc\d+$', r'principal.*component', r'genetic.*pc', r'ancestry.*pc'], joined.columns)
ancestry_cols = find_cols([r'genetic.*ancestry', r'inferred.*ancestry', r'predicted.*ancestry', r'ancestry_group', r'population'], joined.columns)
related_cols = find_cols([r'unrelated', r'kinship', r'relatedness', r'related_pair', r'relationship'], joined.columns)
sample_qc_cols = find_cols([r'sample.*qc', r'qc.*pass', r'call.*rate', r'missingness', r'hard.*filter'], joined.columns)
column_hint_rows = []
for domain, cols in [('pc', pc_cols), ('ancestry', ancestry_cols), ('relatedness_unrelated', related_cols), ('sample_qc', sample_qc_cols)]:
    column_hint_rows.append({'run_id': RUN_ID, 'domain': domain, 'candidate_columns': ';'.join(cols), 'n_columns': len(cols)})
save(pd.DataFrame(column_hint_rows), '17_joined_genetic_qc_column_hints.csv')

add_status('load_genotype_and_optional_qc', 'pass' if not geno.empty else 'missing_genotype', f'geno_rows={len(geno)}; qc_rows={len(qc)}; join={join_status}')

# %% Cell 7
# Cell 5. Genotype QC and HWE checks.
def pct(n, d):
    return round(float(n) * 100.0 / float(d), 3) if d else np.nan

def hwe_summary(x):
    x = pd.to_numeric(x, errors='coerce')
    x = x[x.isin([0, 1, 2])]
    n0 = int((x == 0).sum())
    n1 = int((x == 1).sum())
    n2 = int((x == 2).sum())
    n = n0 + n1 + n2
    if n == 0:
        return {'n_called': 0, 'genotype_0': 0, 'genotype_1': 0, 'genotype_2': 0, 'alt_af': np.nan, 'maf': np.nan, 'hwe_chisq': np.nan, 'hwe_p_chisq': np.nan}
    p_alt = (n1 + 2 * n2) / (2 * n)
    exp0 = n * (1 - p_alt) ** 2
    exp1 = n * 2 * p_alt * (1 - p_alt)
    exp2 = n * p_alt ** 2
    chisq = sum((obs - exp) ** 2 / exp for obs, exp in [(n0, exp0), (n1, exp1), (n2, exp2)] if exp > 0)
    hwe_p = float(chi2.sf(chisq, 1)) if chi2 is not None and np.isfinite(chisq) else np.nan
    return {'n_called': n, 'genotype_0': n0, 'genotype_1': n1, 'genotype_2': n2, 'alt_af': p_alt, 'maf': min(p_alt, 1 - p_alt), 'hwe_chisq': chisq, 'hwe_p_chisq': hwe_p}

qc_rows = []
if not joined.empty:
    for rsid in TIER1_RSIDS:
        dosage = f'{rsid}_alt_dosage'
        if dosage not in joined.columns:
            qc_rows.append({'run_id': RUN_ID, 'scope': 'pooled_all_loaded_samples', 'group': 'all', 'rsid': rsid, 'status': 'missing_dosage_column'})
            continue
        x = joined[dosage]
        s = hwe_summary(x)
        row = {'run_id': RUN_ID, 'scope': 'pooled_all_loaded_samples', 'group': 'all', 'rsid': rsid, 'status': 'descriptive_only_unrelated_ancestry_not_required_here', 'n_total': len(joined), 'n_missing': int(pd.to_numeric(x, errors='coerce').isna().sum()), 'missing_pct': pct(int(pd.to_numeric(x, errors='coerce').isna().sum()), len(joined))}
        row.update(s)
        qc_rows.append(row)

pooled_qc = pd.DataFrame(qc_rows)
save(pooled_qc, '17_tier1_vdr_pooled_genotype_qc_hwe_descriptive.csv')

ancestry_col = os.environ.get('Q3_VDR_ANCESTRY_COL', '').strip()
unrelated_col = os.environ.get('Q3_VDR_UNRELATED_COL', '').strip()
if not ancestry_col and ancestry_cols:
    ancestry_col = ancestry_cols[0]
if not unrelated_col and related_cols:
    unrelated_candidates = [c for c in related_cols if 'unrelated' in c.lower()]
    unrelated_col = unrelated_candidates[0] if unrelated_candidates else ''

hwe_group_rows = []
hwe_gate_status = 'not_available_no_joined_ancestry_unrelated'
if not joined.empty and ancestry_col in joined.columns and unrelated_col in joined.columns:
    u = joined[unrelated_col]
    unrelated_mask = u.astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'unrelated', 'pass']) | (pd.to_numeric(u, errors='coerce') == 1)
    d = joined[unrelated_mask].copy()
    if len(d) > 0:
        for group, sub in d.groupby(ancestry_col, dropna=False):
            for rsid in TIER1_RSIDS:
                dosage = f'{rsid}_alt_dosage'
                if dosage not in sub.columns:
                    continue
                s = hwe_summary(sub[dosage])
                row = {'run_id': RUN_ID, 'scope': 'unrelated_by_ancestry', 'ancestry_col': ancestry_col, 'unrelated_col': unrelated_col, 'ancestry_group': str(group), 'rsid': rsid, 'n_total': len(sub), 'status': 'computed'}
                row.update(s)
                hwe_group_rows.append(row)
        hwe_gate_status = 'computed_unrelated_by_ancestry'
    else:
        hwe_gate_status = 'unrelated_col_found_but_no_unrelated_rows_detected'
else:
    hwe_group_rows.append({'run_id': RUN_ID, 'scope': 'unrelated_by_ancestry', 'status': hwe_gate_status, 'ancestry_col': ancestry_col, 'unrelated_col': unrelated_col})
save(pd.DataFrame(hwe_group_rows), '17_tier1_vdr_hwe_by_unrelated_ancestry_if_available.csv')
save(pd.DataFrame([{'run_id': RUN_ID, 'hwe_by_unrelated_ancestry_status': hwe_gate_status, 'ancestry_col': ancestry_col, 'unrelated_col': unrelated_col, 'note': 'Pooled HWE is descriptive only unless unrelated ancestry grouping is available.'}]), '17_hwe_unrelated_ancestry_gate_status.csv')
add_status('hwe_checks', 'pass' if not pooled_qc.empty else 'skip_no_genotype', hwe_gate_status)

# %% Cell 8
# Cell 6. Pairwise r2 and unphased EM D-prime estimates; haplotype feasibility.
def possible_diplotypes(g1, g2):
    haps = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pairs = []
    for i, h1 in enumerate(haps):
        for j, h2 in enumerate(haps[i:], start=i):
            if h1[0] + h2[0] == g1 and h1[1] + h2[1] == g2:
                pairs.append((i, j))
    return pairs

def em_pairwise_ld(g1, g2, max_iter=500, tol=1e-10):
    dat = pd.DataFrame({'g1': pd.to_numeric(g1, errors='coerce'), 'g2': pd.to_numeric(g2, errors='coerce')}).dropna()
    dat = dat[dat['g1'].isin([0, 1, 2]) & dat['g2'].isin([0, 1, 2])]
    n = len(dat)
    if n == 0:
        return {'n_pair_called': 0, 'r2_corr': np.nan, 'd_prime_em': np.nan, 'r2_em': np.nan, 'D_em': np.nan, 'p_alt_1': np.nan, 'p_alt_2': np.nan}
    p1 = float(dat['g1'].sum() / (2 * n))
    p2 = float(dat['g2'].sum() / (2 * n))
    freqs = np.array([(1 - p1) * (1 - p2), (1 - p1) * p2, p1 * (1 - p2), p1 * p2], dtype=float)
    freqs = np.clip(freqs, 1e-6, None)
    freqs = freqs / freqs.sum()
    diplotypes = [possible_diplotypes(int(a), int(b)) for a, b in dat[['g1', 'g2']].itertuples(index=False)]
    for _ in range(max_iter):
        counts = np.zeros(4, dtype=float)
        for pairs in diplotypes:
            probs = []
            for i, j in pairs:
                prob = freqs[i] * freqs[j] * (2.0 if i != j else 1.0)
                probs.append(prob)
            total = sum(probs)
            if total <= 0:
                continue
            for (i, j), prob in zip(pairs, probs):
                w = prob / total
                counts[i] += w
                counts[j] += w
        new_freqs = counts / counts.sum()
        if np.max(np.abs(new_freqs - freqs)) < tol:
            freqs = new_freqs
            break
        freqs = new_freqs
    h00, h01, h10, h11 = freqs
    pA = h10 + h11
    pB = h01 + h11
    D = h11 - pA * pB
    denom = pA * (1 - pA) * pB * (1 - pB)
    r2_em = (D * D / denom) if denom > 0 else np.nan
    if D >= 0:
        Dmax = min(pA * (1 - pB), (1 - pA) * pB)
    else:
        Dmax = min(pA * pB, (1 - pA) * (1 - pB))
    dprime = D / Dmax if Dmax > 0 else np.nan
    corr = dat['g1'].corr(dat['g2'])
    return {
        'n_pair_called': n,
        'r2_corr': corr ** 2 if pd.notna(corr) else np.nan,
        'd_prime_em': dprime,
        'r2_em': r2_em,
        'D_em': D,
        'p_alt_1': pA,
        'p_alt_2': pB,
        'hap_00_ref_ref': h00,
        'hap_01_ref_alt2': h01,
        'hap_10_alt1_ref': h10,
        'hap_11_alt_alt': h11,
    }

ld_rows = []
if not joined.empty:
    for rsid1, rsid2 in itertools.combinations(TIER1_RSIDS, 2):
        c1 = f'{rsid1}_alt_dosage'
        c2 = f'{rsid2}_alt_dosage'
        if c1 not in joined.columns or c2 not in joined.columns:
            ld_rows.append({'run_id': RUN_ID, 'snp1': rsid1, 'snp2': rsid2, 'status': 'missing_dosage_column'})
            continue
        row = {'run_id': RUN_ID, 'snp1': rsid1, 'snp2': rsid2, 'status': 'computed_unphased_em'}
        row.update(em_pairwise_ld(joined[c1], joined[c2]))
        ld_rows.append(row)
else:
    ld_rows.append({'run_id': RUN_ID, 'snp1': '', 'snp2': '', 'status': 'skip_no_genotype'})
ld_df = pd.DataFrame(ld_rows)
save(ld_df, '17_tier1_vdr_pairwise_ld_r2_dprime_unphased_em.csv')

three_prime = ['rs731236', 'rs7975232', 'rs1544410']
complete_n = 0
if not joined.empty and all(f'{x}_alt_dosage' in joined.columns for x in three_prime):
    complete_n = int(joined[[f'{x}_alt_dosage' for x in three_prime]].dropna().shape[0])
hap_status = pd.DataFrame([{
    'run_id': RUN_ID,
    'haplotype_set': 'VDR_3prime_rs731236_rs7975232_rs1544410',
    'required_snps': ';'.join(three_prime),
    'n_complete_unphased_genotypes': complete_n,
    'pairwise_dprime_available': bool((ld_df['status'] == 'computed_unphased_em').any()) if 'status' in ld_df.columns else False,
    'phased_haplotype_available': False,
    'status': 'pairwise_dprime_estimated_from_unphased_genotypes_only' if complete_n > 0 else 'missing_genotype',
    'note': 'This does not replace phased haplotype analysis. Use only for LD caution unless a phased resource or validated haplotype method is added.',
}])
save(hap_status, '17_tier1_vdr_haplotype_feasibility_status.csv')
add_status('ld_dprime_haplotype_checks', 'pass' if not ld_df.empty else 'skip', hap_status.loc[0, 'status'])

# %% Cell 9
# Cell 7. Final Step 5 gating matrix.
def has_any_pc_resource(columns_df, env_df):
    bq_pc = False
    if not columns_df.empty:
        c = columns_df['column_name'].astype(str).str.lower()
        bq_pc = c.str.match(r'^pc\d+$').any() or c.str.contains('principal.*component|genetic.*pc|ancestry.*pc', regex=True).any()
    env_pc = False
    if not env_df.empty:
        e = (env_df['env_name'].astype(str) + '=' + env_df['env_value'].astype(str)).str.lower()
        env_pc = e.str.contains('principal|pc1|pc_1|genetic.*pc|ancestry.*pc', regex=True).any()
    return bool(bq_pc or env_pc)

def has_related_resource(columns_df, env_df):
    bq_rel = False
    if not columns_df.empty:
        c = (columns_df['table_name'].astype(str) + '.' + columns_df['column_name'].astype(str)).str.lower()
        bq_rel = c.str.contains('kinship|relatedness|unrelated|related_pair|relationship', regex=True).any()
    env_rel = False
    if not env_df.empty:
        e = (env_df['env_name'].astype(str) + '=' + env_df['env_value'].astype(str)).str.lower()
        env_rel = e.str.contains('kinship|relatedness|unrelated|related_pair|relationship', regex=True).any()
    return bool(bq_rel or env_rel)

def safe_read_csv(filename, columns=None):
    path = OUTPUT_DIR / filename
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])

columns_df = safe_read_csv('17_bigquery_pc_relatedness_qc_column_hits.csv', ['dataset', 'table_name', 'column_name', 'data_type'])
env_df = safe_read_csv('17_environment_pc_relatedness_qc_hints.csv', ['run_id', 'env_name', 'env_value'])
harm = safe_read_csv('17_allele_harmonization_gate_status.csv', ['run_id', 'rsid', 'status'])
hwe_gate = safe_read_csv('17_hwe_unrelated_ancestry_gate_status.csv', ['run_id', 'hwe_by_unrelated_ancestry_status'])
hap_gate = safe_read_csv('17_tier1_vdr_haplotype_feasibility_status.csv', ['run_id', 'pairwise_dprime_available', 'phased_haplotype_available'])

allele_pass = (not harm.empty) and harm['status'].astype(str).str.contains('confirmed|pass|complete', case=False, regex=True).all()
pc_candidate = has_any_pc_resource(columns_df, env_df)
pc_joined = len(pc_cols) > 0 if 'pc_cols' in globals() else False
related_candidate = has_related_resource(columns_df, env_df)
related_joined = len(related_cols) > 0 if 'related_cols' in globals() else False
hwe_pass = (not hwe_gate.empty) and hwe_gate['hwe_by_unrelated_ancestry_status'].astype(str).str.contains('computed_unrelated_by_ancestry').any()
dprime_available = (not hap_gate.empty) and bool(hap_gate.get('pairwise_dprime_available', pd.Series([False])).iloc[0])
phased_available = (not hap_gate.empty) and bool(hap_gate.get('phased_haplotype_available', pd.Series([False])).iloc[0])

gate_rows = [
    {
        'run_id': RUN_ID,
        'gate': 'allele_harmonization',
        'status': 'pass' if allele_pass else 'not_solved',
        'evidence': '17_allele_harmonization_gate_status.csv; 17_allele_harmonization_external_search_tasks.csv',
        'q3_writing_decision': 'Directional allele wording allowed only if pass; otherwise ALT remains a coding label.',
    },
    {
        'run_id': RUN_ID,
        'gate': 'genetic_pcs',
        'status': 'joinable' if pc_joined else ('candidate_found_not_joined' if pc_candidate else 'not_found'),
        'evidence': '17_bigquery_pc_relatedness_qc_column_hits.csv; 17_environment_pc_relatedness_qc_hints.csv; 17_joined_genetic_qc_column_hints.csv',
        'q3_writing_decision': 'PC-adjusted wording requires joinable PCs; otherwise population stratification limitation remains.',
    },
    {
        'run_id': RUN_ID,
        'gate': 'relatedness_unrelated_set',
        'status': 'joinable' if related_joined else ('candidate_found_not_joined' if related_candidate else 'not_found'),
        'evidence': '17_bigquery_pc_relatedness_qc_column_hits.csv; 17_environment_pc_relatedness_qc_hints.csv; 17_joined_genetic_qc_column_hints.csv',
        'q3_writing_decision': 'Unrelated-only/HWE wording requires joinable unrelated set; otherwise relatedness limitation remains.',
    },
    {
        'run_id': RUN_ID,
        'gate': 'hwe_by_unrelated_ancestry',
        'status': 'pass' if hwe_pass else 'not_solved_descriptive_pooled_only',
        'evidence': '17_tier1_vdr_pooled_genotype_qc_hwe_descriptive.csv; 17_tier1_vdr_hwe_by_unrelated_ancestry_if_available.csv',
        'q3_writing_decision': 'Do not claim ancestry/unrelated HWE passed unless this gate passes.',
    },
    {
        'run_id': RUN_ID,
        'gate': 'dprime_haplotype',
        'status': 'phased_haplotype_available' if phased_available else ('pairwise_dprime_estimated_unphased' if dprime_available else 'not_available'),
        'evidence': '17_tier1_vdr_pairwise_ld_r2_dprime_unphased_em.csv; 17_tier1_vdr_haplotype_feasibility_status.csv',
        'q3_writing_decision': 'Pairwise D-prime can support LD caution if estimated; phased haplotype wording requires phased haplotype availability.',
    },
]
gating = pd.DataFrame(gate_rows)
save(gating, '17_step5_final_gating_matrix.csv')

note_lines = [
        '# Q3 VDR Step 5 Remaining Validation Summary',
        '',
        f'Run ID: `{RUN_ID}`',
        '',
        'Conservative Q3 final wording can proceed without all five gates passing, as long as unresolved gates are explicitly stated as limitations.',
        '',
        'Stronger wording requires gate-specific evidence:',
        '',
    ]
for _, row in gating.iterrows():
    note_lines.append(f"- {row['gate']}: {row['status']} -> {row['q3_writing_decision']}")
note_lines.extend([
        '',
        'Minimum external/manual information still needed for upgrades:',
        '',
        '- dbSNP/Ensembl/literature mapping from AoU REF/ALT to FokI/BsmI/ApaI/TaqI allele nomenclature and effect/function direction.',
        '- Joinable genetic PC resource with person_id or sample_id.',
        '- Joinable relatedness/kinship or maximal unrelated set resource.',
        '- Ancestry group and unrelated flag for HWE by ancestry.',
        '- Phased haplotype resource or validated haplotype method if haplotype claims are desired.',
    ])
save_text('\n'.join(note_lines), '17_step5_remaining_validation_summary.md')

manifest_rows = []
for path in sorted(OUTPUT_DIR.glob('17_*')):
    if path.is_file():
        manifest_rows.append({'filename': path.name, 'path': str(path), 'size_bytes': path.stat().st_size})
save(pd.DataFrame(manifest_rows), '99_step5_remaining_validation_manifest.csv')
add_status('final_gating_matrix', 'pass', f'gates={len(gating)}; outputs={len(manifest_rows)}')
