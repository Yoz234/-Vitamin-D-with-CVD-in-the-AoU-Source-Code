# Source companion of 14_q3_vdr_hail_stage_once_then_export_self_contained.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Shell/magic cells, if any, require IPython.

# %% Cell 1
# Cell 0. Fixed configuration and local model-prep person_id loading.
import os
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# User-verified writable GCS prefix. Replace here if a new workspace/prefix is used.
os.environ.setdefault('Q3_VDR_ANALYTIC_PER_SNP_EXPORT_URI_PREFIX', (os.environ["WORKSPACE_BUCKET"].rstrip("/") + "/q3_vdr_hail_exports"))
GCS_PREFIX = os.environ['Q3_VDR_ANALYTIC_PER_SNP_EXPORT_URI_PREFIX'].rstrip('/')
if not GCS_PREFIX.startswith('gs://'):
    raise RuntimeError('Q3_VDR_ANALYTIC_PER_SNP_EXPORT_URI_PREFIX must be a writable gs:// prefix.')

CONTROLLED_WGS_SPLIT_MT_PATH = 'gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/acaf_threshold/splitMT/hail.mt'

TARGET_SNPS = [
    {'rsid': 'rs731236', 'label': 'TaqI', 'contig': '12', 'position': 47844974, 'ref': 'A', 'alt': 'G', 'expected_vid': '12-47844974-A-G'},
    {'rsid': 'rs7975232', 'label': 'ApaI', 'contig': '12', 'position': 47845054, 'ref': 'C', 'alt': 'A', 'expected_vid': '12-47845054-C-A'},
    {'rsid': 'rs1544410', 'label': 'BsmI', 'contig': '12', 'position': 47846052, 'ref': 'C', 'alt': 'T', 'expected_vid': '12-47846052-C-T'},
    {'rsid': 'rs2228570', 'label': 'FokI', 'contig': '12', 'position': 47879112, 'ref': 'A', 'alt': 'G', 'expected_vid': '12-47879112-A-G'},
]
TARGET_RSIDS = [x['rsid'] for x in TARGET_SNPS]
TARGET_POSITIONS = {int(x['position']) for x in TARGET_SNPS}
TARGET_VID_TO_RSID = {x['expected_vid']: x['rsid'] for x in TARGET_SNPS}
TARGET_RSID_TO_VID = {x['rsid']: x['expected_vid'] for x in TARGET_SNPS}

def infer_default_data_dir():
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / 'data',
        cwd.parent / 'data',
        Path('/home/dataproc/workspace') / 'VDR',
        Path('/home/jupyter/workspace') / 'VDR',
        Path('/home/jupyter/workspace') / 'VDR' / 'data',
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            pass
    return cwd

DATA_DIR = Path(os.environ.get('Q3_VDR_DATA_DIR', str(infer_default_data_dir()))).expanduser()
OUTPUT_DIR = Path(os.environ.get('Q3_VDR_HAIL_OUTPUT_DIR', str(DATA_DIR / 'Q3_VDR_tier1_hail_genotype_outputs_2026_08_07'))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERWRITE_OUTPUTS = os.environ.get('Q3_VDR_OVERWRITE_OUTPUTS', 'true').strip().lower() in {'1', 'true', 'yes', 'y'}
RUN_TAG = os.environ.get('Q3_VDR_STAGE_ONCE_TAG', pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S'))
STAGING_MT_URI = f'{GCS_PREFIX}/q3_vdr_4snp_analytic_{RUN_TAG}.mt'
ANALYTIC_IDS_URI = f'{GCS_PREFIX}/14_hail_block5e_analytic_person_ids_{RUN_TAG}.tsv'
LONG_STAGE_EXPORT_URI = f'{GCS_PREFIX}/14_tier1_vdr_genotype_long_stage_{RUN_TAG}.tsv'

def infer_google_project():
    for key in ['GOOGLE_PROJECT', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT', 'PROJECT_ID', 'WORKSPACE_NAMESPACE']:
        value = os.environ.get(key, '').strip()
        if value:
            return value
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], check=False, capture_output=True, text=True, timeout=15)
        value = (result.stdout or '').strip()
        if value and value != '(unset)':
            return value
    except Exception:
        pass
    return ''

GOOGLE_PROJECT = infer_google_project()
WORKSPACE_BUCKET = os.environ.get('WORKSPACE_BUCKET', '').strip()
HAIL_TMP_DIR = os.environ.get('Q3_VDR_HAIL_TMP_DIR', '').strip()

status_rows = []

def save(df, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f'{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false')
    df.to_csv(path, index=False)
    print(f'Saved {filename}: {len(df):,} rows')
    return path

def save_text(text, filename):
    path = OUTPUT_DIR / filename
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f'{path} exists and Q3_VDR_OVERWRITE_OUTPUTS is false')
    path.write_text(str(text), encoding='utf-8')
    print(f'Saved {filename}: {len(str(text)):,} chars')
    return path

def add_status(step, status, detail):
    row = {'step': step, 'status': status, 'detail': str(detail)}
    status_rows.append(row)
    save(pd.DataFrame(status_rows), '14_hail_stage_once_status.csv')
    return row

def pct(n, d):
    return round(float(n) / float(d) * 100, 3) if d else np.nan

def run_cmd(args, label):
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    detail = (result.stdout or '') + (result.stderr or '')
    if result.returncode != 0:
        save_text(detail, f'14_hail_stage_once_{label}_failure.txt')
        raise RuntimeError(f'{label} failed with exit code {result.returncode}. See 14_hail_stage_once_{label}_failure.txt')
    return detail.strip()

def resolve_model_prep_dir():
    manual = os.environ.get('Q3_VDR_MODEL_OUTPUT_DIR', '').strip()
    candidate_roots = []
    if manual:
        candidate_roots.append(Path(manual).expanduser())
    for root in [DATA_DIR, DATA_DIR.parent, Path.cwd(), Path.cwd().parent, Path('/home/dataproc/workspace') / 'VDR', Path('/home/jupyter/workspace') / 'VDR']:
        candidate_roots.extend([root / 'Q3_VDR_v9_model_prep_outputs', root / 'prepresult', root])
    seen = set()
    for candidate in candidate_roots:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists() and any((candidate / x).exists() for x in ['02_cvd_model_dataset_v9.csv', '03_crp_model_dataset_v9.csv']):
                return candidate
        except Exception:
            pass
    return Path(manual).expanduser() if manual else (DATA_DIR / 'Q3_VDR_v9_model_prep_outputs')

def read_person_ids(path, label):
    row = {'label': label, 'path': str(path), 'exists': path.exists(), 'rows': 0, 'person_id_n': 0, 'status': 'missing'}
    if not path.exists():
        return [], row
    try:
        df = pd.read_csv(path, usecols=lambda c: c == 'person_id', low_memory=False)
        row['rows'] = len(df)
        if 'person_id' not in df.columns:
            row['status'] = 'missing_person_id_column'
            return [], row
        ids = pd.to_numeric(df['person_id'], errors='coerce').dropna().astype('int64').astype(str).drop_duplicates().tolist()
        row['person_id_n'] = len(ids)
        row['status'] = 'pass' if ids else 'empty_person_id'
        return ids, row
    except Exception:
        row['status'] = 'fail'
        row['detail'] = traceback.format_exc()[:2000]
        return [], row

MODEL_PREP_DIR = resolve_model_prep_dir()
cvd_ids, cvd_row = read_person_ids(MODEL_PREP_DIR / '02_cvd_model_dataset_v9.csv', 'CVD model-prep dataset')
crp_ids, crp_row = read_person_ids(MODEL_PREP_DIR / '03_crp_model_dataset_v9.csv', 'CRP model-prep dataset')
ANALYTIC_IDS = sorted(set(cvd_ids).union(crp_ids), key=lambda x: int(x))
save(pd.DataFrame([cvd_row, crp_row]), '14_hail_stage_once_model_prep_person_id_status.csv')
save(pd.DataFrame([{
    'model_prep_dir': str(MODEL_PREP_DIR),
    'cvd_person_id_n': len(cvd_ids),
    'crp_person_id_n': len(crp_ids),
    'analytic_person_id_union_n': len(ANALYTIC_IDS),
    'run_tag': RUN_TAG,
    'gcs_prefix': GCS_PREFIX,
    'controlled_wgs_split_mt_path': CONTROLLED_WGS_SPLIT_MT_PATH,
    'staging_mt_uri': STAGING_MT_URI,
    'analytic_ids_uri': ANALYTIC_IDS_URI,
    'long_stage_export_uri': LONG_STAGE_EXPORT_URI,
    'workspace_bucket_env': WORKSPACE_BUCKET,
    'google_project': GOOGLE_PROJECT,
}]), '14_hail_stage_once_run_configuration.csv')
save(pd.DataFrame(TARGET_SNPS), '14_tier1_vdr_target_snp_list.csv')
if not ANALYTIC_IDS:
    raise RuntimeError(f'No analytic person_id values found in {MODEL_PREP_DIR}.')
print('Output dir:', OUTPUT_DIR)
print('GCS prefix:', GCS_PREFIX)
print('Analytic person IDs:', len(ANALYTIC_IDS))

# %% Cell 2
# Cell 1. Hail init and lazy read of the AoU v9 WGS split MatrixTable.
import hail as hl

try:
    hl.stop()
except Exception:
    pass

spark_conf = {
    'spark.hadoop.fs.gs.impl': 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem',
    'spark.hadoop.fs.AbstractFileSystem.gs.impl': 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS',
}
if GOOGLE_PROJECT:
    spark_conf.update({
        'spark.hadoop.fs.gs.requester.pays.mode': 'CUSTOM',
        'spark.hadoop.fs.gs.requester.pays.project.id': GOOGLE_PROJECT,
        'spark.hadoop.fs.gs.requester.pays.buckets': 'vwb-aou-datasets-controlled,fc-aou-datasets-controlled,genomics-public-data',
    })

init_kwargs = {'quiet': True, 'spark_conf': spark_conf}
if HAIL_TMP_DIR:
    init_kwargs['tmp_dir'] = HAIL_TMP_DIR
hl.init(**init_kwargs)
try:
    hl.default_reference('GRCh38')
except Exception:
    pass
add_status('hail_init', 'pass', f'hail_version={hl.__version__}; requester_project={GOOGLE_PROJECT}; spark_conf_keys={";".join(sorted(spark_conf))}')

gcs_probe_rows = []
try:
    from pyspark import SparkContext
    sc = SparkContext._active_spark_context
except Exception as sc_exc:
    sc = None
    gcs_probe_rows.append({'check': 'spark_context', 'status': 'fail', 'detail': repr(sc_exc)[:1000]})
if sc is None:
    gcs_probe_rows.append({'check': 'spark_context', 'status': 'missing', 'detail': 'No active SparkContext after hl.init'})
else:
    hconf = sc._jsc.hadoopConfiguration()
    for key in ['fs.gs.impl', 'fs.AbstractFileSystem.gs.impl', 'fs.gs.requester.pays.mode', 'fs.gs.requester.pays.project.id', 'fs.gs.requester.pays.buckets']:
        try:
            gcs_probe_rows.append({'check': key, 'status': 'value', 'detail': str(hconf.get(key))})
        except Exception as key_exc:
            gcs_probe_rows.append({'check': key, 'status': 'fail', 'detail': repr(key_exc)[:1000]})
save(pd.DataFrame(gcs_probe_rows), '14_hail_stage_once_gcs_connector_probe.csv')

mt = hl.read_matrix_table(CONTROLLED_WGS_SPLIT_MT_PATH)
_ = str(mt.row.dtype)
add_status('read_controlled_wgs_split_mt', 'pass_lazy_schema_only', CONTROLLED_WGS_SPLIT_MT_PATH)
print('Read controlled WGS split MT lazily:', CONTROLLED_WGS_SPLIT_MT_PATH)

# %% Cell 3
# Cell 2. Stage once: subset controlled WGS MT to 4 SNPs x analytic persons, then write a small MT to user GCS.
try:
    local_ids_path = OUTPUT_DIR / f'14_hail_stage_once_analytic_person_ids_{RUN_TAG}.tsv'
    pd.DataFrame({'person_id_string': ANALYTIC_IDS}).to_csv(local_ids_path, sep='\t', index=False)
    run_cmd(['gsutil', 'cp', str(local_ids_path), ANALYTIC_IDS_URI], 'copy_analytic_ids_to_gcs')
    add_status('copy_analytic_ids_to_gcs', 'pass', ANALYTIC_IDS_URI)

    ids_ht = hl.import_table(ANALYTIC_IDS_URI, types={'person_id_string': hl.tstr}, impute=False, key='person_id_string')
    target_min = min(x['position'] for x in TARGET_SNPS) - 100
    target_max = max(x['position'] for x in TARGET_SNPS) + 100
    rg = mt.locus.dtype.reference_genome
    contig = 'chr12' if 'chr12' in set(rg.contigs) else '12'
    interval = hl.Interval(
        hl.Locus(contig, target_min, reference_genome=rg),
        hl.Locus(contig, target_max, reference_genome=rg),
        includes_start=True,
        includes_end=True,
    )
    position_literal = hl.literal(TARGET_POSITIONS)
    target_vid_literal = hl.literal(TARGET_VID_TO_RSID)

    mt_region = hl.filter_intervals(mt, [interval])
    mt_target = mt_region.filter_rows(position_literal.contains(mt_region.locus.position) & (hl.len(mt_region.alleles) >= 2))
    mt_target = mt_target.annotate_rows(
        expected_vid=hl.str(mt_target.locus.contig).replace('chr', '') + '-' + hl.str(mt_target.locus.position) + '-' + mt_target.alleles[0] + '-' + mt_target.alleles[1]
    )
    mt_target = mt_target.annotate_rows(target_rsid=target_vid_literal.get(mt_target.expected_vid))
    mt_target = mt_target.filter_rows(hl.is_defined(mt_target.target_rsid))

    col_fields = set(mt_target.col)
    col_key_fields = list(mt_target.col_key)
    sample_source = next((f for f in ['s', 'sample_id', 'person_id', 'sample_name'] if f in col_fields), None)
    if sample_source is None:
        sample_source = col_key_fields[0] if col_key_fields else None
    if not sample_source:
        raise RuntimeError('No joinable sample/person column found in Hail column fields.')

    entry_fields = set(mt_target.entry)
    if 'GT' in entry_fields:
        dosage_expr = mt_target.GT.n_alt_alleles()
        dosage_source = 'GT.n_alt_alleles'
    elif 'LGT' in entry_fields:
        dosage_expr = mt_target.LGT.n_alt_alleles()
        dosage_source = 'LGT.n_alt_alleles'
    elif 'DS' in entry_fields:
        dosage_expr = mt_target.DS
        dosage_source = 'DS'
    else:
        raise RuntimeError(f'No usable genotype dosage field found. Entry fields: {sorted(entry_fields)}')

    mt_target = mt_target.annotate_entries(q3_alt_dosage=hl.float64(dosage_expr))
    mt_stage = mt_target.filter_cols(hl.is_defined(ids_ht[hl.str(mt_target[sample_source])]))
    mt_stage = mt_stage.annotate_cols(sample_id=hl.str(mt_stage[sample_source]), person_id_string=hl.str(mt_stage[sample_source]))
    mt_stage = mt_stage.select_rows('target_rsid', 'expected_vid').select_cols('sample_id', 'person_id_string').select_entries('q3_alt_dosage')

    save(pd.DataFrame([{
        'run_tag': RUN_TAG,
        'controlled_wgs_split_mt_path': CONTROLLED_WGS_SPLIT_MT_PATH,
        'analytic_ids_uri': ANALYTIC_IDS_URI,
        'staging_mt_uri': STAGING_MT_URI,
        'sample_source': sample_source,
        'dosage_source': dosage_source,
        'target_positions': ';'.join(map(str, sorted(TARGET_POSITIONS))),
        'note': 'This write is the only intended heavy action against the controlled WGS split MT.',
    }]), '14_hail_stage_once_staging_plan.csv')
    save_text(STAGING_MT_URI, '14_hail_stage_once_staging_mt_uri.txt')

    mt_stage.write(STAGING_MT_URI, overwrite=True)
    add_status('write_staging_matrix_table', 'pass', STAGING_MT_URI)
    print('Staged small MT:', STAGING_MT_URI)
except Exception:
    tb = traceback.format_exc()
    save_text(tb, '14_hail_stage_once_cell2_staging_failure.txt')
    add_status('write_staging_matrix_table', 'fail', tb[:2000])
    raise

# %% Cell 4
# Cell 3. Export long genotype entries from the small staging MT, then build local long/wide CSVs.
try:
    if 'STAGING_MT_URI' not in globals() or not STAGING_MT_URI:
        STAGING_MT_URI = (OUTPUT_DIR / '14_hail_stage_once_staging_mt_uri.txt').read_text(encoding='utf-8').strip()
    if 'LONG_STAGE_EXPORT_URI' not in globals() or not LONG_STAGE_EXPORT_URI:
        LONG_STAGE_EXPORT_URI = f'{GCS_PREFIX}/14_tier1_vdr_genotype_long_stage_{RUN_TAG}.tsv'

    mt_small = hl.read_matrix_table(STAGING_MT_URI)
    entries_ht = mt_small.entries()
    entries_export = entries_ht.select(
        sample_id=entries_ht.sample_id,
        person_id_string=entries_ht.person_id_string,
        rsid=entries_ht.target_rsid,
        expected_vid=entries_ht.expected_vid,
        q3_alt_dosage=entries_ht.q3_alt_dosage,
    )
    entries_export.export(LONG_STAGE_EXPORT_URI)
    add_status('export_long_from_staging_mt', 'pass', LONG_STAGE_EXPORT_URI)

    local_stage_tsv = OUTPUT_DIR / f'14_tier1_vdr_genotype_long_stage_{RUN_TAG}.tsv'
    run_cmd(['gsutil', 'cp', LONG_STAGE_EXPORT_URI, str(local_stage_tsv)], 'copy_long_stage_tsv_from_gcs')

    long_raw = pd.read_csv(local_stage_tsv, sep='\t', low_memory=False)
    expected_cols = ['sample_id', 'person_id_string', 'rsid', 'expected_vid', 'q3_alt_dosage']
    missing_cols = [c for c in expected_cols if c not in long_raw.columns]
    if missing_cols:
        raise RuntimeError(f'Long stage TSV is missing columns: {missing_cols}')
    long = long_raw[expected_cols].copy()
    long['q3_alt_dosage'] = pd.to_numeric(long['q3_alt_dosage'], errors='coerce')
    long['person_id'] = pd.to_numeric(long['person_id_string'], errors='coerce').astype('Int64')
    long = long[['sample_id', 'person_id', 'rsid', 'expected_vid', 'q3_alt_dosage']].copy()
    long = long.sort_values(['person_id', 'rsid'], na_position='last').reset_index(drop=True)
    save(long, '14_tier1_vdr_genotype_long.csv')

    wide = long.pivot_table(index=['sample_id', 'person_id'], columns='rsid', values='q3_alt_dosage', aggfunc='first').reset_index()
    wide.columns.name = None
    for rsid in TARGET_RSIDS:
        if rsid not in wide.columns:
            wide[rsid] = np.nan
    wide = wide[['sample_id', 'person_id'] + TARGET_RSIDS].copy()
    wide = wide.drop_duplicates(subset=['person_id'], keep='first').sort_values('person_id', na_position='last').reset_index(drop=True)
    save(wide, '14_tier1_vdr_genotype_wide.csv')

    save(pd.DataFrame([{
        'run_tag': RUN_TAG,
        'staging_mt_uri': STAGING_MT_URI,
        'long_stage_export_uri': LONG_STAGE_EXPORT_URI,
        'local_stage_tsv': str(local_stage_tsv),
        'long_rows': len(long),
        'wide_rows': len(wide),
        'export_method': 'entries_table_export_from_small_staging_mt',
    }]), '14_hail_stage_once_export_status.csv')
    add_status('build_local_long_wide_csv', 'pass', f'long_rows={len(long)}; wide_rows={len(wide)}')
    print('Built:', OUTPUT_DIR / '14_tier1_vdr_genotype_wide.csv')
except Exception:
    tb = traceback.format_exc()
    save_text(tb, '14_hail_stage_once_cell3_export_failure.txt')
    add_status('export_from_staging_matrix_table', 'fail', tb[:2000])
    raise

# %% Cell 5
# Cell 4. Local validation and manifest. No Hail/Spark action is used here.
wide_path = OUTPUT_DIR / '14_tier1_vdr_genotype_wide.csv'
long_path = OUTPUT_DIR / '14_tier1_vdr_genotype_long.csv'
validation_rows = []

if not wide_path.exists():
    validation_rows.append({'check': 'wide_csv_exists', 'status': 'fail', 'detail': str(wide_path)})
    save(pd.DataFrame(validation_rows), '14_hail_stage_once_local_validation.csv')
    raise FileNotFoundError(wide_path)
if not long_path.exists():
    validation_rows.append({'check': 'long_csv_exists', 'status': 'fail', 'detail': str(long_path)})
    save(pd.DataFrame(validation_rows), '14_hail_stage_once_local_validation.csv')
    raise FileNotFoundError(long_path)

wide = pd.read_csv(wide_path, low_memory=False)
long = pd.read_csv(long_path, low_memory=False)
required_cols = ['sample_id', 'person_id'] + TARGET_RSIDS
missing_cols = [c for c in required_cols if c not in wide.columns]
validation_rows.append({'check': 'wide_required_columns', 'status': 'pass' if not missing_cols else 'fail', 'detail': ';'.join(missing_cols)})
for rsid in TARGET_RSIDS:
    value = pd.to_numeric(wide[rsid], errors='coerce') if rsid in wide.columns else pd.Series(dtype=float)
    validation_rows.append({'check': f'{rsid}_nonmissing', 'status': 'pass' if int(value.notna().sum()) > 0 else 'fail', 'detail': int(value.notna().sum())})
complete_four = int(wide[TARGET_RSIDS].apply(pd.to_numeric, errors='coerce').dropna().shape[0]) if not missing_cols else 0
validation_rows.append({'check': 'wide_row_count', 'status': 'pass' if len(wide) > 0 else 'fail', 'detail': len(wide)})
validation_rows.append({'check': 'long_row_count', 'status': 'pass' if len(long) > 0 else 'fail', 'detail': len(long)})
validation_rows.append({'check': 'complete_four_snp_count', 'status': 'pass' if complete_four > 0 else 'needs_review', 'detail': complete_four})
save(pd.DataFrame(validation_rows), '14_hail_stage_once_local_validation.csv')

sample_summary = pd.DataFrame([{
    'n_wide_rows': len(wide),
    'n_long_rows': len(long),
    'n_complete_four_snp_genotype': complete_four,
    'rs731236_nonmissing': int(pd.to_numeric(wide.get('rs731236', pd.Series(dtype=float)), errors='coerce').notna().sum()),
    'rs7975232_nonmissing': int(pd.to_numeric(wide.get('rs7975232', pd.Series(dtype=float)), errors='coerce').notna().sum()),
    'rs1544410_nonmissing': int(pd.to_numeric(wide.get('rs1544410', pd.Series(dtype=float)), errors='coerce').notna().sum()),
    'rs2228570_nonmissing': int(pd.to_numeric(wide.get('rs2228570', pd.Series(dtype=float)), errors='coerce').notna().sum()),
    'staging_mt_uri': STAGING_MT_URI if 'STAGING_MT_URI' in globals() else '',
    'long_stage_export_uri': LONG_STAGE_EXPORT_URI if 'LONG_STAGE_EXPORT_URI' in globals() else '',
}])
save(sample_summary, '14_tier1_vdr_sample_id_person_id_summary.csv')

manifest_files = [
    '14_hail_stage_once_run_configuration.csv',
    '14_hail_stage_once_model_prep_person_id_status.csv',
    '14_hail_stage_once_status.csv',
    '14_hail_stage_once_staging_plan.csv',
    '14_hail_stage_once_export_status.csv',
    '14_hail_stage_once_local_validation.csv',
    '14_tier1_vdr_target_snp_list.csv',
    '14_tier1_vdr_genotype_long.csv',
    '14_tier1_vdr_genotype_wide.csv',
    '14_tier1_vdr_sample_id_person_id_summary.csv',
]
manifest_rows = []
for filename in manifest_files:
    path = OUTPUT_DIR / filename
    if path.exists():
        manifest_rows.append({'filename': filename, 'path': str(path), 'size_bytes': path.stat().st_size})
save(pd.DataFrame(manifest_rows), '99_hail_stage_once_output_manifest.csv')
add_status('local_validation_manifest', 'pass', f'wide_rows={len(wide)}; complete_four={complete_four}')
print('Validation complete. Wide CSV:', wide_path)
