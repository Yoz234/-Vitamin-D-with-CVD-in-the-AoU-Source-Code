"""Validate the source archive without accessing participant data or networks."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SUMS = ROOT / 'provenance/SHA256SUMS.txt'
REPORT = ROOT / 'provenance/validation_report.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def publication_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and '.git' not in p.parts and '__pycache__' not in p.parts
                  and p not in [SUMS, REPORT])


def python_cell(s):
    lines = []
    for line in s.splitlines():
        indent = line[:len(line)-len(line.lstrip())]
        stripped = line.lstrip()
        if stripped.startswith('!'):
            line = indent + 'get_ipython().system(' + repr(stripped[1:]) + ')'
        elif stripped.startswith('%') and not stripped.startswith('%%'):
            magic, _, arg = stripped[1:].partition(' ')
            line = indent + f'get_ipython().run_line_magic({magic!r}, {arg!r})'
        lines.append(line)
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Regenerate the checksum inventory after an intentional edit.')
    args = parser.parse_args()
    files = publication_files()
    errors = []
    allowed = {'.md','.py','.R','.ipynb','.txt','.json'}
    manifest = json.loads((ROOT/'provenance/source_manifest.json').read_text(encoding='utf-8'))
    entries = manifest['notebooks']
    mapped = {x['destination'] for x in entries}
    notebooks = {p.relative_to(ROOT).as_posix() for p in files if p.suffix=='.ipynb'}
    if mapped != notebooks:
        errors.append('Notebook inventory differs from provenance manifest')
    if len(mapped) != len(entries):
        errors.append('Duplicate notebook provenance entries')
    secret_patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                       r'gh[pousr]_[A-Za-z0-9_]{20,}', r'github_pat_[A-Za-z0-9_]{20,}',
                       r'AIza[A-Za-z0-9_-]{30,}', r'AKIA[A-Z0-9]{16}',
                       r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
                       r'person_id\s*(?:==|IN|in|=)\s*(?:\[\s*)?\d{5,}']
    py_count = r_count = code_cells = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix not in allowed and p.name not in {'.gitignore','.gitattributes'}:
            errors.append('Non-source file: '+rel)
        s = p.read_text(encoding='utf-8')
        for pattern in secret_patterns:
            if re.search(pattern,s):
                errors.append('Potential credential/contact/literal participant ID: '+rel)
        if p.suffix=='.py':
            py_count += 1
            try:
                compile(ast.parse(s),rel,'exec')
            except SyntaxError as e:
                errors.append(f'Python syntax: {rel}:{e.lineno}: {e.msg}')
        if p.suffix=='.R':
            r_count += 1
    for entry in entries:
        p=ROOT/entry['destination']
        n=json.loads(p.read_text(encoding='utf-8'))
        is_r=n['metadata']['language_info']['name'].lower()=='r'
        if n['nbformat']!=4 or n['nbformat_minor']!=5:
            errors.append('Unexpected notebook format: '+entry['destination'])
        if n['metadata']['kernelspec']['language'].lower()!=('r' if is_r else 'python'):
            errors.append('Kernel-language mismatch: '+entry['destination'])
        companion=ROOT/entry['source_companion']
        expected=['# Source companion of '+p.name+'\n',
                  '# Retains notebook cell order. Run in an authorized AoU environment.\n',
                  '# Use Rscript or an R notebook kernel.\n' if is_r else '# Shell/magic cells, if any, require IPython.\n']
        for i,c in enumerate(n['cells']):
            if c.get('metadata') or c.get('attachments'):
                errors.append('Residual cell metadata/attachment: '+entry['destination'])
            if c['cell_type']=='code':
                code_cells += 1
                if c.get('outputs')!=[] or c.get('execution_count') is not None:
                    errors.append('Residual output/execution count: '+entry['destination'])
                s=''.join(c['source'])
                expected.extend([f'\n# %% Cell {i}\n',s+'\n' if is_r else python_cell(s)])
        if not companion.exists() or companion.read_text(encoding='utf-8')!=''.join(expected):
            errors.append('Companion differs from notebook: '+entry['destination'])
    current={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    if args.refresh and not errors:
        SUMS.write_text(''.join(f'{sha}  {rel}\n' for rel,sha in current.items()),encoding='utf-8',newline='\n')
    if not SUMS.exists():
        errors.append('Missing checksum inventory; use --refresh after review')
    else:
        recorded=dict(line.split('  ',1)[::-1] for line in SUMS.read_text(encoding='utf-8').splitlines())
        if recorded!=current:
            errors.append('File inventory or checksum mismatch')
    report={'status':'pass' if not errors else 'fail','snapshot_date':'2026-10-17',
            'checksummed_files':len(files),'notebooks':len(notebooks),
            'code_cells':code_cells,'python_files_compiled_without_execution':py_count,
            'r_companions_preserved_without_r_execution':r_count,
            'outputs_execution_counts_attachments_removed':True,
            'notebook_companions_and_manifest_checked':True,
            'source_file_allowlist_and_pattern_scan_checked':True,
            'checksum_inventory_sha256':digest(SUMS) if SUMS.exists() else None,
            'participant_data_accessed':False,'bigquery_hail_r_models_executed':False,
            'limits':['Pattern scans are not a formal disclosure certification.',
                      'Python syntax checks do not validate model results or runtime dependencies.',
                      'R syntax, IRkernel execution, and AoU analyses require server validation.'],
            'errors':errors}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':
    main()
