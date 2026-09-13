"""Read-only handoff inspection; never execute notebook cells or load pickle."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickletools


def inspect(path):
    result = {'path': str(path), 'size_bytes': path.stat().st_size}
    with path.open('rb') as stream:
        result['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text(encoding='utf-8'))
        result['kernel'] = notebook.get('metadata', {}).get('kernelspec')
        result['language'] = notebook.get('metadata', {}).get('language_info')
        terms = ('import ', 'pip install', 'fit(', 'fit_transform(', 'transform(', 'split', 'scaler', 'encoder',
                 'log1p', 'normalize', 'rolling', 'shift(', 'pickle', 'dump(', 'save(', 'feature', 'reshape', 'pad_sequences')
        result['relevant_cells'] = [{'cell': i, 'execution_count': c.get('execution_count'), 'source': ''.join(c.get('source', []))}
            for i, c in enumerate(notebook.get('cells', [])) if c.get('cell_type') == 'code' and any(t in ''.join(c.get('source', [])) for t in terms)]
    elif path.suffix == '.py':
        tree = ast.parse(path.read_text(encoding='utf-8'))
        result['imports'] = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    elif path.suffix in {'.pickle', '.pkl'}:
        if path.stat().st_size > 128 * 1024 * 1024:
            result['opcodes'] = 'Skipped: artifact exceeds 128 MiB static-analysis limit.'
        else:
            with path.open('rb') as stream:
                result['global_references'] = [{'position': pos, 'opcode': op.name, 'argument': arg}
                    for op, arg, pos in pickletools.genops(stream) if op.name in {'GLOBAL', 'STACK_GLOBAL'}]
    result['warning'] = 'Static evidence only; this report does not prove trust, compatibility, or preprocessing parity.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([inspect(path) for path in args.paths], indent=2, ensure_ascii=False))
