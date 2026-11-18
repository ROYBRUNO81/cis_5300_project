import ast
import json
from pathlib import Path
import sys
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import digest


def standalone_script(name):
    source = (ROOT / name).read_text()
    helpers = (ROOT / 'common.py').read_text()
    nodes = ast.parse(helpers).body
    imports = [ast.get_source_segment(helpers, node) for node in nodes
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == 'common':
            names = {alias.name for alias in node.names}
            definitions = [ast.get_source_segment(helpers, item) for item in nodes
                           if isinstance(item, ast.FunctionDef) and item.name in names]
            source = source.replace(ast.get_source_segment(source, node),
                                    '\n'.join(imports) + '\n\n\n' + '\n\n\n'.join(definitions))
    ast.parse(source)
    return source


def standalone_guide(name):
    source = (ROOT / name).read_text()
    packages = ['torch>=2.7,<3', 'transformers==4.57.3', 'accelerate==1.11.0',
                'safetensors==0.6.2', 'rouge-score==0.1.2', 'bert-score==0.3.13', 'numpy==2.2.6']
    install = 'pip install ' + ' '.join(f'"{package}"' for package in packages)
    downloads = ('python -c "from huggingface_hub import snapshot_download; '
                 "snapshot_download('Qwen/Qwen2.5-1.5B', revision='8faed761d45a263340a0528343f099c05c9a4323', local_dir='.models/qwen'); "
                 "snapshot_download('distilbert/distilbert-base-uncased', revision='12040accade4e8a0f71eabdb258fecc2e7e948be', local_dir='.models/bertscore')\"")
    source = source.replace('pip install -r requirements-ms2.txt', install)
    source = source.replace('python scripts/download_models.py --output .models', downloads)
    title, rest = source.split('\n', 1)
    note = ('\n\nThis submission contains self-contained Python scripts; no `common.py` or requirements file is needed. '
            'Install the packages and models using the commands in `scoring.md`, then use the MS1 train/test JSONL splits. '
            'Referenced notebooks and saved run artifacts remain in the '
            '[project repository](https://github.com/ROYBRUNO81/cis_5300_project/tree/ms2); they are not part of this hand-in.\n')
    return title + note + rest


def main():
    required = ['report/milestone2.pdf', 'scoring.md', 'score.py',
                'simple-baseline.py', 'simple-baseline.md',
                'strong-baseline.py', 'strong-baseline.md']
    for name in required:
        if not (ROOT / name).is_file():
            raise ValueError(f'missing required deliverable: {name}')
    pages = len(PdfReader(ROOT / 'report/milestone2.pdf').pages)
    if not 1 <= pages <= 2:
        raise ValueError(f'the report must have 1–2 pages; got {pages}')
    for system in ['simple', 'strong']:
        result = json.loads((ROOT / f'results/{system}-test-metrics.json').read_text())
        if result.get('n') != 150 or 'bertscore_f1' not in result:
            raise ValueError('incomplete evaluation')
        if result['configuration']['predictions_sha256'] != digest(ROOT / f'results/{system}-test.jsonl'):
            raise ValueError('metrics/predictions mismatch')
        if result['configuration']['gold_sha256'] != digest(ROOT / 'data/processed/test.jsonl'):
            raise ValueError('metrics/gold mismatch')
    files = set(required)
    output = ROOT / 'MS2_submission.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            target = Path(name).name
            if name.endswith('.py'):
                archive.writestr(target, standalone_script(name))
            elif name.endswith('.md'):
                archive.writestr(target, standalone_guide(name))
            else:
                archive.write(ROOT / name, target)
    print(f'Created {output.name}: {output.stat().st_size / 1e6:.2f} MB; {pages}-page report; {len(files)} files.')


if __name__ == '__main__':
    main()
