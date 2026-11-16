import json
from pathlib import Path
import sys
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import digest


def main():
    required = ['score.py', 'scoring.md', 'common.py', 'simple-baseline.py', 'simple-baseline.md',
                'strong-baseline.py', 'strong-baseline.md', 'requirements-ms2.txt', 'MS2_ADR.md',
                'MS2_README.md', 'report/milestone2.pdf', 'results/simple-test.jsonl',
                'results/strong-test.jsonl', 'results/simple-test-metrics.json', 'results/strong-test-metrics.json']
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
    files = set(required + ['README.md', 'data.md'])
    for directory in ['scripts', 'tests', 'notebooks', 'results', 'data/processed']:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc', '.tmp']:
                files.add(str(path.relative_to(ROOT)))
    output = ROOT / 'MS2_submission.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.write(ROOT / name, name)
    print(f'Created {output.name}: {output.stat().st_size / 1e6:.2f} MB; {pages}-page report; {len(files)} files.')


if __name__ == '__main__':
    main()
