import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})


def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                  'source': text.strip().splitlines(True)})


markdown('''# ContextComplete — Milestone 2 GPU workflow
Select **Runtime → Change runtime type → GPU**, then run top to bottom. This notebook uses the committed 297-paper MS1 dataset, a trigram baseline, and frozen prefix-only Qwen2.5-1.5B. It runs no paid APIs and never pushes to GitHub. Full generation may take several minutes; progress is saved after every example. Download results before disconnecting.

The report and baseline markdowns are built only from complete measured results. Human calibration results are not claimed without real labels.''')
code('''
from pathlib import Path
import os
import subprocess
import sys

repo = Path('/content/cis_5300_project')
if not repo.exists():
    subprocess.run(['git', 'clone', '--branch', 'ms2', '--single-branch',
                    'https://github.com/ROYBRUNO81/cis_5300_project.git', str(repo)], check=True)
os.chdir(repo)
branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
assert branch == 'ms2', f'Expected ms2 branch, got {branch}'
print('Repository commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
''')
code('''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-ms2.txt'], check=True)
''')
code('''
import torch
assert torch.cuda.is_available(), 'Choose a GPU runtime before proceeding.'
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__)
RUN_NAME = 'colab_run_01'
run_dir = Path('runs') / RUN_NAME
run_dir.mkdir(parents=True, exist_ok=True)

def run(*arguments):
    subprocess.run([sys.executable, '-u', *map(str, arguments)], check=True)

run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
''')
markdown('## Download model snapshots\nAbout 3.4 GB of model files are needed. The downloader records revisions and checksums and reuses complete cached files. Weights are not included in the final submission zip.')
code("run('scripts/download_models.py', '--output', '.models')")
markdown('## Generate both baselines\nNo test targets or context enter Qwen generation. The trigram fits training papers only. `--resume` requires identical input hashes, script, device, model, and package versions. Do not change settings based on test scores.')
code('''
run('simple-baseline.py', '--train', 'data/processed/train.jsonl', '--inputs', 'data/processed/test.jsonl',
    '--output', run_dir / 'simple-test.jsonl')
run('strong-baseline.py', '--inputs', 'data/processed/test.jsonl', '--output', run_dir / 'strong-test.jsonl',
    '--model', '.models/qwen', '--device', 'cuda', '--resume')
''')
markdown('## Evaluate\nBoth systems use the same frozen DistilBERT scorer. All scores are on the full 150-example test split; generation uses the MS1 word budgets, not token budgets.')
code('''
for system in ['simple', 'strong']:
    run('score.py', '--gold', 'data/processed/test.jsonl', '--predictions', run_dir / f'{system}-test.jsonl',
        '--bert-model', '.models/bertscore', '--device', 'cuda',
        '--output', run_dir / f'{system}-test-metrics.json',
        '--per-example', run_dir / f'{system}-test-scores.jsonl')
''')
markdown('## Build and validate deliverables\nThis generates `simple-baseline.md`, `strong-baseline.md`, and `report/milestone2.pdf` from measured outputs. The archive validator checks that the report has at most two pages. No fabricated labels or provisional strong scores are allowed.')
code('''
import shutil
Path('results').mkdir(exist_ok=True)
for path in run_dir.glob('*'):
    if path.is_file() and not path.name.endswith('.tmp'):
        shutil.copy2(path, Path('results') / path.name)
run('scripts/build_ms2_deliverables.py')
run('scripts/package_ms2.py')
''')
code('''
import json
for system in ['simple', 'strong']:
    metrics = json.loads((run_dir / f'{system}-test-metrics.json').read_text())
    print(system, {key: metrics[key] for key in ['n', 'rougeL_f1', 'bertscore_f1', 'exact_length']})
from pypdf import PdfReader
print('Report pages:', len(PdfReader('report/milestone2.pdf').pages))
''')
markdown('## Download before ending the runtime\nUpload the downloaded archive back to our project workspace so the measured results and report can be reviewed and committed to `ms2`. The notebook does not update the repository checklist automatically; completion must be verified first.')
code('''
from google.colab import files
files.download('MS2_submission.zip')
''')
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
            'nbformat': 4, 'nbformat_minor': 4}
destination = ROOT / 'notebooks/MS2_Colab.ipynb'
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, indent=2) + '\n')
print('Built', destination)
