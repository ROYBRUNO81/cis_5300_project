import argparse
import hashlib
import json
from pathlib import Path
import time

import requests


def download(repo, directory):
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://huggingface.co/api/models/{repo}', timeout=60)
    response.raise_for_status()
    info = response.json()
    revision = info['sha']
    wanted = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
              'special_tokens_map.json', 'vocab.json', 'vocab.txt', 'merges.txt', 'LICENSE',
              'model.safetensors', 'model.safetensors.index.json'}
    records = []
    for item in info['siblings']:
        name = item['rfilename']
        if name not in wanted and not (name.startswith('model-') and name.endswith('.safetensors')):
            continue
        target = directory / name
        if target.exists():
            print('cached', repo, name, flush=True)
        else:
            part = target.with_suffix(target.suffix + '.part')
            for attempt in range(5):
                try:
                    offset = part.stat().st_size if part.exists() else 0
                    headers = {'Range': f'bytes={offset}-'} if offset else {}
                    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true'
                    with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as data:
                        data.raise_for_status()
                        mode = 'ab' if offset and data.status_code == 206 else 'wb'
                        with part.open(mode) as file:
                            for chunk in data.iter_content(4 * 1024 * 1024):
                                file.write(chunk)
                    part.rename(target)
                    print('downloaded', repo, name, target.stat().st_size, flush=True)
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        records.append({'file': name, 'bytes': target.stat().st_size, 'sha256': digest})
    (directory / 'download_manifest.json').write_text(json.dumps({'repo': repo, 'revision': revision, 'files': records}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.models'))
    args = parser.parse_args()
    download('Qwen/Qwen2.5-1.5B', args.output / 'qwen')
    download('distilbert/distilbert-base-uncased', args.output / 'bertscore')
