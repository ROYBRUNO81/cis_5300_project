import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import time

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed

from common import digest, input_only, mean, read_jsonl, write_json, write_jsonl


class WordBudget(StoppingCriteria):
    def __init__(self, tokenizer, prefix_tokens, words):
        self.tokenizer = tokenizer
        self.prefix_tokens = prefix_tokens
        self.words = words

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prefix_tokens:].tolist(), skip_special_tokens=True)
        return len(text.split()) > self.words


def trim_to_words(text, words):
    matches = list(re.finditer(r'\S+', text))
    if not matches:
        return '', 0
    end = matches[min(words, len(matches)) - 1].end()
    return text[:end].strip(), end


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--revision')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--max-input-tokens', type=int, default=2048)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=5300)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rows = [input_only(row) for row in read_jsonl(args.inputs)]
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    set_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.truncation_side = 'left'
    dtype = torch.float32 if device == 'cpu' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype,
                                               low_cpu_mem_usage=True, device_map={'': device},
                                               attn_implementation='sdpa')
    model.eval().requires_grad_(False)
    revision = args.revision or getattr(model.config, '_commit_hash', None)
    model_id = args.model
    manifest = Path(args.model) / 'download_manifest.json'
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        model_id, revision = metadata['repo'], metadata['revision']
    config = {'system': 'qwen_prefix_only', 'model': model_id, 'revision': revision,
              'device': device, 'dtype': str(dtype), 'do_sample': False, 'num_beams': 1,
              'max_input_tokens': args.max_input_tokens, 'max_new_tokens': args.max_new_tokens,
              'seed': args.seed, 'inputs_sha256': digest(args.inputs), 'script_sha256': digest(__file__),
              'versions': {package: version(package) for package in ['torch', 'transformers', 'tokenizers']},
              'prompt': 'raw paragraph prefix; no instruction, retrieval, target, or chat template'}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    predictions = []
    if Path(args.output).exists():
        if not args.resume:
            raise ValueError('output already exists; use --resume to continue the same run')
        predictions = read_jsonl(args.output)
        if any(row.get('run_id') != run_id for row in predictions):
            raise ValueError('existing predictions have a different run configuration')
    complete = {row['example_id'] for row in predictions}
    if not complete <= {row['example_id'] for row in rows}:
        raise ValueError('existing output includes IDs absent from inputs')
    started = time.perf_counter()
    write_json(args.output + '.run.json', {**config, 'run_id': run_id, 'status': 'running', 'n': len(predictions)})
    for row in rows:
        if row['example_id'] in complete:
            continue
        synchronize(device)
        begin = time.perf_counter()
        encoded = tokenizer(row['prefix'], return_tensors='pt', truncation=True,
                            max_length=args.max_input_tokens, add_special_tokens=False).to(device)
        prefix_tokens = encoded['input_ids'].shape[1]
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, num_beams=1, return_dict_in_generate=True,
                                    output_scores=True, pad_token_id=tokenizer.eos_token_id,
                                    stopping_criteria=StoppingCriteriaList([WordBudget(tokenizer, prefix_tokens, row['target_words'])]))
        generated = output.sequences[0, prefix_tokens:].tolist()
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        prediction, retained_characters = trim_to_words(decoded, row['target_words'])
        retained_tokens = 0
        if prediction:
            for end in range(1, len(generated) + 1):
                if len(tokenizer.decode(generated[:end], skip_special_tokens=True)) >= retained_characters:
                    retained_tokens = end
                    break
        log_probabilities = []
        for index, token in enumerate(generated[:retained_tokens]):
            value = output.scores[index].float().log_softmax(dim=-1)[0, token].item()
            log_probabilities.append(value)
        confidence = mean([math.exp(value) for value in log_probabilities])
        synchronize(device)
        seconds = time.perf_counter() - begin
        predictions.append({'example_id': row['example_id'], 'prediction': prediction,
                            'confidence': confidence, 'confidence_type': 'uncalibrated_mean_subword_probability',
                            'system': 'qwen_prefix_only', 'run_id': run_id,
                            'generated_tokens': len(generated), 'retained_tokens': retained_tokens,
                            'mean_log_probability': mean(log_probabilities) if log_probabilities else None,
                            'elapsed_seconds': seconds})
        write_jsonl(args.output, predictions)
        print(f"{len(predictions)}/{len(rows)} {row['example_id']} {len(prediction.split())}/{row['target_words']} words {seconds:.2f}s", flush=True)
        del output
    write_json(args.output + '.run.json', {**config, 'run_id': run_id, 'status': 'complete',
               'n': len(predictions), 'wall_seconds_this_invocation': time.perf_counter() - started,
               'generation_seconds': sum(row['elapsed_seconds'] for row in predictions)})


if __name__ == '__main__':
    main()
