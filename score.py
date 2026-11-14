import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random

from rouge_score import rouge_scorer

from common import calibration_metrics, digest, mean, read_jsonl, validate_gold, write_json, write_jsonl


def choose_device(requested):
    import torch
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def group_scores(rows):
    fields = ['rougeL_precision', 'rougeL_recall', 'rougeL_f1',
              'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
              'exact_length', 'prediction_words', 'empty']
    return {'n': len(rows), **{key: mean([row[key] for row in rows])
                              for key in fields if all(key in row for row in rows)}}


def bootstrap(rows, field, iterations, seed):
    import numpy as np
    groups = defaultdict(list)
    for row in rows:
        groups[row['paper_id']].append(row[field])
    papers = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choices(papers, k=len(papers))
        values.append(mean([value for paper in sampled for value in groups[paper]]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def evaluate(predictions, gold, bert_model=None, bert_layers=5, device='cpu', batch_size=16):
    validate_gold(gold)
    lookup = {row['example_id']: row for row in predictions}
    expected = {row['example_id'] for row in gold}
    if set(lookup) != expected:
        raise ValueError(f'prediction IDs do not match gold: missing={len(expected-set(lookup))}, extra={len(set(lookup)-expected)}')
    if len(lookup) != len(predictions):
        raise ValueError('duplicate prediction IDs')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rows = []
    for reference in gold:
        prediction = lookup[reference['example_id']].get('prediction')
        if not isinstance(prediction, str):
            raise ValueError('every prediction must be a string')
        score = scorer.score(reference['target'], prediction)['rougeL']
        words = len(prediction.split())
        rows.append({'example_id': reference['example_id'], 'paper_id': reference['paper_id'],
                     'domain': reference['domain'], 'target_words': reference['target_words'],
                     'rougeL_precision': score.precision, 'rougeL_recall': score.recall,
                     'rougeL_f1': score.fmeasure, 'prediction_words': words,
                     'exact_length': float(words == reference['target_words']), 'empty': float(words == 0)})
    if bert_model is not None:
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        from bert_score import BERTScorer
        bert = BERTScorer(model_type=bert_model, num_layers=bert_layers, lang='en', idf=False,
                          rescale_with_baseline=False, device=device, batch_size=batch_size)
        nonempty = [i for i, row in enumerate(rows) if not row['empty']]
        for row in rows:
            row.update(bertscore_precision=0.0, bertscore_recall=0.0, bertscore_f1=0.0)
        if nonempty:
            p, r, f = bert.score([lookup[gold[i]['example_id']]['prediction'] for i in nonempty],
                                 [gold[i]['target'] for i in nonempty], verbose=False)
            for index, precision, recall, f1 in zip(nonempty, p.tolist(), r.tolist(), f.tolist()):
                if not all(math.isfinite(x) for x in [precision, recall, f1]):
                    raise ValueError('non-finite BERTScore')
                rows[index].update(bertscore_precision=precision, bertscore_recall=recall, bertscore_f1=f1)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--output')
    parser.add_argument('--per-example')
    parser.add_argument('--bert-model', default='distilbert/distilbert-base-uncased')
    parser.add_argument('--bert-layers', type=int, default=5)
    parser.add_argument('--no-bertscore', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=5300)
    parser.add_argument('--labels', help='optional human labels bound to prediction_sha256')
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    predictions = read_jsonl(args.predictions)
    bert_model = None if args.no_bertscore else args.bert_model
    rows = evaluate(predictions, gold, bert_model, args.bert_layers,
                    choose_device(args.device) if bert_model else 'cpu', args.batch_size)
    result = {'primary_metric': 'rougeL_f1', **group_scores(rows)}
    for field, name in [('domain', 'by_domain'), ('target_words', 'by_target_words')]:
        keys = sorted({row[field] for row in rows})
        result[name] = {str(key): group_scores([row for row in rows if row[field] == key]) for key in keys}
    result['calibration'] = {'status': 'not evaluated: no human acceptability labels supplied'}
    if args.labels:
        lookup = {row['example_id']: row for row in predictions}
        labels = read_jsonl(args.labels)
        probabilities, outcomes = [], []
        for row in labels:
            if row['example_id'] not in lookup:
                raise ValueError('label ID absent from predictions')
            prediction = lookup[row['example_id']]
            fingerprint = hashlib.sha256(prediction['prediction'].encode()).hexdigest()
            if row.get('prediction_sha256') != fingerprint:
                raise ValueError('human label does not match this generated suggestion')
            if row.get('acceptable') not in [0, 1]:
                raise ValueError('human acceptable must be 0 or 1')
            probabilities.append(float(prediction['confidence']))
            outcomes.append(row['acceptable'])
        result['calibration'] = calibration_metrics(probabilities, outcomes)
        result['calibration']['warning'] = 'raw token probabilities are uncalibrated acceptability estimates; never interpret overlap metrics as labels'
    result['confidence_intervals_95'] = {}
    if args.bootstrap > 0:
        for field in ['rougeL_f1'] + (['bertscore_f1'] if bert_model else []):
            result['confidence_intervals_95'][field] = bootstrap(rows, field, args.bootstrap, args.seed)
    model_info = {'model': args.bert_model, 'layers': args.bert_layers, 'idf': False, 'rescale_with_baseline': False}
    manifest = Path(args.bert_model) / 'download_manifest.json'
    if bert_model and manifest.exists():
        info = json.loads(manifest.read_text())
        model_info.update(model=info['repo'], revision=info['revision'])
    result['configuration'] = {'rouge_stemmer': True, 'bertscore': model_info if bert_model else None,
                               'bootstrap_papers': args.bootstrap, 'seed': args.seed,
                               'gold_sha256': digest(args.gold), 'predictions_sha256': digest(args.predictions),
                               'versions': {package: version(package) for package in ['rouge-score'] + (['bert-score', 'transformers', 'torch'] if bert_model else [])}}
    if args.output:
        write_json(args.output, result)
    if args.per_example:
        write_jsonl(args.per_example, rows)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
