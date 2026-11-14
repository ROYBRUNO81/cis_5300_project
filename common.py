import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path):
    rows = []
    seen = set()
    with Path(path).open() as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get('example_id')
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError(f'{path}:{line_number}: missing or duplicate example_id')
            seen.add(identifier)
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no examples')
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def input_only(row):
    return {'example_id': row['example_id'], 'paper_id': row['paper_id'], 'domain': row['domain'],
            'prefix': row['prefix'], 'target_words': int(row['target_words'])}


def validate_gold(rows):
    for row in rows:
        if not isinstance(row.get('target'), str) or len(row['target'].split()) != row.get('target_words'):
            raise ValueError(f"invalid target length: {row['example_id']}")
        if int(row['target_words']) not in [10, 30, 60]:
            raise ValueError('expected target_words in {10,30,60}')


def mean(values):
    return sum(values) / len(values) if values else 0.0


def calibration_metrics(probabilities, labels, bins=10):
    if len(probabilities) != len(labels) or not labels:
        raise ValueError('nonempty aligned labels required')
    if any(y not in [0, 1] for y in labels):
        raise ValueError('acceptability labels must be 0/1')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('confidence must be a finite probability')
    brier = mean([(p - y) ** 2 for p, y in zip(probabilities, labels)])
    reliability = []
    ece = 0.0
    for index in range(bins):
        group = [i for i, p in enumerate(probabilities) if min(int(p * bins), bins - 1) == index]
        confidence = mean([probabilities[i] for i in group])
        accuracy = mean([labels[i] for i in group])
        ece += len(group) / len(labels) * abs(confidence - accuracy)
        reliability.append({'lower': index / bins, 'upper': (index + 1) / bins,
                            'count': len(group), 'mean_confidence': confidence, 'acceptability_rate': accuracy})
    # include entire confidence-tie groups so coverage does not depend on row order
    curve = [{'coverage': 0.0, 'risk': 0.0, 'threshold': None}]
    for threshold in sorted(set(probabilities), reverse=True):
        kept = [i for i, p in enumerate(probabilities) if p >= threshold]
        curve.append({'coverage': len(kept) / len(labels),
                      'risk': mean([1 - labels[i] for i in kept]), 'threshold': threshold})
    aurc = sum((b['coverage'] - a['coverage']) * b['risk'] for a, b in zip(curve, curve[1:]))
    return {'n': len(labels), 'brier': brier, 'ece_10': ece, 'aurc_tie_grouped': aurc,
            'reliability': reliability, 'risk_coverage': curve}
