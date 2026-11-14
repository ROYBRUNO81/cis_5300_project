import argparse
from collections import Counter, defaultdict
import time

from common import digest, input_only, mean, read_jsonl, write_json, write_jsonl


class Trigram:
    def __init__(self):
        self.counts = [Counter(), defaultdict(Counter), defaultdict(Counter)]

    def fit(self, rows):
        paragraphs = {}
        for row in rows:
            for paragraph in row['context']:
                paragraphs[(row['paper_id'], paragraph['paragraph_index'])] = paragraph['text']
        for row in rows:
            paragraphs.setdefault((row['paper_id'], row['paragraph_index']), row['prefix'] + ' ' + row['target'])
        for key in sorted(paragraphs):
            words = ['<BOS>', '<BOS>'] + paragraphs[key].split()
            for index in range(2, len(words)):
                word = words[index]
                self.counts[0][word] += 1
                self.counts[1][words[index-1]][word] += 1
                self.counts[2][tuple(words[index-2:index])][word] += 1
        if not self.counts[0]:
            raise ValueError('empty training corpus')
        self.paragraphs = len(paragraphs)
        return self

    def generate(self, prefix, length):
        words = ['<BOS>', '<BOS>'] + prefix.split()
        generated, probabilities = [], []
        for _ in range(length):
            choices = self.counts[2].get(tuple(words[-2:])) or self.counts[1].get(words[-1]) or self.counts[0]
            word = min(choices, key=lambda candidate: (-choices[candidate], candidate))
            probabilities.append(choices[word] / sum(choices.values()))
            words.append(word)
            generated.append(word)
        return ' '.join(generated), mean(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training = read_jsonl(args.train)
    inputs = read_jsonl(args.inputs)
    if {r['paper_id'] for r in training} & {r['paper_id'] for r in inputs}:
        raise ValueError('training and evaluation papers overlap')
    started = time.perf_counter()
    model = Trigram().fit(training)
    predictions = []
    for original in inputs:
        row = input_only(original)
        prediction, confidence = model.generate(row['prefix'], row['target_words'])
        predictions.append({'example_id': row['example_id'], 'prediction': prediction,
                            'confidence': confidence, 'confidence_type': 'uncalibrated_mean_word_probability',
                            'system': 'trigram_backoff'})
    write_jsonl(args.output, predictions)
    write_json(args.output + '.run.json', {'system': 'trigram_backoff', 'n': len(predictions),
               'training_examples': len(training), 'unique_training_paragraphs': model.paragraphs,
               'train_sha256': digest(args.train), 'inputs_sha256': digest(args.inputs),
               'elapsed_seconds': time.perf_counter() - started})
    print(f'wrote {len(predictions)} predictions to {args.output}')


if __name__ == '__main__':
    main()
