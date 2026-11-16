import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import calibration_metrics, read_jsonl, validate_gold
from score import evaluate


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


simple = module('simple_baseline', 'simple-baseline.py')
strong = module('strong_baseline', 'strong-baseline.py')


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.gold = [{'example_id': 'a', 'paper_id': 'p', 'domain': 'nlp', 'target_words': 10,
                      'target': 'one two three four five six seven eight nine ten'}]

    def test_identity(self):
        row = evaluate([{'example_id': 'a', 'prediction': self.gold[0]['target']}], self.gold)[0]
        self.assertEqual(row['rougeL_f1'], 1.0)
        self.assertEqual(row['exact_length'], 1.0)

    def test_empty(self):
        row = evaluate([{'example_id': 'a', 'prediction': ''}], self.gold)[0]
        self.assertEqual(row['rougeL_f1'], 0.0)
        self.assertEqual(row['empty'], 1.0)

    def test_no_overlap(self):
        row = evaluate([{'example_id': 'a', 'prediction': 'alpha bravo charlie delta echo'}], self.gold)[0]
        self.assertEqual(row['rougeL_f1'], 0.0)

    def test_alignment_and_duplicate_rejection(self):
        for predictions in [[], [{'example_id': 'b', 'prediction': ''}],
                            [{'example_id': 'a', 'prediction': ''}] * 2]:
            with self.assertRaises(ValueError):
                evaluate(predictions, self.gold)

    def test_jsonl_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.jsonl'
            path.write_text('{"example_id":"a"}\n{"example_id":"a"}\n')
            with self.assertRaises(ValueError):
                read_jsonl(path)

    def test_invalid_reference_length(self):
        self.gold[0]['target_words'] = 30
        with self.assertRaises(ValueError):
            validate_gold(self.gold)


class ConfidenceTests(unittest.TestCase):
    def test_perfect_and_wrong(self):
        self.assertEqual(calibration_metrics([0, 1], [0, 1])['brier'], 0.0)
        self.assertEqual(calibration_metrics([1, 0], [0, 1])['brier'], 1.0)

    def test_known_metrics(self):
        score = calibration_metrics([0.8, 0.2], [1, 0])
        self.assertAlmostEqual(score['brier'], 0.04)
        self.assertAlmostEqual(score['ece_10'], 0.2)
        self.assertAlmostEqual(score['aurc_tie_grouped'], 0.25)

    def test_confidence_ties(self):
        self.assertEqual(calibration_metrics([0.5, 0.5], [1, 0]),
                         calibration_metrics([0.5, 0.5], [0, 1]))
        self.assertEqual(calibration_metrics([0.5, 0.5], [1, 0])['aurc_tie_grouped'], 0.5)

    def test_invalid_inputs(self):
        for probabilities, labels in [([], []), ([float('nan')], [1]), ([1.1], [1]), ([0.5], [2])]:
            with self.assertRaises(ValueError):
                calibration_metrics(probabilities, labels)


class BaselineTests(unittest.TestCase):
    def test_trigram_deduplicates_training_paragraphs(self):
        context = [{'paragraph_index': 0, 'text': 'zero alpha'}]
        rows = [{'paper_id': 'p', 'paragraph_index': 1, 'prefix': 'alpha', 'target': 'beta', 'context': context},
                {'paper_id': 'p', 'paragraph_index': 2, 'prefix': 'beta', 'target': 'gamma', 'context': context}]
        model = simple.Trigram().fit(rows)
        self.assertEqual(model.paragraphs, 3)
        self.assertEqual(model.generate('unseen prefix', 30), model.generate('unseen prefix', 30))
        self.assertEqual(len(model.generate('unseen prefix', 30)[0].split()), 30)

    def test_trim_whitespace_and_empty(self):
        self.assertEqual(strong.trim_to_words('', 10), ('', 0))
        self.assertEqual(strong.trim_to_words('  one two\nthree four', 2)[0], 'one two')
        self.assertEqual(strong.trim_to_words('short output', 60)[0], 'short output')

    def test_word_budget_waits_for_next_word(self):
        import torch
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ' '.join(str(i) for i in ids)
        rule = strong.WordBudget(Tokenizer(), 2, 2)
        self.assertFalse(rule(torch.tensor([[0, 0, 1, 2]]), None))
        self.assertTrue(rule(torch.tensor([[0, 0, 1, 2, 3]]), None))


class DatasetTests(unittest.TestCase):
    def test_ms1_split_integrity(self):
        papers = []
        for split, n_examples, n_papers in [('train', 1182, 237), ('dev', 150, 30), ('test', 150, 30)]:
            rows = read_jsonl(ROOT / 'data/processed' / f'{split}.jsonl')
            validate_gold(rows)
            self.assertEqual(len(rows), n_examples)
            current = {row['paper_id'] for row in rows}
            self.assertEqual(len(current), n_papers)
            self.assertTrue(all(not (current & previous) for previous in papers))
            papers.append(current)
            for row in rows:
                self.assertTrue(all(p['paragraph_index'] < row['paragraph_index'] for p in row['context']))


class ReportTests(unittest.TestCase):
    def test_report_layout_with_synthetic_fixture_only(self):
        from pypdf import PdfReader
        from common import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            (root / 'data/processed').mkdir(parents=True)
            shutil.copy2(ROOT / 'data/processed/test.jsonl', root / 'data/processed/test.jsonl')
            gold = read_jsonl(root / 'data/processed/test.jsonl')
            for system in ['simple', 'strong']:
                rows = [{'example_id': row['example_id'], 'prediction': 'synthetic layout fixture only',
                         'system': 'qwen_prefix_only' if system == 'strong' else 'trigram_backoff',
                         'run_id': 'synthetic-unit-test-only'} for row in gold]
                predictions = root / f'results/{system}-test.jsonl'
                predictions.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                metrics = {key: 0.1 for key in ['rougeL_precision','rougeL_recall','rougeL_f1',
                                               'bertscore_precision','bertscore_recall','bertscore_f1']}
                metrics.update(n=150, exact_length=1.0, configuration={
                    'gold_sha256': digest(root / 'data/processed/test.jsonl'),
                    'predictions_sha256': digest(predictions), 'bertscore': {'model': 'synthetic-fixture'}},
                    confidence_intervals_95={'rougeL_f1': [0.05, 0.15]})
                metrics['by_domain'] = {key: {**metrics, 'n': 50} for key in ['nlp', 'ml', 'cv']}
                metrics['by_target_words'] = {str(key): {'n': n, 'rougeL_f1': 0.1, 'bertscore_f1': 0.1}
                                              for key,n in [(10,59),(30,59),(60,32)]}
                (root / f'results/{system}-test-metrics.json').write_text(json.dumps(metrics))
                Path(str(predictions) + '.run.json').write_text(json.dumps({
                    'status': 'complete', 'n': 150, 'model': 'Qwen/Qwen2.5-1.5B',
                    'run_id': 'synthetic-unit-test-only'}))
            subprocess.run([sys.executable, str(ROOT / 'scripts/build_ms2_deliverables.py'), '--root', str(root)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(len(PdfReader(root / 'report/milestone2.pdf').pages), 2)


if __name__ == '__main__':
    unittest.main()
