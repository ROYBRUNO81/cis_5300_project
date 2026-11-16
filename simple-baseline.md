# Simple baseline: trigram with hard backoff

Train a whitespace-word trigram on deduplicated paragraphs from the **training papers only**. For a seen two-word context, choose its most frequent continuation; otherwise back off to one word, then to the unigram mode. Ties are lexicographic. Generate exactly the requested 10/30/60 words. The program rejects overlapping training/evaluation paper IDs. No test gold or test context is used to generate output.

Training contexts are deduplicated by paper and paragraph index. A full training-context paragraph takes precedence over a shorter prefix-plus-target span for that same paragraph. Two BOS markers initialize each paragraph; no EOS stop is used in this fixed-word-budget baseline. The saved mean selected-word probability is an uncalibrated diagnostic, not a writer-acceptance probability.

## Run from the repository root

```bash
python simple-baseline.py --train data/processed/train.jsonl --inputs data/processed/test.jsonl --output results/simple-test.jsonl
python score.py --gold data/processed/test.jsonl --predictions results/simple-test.jsonl --bert-model .models/bertscore --output results/simple-test-metrics.json --per-example results/simple-test-scores.jsonl
```

## Measured test results

All **150** examples from 30 papers; 50 examples each in NLP, ML, and CV. Scores are fractions.

| Metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| ROUGE-L | 0.1095 | 0.1068 | 0.1079 |
| BERTScore | 0.7151 | 0.7132 | 0.7137 |

Exact requested length: **100%**; empty outputs: **0%**. BERTScore uses five-layer DistilBERT, no IDF weighting or rescaling. See `scoring.md` for definitions and `results/simple-test-metrics.json` for configuration, hashes, bootstrap intervals, and breakdowns. These are reference-similarity scores, not factuality or calibration results.

## Actual sample output

```json
{"example_id": "2405.14039:0", "prediction": "is a set of the model to learn the best of our proposed method is to use the same as the number of parameters and the corresponding UEFA EURO 2020", "confidence": 0.19863489500787113, "confidence_type": "uncalibrated_mean_word_probability", "system": "trigram_backoff"}
```

Greedy trigram generation often repeats common phrases and cannot use long-range dependencies. The strong baseline comparison remains pending the Colab run; no strong score is claimed here. The report builder refreshes this file with the full comparison once both evaluations are complete.
