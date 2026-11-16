# Simple baseline: trigram with hard backoff

I fit a whitespace-word trigram model using only the 237 training papers. Training contexts are deduplicated by `(paper_id, paragraph_index)`; a full training-context paragraph takes precedence over its shorter prefix-plus-target span. Two BOS markers mark paragraph starts. At each step, select the most frequent continuation of the last two words. If that context is unseen, back off to the last word, then to the unigram mode. Ties use lexicographic order. The model emits exactly the requested number of words and never reads the test target or test context for generation. Training/evaluation paper overlap is rejected.

This is a simple open-ended-generation baseline, not a majority-class classifier. No model selection or tuning uses test scores. Hard backoff is not interpolated smoothing. The saved mean word probability is an uncalibrated diagnostic, not the probability a writer accepts the continuation.

## Run

```bash
python simple-baseline.py --train data/processed/train.jsonl --inputs data/processed/test.jsonl --output results/simple-test.jsonl
python score.py --gold data/processed/test.jsonl --predictions results/simple-test.jsonl --bert-model .models/bertscore --output results/simple-test-metrics.json --per-example results/simple-test-scores.jsonl
```

## Measured full-test results

| System | ROUGE-L P | ROUGE-L R | ROUGE-L F1 | BERTScore P | BERTScore R | BERTScore F1 | Exact word length |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trigram | 0.1095 | 0.1068 | 0.1079 | 0.7151 | 0.7132 | 0.7137 | 100.0% |
| Qwen2.5-1.5B | 0.2343 | 0.2226 | 0.2262 | 0.7963 | 0.7776 | 0.7865 | 94.7% |

The table includes both baselines for comparison. See `results/simple-test-metrics.json` for paper-bootstrap intervals and domain/length breakdowns. These metrics quantify reference similarity, not factual correctness. Repetition is a known limitation of greedy count-based generation.

## Actual sample output

Example `2405.14039:0` (30 requested words):

```json
{"example_id": "2405.14039:0", "prediction": "is a set of the model to learn the best of our proposed method is to use the same as the number of parameters and the corresponding UEFA EURO 2020", "confidence": 0.19863489500787115, "confidence_type": "uncalibrated_mean_word_probability", "system": "trigram_backoff"}
```

The program writes one JSON object per test ID with `prediction`, raw `confidence`, and its unit/type. Output order is aligned by ID by the scorer.
