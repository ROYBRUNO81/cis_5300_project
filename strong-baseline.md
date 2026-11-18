# Strong baseline: frozen prefix-only Qwen2.5-1.5B

I use the public **base** checkpoint `Qwen/Qwen2.5-1.5B`, not an instruction-tuned model. A frozen causal LM extends the paragraph prefix directly, without a chat template, instructions, retrieved passages, or fine-tuning. This standard pretrained-LM baseline is supported by the [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115). It is not a reproduction of REPLUG's retriever training.

Qwen is a frozen pretrained model: we did not train or fine-tune it on our dataset. We ran inference and scoring on **all 150 held-out test examples**, not all examples across the dataset. The 1,182 training and 150 development examples remain separate. The simple trigram baseline **was fitted on the training split**; neither baseline was tuned or selected using test scores.

Each example comes from a cleaned scientific-paper paragraph with at least 60 words, excluding abstracts and requiring at least three earlier paragraphs. Using a fixed random seed, we choose an eligible paragraph, select a 10-, 30-, or 60-word target length that fits, and cut at a word boundary with at least 15 prefix words. The prefix is the input; the next original words are the gold continuation. Earlier paragraphs are saved as context, but neither MS2 baseline uses them at inference. Papers, including all their examples, stay in separate training/development/test splits.

Greedy decoding uses one beam, seed 5300, at most 2,048 prefix tokens and 256 new tokens. To avoid counting a partial last word, generation stops after more than the requested number of whitespace words appears, and the output is trimmed to the first 10/30/60 words. Early EOS/token-budget limits can yield shorter suggestions; we report length adherence rather than padding them. Mean probabilities cover the emitted subword tokens and are raw diagnostics, not calibrated acceptance probabilities. The model is loaded directly on the chosen device to reduce transient memory.

## Run

```bash
python scripts/download_models.py --output .models
python strong-baseline.py --inputs data/processed/test.jsonl --output results/strong-test.jsonl --model .models/qwen --device cuda
python score.py --gold data/processed/test.jsonl --predictions results/strong-test.jsonl --bert-model .models/bertscore --device cuda --output results/strong-test-metrics.json --per-example results/strong-test-scores.jsonl
```

Use `--device auto`/`mps`/`cpu` where appropriate. A Colab GPU is recommended; the local 8 GB Mac encountered memory/swap pressure. `notebooks/MS2_Colab.ipynb` runs the complete workflow. `--resume` continues the **same** saved configuration; a changed model, input hash, or script is rejected. Output is saved atomically after each example. Keep outputs before disconnecting Colab. Downloaded weights are not committed.

## Measured full-test results

| System | ROUGE-L P | ROUGE-L R | ROUGE-L F1 | BERTScore P | BERTScore R | BERTScore F1 | Exact word length |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trigram | 0.1095 | 0.1068 | 0.1079 | 0.7151 | 0.7132 | 0.7137 | 100.0% |
| Qwen2.5-1.5B | 0.2343 | 0.2226 | 0.2262 | 0.7963 | 0.7776 | 0.7865 | 94.7% |

See `results/strong-test.jsonl.run.json` for the actual checkpoint revision, precision, hardware, seed, software versions, input hashes, and generation time. See `results/strong-test-metrics.json` for the scorer settings, paper-bootstrap intervals, and breakdowns. The baseline was chosen before scoring, not selected by test performance.

## Actual sample output

Example `2405.14039:0` (30 requested words):

```json
{"example_id": "2405.14039:0", "prediction": "as the OOD dataset, which is far from the ID dataset in terms of the distribution of the training data. (ii) Near-shift OOD scenario, we select the MATH [11] dataset", "confidence": 0.6106921394413811, "confidence_type": "uncalibrated_mean_subword_probability", "system": "qwen_prefix_only", "run_id": "56ccf21ff29c131aa467048bf07c58f9ac43b896638754feaebe6d02ad92ebc7", "generated_tokens": 43, "retained_tokens": 42, "mean_log_probability": -0.7075012092760257, "elapsed_seconds": 6.728458571000033}
```

These are single-reference, single-run similarity results. They do not establish calibrated confidence, factual grounding, or robustness to pretrained-data overlap.
