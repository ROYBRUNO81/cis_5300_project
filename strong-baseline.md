# Strong baseline: frozen prefix-only Qwen2.5-1.5B

**Status: implemented; the complete GPU test evaluation is pending. No strong-baseline score is claimed yet.** The Colab workflow replaces this draft with measured results and sample output once all 150 predictions have been scored.

## Model and decoding

Use the public **base** checkpoint `Qwen/Qwen2.5-1.5B`, frozen, without a chat template, instructions, retrieval, or fine-tuning. The input is only the partially written paragraph. This is a standard pretrained causal-LM baseline, supported by the [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115), not a reproduction of REPLUG's retriever training.

Decode greedily with one beam, seed 5300, at most 2,048 prefix tokens and 256 new tokens. The task's output budget is **words**, not subword tokens. Stop after one more whitespace word than requested appears, then retain the first 10/30/60 complete words. Early EOS or a token-budget limit can yield a shorter suggestion; score and report that rather than padding with reference text.

The output includes the actual generated continuation, a raw mean emitted-subword probability, and a run fingerprint. This confidence is not calibrated acceptability. Save checkpoint revisions, exact package versions, device/precision, input hashes, and runtime in the sidecar JSON. Generation outputs are saved atomically after each example; `--resume` requires the same configuration.

## Run

Recommended: open `notebooks/MS2_Colab.ipynb` with a Colab GPU. Or run from the repository root:

```bash
pip install -r requirements-ms2.txt
python scripts/download_models.py --output .models
python strong-baseline.py --inputs data/processed/test.jsonl --output results/strong-test.jsonl --model .models/qwen --device cuda
python score.py --gold data/processed/test.jsonl --predictions results/strong-test.jsonl --bert-model .models/bertscore --device cuda --output results/strong-test-metrics.json --per-example results/strong-test-scores.jsonl
python scripts/build_ms2_deliverables.py
```

The complete notebook uses a separate run folder, preserves per-example outputs, builds the PDF report, validates the package, and downloads a zip. No test gold/context enters the model prompt. No model settings are chosen using test scores. The local 8 GB Mac encountered swap/disk pressure; use a GPU with sufficient memory instead. Weights are not committed or included in the submission.
