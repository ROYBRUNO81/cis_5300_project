# Milestone 2: run and submission guide

The live checklist and decisions are in [MS2_ADR.md](MS2_ADR.md). This branch uses MS1's **297-paper NLP/ML/CV dataset**, not the earlier 300-paper physics/economics pilot.

## Recommended: Colab GPU

Open [`notebooks/MS2_Colab.ipynb`](notebooks/MS2_Colab.ipynb), select **Runtime → Change runtime type → GPU**, then run the cells in order. A T4-class GPU should fit the frozen 1.5B model. The notebook:

1. Clones the `ms2` branch without modifying `main`.
2. Installs dependencies and checks that CUDA is available.
3. Runs unit tests and downloads pinned model snapshots (no paid APIs).
4. Generates the simple and strong baselines on all 150 test examples.
5. Scores both with the same ROUGE-L/BERTScore configuration.
6. Builds measured baseline documentation and the 1–2 page PDF report.
7. Validates and downloads `MS2_submission.zip`.

Outputs are saved per example under `runs/colab_run_01/`; rerunning the generation cell resumes only an identical configuration. Download the final archive before disconnecting Colab. Runtime storage is temporary; a fresh runtime cannot recover undownloaded outputs. Use a new run name when changing settings or software. The final cells copy completed outputs into `results/` in the Colab checkout, but never commit or push them automatically.

The local Mac has 8 GB unified memory and encountered swap/disk pressure loading Qwen. These failures are **not benchmark results**. Use Colab for full inference; no changes to model quality settings are needed. Dependencies allow an already-installed PyTorch 2.x version >=2.7 to avoid a large Colab CUDA downgrade; actual versions are logged in run metadata.

## Command-line workflow

```bash
git clone --branch ms2 https://github.com/ROYBRUNO81/cis_5300_project.git
cd cis_5300_project
python3 -m venv .venv-ms2
source .venv-ms2/bin/activate
pip install -r requirements-ms2.txt
export HF_ENDPOINT=https://huggingface.co
python -m unittest discover -s tests -v
python scripts/download_models.py --output .models

python simple-baseline.py --train data/processed/train.jsonl --inputs data/processed/test.jsonl --output results/simple-test.jsonl
python strong-baseline.py --inputs data/processed/test.jsonl --output results/strong-test.jsonl --model .models/qwen --device cuda

python score.py --gold data/processed/test.jsonl --predictions results/simple-test.jsonl --bert-model .models/bertscore --device cuda --output results/simple-test-metrics.json --per-example results/simple-test-scores.jsonl
python score.py --gold data/processed/test.jsonl --predictions results/strong-test.jsonl --bert-model .models/bertscore --device cuda --output results/strong-test-metrics.json --per-example results/strong-test-scores.jsonl

python scripts/build_ms2_deliverables.py
python scripts/package_ms2.py
```

The report builder refuses incomplete or mismatched measurements. It will not invent a strong-baseline score. The package validator also checks report page count, complete test coverage, and prediction/gold hashes. Model weights, virtual environments, raw PDFs, and caches are excluded from the zip and from Git.

## Current measured evidence

Qwen is a frozen pretrained model: we did not train or fine-tune it on our dataset. We ran inference and scoring on **all 150 held-out test examples**, not all examples across the dataset. The 1,182 training and 150 development examples remain separate. The simple trigram baseline **was fitted on the training split**; neither baseline was tuned or selected using test scores.

Each example comes from a cleaned scientific-paper paragraph with at least 60 words, excluding abstracts and requiring at least three earlier paragraphs. Using a fixed random seed, we choose an eligible paragraph, select a 10-, 30-, or 60-word target length that fits, and cut at a word boundary with at least 15 prefix words. The prefix is the input; the next original words are the gold continuation. Earlier paragraphs are saved as context, but neither MS2 baseline uses them at inference. Papers, including all their examples, stay in separate training/development/test splits.

Both complete Colab evaluations are saved in `results/` and have been checked locally against their hashes and per-example scores:

| System | Test examples | ROUGE-L F1 | BERTScore F1 | Exact requested word length |
|---|---:|---:|---:|---:|
| Trigram | 150 | 0.107934 | 0.713692 | 100.00% |
| Frozen Qwen2.5-1.5B | 150 | 0.226200 | 0.786479 | 94.67% |

The final two-page report is [report/milestone2.pdf](report/milestone2.pdf). The eight shorter Qwen outputs are included in all scores; none is silently dropped or padded. Full precision/recall, domain/length breakdowns, uncertainty intervals, model revision, and runtime provenance are preserved. All 15 tests pass. Run `python scripts/package_ms2.py` to recreate the submission zip from these artifacts.

## Human-label scope

There are no genuine acceptability labels yet. `score.py` implements optional Brier/ECE/risk–coverage checks, but reports them as not evaluated without labels. Neither ROUGE-L nor BERTScore proves factual support. No human judgments, calibration improvements, or abstention results are fabricated for M2.
