# ContextComplete: Grounded Paragraph Autocompletion That Knows When It Might Be Wrong

CIS 5300 Term Project, Fall 2026: Bruno Ndiba Mbwaye Roy, Sheila Nalubega, Hansheng Zhu

## Milestone 2 work

**Status: all Milestone 2 deliverables are complete on the `ms2` branch.** We implemented a shared evaluation script, trained a simple trigram baseline, and evaluated a frozen prefix-only Qwen2.5-1.5B baseline on the same 150 test examples. Full inference ran on Colab GPU; measured predictions, scores, and run metadata are committed under [`results/`](results/).

### MS2 checklist

- [x] Implement [`score.py`](score.py), taking system predictions and gold answers with strict example-ID matching.
- [x] Define ROUGE-L and BERTScore, cite their papers, and provide CLI examples in [`scoring.md`](scoring.md).
- [x] Implement [`simple-baseline.py`](simple-baseline.py): a training-only trigram language model with backoff.
- [x] Document simple-baseline usage, sample output, and full-test scores in [`simple-baseline.md`](simple-baseline.md).
- [x] Integrate and evaluate the frozen pretrained Qwen2.5-1.5B model in [`strong-baseline.py`](strong-baseline.py).
- [x] Document strong-baseline usage, decoding settings, and measured results in [`strong-baseline.md`](strong-baseline.md).
- [x] Evaluate both systems with the same metric configuration on all 150 test examples.
- [x] Prepare the required two-page PDF: [`report/milestone2.pdf`](report/milestone2.pdf).
- [x] Validate all 15 automated tests, result hashes, complete prediction coverage, and report length.
- [x] Provide a reproducible Colab workflow and validated submission-packaging script.
- [x] Publish results and documentation to the `ms2` branch.
- [ ] Upload the final archive and report to the course submission portal (manual submission).

### Measured test results

| System | Examples | ROUGE-L F1 (primary) | BERTScore F1 | Exact requested word length |
|---|---:|---:|---:|---:|
| Trigram with backoff | 150 | 0.107934 | 0.713692 | 100.00% |
| Frozen Qwen2.5-1.5B | 150 | 0.226200 | 0.786479 | 94.67% |

Both systems generate 10-, 30-, or 60-word continuations from the prefix alone. Qwen's eight shorter outputs remain in the evaluation. BERTScore uses DistilBERT with five encoder layers, without IDF or baseline rescaling. Full precision/recall/F1, paper-bootstrap intervals, and domain/length breakdowns are saved in the metric JSON files. These scores measure reference similarity, not factual support or calibrated acceptability.

### Reproduce and package

See [`MS2_README.md`](MS2_README.md) for setup and exact generation/scoring commands, or run [`notebooks/MS2_Colab.ipynb`](notebooks/MS2_Colab.ipynb) on a GPU runtime. Dependencies are in [`requirements-ms2.txt`](requirements-ms2.txt). Rebuild the submission archive from the committed results with:

```bash
python scripts/package_ms2.py
```

This creates `MS2_submission.zip` locally. Model weights, caches, virtual environments, and the generated zip are excluded from Git. Decisions and the detailed implementation checklist are tracked in [`MS2_ADR.md`](MS2_ADR.md).

### Next steps after MS2

These are future project extensions, not missing baseline deliverables:

- [ ] Compare retrieved context against prefix-only and random-context generation.
- [ ] Collect human labels for relevance, fluency, factual support, and acceptability.
- [ ] Train a quality predictor and calibrate confidence on held-out development labels.
- [ ] Evaluate Brier score, calibration error, and confidence-based abstention on separate test labels.

The scorer supports optional labeled confidence evaluation, but calibration and abstention results have not yet been measured.

## Research question

Autocomplete tools mostly look at the few words right before the cursor. When people write a paper or a report, much of what comes next is already written somewhere else in the document, such as the methods section, a results table or an earlier draft. A model that ignores those sources can produce fluent text that contradicts the facts, for example by inventing an accuracy number. It also gives the writer no signal about how far to trust a suggestion.

We ask two questions:

1. **Grounding.** Does retrieving passages from the rest of the document make paragraph continuations more faithful to what the author actually wrote?
2. **Confidence.** Can the system produce a calibrated score between 0 and 1 that predicts whether its continuation is acceptable? With that score, the system can stay silent when it is likely wrong.

### Task definition

- **Input:** a partially written paragraph (the *prefix*) plus *context*: every paragraph of the same paper that comes before it, which is what the writer has already written. The target paragraph and everything after it are never in the context.
- **Output:** (a) a continuation of 10, 30 or 60 words, and (b) a confidence score `p ∈ [0, 1]` that the continuation is acceptable.

### Planned systems

| System | Description |
|---|---|
| Simple baseline | Trigram language model trained on the training papers |
| Strong baseline | Small pretrained causal LM (Qwen2.5-1.5B / Llama-3.2-1B), prefix only |
| Extension 1: retrieval | BM25, then dense embeddings, with retrieved passages added to the prompt; optional LoRA fine-tuning |
| Extension 2: confidence | (i) mean token probability, (ii) NLI evidence score against retrieved passages, (iii) learned quality predictor, plus P(True)-style self-evaluation (Kadavath et al., 2022) |

### Evaluation

- **Completion quality:** BERTScore and ROUGE-L against the real continuation.
- **Confidence:** Brier score, expected calibration error (ECE) and AUROC on a few hundred hand-labelled acceptable/unacceptable completions, plus risk–coverage curves that show suggestion quality as the system abstains more often.

## Repository layout

```
README.md                 project overview (this file)
data.md                   data description (Milestone 1 deliverable)
requirements.txt
scripts/
  download_papers.py      step 1: download raw arXiv PDFs + metadata
  build_dataset.py        step 2: PDFs -> prefix/target/context examples, 80/10/10 split, data.tar.gz
report/
  milestone1_data_description.md
data/
  raw/                    PDFs (not committed) + metadata.jsonl
  processed/              train/dev/test.jsonl, paper_splits.jsonl, stats.json
  data.tar.gz             Milestone 1 submission archive (not committed)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_papers.py          # 100 papers each from cs.CL, cs.LG, cs.CV
python scripts/build_dataset.py            # build examples + splits + archive
```

## Milestones

| Milestone | Due | Deliverables |
|---|---|---|
| Proposal | Oct 14 | Proposal PDF |
| Milestone 1 | Nov 4 | Literature review (3 papers) + data description report, `data.md`, train/dev/test tar.gz |
| Milestone 2 | Nov 18 | `score.py` + `scoring.md`, simple and strong baselines + markdowns, report |
| Milestone 3 | Dec 7 | First extension + report, draft slides |
| Final | Dec 17 | ACL-style report, slides, video, `data/` `code/` `output/` |

## Literature review (Milestone 1)

| Paper | Area | Reader |
|---|---|---|
| Shi et al., *REPLUG: Retrieval-Augmented Black-Box Language Models* (NAACL 2024) | Retrieval-augmented generation | Teammate |
| Bavarian et al., *Efficient Training of Language Models to Fill in the Middle* (2022) | Contextual text completion | Teammate |
| Kadavath et al., *Language Models (Mostly) Know What They Know* (2022) | Confidence / calibration | Bruno |
