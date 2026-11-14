# Milestone 2 — checklist and architecture decisions

## To-do / completion status

- [x] Create a separate `ms2` branch without changing `main`.
- [x] Verify that the repository uses the MS1 297-paper NLP/ML/CV dataset.
- [x] Implement strict ID-aligned scoring with ROUGE-L and optional BERTScore.
- [x] Implement a training-only trigram/backoff simple baseline.
- [x] Run the simple baseline on all 150 test examples (ROUGE-L measured).
- [x] Implement frozen prefix-only Qwen2.5-1.5B generation.
- [ ] Test scoring, leakage guards, generation-length handling, and calibration edge cases.
- [ ] Complete strong-baseline inference on all 150 test examples.
- [ ] Measure both baselines with the same ROUGE-L and BERTScore configuration.
- [ ] Write `scoring.md`, `simple-baseline.md`, and `strong-baseline.md` with actual results and commands.
- [ ] Write and compile the required 1–2 page report.
- [ ] Verify deliverables, publish the branch, and prepare a submission archive.

Tick an item only after its implementation or measurement is verified. Do not treat generated code as a completed experiment.

## ADR-001 — dataset and evaluation units

**Status:** accepted, 2026-11-14.

Use the dataset committed under `data/processed/`: 1,182 training, 150 development, and 150 test examples from 297 papers. Domains are NLP, ML, and CV, not the older physics/economics pilot. Reference lengths are **whitespace-separated words**, not model tokens. Context consists exclusively of paragraphs earlier than the target paragraph. The local MS1 archive and repository splits have identical SHA-256 hashes.

**Reason:** maintain consistency with the submitted MS1 and prevent silently mixing dataset versions. Keep paper-level splits fixed; do not select settings using test scores.

## ADR-002 — scoring

**Status:** accepted.

Primary score: example-macro ROUGE-L F1 with the official `rouge-score` package's English normalization and Porter stemming. Secondary score: example-macro BERTScore F1 with frozen DistilBERT, five encoder layers, no IDF weighting, and no baseline rescaling. Report precision/recall/F1, domain/length breakdowns, empty outputs, and exact requested-word-length rate. Require every gold ID exactly once and reject duplicates or extra/missing IDs. Optional confidence intervals use 1,000 paper-cluster bootstrap samples, seed 5300.

**Reason:** ROUGE-L is cheap and reproducible; BERTScore adds contextual semantic matching. Neither measures factual support or human acceptability directly. DistilBERT keeps evaluation feasible locally; this is not the paper's default RoBERTa-large configuration and scores must not be compared as though it were.

## ADR-003 — simple baseline

**Status:** implemented; full-test ROUGE-L measured, BERTScore pending.

Train a whitespace-word trigram model on deduplicated training-paper paragraphs available through training contexts and prefix–target spans. Use greedy maximum-count continuation, backing off from trigram to bigram to unigram when the context is unseen. Resolve ties lexicographically. Emit exactly the requested number of words. No development/test text is used for fitting. Test ROUGE-L F1 from the initial run: 0.107934; rerun and archive results in this branch before final reporting.

**Reason:** a majority-label classifier is not meaningful for open-ended generation. This is the simple language-model baseline promised in the proposal.

## ADR-004 — strong baseline

**Status:** implementation complete; full evaluation pending.

Use the public **base** checkpoint `Qwen/Qwen2.5-1.5B`, frozen, without fine-tuning, chat formatting, instructions, or retrieval. Input is only the paragraph prefix. Greedy decoding, one beam, seed 5300, at most 2,048 input tokens and 256 newly generated tokens. Stop after observing one more whitespace word than requested, then retain the first 10/30/60 complete words. Early EOS or a token-budget limit can produce a short output and must be reported, not padded with gold text. Save model revision, input hashes, configuration, and resumable predictions.

**Reason:** this isolates the benefit of pretrained language modeling and leaves retrieval and calibrated selective suggestion as later extensions. It follows the standard pretrained causal-LM baseline setting; it does not attempt to reproduce REPLUG or ScholarCopilot's full training pipeline.

## ADR-005 — confidence and calibration scope

**Status:** accepted; human annotation deferred.

Store the mean probability of emitted units as a raw diagnostic: words for the trigram, subword tokens for Qwen. These are **not calibrated acceptability probabilities**, and their units differ. Implement optional Brier/ECE/risk–coverage evaluation only when genuine binary human labels are supplied and bound to the exact generated string by SHA-256. Do not manufacture acceptable/unacceptable labels from ROUGE/BERTScore. Synthetic labels may be used only for unit tests.

**Reason:** MS1 has reference continuations but no human judgments of new suggestions. Reporting calibration without such labels would misrepresent the experiment. Later predictor training, calibration, threshold selection, and final testing must use disjoint paper partitions.

## ADR-006 — reproducibility and publication

**Status:** accepted.

Keep code and small measured outputs on `ms2`. Do not commit model weights, virtual environments, caches, access tokens, or raw PDFs. Preserve the existing MS1 files and dependencies; install M2 dependencies using `requirements-ms2.txt`. Record model revisions/checksums, software versions, seeds, and hardware. The report must clearly distinguish measured results from planned extensions. No training runs or paid APIs are required for these baselines.
