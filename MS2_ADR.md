# Milestone 2 — checklist and architecture decisions

## To-do / completion status

- [x] Create a separate `ms2` branch without changing `main`.
- [x] Verify that the repository uses the MS1 297-paper NLP/ML/CV dataset.
- [x] Implement strict ID-aligned scoring with ROUGE-L and optional BERTScore.
- [x] Implement a training-only trigram/backoff simple baseline.
- [x] Run the simple baseline on all 150 test examples (ROUGE-L measured).
- [x] Implement frozen prefix-only Qwen2.5-1.5B generation.
- [x] Test scoring, leakage guards, generation-length handling, and calibration edge cases (15 tests passed).
- [x] Measure simple-baseline BERTScore on all 150 test examples.
- [x] Add and validate a Colab GPU notebook for the complete M2 workflow.
- [x] Write metric definitions and CLI examples in `scoring.md`.
- [x] Write `simple-baseline.md` with actual output and full-test measurements.
- [x] Write strong-baseline usage documentation and replace the initial pending draft with actual results.
- [x] Complete strong-baseline inference on all 150 test examples.
- [x] Measure both baselines with the same ROUGE-L and BERTScore configuration.
- [x] Write `scoring.md`, `simple-baseline.md`, and `strong-baseline.md` with actual results and commands.
- [x] Generate and verify the required two-page report.
- [x] Verify deliverables, publish the branch, and prepare a submission archive.

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

**Status:** implemented and evaluated on all 150 test examples.

Train a whitespace-word trigram model on deduplicated training-paper paragraphs available through training contexts and prefix–target spans. Use greedy maximum-count continuation, backing off from trigram to bigram to unigram when the context is unseen. Resolve ties lexicographically. Emit exactly the requested number of words. No development/test text is used for fitting. The full run and score artifacts are archived in this branch.

The branch now contains the measured full-test predictions and scores: ROUGE-L F1 **0.107934**, BERTScore F1 **0.713691**, and 100% exact requested word length. Full precision/recall and group breakdowns are in `results/simple-test-metrics.json`.

**Reason:** a majority-label classifier is not meaningful for open-ended generation. This is the simple language-model baseline promised in the proposal.

## ADR-004 — strong baseline

**Status:** implemented and evaluated on all 150 test examples.

**Execution update, 2026-11-16:** local inference on the 8 GB Mac encountered memory/swap pressure. An incomplete local output is retained only in ignored `runs/`, not presented as a test result. Continue on Colab GPU using `notebooks/MS2_Colab.ipynb`; retain the same model/decoding settings. Direct-device loading avoids a simultaneous CPU/GPU parameter copy. Actual software versions are logged; the dependency range permits Colab's existing PyTorch >=2.7 rather than forcing a large CUDA downgrade.

**Completion update, 2026-11-16:** the user returned the Colab archive. It contains a complete CUDA float16 run of all 150 examples, checkpoint revision `8faed761d45a263340a0528343f099c05c9a4323`, and 187.6 seconds of recorded generation time (excluding model loading). Measured ROUGE-L F1 is **0.226200** and BERTScore F1 is **0.786479**. Exact word-length adherence is **142/150 (94.67%)**; eight suggestions were shorter, none hit the 256-token cap, and all are retained in scoring. No suggestion is empty. These are similarity results, not proof of factuality or calibration.

Use the public **base** checkpoint `Qwen/Qwen2.5-1.5B`, frozen, without fine-tuning, chat formatting, instructions, or retrieval. Input is only the paragraph prefix. Greedy decoding, one beam, seed 5300, at most 2,048 input tokens and 256 newly generated tokens. Stop after observing one more whitespace word than requested, then retain the first 10/30/60 complete words. Early EOS or a token-budget limit can produce a short output and must be reported, not padded with gold text. Save model revision, input hashes, configuration, and resumable predictions.

**Reason:** this isolates the benefit of pretrained language modeling and leaves retrieval and calibrated selective suggestion as later extensions. It follows the standard pretrained causal-LM baseline setting; it does not attempt to reproduce REPLUG or ScholarCopilot's full training pipeline.

## ADR-005 — confidence and calibration scope

**Status:** accepted; human annotation deferred.

Store the mean probability of emitted units as a raw diagnostic: words for the trigram, subword tokens for Qwen. These are **not calibrated acceptability probabilities**, and their units differ. Implement optional Brier/ECE/risk–coverage evaluation only when genuine binary human labels are supplied and bound to the exact generated string by SHA-256. Do not manufacture acceptable/unacceptable labels from ROUGE/BERTScore. Synthetic labels may be used only for unit tests.

**Reason:** MS1 has reference continuations but no human judgments of new suggestions. Reporting calibration without such labels would misrepresent the experiment. Later predictor training, calibration, threshold selection, and final testing must use disjoint paper partitions.

## ADR-006 — reproducibility and publication

**Status:** accepted.

Keep code and small measured outputs on `ms2`. Do not commit model weights, virtual environments, caches, access tokens, or raw PDFs. Preserve the existing MS1 files and dependencies; install M2 dependencies using `requirements-ms2.txt`. Record model revisions/checksums, software versions, seeds, and hardware. The report must clearly distinguish measured results from planned extensions. No training runs or paid APIs are required for these baselines.

The real Colab report has been verified to contain **two pages**. Full-test IDs, dataset/prediction/script hashes, word-length adherence, and score aggregation were checked locally; ROUGE-L was independently recomputed. Both baselines use the same DistilBERT revision and scorer settings. The 15 automated tests also pass; their synthetic report-layout fixture is not a benchmark result. The final archive includes actual measured predictions and metrics, formal scoring definitions, usage documentation, and the report. Human calibration remains an explicitly deferred extension, not an unfinished M2 baseline requirement.
