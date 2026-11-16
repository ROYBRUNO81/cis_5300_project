# Evaluation: continuation similarity and optional confidence evaluation

## Task and gold data

The system extends a scientific paragraph prefix by 10, 30, or 60 **whitespace-separated words**. Gold answers are the original author continuations in `data/processed/{train,dev,test}.jsonl`. M2 evaluates all 150 test examples from 30 papers, balanced at 50 examples in each of NLP, ML, and CV. Test lengths are not balanced: 59 ten-word, 59 thirty-word, and 32 sixty-word targets.

The primary number is **example-macro ROUGE-L F1**. Secondary BERTScore F1 measures contextual semantic similarity. Both are reference-similarity metrics, **not factuality, acceptability, or calibration measurements**. A valid alternative continuation can score poorly; an unsupported continuation can score well. We retain the single-reference limitation and plan human judgments for the later confidence extension.

## Formal metric definitions

For one candidate token sequence \(C\) and reference \(R\), let \(L\) be their longest common subsequence length. We use the F1 variant of ROUGE-L:

\[
P_L=\frac{L}{|C|},\qquad R_L=\frac{L}{|R|},\qquad
F_L=\frac{2P_LR_L}{P_L+R_L}.
\]

If a denominator is zero, the corresponding score is zero. `rouge-score==0.1.2` supplies the implementation with `RougeScorer(['rougeL'], use_stemmer=True)`: lowercase English alphanumeric tokenization followed by Porter stemming. This metric tokenization differs from the whitespace word count used for requested output lengths. We score only the generated continuation, not the prefix or context. The dataset score is \(N^{-1}\sum_{i=1}^{N}F_{L,i}\), not ROUGE over concatenated papers. Precision and recall are also macro-averaged and reported.

For BERTScore, independently encode the candidate and reference with a frozen contextual encoder. Let normalized contextual embeddings be \(c_i\) and \(r_j\). Without IDF weights:

\[
P_B=\frac{1}{|C|}\sum_i\max_j c_i^\top r_j,\qquad
R_B=\frac{1}{|R|}\sum_j\max_i c_i^\top r_j,\qquad
F_B=\frac{2P_BR_B}{P_B+R_B}.
\]

We average per-example F1, precision, and recall separately. The M2 configuration is `bert-score==0.3.13`, `distilbert/distilbert-base-uncased`, **five retained Transformer layers**, `idf=False`, and `rescale_with_baseline=False`. Padding/special-token handling follows BERTScore's implementation. Empty candidate strings receive zero. The exact checkpoint revision is stored in result JSON. This deliberately lightweight encoder is **not** the standard RoBERTa-large setting; do not compare the raw numbers directly to papers using a different encoder, layer, IDF, or rescaling configuration.

Also report exact requested-word-length rate, mean output length, empty-output rate, and breakdowns by domain and target length. Default 95% uncertainty intervals are percentile intervals from 1,000 **paper-cluster** bootstrap samples with replacement, seed 5300; all examples of each drawn paper are retained. These intervals describe this small test sample, not pretrained-data contamination or annotation uncertainty.

## Command-line use

From the repository root:

```bash
python3 -m venv .venv-ms2
source .venv-ms2/bin/activate
pip install -r requirements-ms2.txt
python scripts/download_models.py --output .models

python score.py \
  --gold data/processed/test.jsonl \
  --predictions results/simple-test.jsonl \
  --bert-model .models/bertscore \
  --output results/simple-test-metrics.json \
  --per-example results/simple-test-scores.jsonl
```

For a fast lexical-only check, add `--no-bertscore` (and omit `--bert-model`). This **does not produce the full M2 evaluation**, so missing BERTScore is not reported as zero. `--device cpu`, `mps`, or `cuda` selects hardware; default `auto` prefers CUDA, then MPS. CLI output is JSON on stdout, optionally saved with `--output`. Scores are fractions, not percentages.

Example measured simple-baseline output (abridged):

```json
{
  "primary_metric": "rougeL_f1",
  "n": 150,
  "rougeL_f1": 0.1079339534,
  "bertscore_f1": 0.7136913951,
  "exact_length": 1.0,
  "empty": 0.0
}
```

## Input/output contracts and checks

Gold uses the existing example fields `example_id`, `paper_id`, `domain`, `target`, and `target_words`. Predictions are JSONL with exactly one row per gold ID:

```json
{"example_id":"paper:example", "prediction":"generated continuation", "confidence":0.45}
```

`prediction` must be a string; additional provenance fields are allowed. Rows may be in any order. Missing/extra IDs, duplicates, and incorrect gold word counts are errors, rather than silently scoring a subset. Confidence is optional unless human-label evaluation is requested. Raw mean word/subword probabilities are saved by the baselines but are **not calibrated acceptability estimates**, and are not directly comparable across their different units.

## Confidence metrics: implemented but not yet measured on human data

For true binary acceptability labels \(y_i\) and predicted confidence \(p_i\):

\[
\operatorname{Brier}=\frac{1}{N}\sum_i(p_i-y_i)^2,
\qquad \operatorname{ECE}=\sum_{b=1}^{10}\frac{|B_b|}{N}
\left|\operatorname{mean}_{i\in B_b}p_i-\operatorname{mean}_{i\in B_b}y_i\right|.
\]

Use ten equal-width bins on [0,1], placing 1 in the last bin and ignoring empty bins. At threshold \(t\), coverage is the fraction with \(p_i\ge t\); risk is the unacceptable fraction among those retained. Evaluate thresholds at distinct confidences, keeping entire tie groups together. We additionally report a right-endpoint step-function area under this risk–coverage curve. At zero coverage set the plotting origin to risk zero; this is a convention, not observed quality.

The optional `--labels human-labels.jsonl` file contains `example_id`, binary `acceptable`, and `prediction_sha256` (the SHA-256 of that exact prediction's UTF-8 text). Labels may cover a subset, and calibration output reports that subset size. A mismatched prediction hash is rejected so labels for one system cannot accidentally be reused for another. There are **no genuine acceptability labels in MS1**, so normal M2 output explicitly marks calibration as not evaluated. Unit-test labels are synthetic and are not experimental results. Do not turn ROUGE/BERTScore into alleged human labels. AUROC, calibration fitting, NLI support scoring, and learned abstention thresholds remain later extensions.

## References

- Lin, C.-Y. (2004). [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/).
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2020). [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675).
- Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of Probability. *Monthly Weather Review*, 78(1), 1–3. [DOI](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2).
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
