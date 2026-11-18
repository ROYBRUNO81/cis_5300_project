"""Build baseline documentation and the report from complete measured outputs."""
import argparse
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import digest, read_jsonl


def load_result(root, system):
    predictions = root / 'results' / f'{system}-test.jsonl'
    path = root / 'results' / f'{system}-test-metrics.json'
    if not path.exists():
        raise ValueError(f'missing {path}; finish the full evaluation first')
    result = json.loads(path.read_text())
    if result.get('n') != 150 or 'bertscore_f1' not in result:
        raise ValueError(f'{path}: expected all 150 examples and BERTScore')
    if result['configuration']['predictions_sha256'] != digest(predictions):
        raise ValueError(f'{path}: scores do not match the current predictions')
    if result['configuration']['gold_sha256'] != digest(root / 'data/processed/test.jsonl'):
        raise ValueError('metrics refer to a different gold dataset')
    rows = read_jsonl(predictions)
    if system == 'strong':
        run = json.loads(Path(str(predictions) + '.run.json').read_text())
        if run.get('status') != 'complete' or run.get('n') != 150 or run.get('model') != 'Qwen/Qwen2.5-1.5B':
            raise ValueError('a completed Qwen2.5-1.5B run is required')
        if any(row.get('system') != 'qwen_prefix_only' or row.get('run_id') != run['run_id'] for row in rows):
            raise ValueError('strong predictions do not match their run metadata')
    return result, rows


def metric_table(simple, strong):
    lines = ['| System | ROUGE-L P | ROUGE-L R | ROUGE-L F1 | BERTScore P | BERTScore R | BERTScore F1 | Exact word length |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in [('Trigram', simple), ('Qwen2.5-1.5B', strong)]:
        fields = [result[key] for key in ['rougeL_precision', 'rougeL_recall', 'rougeL_f1',
                                          'bertscore_precision', 'bertscore_recall', 'bertscore_f1']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{value:.4f}' for value in fields) + f" | {100*result['exact_length']:.1f}% |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    simple, simple_predictions = load_result(root, 'simple')
    strong, strong_predictions = load_result(root, 'strong')
    if simple['configuration']['bertscore'] != strong['configuration']['bertscore']:
        raise ValueError('the two baselines must use the same BERTScore settings')
    table = metric_table(simple, strong)
    example = read_jsonl(root / 'data/processed/test.jsonl')[0]
    simple_sample = next(r for r in simple_predictions if r['example_id'] == example['example_id'])
    strong_sample = next(r for r in strong_predictions if r['example_id'] == example['example_id'])
    simple_doc = f'''# Simple baseline: trigram with hard backoff

I fit a whitespace-word trigram model using only the 237 training papers. Training contexts are deduplicated by `(paper_id, paragraph_index)`; a full training-context paragraph takes precedence over its shorter prefix-plus-target span. Two BOS markers mark paragraph starts. At each step, select the most frequent continuation of the last two words. If that context is unseen, back off to the last word, then to the unigram mode. Ties use lexicographic order. The model emits exactly the requested number of words and never reads the test target or test context for generation. Training/evaluation paper overlap is rejected.

This is a simple open-ended-generation baseline, not a majority-class classifier. No model selection or tuning uses test scores. Hard backoff is not interpolated smoothing. The saved mean word probability is an uncalibrated diagnostic, not the probability a writer accepts the continuation.

## Run

```bash
python simple-baseline.py --train data/processed/train.jsonl --inputs data/processed/test.jsonl --output results/simple-test.jsonl
python score.py --gold data/processed/test.jsonl --predictions results/simple-test.jsonl --bert-model .models/bertscore --output results/simple-test-metrics.json --per-example results/simple-test-scores.jsonl
```

## Measured full-test results

{table}

The table includes both baselines for comparison. See `results/simple-test-metrics.json` for paper-bootstrap intervals and domain/length breakdowns. These metrics quantify reference similarity, not factual correctness. Repetition is a known limitation of greedy count-based generation.

## Actual sample output

Example `{example['example_id']}` ({example['target_words']} requested words):

```json
{json.dumps(simple_sample, ensure_ascii=False)}
```

The program writes one JSON object per test ID with `prediction`, raw `confidence`, and its unit/type. Output order is aligned by ID by the scorer.
'''
    strong_doc = f'''# Strong baseline: frozen prefix-only Qwen2.5-1.5B

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

{table}

See `results/strong-test.jsonl.run.json` for the actual checkpoint revision, precision, hardware, seed, software versions, input hashes, and generation time. See `results/strong-test-metrics.json` for the scorer settings, paper-bootstrap intervals, and breakdowns. The baseline was chosen before scoring, not selected by test performance.

## Actual sample output

Example `{example['example_id']}` ({example['target_words']} requested words):

```json
{json.dumps(strong_sample, ensure_ascii=False)}
```

These are single-reference, single-run similarity results. They do not establish calibrated confidence, factual grounding, or robustness to pretrained-data overlap.
'''
    (root / 'simple-baseline.md').write_text(simple_doc)
    (root / 'strong-baseline.md').write_text(strong_doc)
    (root / 'report').mkdir(exist_ok=True)
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyMS2', fontName='Helvetica', fontSize=10, leading=13, spaceAfter=7))
    styles.add(ParagraphStyle(name='TitleMS2', fontName='Helvetica-Bold', fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=8))
    story = []
    def paragraph(text):
        story.append(Paragraph(text, styles['BodyMS2']))
    def heading(text):
        story.append(Paragraph(text, styles['Heading2']))
    def add_table(data, widths=None):
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='CENTER')
        t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,-1),8),
                              ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
                              ('LINEABOVE',(0,0),(-1,0),0.7,colors.black), ('LINEBELOW',(0,0),(-1,0),0.5,colors.black),
                              ('LINEBELOW',(0,-1),(-1,-1),0.7,colors.black)]))
        story.extend([t, Spacer(1,8)])
    story.append(Paragraph('ContextComplete: Milestone 2', styles['TitleMS2']))
    paragraph('Bruno Ndiba Mbwaye Roy, Sheila Nalubega, Hansheng Zhu<br/>CIS 5300, Fall 2026')
    heading('1. Task and evaluation')
    paragraph('We predict the next 10, 30, or 60 words of a partially written scientific paragraph. We use the MS1 dataset: 1,182 training examples from 237 papers and 150 examples each for development and test from 30 papers per split. Test examples are balanced across NLP, ML, and CV. Splits are paper-disjoint; context contains only earlier paragraphs. Both M2 baselines are prefix-only, leaving retrieval and calibrated selective suggestion as later extensions.')
    paragraph('Our primary measure is example-macro ROUGE-L F1 [1]. Let L be the longest common subsequence of candidate C and reference R after English normalization and Porter stemming:')
    paragraph('<b>P = L/|C|; R = L/|R|; F1 = 2PR/(P+R); Score = (1/N) sum_i F1_i.</b>')
    paragraph('Zero-denominator cases receive zero. We report precision and recall as well. The secondary measure is BERTScore [2]: each contextual candidate token greedily matches the reference token with highest cosine similarity, and vice versa. Average these maxima to obtain precision and recall, then compute F1 per example. We use frozen DistilBERT with five layers, no IDF weights, and no baseline rescaling, and macro-average over examples. This lightweight setting is not directly comparable to RoBERTa-large scores.')
    paragraph('The scoring script aligns IDs and rejects missing, extra, or duplicate predictions. It scores only continuations and also reports word-length adherence and domain/length breakdowns. ROUGE-L and BERTScore measure similarity to one reference, not factuality or writer acceptability. A different valid continuation may score poorly.')
    heading('2. Baselines')
    paragraph('<b>Simple: trigram with backoff.</b> We count whitespace-word trigrams in deduplicated training-paper paragraphs and generate greedily. Unseen contexts back off to bigrams and then unigrams; ties are lexicographic. No development/test paper enters fitting. Exactly the requested number of words is emitted. This baseline can repeat common phrases and cannot represent long-range dependencies.')
    paragraph('<b>Strong: frozen Qwen2.5-1.5B.</b> We use the public base causal LM [3], not a chat model, with the raw prefix and no retrieved context. Greedy decoding uses one beam, seed 5300, at most 2,048 input tokens and 256 new tokens. We stop after an extra word appears and trim to the requested complete-word budget. Short EOS/budget-limited outputs remain short. No hyperparameter tuning or selection uses test scores. Checkpoint revisions and actual runtime versions are recorded with the outputs.')
    story.append(PageBreak())
    heading('3. Measured test results')
    data = [['System','RL P','RL R','RL F1','BS P','BS R','BS F1','Length']]
    for name, result in [('Trigram',simple),('Qwen 1.5B',strong)]:
        data.append([name]+[f'{result[k]:.4f}' for k in ['rougeL_precision','rougeL_recall','rougeL_f1','bertscore_precision','bertscore_recall','bertscore_f1']]+[f"{100*result['exact_length']:.1f}%"])
    add_table(data)
    paragraph('<b>Table 1.</b> All 150 test examples. RL = ROUGE-L; BS = BERTScore. Higher is better. Length is the fraction with exactly the requested number of whitespace words.')
    delta = strong['rougeL_f1'] - simple['rougeL_f1']
    paragraph(f"Qwen's ROUGE-L F1 differs from the trigram by {delta:+.4f}. The two models' BERTScore F1 values are {strong['bertscore_f1']:.4f} and {simple['bertscore_f1']:.4f}, respectively. We report these as observed similarity differences, not a demonstrated gain in factual correctness.")
    for name, result in [('Trigram',simple),('Qwen',strong)]:
        ci = result['confidence_intervals_95']['rougeL_f1']
        paragraph(f'{name}: paper-cluster bootstrap 95% ROUGE-L F1 interval [{ci[0]:.4f}, {ci[1]:.4f}] (1,000 resamples; seed 5300). These are separate model intervals, not a paired significance test.')
    breakdown = [['Group','N','Trigram RL F1','Qwen RL F1','Trigram BS F1','Qwen BS F1']]
    for key, label in [('by_domain','Domain'),('by_target_words','Words')]:
        for group in sorted(simple[key], key=lambda x: int(x) if x.isdigit() else x):
            a,b = simple[key][group],strong[key][group]
            breakdown.append([group.upper() if key=='by_domain' else group+' words',str(a['n'])]+[f'{v:.4f}' for v in [a['rougeL_f1'],b['rougeL_f1'],a['bertscore_f1'],b['bertscore_f1']]])
    add_table(breakdown)
    paragraph('<b>Table 2.</b> Domain and requested-length breakdowns. Length bins contain 59, 59, and 32 examples, so their uncertainty differs.')
    heading('4. Limitations and next steps')
    paragraph('The test set has only 30 papers, and paper-level splitting does not exclude overlap with pretraining data. PDF extraction can retain artifacts, and a single author continuation does not exhaust valid answers. Unrescaled BERTScore can be relatively high even for weak continuations. We will next compare retrieved context against prefix-only and random context, then collect genuine acceptability labels for calibration and abstention. The scorer implements optional Brier/ECE/risk-coverage evaluation, but no human calibration results are claimed in M2. Raw token probabilities are not calibrated acceptability probabilities.')
    paragraph('<b>Reproduction:</b> see scoring.md, simple-baseline.md, and strong-baseline.md. Results include per-example predictions/scores, paper/length breakdowns, hashes, and model revisions. The Colab notebook runs generation, scoring, and this report builder; model weights are excluded from submission.')
    paragraph('<b>References.</b> [1] Lin (2004), ROUGE: A Package for Automatic Evaluation of Summaries. [2] Zhang et al. (2020), BERTScore: Evaluating Text Generation with BERT, ICLR. [3] Yang et al. (2025), Qwen2.5 Technical Report, arXiv:2412.15115.')
    document = SimpleDocTemplate(str(root/'report/milestone2.pdf'), pagesize=letter,
                                 leftMargin=0.55*inch, rightMargin=0.55*inch,
                                 topMargin=0.5*inch, bottomMargin=0.5*inch)
    document.build(story)
    (root/'report/milestone2_results.md').write_text('# Milestone 2 measured results\n\n'+table+'\n\nSee milestone2.pdf for the report and results/*-metrics.json for complete measurements.\n')
    print('Built simple-baseline.md, strong-baseline.md, report/milestone2.pdf, and the result summary.')


if __name__ == '__main__':
    main()
