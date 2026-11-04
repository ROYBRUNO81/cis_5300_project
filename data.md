# ContextComplete Data

## Source

We built the dataset ourselves from **arXiv** papers, downloaded through the official arXiv API with the [`arxiv`](https://pypi.org/project/arxiv/) Python package. It covers three related computer-science domains:

| Domain | arXiv category | Papers |
|---|---|---|
| Natural Language Processing | `cs.CL` | 100 |
| Machine Learning | `cs.LG` | 100 |
| Computer Vision | `cs.CV` | 100 |

How we sampled papers:

- **Years:** submissions from 2022 to 2025, spread evenly over all 48 months. For each month the script picks a random submission day and takes at most 3 papers. This keeps any one month's popular topic from dominating the corpus.
- **Primary category only:** a paper is kept only if its *primary* arXiv category matches. A cs.CL paper cross-listed to cs.LG therefore appears once, so no paper is in two domains.
- **Minimum length:** PDFs with fewer than 4 pages, such as extended abstracts, are dropped.
- **Reproducible:** the random seed is fixed (`--seed 5300`). Re-running the script reuses PDFs that are already on disk.

Paper metadata is stored in `data/raw/metadata.jsonl`, one JSON object per paper:

```json
{"paper_id": "2405.13622", "title": "Automated Evaluation of Retrieval-Augmented Language Models with Task-Specific Exam Generation",
 "authors": ["..."], "primary_category": "cs.CL", "categories": ["cs.CL", "cs.AI"], "domain": "nlp",
 "published": "2024-05-22", "abs_url": "http://arxiv.org/abs/2405.13622v1",
 "pdf_path": "pdfs/nlp/2405.13622.pdf", "license": "..."}
```

Command:

```bash
python scripts/download_papers.py --per-category 100 --categories cs.CL cs.LG cs.CV
```

## Building examples from papers

Each paper is turned into autocomplete examples:

1. **Extract paragraphs** from the PDF with PyMuPDF. We drop references, acknowledgements, captions, tables and formula-heavy blocks.
2. **Choose target paragraphs** with enough words, and cut each one at a random point into a **prefix** and a **target** continuation of 10, 30 or 60 words.
3. **Build the linked resources** from every other paragraph of the same paper. The target paragraph is **removed**, so the model can't copy the answer.

## Splits

We split **by paper, not by paragraph**, about 80/10/10 within each domain. All examples from one paper go to the same split, so no paper, and none of its context, appears in more than one split.

| Split | Papers | Examples |
|---|---|---|
| train | _tbd_ | _tbd_ |
| dev | _tbd_ | _tbd_ |
| test | _tbd_ | _tbd_ |

## Licensing

arXiv papers have individual licenses (stored in the `license` field). The raw PDFs are used locally for research and are **not** redistributed in this repo or in the submission archive.
