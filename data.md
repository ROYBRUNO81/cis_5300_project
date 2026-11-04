# ContextComplete Data

## Overview

Each example is a point in a real scientific paper where the author is partway through a paragraph. The model sees:

- the **prefix**: the start of the paragraph, up to a random cut point;
- the **context**: every paragraph of the same paper that comes before the current paragraph, which is what the writer has already written.

It must predict the **target**: the next 10, 30 or 60 words the author actually wrote. The target paragraph and everything after it never appear in the context.

| Split | Papers | Examples | NLP / ML / CV examples |
|---|---|---|---|
| train | 237 | 1,182 | 392 / 390 / 400 |
| dev | 30 | 150 | 50 / 50 / 50 |
| test | 30 | 150 | 50 / 50 / 50 |
| **total** | **297** | **1,482** | |

| Target length | 10 words | 30 words | 60 words |
|---|---|---|---|
| Examples | 554 | 548 | 380 |

On average an example has about 30 context paragraphs.

## Where the data came from

We built the dataset ourselves from **arXiv** papers, downloaded through the official arXiv API with the `arxiv` Python package. We took 100 papers from each of three computer-science categories:

| Domain | arXiv category | Papers downloaded | Papers used |
|---|---|---|---|
| Natural Language Processing (`nlp`) | `cs.CL` | 100 | 99 |
| Machine Learning (`ml`) | `cs.LG` | 100 | 98 |
| Computer Vision (`cv`) | `cs.CV` | 100 | 100 |

How we sampled papers:

- **Years:** submissions from 2022 to 2025, spread over all 48 months. For each month we picked a random submission day and took at most 3 papers, so no single month's popular topic dominates.
- **Primary category only:** a paper counts only for its *primary* arXiv category, so no paper appears in two domains.
- **Minimum length:** papers with fewer than 4 pages are dropped.
- **Unusable papers:** 3 papers had too little clean text and were skipped (listed in `skipped.txt`).

## How examples are built

Script: `scripts/build_dataset.py`.

1. **Paragraph extraction.** We read the PDF text blocks in reading order with PyMuPDF, then:
   - rejoin words hyphenated across lines;
   - merge paragraphs split across a column or page break;
   - drop running headers and footers, figure and table captions, table and equation fragments, and blocks shorter than 25 words;
   - stop at the References section and skip Acknowledgements, Limitations and similar sections.

   Each paragraph keeps the name of the section it was found in.
2. **Target selection.** For each paper we sample up to 5 target paragraphs. A target paragraph must have at least 60 words, must have at least 3 paragraphs before it, and must not be in the abstract.
3. **Cutting.** Each target paragraph is cut at a random word. The prefix is at least 15 words long, and the target is the next 10, 30 or 60 words. Lengths are counted in whitespace-separated words, not model tokens.
4. **Context.** The context is every extracted paragraph that comes before the target paragraph. This copies a writer working through the paper: what has already been written is available, and the rest of the current paragraph and the later sections are not.

## Splits

We split **by paper, not by paragraph**, 80/10/10 within each domain, using a fixed seed (5300). All examples from one paper are in the same split, so no paper, and none of its context, appears in more than one split. `paper_splits.jsonl` records the split of each paper.

## File format

The data is in JSON Lines format: `train.jsonl`, `dev.jsonl` and `test.jsonl` hold one example per line. Fields:

| Field | Description |
|---|---|
| `example_id` | `<arxiv id>:<n>` |
| `paper_id` | arXiv id |
| `domain` | `nlp`, `ml` or `cv` |
| `title` | paper title |
| `section` | section the target paragraph is in |
| `paragraph_index` | position of the target paragraph in the paper |
| `prefix` | text the writer has typed so far in this paragraph (model input) |
| `target` | the author's real next 10/30/60 words (gold output) |
| `target_words` | 10, 30 or 60 |
| `context` | list of `{paragraph_index, section, text}` for all earlier paragraphs (model input) |

Example from `dev.jsonl`, with the context shortened to its last 2 of 33 paragraphs:

```json
{
  "example_id": "2502.11175:3",
  "paper_id": "2502.11175",
  "domain": "nlp",
  "title": "Investigating Language Preference of Multilingual RAG Systems",
  "section": "Results",
  "paragraph_index": 33,
  "prefix": "Modest Preference for the Query Languages. In addition, the generator shows a slight increase in consistency when the output language matches a query language. For instance, Korean (ko) queries yield somewhat more consistent responses",
  "target": "when the generator replies in Korean rather than when the query is in English. However, this improvement is marginal, suggesting that the overall preference toward Latin scripts remains influential. Despite",
  "target_words": 30,
  "context": [
    {"paragraph_index": 31, "section": "Experimental Setup", "text": "We use LaBSE for measuring multilingual semantic similarity (Feng et al., 2022). And we use aya-expanse-8B, Llama-3.1-8B ..."},
    {"paragraph_index": 32, "section": "Results", "text": "Strong Preference for Latin Script Languages. Figure 3 indicates that the generator produces more consistent responses i ..."}
  ]
}
```

Other files:

| File | Contents |
|---|---|
| `paper_splits.jsonl` | `{paper_id, split}` for every paper used |
| `stats.json` | counts per split, domain and target length |
| `data/raw/metadata.jsonl` (repo only) | title, authors, categories, date, URL and license for all 300 downloaded papers |

## Reproducing

```bash
pip install -r requirements.txt
python scripts/download_papers.py      # ~45 min, 300 PDFs (1.1 GB) into data/raw/pdfs/
python scripts/build_dataset.py        # ~20 s, writes data/processed/ and data/data.tar.gz
```

## Known limitations

- PDF extraction is imperfect. A few paragraphs still contain inline math symbols, and some paragraphs that continue after a displayed equation are merged with the text before it.
- Section names come from the headings as printed in each paper, so they are not standardized: "Experiments", "Main Results" and "Results" all appear.

## Licensing

arXiv papers carry individual licenses (stored in `metadata.jsonl`). The raw PDFs are used locally for research and are not redistributed. The archive contains only extracted text excerpts.
