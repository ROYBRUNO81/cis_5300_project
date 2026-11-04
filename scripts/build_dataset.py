#!/usr/bin/env python3
"""Step 2 of the ContextComplete data pipeline: PDFs -> autocomplete examples.

For every downloaded paper:
  1. Extract clean body paragraphs in reading order (references, captions,
     tables and equation fragments are dropped; extraction stops at References).
  2. Pick up to K target paragraphs and cut each at a random word into a
     prefix and a target continuation of 10, 30 or 60 words.
  3. Attach as context every paragraph that comes BEFORE the target paragraph,
     i.e. what the writer has already written. The target paragraph and
     everything after it are never in the context.

Papers are split 80/10/10 by paper within each domain, so no paper (and none
of its text) appears in more than one split.

Usage:
  python scripts/build_dataset.py                    # reads data/raw, writes data/processed
"""
import argparse
import json
import random
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

TARGET_LENGTHS = [10, 30, 60]       # words
MIN_PREFIX_WORDS = 15
MIN_PARAGRAPH_WORDS = 25            # shorter blocks are fragments, not paragraphs
MIN_TARGET_PARAGRAPH_WORDS = 60
MIN_PRECEDING_PARAGRAPHS = 3        # target must have some context before it
MIN_PARAGRAPHS_PER_PAPER = 10

STOP_HEADINGS = re.compile(r"^(\d+\.?\s*)?(references|bibliography|reference)\s*$", re.I)
SKIP_HEADINGS = re.compile(r"^(\d+\.?\s*)?(acknowledge?ments?|author contributions|ethics statement|"
                           r"impact statement|broader impacts?|limitations)\s*$", re.I)
HEADING = re.compile(r"^((\d+(\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+)?[A-Z][A-Za-z0-9 ,:&()\-–/']{1,80}$")
NAMED_HEADINGS = {"abstract", "introduction", "related work", "background", "method", "methods",
                  "methodology", "experiments", "experimental setup", "results", "discussion",
                  "conclusion", "conclusions", "evaluation", "analysis"}
CAPTION = re.compile(r"^(figure|fig\.|table|algorithm|listing)\s*\d+", re.I)


def clean(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    return " ".join(text.split())


def is_heading(text):
    words = text.split()
    if not words or len(words) > 12:
        return False
    if text.lower().strip(" .") in NAMED_HEADINGS:
        return True
    return bool(HEADING.match(text)) and not text.endswith(".") and re.match(r"^(\d|[A-Z]\.|[IVX]+\.)", text) is not None


def alpha_ratio(text):
    return sum(c.isalpha() for c in text) / max(len(text), 1)


def extract_paragraphs(pdf_path):
    """Return ordered body paragraphs: [{"index", "section", "text"}]."""
    paragraphs, section, skipping = [], "front", False
    with pymupdf.open(pdf_path) as doc:
        blocks = [clean(b[4]) for page in doc for b in page.get_text("blocks") if b[6] == 0]
    # Short text repeated on several pages is a running header/footer.
    counts = Counter(b for b in blocks if len(b.split()) <= 12)
    repeated = {b for b, n in counts.items() if n >= 3}
    for text in blocks:
        if not text or text.startswith("arXiv:") or text in repeated:
            continue
        if STOP_HEADINGS.match(text):
            return paragraphs
        if is_heading(text):
            section = re.sub(r"^(\d+(\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+", "", text)
            if section.isupper():
                section = section.title()
            skipping = bool(SKIP_HEADINGS.match(text))
            continue
        if text.lower() == "abstract":
            section, skipping = "Abstract", False
            continue
        if skipping or CAPTION.match(text) or alpha_ratio(text) < 0.6:
            continue
        # A block that starts lowercase continues the previous paragraph
        # (split by a column/page break or an equation).
        prev = paragraphs[-1] if paragraphs else None
        if prev and text[0].islower() and not prev["text"].endswith((".", "!", "?")):
            prev["text"] += " " + text
            continue
        paragraphs.append({"section": section, "text": text})
    return paragraphs


def finalize(paragraphs):
    kept = [p for p in paragraphs if len(p["text"].split()) >= MIN_PARAGRAPH_WORDS]
    for i, p in enumerate(kept):
        p["index"] = i
    return kept


def make_examples(paper, paragraphs, rng, per_paper):
    candidates = [p for p in paragraphs
                  if p["index"] >= MIN_PRECEDING_PARAGRAPHS
                  and len(p["text"].split()) >= MIN_TARGET_PARAGRAPH_WORDS
                  and p["section"].lower() != "abstract"]
    rng.shuffle(candidates)
    examples = []
    for p in sorted(candidates[:per_paper], key=lambda p: p["index"]):
        words = p["text"].split()
        length = rng.choice([L for L in TARGET_LENGTHS if len(words) >= MIN_PREFIX_WORDS + L] or [10])
        cut = rng.randint(MIN_PREFIX_WORDS, len(words) - length)
        examples.append({
            "example_id": f"{paper['paper_id']}:{len(examples)}",
            "paper_id": paper["paper_id"],
            "domain": paper["domain"],
            "title": paper["title"],
            "section": p["section"],
            "paragraph_index": p["index"],
            "prefix": " ".join(words[:cut]),
            "target": " ".join(words[cut:cut + length]),
            "target_words": length,
            "context": [{"paragraph_index": c["index"], "section": c["section"], "text": c["text"]}
                        for c in paragraphs[:p["index"]]],
        })
    return examples


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--per-paper", type=int, default=5)
    ap.add_argument("--seed", type=int, default=5300)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    papers = [json.loads(l) for l in (args.raw / "metadata.jsonl").open()]
    examples_by_paper, skipped = {}, []
    for i, paper in enumerate(papers, 1):
        try:
            paragraphs = finalize(extract_paragraphs(args.raw / paper["pdf_path"]))
        except Exception as e:
            skipped.append((paper["paper_id"], f"error: {e}"))
            continue
        if len(paragraphs) < MIN_PARAGRAPHS_PER_PAPER:
            skipped.append((paper["paper_id"], f"only {len(paragraphs)} paragraphs"))
            continue
        ex = make_examples(paper, paragraphs, rng, args.per_paper)
        if not ex:
            skipped.append((paper["paper_id"], "no usable target paragraph"))
            continue
        examples_by_paper[paper["paper_id"]] = ex
        if i % 50 == 0:
            print(f"processed {i}/{len(papers)}", flush=True)

    # 80/10/10 split by paper, within each domain.
    split_of = {}
    by_domain = defaultdict(list)
    for p in papers:
        if p["paper_id"] in examples_by_paper:
            by_domain[p["domain"]].append(p["paper_id"])
    for domain, ids in by_domain.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        a, b = round(0.8 * len(ids)), round(0.9 * len(ids))
        for pid in ids[:a]: split_of[pid] = "train"
        for pid in ids[a:b]: split_of[pid] = "dev"
        for pid in ids[b:]: split_of[pid] = "test"

    stats = {"papers_downloaded": len(papers), "papers_used": len(split_of),
             "papers_skipped": len(skipped), "splits": {}}
    for split in ("train", "dev", "test"):
        rows = [e for pid, ex in examples_by_paper.items() if split_of[pid] == split for e in ex]
        write_jsonl(args.out / f"{split}.jsonl", rows)
        stats["splits"][split] = {
            "papers": sum(1 for s in split_of.values() if s == split),
            "examples": len(rows),
            "by_domain": dict(Counter(e["domain"] for e in rows)),
            "by_target_words": dict(sorted(Counter(e["target_words"] for e in rows).items())),
            "avg_context_paragraphs": round(sum(len(e["context"]) for e in rows) / max(len(rows), 1), 1),
        }
    write_jsonl(args.out / "paper_splits.jsonl",
                [{"paper_id": pid, "split": s} for pid, s in sorted(split_of.items())])
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "skipped.txt").write_text("\n".join(f"{p}\t{r}" for p, r in skipped))

    archive = Path("data/data.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "paper_splits.jsonl", "stats.json"):
            tar.add(args.out / name, arcname=f"data/{name}")
        if Path("data.md").exists():
            tar.add("data.md", arcname="data/data.md")
    print(json.dumps(stats, indent=2))
    print(f"archive: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
