#!/usr/bin/env python3
"""Step 1 of the ContextComplete data pipeline: download raw arXiv PDFs.

Downloads N papers per arXiv category (default 100), spread evenly over
2022-2025 so the corpus is not dominated by one month's hot topic.
Only papers whose PRIMARY category matches are kept, so a paper can't
show up under two domains. Re-running resumes: PDFs already on disk are
reused and metadata is rebuilt.

Usage:
  python scripts/download_papers.py                 # 100 each of cs.CL, cs.LG, cs.CV
  python scripts/download_papers.py --per-category 20 --categories cs.CL
"""
import argparse
import calendar
import json
import random
import time
import urllib.request
from pathlib import Path

import arxiv
import pymupdf as fitz

DEFAULT_CATEGORIES = {"cs.CL": "nlp", "cs.LG": "ml", "cs.CV": "cv"}
YEARS = [2022, 2023, 2024, 2025]
MIN_PAGES = 4          # skip extended abstracts / short notes
USER_AGENT = "ContextComplete-CIS5300/0.1 (student research project)"


def month_windows(rng):
    """All (year, month) pairs, shuffled so any shortfall is spread out."""
    months = [(y, m) for y in YEARS for m in range(1, 13)]
    rng.shuffle(months)
    return months


def random_day_query(cat, year, month, rng):
    """Query for one random day in the given month (avoids month-end clustering)."""
    day = rng.randint(1, calendar.monthrange(year, month)[1])
    d = f"{year}{month:02d}{day:02d}"
    return f"cat:{cat} AND submittedDate:[{d}0000 TO {d}2359]"


def valid_pdf(path):
    try:
        with fitz.open(path) as doc:
            return doc.page_count >= MIN_PAGES
    except Exception:
        return False


def download(url, path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            path.write_bytes(data)
            return True
        except Exception as e:
            print(f"    retry {attempt + 1}/{retries}: {str(e)[:100]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return False


def collect_category(cat, domain, wanted, out_dir, rng, delay, seen_global):
    pdf_dir = out_dir / "pdfs" / domain
    pdf_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    kept = []
    months = month_windows(rng)
    per_month = -(-wanted // len(months))  # ceil, so 100 / 48 -> up to 3 a month
    passes = 0
    while len(kept) < wanted and passes < 4:
        passes += 1
        for year, month in months:
            if len(kept) >= wanted:
                break
            query = random_day_query(cat, year, month, rng)
            try:
                results = list(client.results(arxiv.Search(query=query, max_results=50)))
            except Exception as e:
                print(f"  search failed ({query}): {str(e)[:100]}", flush=True)
                continue
            results = [r for r in results if r.primary_category == cat]
            rng.shuffle(results)
            taken = 0
            for r in results:
                if taken >= per_month or len(kept) >= wanted:
                    break
                pid = r.get_short_id().split("v")[0]
                if pid in seen_global:
                    continue
                path = pdf_dir / f"{pid}.pdf"
                if not path.exists():
                    if not download(r.pdf_url, path):
                        continue
                    time.sleep(delay)
                if not valid_pdf(path):
                    path.unlink(missing_ok=True)
                    continue
                seen_global.add(pid)
                taken += 1
                kept.append({
                    "paper_id": pid,
                    "title": " ".join(r.title.split()),
                    "authors": [a.name for a in r.authors],
                    "primary_category": r.primary_category,
                    "categories": r.categories,
                    "domain": domain,
                    "published": r.published.date().isoformat(),
                    "abs_url": r.entry_id,
                    "pdf_path": str(path.relative_to(out_dir)),
                    "license": getattr(r, "license", None),
                })
                print(f"  [{domain}] {len(kept)}/{wanted}  {pid}  {kept[-1]['title'][:70]}", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--seed", type=int, default=5300)
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between PDF downloads")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seen = set()
    all_meta = []
    for cat in args.categories:
        domain = DEFAULT_CATEGORIES.get(cat, cat.replace(".", "_"))
        print(f"\n== {cat} ({domain}) ==", flush=True)
        all_meta += collect_category(cat, domain, args.per_category, args.out, rng, args.delay, seen)

    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in all_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in all_meta:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    print("\nDone. Papers per domain:", json.dumps(counts))
    print("Metadata:", args.out / "metadata.jsonl")


if __name__ == "__main__":
    main()
