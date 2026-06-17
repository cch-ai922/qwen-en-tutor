"""build_vocab_bands.py  -  convert external wordlists like CEFR-J into cefr_vocab_bands.json.

The default cefr_vocab_bands.json is only a demo and contains about 90 words per
band, which makes the denominator for ``CEFRVocabFilter`` too small to trust.
This script consumes a public CEFR wordlist (e.g. CEFR-J Wordlist v1.6) and
rebuilds it into the same JSON format.

Supported input formats:
    *.csv  / *.tsv  / *.txt    (tab or comma-separated, UTF-8 recommended)
    *.xlsx                    (only if openpyxl is installed)

The input file must contain at least these two columns (case-insensitive):
    headword  -  lemma/headword
    CEFR      -  one of A1 / A2 / B1 / B2 / C1 / C2

Usage:
    # Process CEFR-J txt file directly
    python scripts/build_vocab_bands.py \
        --input downloads/CEFR-J_Wordlist_Ver1.6.txt \
        --out config/cefr_vocab_bands.json

    # For other formats, specify column names explicitly
    python scripts/build_vocab_bands.py \
        --input wordlist.csv --headword-col WORD --cefr-col LEVEL

If manual conversion is burdensome:
    1. Download CEFR-J Wordlist v1.6 from https://cefr-j.org/download.html.
    2. Extract and place the xlsx or txt file under ``downloads/``.
    3. Run the command above.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# Support Singapore output in the Windows console
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

VALID_BANDS = ("A1", "A2", "B1", "B2", "C1", "C2")
BAND_RANK = {b: i for i, b in enumerate(VALID_BANDS)}


# ---------------------------------------------------------------------------
# Input parser - auto-branch between tab/comma/xlsx
# ---------------------------------------------------------------------------


def _sniff_delimiter(sample: str) -> str:
    """Guess tab/comma/semicolon delimiter from the first line."""
    counts = {d: sample.count(d) for d in ("\t", ",", ";")}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def _iter_rows_text(path: Path) -> Iterable[dict[str, str]]:
    """Yield headered rows as dicts from CSV / TSV / TXT files."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        head = fh.readline()
        delim = _sniff_delimiter(head)
        fh.seek(0)
        yield from csv.DictReader(fh, delimiter=delim)


def _iter_rows_xlsx(path: Path) -> Iterable[dict[str, str]]:
    """Read xlsx files with openpyxl and yield dictionaries."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise SystemExit(
            "openpyxl is required to process xlsx files.\n"
            "  pip install openpyxl   (if no network, add it from vendor/wheels/)\n"
            "Or save the CEFR-J file again as .csv / .txt."
        ) from exc
    wb = load_workbook(filename=str(path), read_only=True, data_only=True)
    sheet = wb.active
    rows = sheet.iter_rows(values_only=True)
    headers = [str(c).strip() if c is not None else "" for c in next(rows)]
    for raw in rows:
        if raw is None:
            continue
        yield {h: ("" if v is None else str(v)) for h, v in zip(headers, raw)}


def _iter_rows(path: Path) -> Iterable[dict[str, str]]:
    if path.suffix.lower() == ".xlsx":
        return _iter_rows_xlsx(path)
    return _iter_rows_text(path)


# ---------------------------------------------------------------------------
# Column auto-matching
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """Normalize for case-insensitive, underscore/space-insensitive comparison."""
    return s.lower().replace("_", "").replace(" ", "").replace("-", "")


def _find_column(sample_row: dict[str, str], candidates: list[str]) -> str | None:
    """Find a common alias when the user did not specify a column name."""
    keys = {_norm(k): k for k in sample_row.keys()}
    for cand in candidates:
        actual = keys.get(_norm(cand))
        if actual is not None:
            return actual
    return None


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------


def build_bands(
    rows: Iterable[dict[str, str]],
    headword_col: str | None,
    cefr_col: str | None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Take input rows and return {band: [words]} plus statistics.

    If the same word appears in multiple bands, the lowest (=easiest) band is
    adopted (charitable default; matches ``CEFRVocabFilter`` setdefault behavior).
    """
    word_to_band: dict[str, str] = {}
    skipped_unknown_band = 0
    skipped_empty = 0
    total = 0

    first = None
    for first in rows:
        break
    if first is None:
        raise SystemExit("Input file is empty.")

    auto_head = headword_col or _find_column(
        first, ["headword", "lemma", "word", "term", "entry"]
    )
    auto_cefr = cefr_col or _find_column(
        first, ["CEFR", "level", "band", "CEFR_Level", "cefr_level"]
    )
    if auto_head is None or auto_cefr is None:
        raise SystemExit(
            f"Could not find required columns.\n"
            f"  Found columns: {list(first.keys())}\n"
            f"  Please specify them with --headword-col / --cefr-col."
        )

    print(f"headword column: '{auto_head}',  CEFR column: '{auto_cefr}'")

    # Iterate first row plus the remaining rows again
    def _all():
        yield first
        yield from rows

    for row in _all():
        total += 1
        word = (row.get(auto_head) or "").strip().lower()
        band = (row.get(auto_cefr) or "").strip().upper()
        if not word:
            skipped_empty += 1
            continue
        # If there is a detailed label like "B2.1", use only the first two characters
        band = band[:2]
        if band not in BAND_RANK:
            skipped_unknown_band += 1
            continue
        # earliest-band-wins
        cur = word_to_band.get(word)
        if cur is None or BAND_RANK[band] < BAND_RANK[cur]:
            word_to_band[word] = band

    bands: dict[str, list[str]] = {b: [] for b in VALID_BANDS}
    for w, b in word_to_band.items():
        bands[b].append(w)
    for b in bands:
        bands[b].sort()

    stats = {
        "total_input_rows": total,
        "skipped_unknown_band": skipped_unknown_band,
        "skipped_empty_headword": skipped_empty,
        "unique_words": len(word_to_band),
    }
    return bands, stats


def _load_existing_stopwords(path: Path) -> list[str]:
    """Load only stopwords from an existing cefr_vocab_bands.json."""
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(doc.get("stopwords") or [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild cefr_vocab_bands.json from an external CEFR wordlist like CEFR-J.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="CEFR wordlist file (csv / tsv / txt / xlsx)",
    )
    p.add_argument(
        "--out",
        default="config/cefr_vocab_bands.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--headword-col",
        default=None,
        help="Headword column name. Automatically detected if empty (headword, lemma, word, ...).",
    )
    p.add_argument(
        "--cefr-col",
        default=None,
        help="CEFR level column name. Automatically detected if empty (CEFR, level, band, ...).",
    )
    p.add_argument(
        "--keep-stopwords",
        action="store_true",
        default=True,
        help="Keep stopwords from the existing JSON (default).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    print(f"Input: {in_path}")
    rows = _iter_rows(in_path)
    bands, stats = build_bands(
        rows, headword_col=args.headword_col, cefr_col=args.cefr_col
    )

    stopwords = _load_existing_stopwords(out_path) if args.keep_stopwords else []

    out_doc = {
        "_meta": {
            "source": str(in_path.name),
            "rebuilt_by": "scripts/build_vocab_bands.py",
            "stats": stats,
        },
        "bands": bands,
        "stopwords": stopwords,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out_doc, fh, ensure_ascii=False, indent=2)

    # Output summary
    print(f"\nOutput: {out_path}")
    counts = Counter({b: len(ws) for b, ws in bands.items()})
    for b in VALID_BANDS:
        bar = "#" * min(40, counts[b] // 50)
        print(f"  {b}: {counts[b]:>5} words  {bar}")
    print(f"\nTotal unique words: {stats['unique_words']}")
    if stats["skipped_unknown_band"]:
        print(f"  Skipped rows with unknown band: {stats['skipped_unknown_band']}")
    if stats["skipped_empty_headword"]:
        print(f"  Skipped rows with empty headword: {stats['skipped_empty_headword']}")
    print(f"  stopwords (kept): {len(stopwords)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
