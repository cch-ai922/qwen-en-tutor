"""build_vocab_bands.py  -  CEFR-J 등 외부 워드리스트를 cefr_vocab_bands.json 으로 변환.

기본 cefr_vocab_bands.json 은 데모용으로 밴드당 ~90 단어밖에 안 들어 있어
``CEFRVocabFilter`` 의 분모가 너무 작아져 신뢰할 수 없습니다. 이 스크립트는
공개 CEFR 워드리스트(예: CEFR-J Wordlist v1.6)를 받아 같은 JSON 포맷으로
다시 만들어 줍니다.

지원 입력 포맷
    *.csv  / *.tsv  / *.txt    (탭 또는 콤마 구분, UTF-8 권장)
    *.xlsx                       (openpyxl 이 설치된 경우만)

입력 파일은 최소 다음 두 컬럼을 가져야 합니다 (대소문자 무관):
    headword  -  표제어
    CEFR      -  A1 / A2 / B1 / B2 / C1 / C2 중 하나

사용 예
    # CEFR-J 의 txt 파일을 그대로 처리
    python scripts/build_vocab_bands.py \
        --input downloads/CEFR-J_Wordlist_Ver1.6.txt \
        --out config/cefr_vocab_bands.json

    # 다른 포맷이라면 컬럼 이름을 직접 지정
    python scripts/build_vocab_bands.py \
        --input wordlist.csv --headword-col WORD --cefr-col LEVEL

수동 변환이 부담스러우면:
    1. https://cefr-j.org/download.html 에서 CEFR-J Wordlist v1.6 다운로드.
    2. 압축을 풀어 xlsx 또는 txt 파일을 ``downloads/`` 등에 놓습니다.
    3. 위 명령을 실행합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# Windows 콘솔 한글 출력 대응
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

VALID_BANDS = ("A1", "A2", "B1", "B2", "C1", "C2")
BAND_RANK = {b: i for i, b in enumerate(VALID_BANDS)}


# ---------------------------------------------------------------------------
# 입력 파서 - 탭/콤마/xlsx 자동 분기
# ---------------------------------------------------------------------------


def _sniff_delimiter(sample: str) -> str:
    """첫 줄을 보고 탭/콤마/세미콜론 구분자를 추측합니다."""
    counts = {d: sample.count(d) for d in ("\t", ",", ";")}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def _iter_rows_text(path: Path) -> Iterable[dict[str, str]]:
    """CSV / TSV / TXT 파일에서 헤더-있는 dict 한 줄씩 yield."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        head = fh.readline()
        delim = _sniff_delimiter(head)
        fh.seek(0)
        yield from csv.DictReader(fh, delimiter=delim)


def _iter_rows_xlsx(path: Path) -> Iterable[dict[str, str]]:
    """xlsx 파일을 openpyxl 로 읽어 dict 형태로 yield."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise SystemExit(
            "xlsx 파일을 처리하려면 openpyxl 이 필요합니다.\n"
            "  pip install openpyxl   (네트워크 없으면 vendor/wheels/ 에 추가 필요)\n"
            "또는 CEFR-J 파일을 .csv / .txt 로 다시 저장해 주세요."
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
# 컬럼 자동 매칭
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """대소문자 + 언더스코어/공백 무시 비교용."""
    return s.lower().replace("_", "").replace(" ", "").replace("-", "")


def _find_column(sample_row: dict[str, str], candidates: list[str]) -> str | None:
    """사용자 지정 컬럼명이 없을 때 흔한 별칭에서 찾아 줍니다."""
    keys = {_norm(k): k for k in sample_row.keys()}
    for cand in candidates:
        actual = keys.get(_norm(cand))
        if actual is not None:
            return actual
    return None


# ---------------------------------------------------------------------------
# 메인 변환 로직
# ---------------------------------------------------------------------------


def build_bands(
    rows: Iterable[dict[str, str]],
    headword_col: str | None,
    cefr_col: str | None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """입력 행을 받아 {band: [words]} 와 통계를 반환합니다.

    같은 단어가 여러 밴드에 나타나면 가장 낮은(=쉬운) 밴드를 채택합니다
    (charitable default; ``CEFRVocabFilter`` 의 setdefault 동작과 일치).
    """
    word_to_band: dict[str, str] = {}
    skipped_unknown_band = 0
    skipped_empty = 0
    total = 0

    first = None
    for first in rows:
        break
    if first is None:
        raise SystemExit("입력 파일이 비어 있습니다.")

    auto_head = headword_col or _find_column(
        first, ["headword", "lemma", "word", "term", "entry"]
    )
    auto_cefr = cefr_col or _find_column(
        first, ["CEFR", "level", "band", "CEFR_Level", "cefr_level"]
    )
    if auto_head is None or auto_cefr is None:
        raise SystemExit(
            f"필요한 컬럼을 찾지 못했습니다.\n"
            f"  발견된 컬럼: {list(first.keys())}\n"
            f"  --headword-col / --cefr-col 로 직접 지정해 주세요."
        )

    print(f"headword 컬럼: '{auto_head}',  CEFR 컬럼: '{auto_cefr}'")

    # 첫 행 + 나머지 행을 합쳐 다시 순회
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
        # "B2.1" 같은 세부 라벨이 있으면 첫 두 글자만 사용
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
    """기존 cefr_vocab_bands.json 에서 stopwords 만 끌어옵니다."""
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
        description="CEFR-J 같은 외부 워드리스트로 cefr_vocab_bands.json 을 다시 만듭니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="CEFR 워드리스트 파일 (csv / tsv / txt / xlsx)",
    )
    p.add_argument(
        "--out",
        default="config/cefr_vocab_bands.json",
        help="출력 JSON 경로",
    )
    p.add_argument(
        "--headword-col",
        default=None,
        help="표제어 컬럼명. 비워 두면 자동 탐색 (headword, lemma, word, ...).",
    )
    p.add_argument(
        "--cefr-col",
        default=None,
        help="CEFR 레벨 컬럼명. 비워 두면 자동 탐색 (CEFR, level, band, ...).",
    )
    p.add_argument(
        "--keep-stopwords",
        action="store_true",
        default=True,
        help="기존 JSON 의 stopwords 를 그대로 유지 (기본값).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)
    if not in_path.exists():
        raise SystemExit(f"입력 파일 없음: {in_path}")
    print(f"입력: {in_path}")
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

    # 출력 요약
    print(f"\n출력: {out_path}")
    counts = Counter({b: len(ws) for b, ws in bands.items()})
    for b in VALID_BANDS:
        bar = "#" * min(40, counts[b] // 50)
        print(f"  {b}: {counts[b]:>5} words  {bar}")
    print(f"\n총 unique words: {stats['unique_words']}")
    if stats["skipped_unknown_band"]:
        print(f"  스킵된 알 수 없는 밴드 행: {stats['skipped_unknown_band']}")
    if stats["skipped_empty_headword"]:
        print(f"  스킵된 빈 표제어 행: {stats['skipped_empty_headword']}")
    print(f"  stopwords (유지): {len(stopwords)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
