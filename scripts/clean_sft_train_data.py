"""clean_sft_train_data.py  -  strip filter-pipeline cruft from filtered data.

The training stage's existing loaders (``load_sft_examples`` /
``load_evaluation_examples`` in ``src/qwen_tutor/training/sft.py``) read from
``data/sft_filtered/`` and ``data/eval_filtered/``. Those directories are
written by the filter pipeline and contain three layers of noise that the
trainer does NOT need:

  1. A top-level ``{"example": {...}, "pipeline": {...}}`` wrapper added by
     the filter runner. The trainer unwraps this on the fly, but it bloats
     every line.
  2. ``quality_signals`` -- diagnostic data from the filter stage. The
     trainer never reads it.
  3. ``metadata.generation`` -- which teacher / provider / variant produced
     the record. Useful for debugging the data-gen pipeline, irrelevant to
     training.

The SFT and evaluation files have **different schemas** -- ``ExampleMetadata``
vs ``EvaluationMetadata`` -- so this script writes them to two parallel
output directories so the trainer can keep loading them as separate
populations (which it must, since the formatter routes them through
different code paths: ``/no_think`` for SFT, ``/think`` for eval).

What the trainer needs from each record
---------------------------------------
SFT  (mode == "conversation"):
    id, metadata.topic, metadata.subtopics, metadata.user_role,
    metadata.model_role, metadata.cefr_level, metadata.scenario_type,
    metadata.locale, metadata.category, system_prompt, messages

Eval (mode == "evaluation"):
    id, metadata.source_dialogue_id, metadata.learner_cefr_target,
    metadata.locale, metadata.scenario_type, metadata.category,
    system_prompt, messages

``system_prompt`` is kept because ``SFTExample`` / ``EvaluationExample``
declare it required. The formatter then ignores it and rebuilds the
prompt from ``training.yaml`` at training time -- but the strict schema
demands the field be present. The ``--drop-system-prompt`` flag below
lets you remove it anyway if you have a custom loader that re-renders.

Usage
-----
    # Default: read filtered dirs, write parallel clean dirs.
    python scripts/clean_sft_train_data.py

    # Subset by CEFR level
    python scripts/clean_sft_train_data.py --levels A2,B1

    # Subset by locale
    python scripts/clean_sft_train_data.py --locales china

    # Custom in/out dirs
    python scripts/clean_sft_train_data.py \
        --sft-in data/sft_filtered  --sft-out  data/sft_clean \
        --eval-in data/eval_filtered --eval-out data/eval_clean

After running, point training.yaml at the new dirs:

    data:
      sft_filtered_dir:  data/sft_clean
      eval_filtered_dir: data/eval_clean
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# UTF-8 stdout on Windows so non-ASCII previews print cleanly.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger("clean_sft_train_data")

# Field allow-lists. Anything NOT in these lists is stripped from output,
# even if the input has it. Easier to audit than a deny-list.
_SFT_METADATA_KEEP = (
    "topic", "subtopics", "user_role", "model_role",
    "cefr_level", "scenario_type", "locale", "category",
)
_EVAL_METADATA_KEEP = (
    "source_dialogue_id", "learner_cefr_target",
    "locale", "scenario_type", "category",
)
_SFT_RECORD_KEEP = ("id", "metadata", "system_prompt", "messages")
_EVAL_RECORD_KEEP = ("id", "metadata", "system_prompt", "messages")


def _unwrap_filter_wrapper(rec: dict[str, Any]) -> dict[str, Any]:
    """Filter pipeline wraps each record as ``{"example": {...}, "pipeline": {...}}``.

    Unwrap to the inner example dict. If the input is already unwrapped
    (raw-format) we return it as-is so the script also works on
    ``data/sft_raw/`` and ``data/eval_raw/`` inputs.
    """
    if isinstance(rec.get("example"), dict):
        return rec["example"]
    return rec


def _strip_dict(d: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    """Return a new dict containing only keys from ``keep`` that exist in ``d``,
    in ``keep`` order so output is deterministic and diff-friendly."""
    return {k: d[k] for k in keep if k in d}


def _clean_message(msg: dict[str, Any]) -> dict[str, str] | None:
    """Strict ``{role, content}`` shape. Drop messages missing either field."""
    role = msg.get("role")
    content = msg.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        return None
    return {"role": role, "content": content}


def _clean_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    out: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        cm = _clean_message(m)
        if cm is not None:
            out.append(cm)
    return out


def _clean_record(
    rec: dict[str, Any],
    *,
    record_keep: tuple[str, ...],
    metadata_keep: tuple[str, ...],
    drop_system_prompt: bool,
) -> dict[str, Any] | None:
    """Strip a single record to the trainer-needed subset.

    Returns ``None`` if the record is malformed enough to skip (missing
    id, messages, or metadata).
    """
    rec = _unwrap_filter_wrapper(rec)
    if not isinstance(rec, dict):
        return None
    if not rec.get("id") or not isinstance(rec.get("messages"), list):
        return None
    if not isinstance(rec.get("metadata"), dict):
        return None

    cleaned_metadata = _strip_dict(rec["metadata"], metadata_keep)
    cleaned_messages = _clean_messages(rec["messages"])
    if not cleaned_messages:
        return None

    out: dict[str, Any] = {}
    for k in record_keep:
        if k == "metadata":
            out["metadata"] = cleaned_metadata
        elif k == "messages":
            out["messages"] = cleaned_messages
        elif k == "system_prompt":
            if drop_system_prompt:
                continue
            sp = rec.get("system_prompt", "")
            if isinstance(sp, str):
                out["system_prompt"] = sp
        elif k in rec:
            out[k] = rec[k]
    return out


def _passes_filters(
    metadata: dict[str, Any],
    *,
    level_set: set[str] | None,
    locale_set: set[str] | None,
    cefr_field: str,
) -> bool:
    """SFT uses ``cefr_level``, eval uses ``learner_cefr_target``. Same shape
    of metadata-as-dict so one helper covers both."""
    if level_set is not None:
        if metadata.get(cefr_field) not in level_set:
            return False
    if locale_set is not None:
        if metadata.get("locale", "china") not in locale_set:
            return False
    return True


def _iter_passed_files(in_dir: Path) -> list[Path]:
    """Yield every ``*_passed.jsonl`` file in ``in_dir``.

    Skips ``*_failed.jsonl`` and anything else. Sorted for deterministic
    output ordering.
    """
    if not in_dir.exists():
        return []
    return sorted(
        p for p in in_dir.glob("*_passed.jsonl") if p.is_file()
    )


def _convert_dir(
    *,
    in_dir: Path,
    out_dir: Path,
    record_keep: tuple[str, ...],
    metadata_keep: tuple[str, ...],
    cefr_field: str,
    level_set: set[str] | None,
    locale_set: set[str] | None,
    drop_system_prompt: bool,
    label: str,
) -> dict[str, Any]:
    """Walk one input dir, write one output dir. Return stats dict."""
    in_files = _iter_passed_files(in_dir)
    if not in_files:
        logger.warning(
            "[%s] no '*_passed.jsonl' files found in %s -- skipping",
            label, in_dir,
        )
        return {
            "label": label,
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "files_processed": 0,
            "input_records": 0,
            "kept_records": 0,
            "dropped_malformed": 0,
            "dropped_filter": 0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    total_input = 0
    total_kept = 0
    total_malformed = 0
    total_filtered = 0
    files_processed = 0

    for in_path in in_files:
        # Drop the trailing "_passed" so downstream tooling sees a clean
        # name (e.g. normal_A2_passed.jsonl -> normal_A2.jsonl).
        stem = in_path.stem[: -len("_passed")] if in_path.stem.endswith("_passed") else in_path.stem
        out_path = out_dir / f"{stem}.jsonl"
        n_in = 0
        n_kept = 0
        n_malformed = 0
        n_filtered = 0
        with in_path.open("r", encoding="utf-8") as fh_in, \
             out_path.open("w", encoding="utf-8") as fh_out:
            for line_no, line in enumerate(fh_in, start=1):
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "%s:%d skipping malformed JSON: %s",
                        in_path, line_no, exc,
                    )
                    n_malformed += 1
                    continue
                cleaned = _clean_record(
                    rec,
                    record_keep=record_keep,
                    metadata_keep=metadata_keep,
                    drop_system_prompt=drop_system_prompt,
                )
                if cleaned is None:
                    n_malformed += 1
                    continue
                if not _passes_filters(
                    cleaned.get("metadata", {}),
                    level_set=level_set,
                    locale_set=locale_set,
                    cefr_field=cefr_field,
                ):
                    n_filtered += 1
                    continue
                fh_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                n_kept += 1
        files_processed += 1
        total_input += n_in
        total_kept += n_kept
        total_malformed += n_malformed
        total_filtered += n_filtered
        logger.info(
            "[%s] %s -> %s: in=%d kept=%d malformed=%d filtered=%d",
            label, in_path.name, out_path.name,
            n_in, n_kept, n_malformed, n_filtered,
        )

    return {
        "label": label,
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "files_processed": files_processed,
        "input_records": total_input,
        "kept_records": total_kept,
        "dropped_malformed": total_malformed,
        "dropped_filter": total_filtered,
    }


def _parse_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {s.strip() for s in value.split(",") if s.strip()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strip filter-pipeline wrappers + diagnostic fields from "
            "filtered SFT and eval data so the trainer sees a minimal, "
            "deterministic record shape."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sft-in", default="data/sft_filtered")
    p.add_argument("--eval-in", default="data/eval_filtered")
    p.add_argument("--sft-out", default="data/sft_clean")
    p.add_argument("--eval-out", default="data/eval_clean")
    p.add_argument(
        "--levels", default=None,
        help="Comma-separated CEFR levels to keep (A1,A2,...). Default: all.",
    )
    p.add_argument(
        "--locales", default=None,
        help="Comma-separated locale names to keep (china,japan,...). Default: all.",
    )
    p.add_argument(
        "--drop-system-prompt", action="store_true",
        help="Also drop the on-disk system_prompt (which the formatter "
             "ignores anyway). NOTE: breaks strict SFTExample/EvaluationExample "
             "validation in the existing trainer loader -- use only with a "
             "custom loader that re-renders the prompt from training.yaml.",
    )
    p.add_argument(
        "--skip-sft", action="store_true",
        help="Only convert eval data.",
    )
    p.add_argument(
        "--skip-eval", action="store_true",
        help="Only convert SFT data.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    level_set = _parse_csv(args.levels)
    locale_set = _parse_csv(args.locales)

    all_stats: list[dict[str, Any]] = []

    if not args.skip_sft:
        sft_stats = _convert_dir(
            in_dir=Path(args.sft_in),
            out_dir=Path(args.sft_out),
            record_keep=_SFT_RECORD_KEEP,
            metadata_keep=_SFT_METADATA_KEEP,
            cefr_field="cefr_level",
            level_set=level_set,
            locale_set=locale_set,
            drop_system_prompt=args.drop_system_prompt,
            label="sft",
        )
        all_stats.append(sft_stats)

    if not args.skip_eval:
        eval_stats = _convert_dir(
            in_dir=Path(args.eval_in),
            out_dir=Path(args.eval_out),
            record_keep=_EVAL_RECORD_KEEP,
            metadata_keep=_EVAL_METADATA_KEEP,
            cefr_field="learner_cefr_target",
            level_set=level_set,
            locale_set=locale_set,
            drop_system_prompt=args.drop_system_prompt,
            label="eval",
        )
        all_stats.append(eval_stats)

    print("\n=== clean_sft_train_data ===")
    for s in all_stats:
        print(f"  [{s['label']}] {s['in_dir']} -> {s['out_dir']}")
        print(f"    files processed:    {s['files_processed']}")
        print(f"    input records:      {s['input_records']}")
        print(f"    kept:               {s['kept_records']}")
        print(f"    dropped (malformed):{s['dropped_malformed']:>4d}")
        print(f"    dropped (filter):   {s['dropped_filter']:>4d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
