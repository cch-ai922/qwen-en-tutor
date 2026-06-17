"""clean_dpo_train_data.py  -  strip filter-pipeline cruft from DPO data.

Counterpart to ``scripts/clean_sft_train_data.py`` but for DPO records.
The DPO trainer's loader (``_load_dpo_by_source`` in
``src/qwen_tutor/training/dpo.py``) reads ``data/dpo_filtered/`` and
already unwraps the filter pipeline ``{"example": ...}`` wrapper on the
fly. This script does the same unwrap once and strips fields the trainer
never reads, producing a cleaner ``data/dpo_clean/`` directory you can
point the trainer at.

What the trainer needs from each DPO record
-------------------------------------------
``id``, ``metadata.{topic, subtopics, user_role, model_role, cefr_level,
scenario_type, locale, category}``, ``system_prompt``, ``prompt_messages``,
``chosen``, ``rejected``, ``rejection_axis``, ``rejection_note``.

What gets stripped
------------------
  1. The top-level ``{"example": {...}, "pipeline": {...}}`` wrapper
     added by the filter runner.
  2. ``metadata.generation`` -- provider / model / variant lineage from
     the data-gen stage. Useful for debugging the data pipeline,
     irrelevant to DPO training.

``system_prompt`` is kept because ``DPOExample`` declares it required
(strict pydantic). The DPO renderer in ``dpo._render_dpo_row`` rebuilds
the system prompt from ``metadata`` anyway (so the on-disk string is
ignored) but the schema demands the field. Pass ``--drop-system-prompt``
if you have a custom loader that re-renders.

Filename preservation
---------------------
The DPO trainer loads records by **filename prefix** -- ``register_*.jsonl``
and ``on_policy_*.jsonl`` are loaded into separate pools and mixed
according to ``mix_ratio_register`` / ``mix_ratio_on_policy``. This
script drops the trailing ``_passed`` from each filename but preserves
the ``register_`` / ``on_policy_`` prefix so the trainer's prefix-based
loader keeps working:

    register_A2_passed.jsonl   ->  register_A2.jsonl
    on_policy_A2_passed.jsonl  ->  on_policy_A2.jsonl

Usage
-----
    # Default: read data/dpo_filtered, write data/dpo_clean
    python scripts/clean_dpo_train_data.py

    # Subset by level / locale
    python scripts/clean_dpo_train_data.py --levels A2,B1 --locales china

    # Only one DPO source
    python scripts/clean_dpo_train_data.py --sources register
    python scripts/clean_dpo_train_data.py --sources on_policy
    python scripts/clean_dpo_train_data.py --sources register,on_policy   # both (default)

After running, point training.yaml at the new dir:

    dpo:
      data:
        dpo_filtered_dir: data/dpo_clean
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger("clean_dpo_train_data")

# Allow-list of fields kept in the output. Anything else is stripped.
_DPO_METADATA_KEEP = (
    "topic", "subtopics", "user_role", "model_role",
    "cefr_level", "scenario_type", "locale", "category",
)
_DPO_RECORD_KEEP = (
    "id", "metadata", "system_prompt",
    "prompt_messages", "chosen", "rejected",
    "rejection_axis", "rejection_note",
)

# Trainer's prefix-based loader globs <prefix>_*.jsonl per pool. Each
# source the trainer knows about must have its prefix preserved on
# output so _load_dpo_by_source can still find the records.
_KNOWN_SOURCES = ("register", "on_policy")


def _unwrap_filter_wrapper(rec: dict[str, Any]) -> dict[str, Any]:
    """Filter pipeline wraps each record as ``{"example": {...}, "pipeline": {...}}``."""
    if isinstance(rec.get("example"), dict):
        return rec["example"]
    return rec


def _strip_dict(d: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    """Return a new dict with only ``keep`` keys, in ``keep`` order."""
    return {k: d[k] for k in keep if k in d}


def _clean_message(msg: Any) -> dict[str, str] | None:
    """Validate a single message dict ``{role, content}``."""
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    content = msg.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        return None
    return {"role": role, "content": content}


def _clean_messages(messages: Any) -> list[dict[str, str]] | None:
    """Validate a list of message dicts. Returns None if the list is
    not a list or yields zero valid messages (DPO needs at least one
    prompt turn)."""
    if not isinstance(messages, list):
        return None
    out: list[dict[str, str]] = []
    for m in messages:
        cm = _clean_message(m)
        if cm is not None:
            out.append(cm)
    return out if out else None


def _clean_dpo_record(
    rec: dict[str, Any],
    *,
    drop_system_prompt: bool,
) -> dict[str, Any] | None:
    """Strip one DPO record to the trainer-needed subset.

    Returns ``None`` if the record is malformed enough to skip
    (missing id, no prompt_messages, no chosen, no rejected, or no
    metadata).
    """
    rec = _unwrap_filter_wrapper(rec)
    if not isinstance(rec, dict):
        return None
    if not rec.get("id"):
        return None
    if not isinstance(rec.get("metadata"), dict):
        return None

    cleaned_prompt = _clean_messages(rec.get("prompt_messages"))
    if cleaned_prompt is None:
        return None
    cleaned_chosen = _clean_message(rec.get("chosen"))
    if cleaned_chosen is None:
        return None
    cleaned_rejected = _clean_message(rec.get("rejected"))
    if cleaned_rejected is None:
        return None

    cleaned_metadata = _strip_dict(rec["metadata"], _DPO_METADATA_KEEP)

    out: dict[str, Any] = {}
    for k in _DPO_RECORD_KEEP:
        if k == "metadata":
            out["metadata"] = cleaned_metadata
        elif k == "system_prompt":
            if drop_system_prompt:
                continue
            sp = rec.get("system_prompt", "")
            if isinstance(sp, str):
                out["system_prompt"] = sp
        elif k == "prompt_messages":
            out["prompt_messages"] = cleaned_prompt
        elif k == "chosen":
            out["chosen"] = cleaned_chosen
        elif k == "rejected":
            out["rejected"] = cleaned_rejected
        elif k in rec:
            out[k] = rec[k]
    return out


def _passes_filters(
    metadata: dict[str, Any],
    *,
    level_set: set[str] | None,
    locale_set: set[str] | None,
) -> bool:
    if level_set is not None and metadata.get("cefr_level") not in level_set:
        return False
    if locale_set is not None and metadata.get("locale", "china") not in locale_set:
        return False
    return True


def _passed_files_for_source(in_dir: Path, source: str) -> list[Path]:
    """Return ``<source>_*_passed.jsonl`` files for a given DPO source
    prefix (``register`` or ``on_policy``)."""
    if not in_dir.exists():
        return []
    return sorted(p for p in in_dir.glob(f"{source}_*_passed.jsonl") if p.is_file())


def _convert(
    *,
    in_dir: Path,
    out_dir: Path,
    sources: tuple[str, ...],
    level_set: set[str] | None,
    locale_set: set[str] | None,
    drop_system_prompt: bool,
) -> dict[str, Any]:
    """Walk every requested source's passed files, write cleaned outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_per_source: dict[str, dict[str, int]] = {}
    grand_total_in = 0
    grand_total_kept = 0
    grand_total_malformed = 0
    grand_total_filtered = 0
    files_processed = 0

    for source in sources:
        in_files = _passed_files_for_source(in_dir, source)
        if not in_files:
            logger.warning(
                "[dpo:%s] no '%s_*_passed.jsonl' files found in %s -- skipping source",
                source, source, in_dir,
            )
            stats_per_source[source] = {
                "files": 0, "in": 0, "kept": 0, "malformed": 0, "filtered": 0,
            }
            continue
        n_in_src = 0
        n_kept_src = 0
        n_malformed_src = 0
        n_filtered_src = 0
        files_src = 0

        for in_path in in_files:
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
                    cleaned = _clean_dpo_record(
                        rec, drop_system_prompt=drop_system_prompt
                    )
                    if cleaned is None:
                        n_malformed += 1
                        continue
                    if not _passes_filters(
                        cleaned.get("metadata", {}),
                        level_set=level_set,
                        locale_set=locale_set,
                    ):
                        n_filtered += 1
                        continue
                    fh_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    n_kept += 1
            files_src += 1
            n_in_src += n_in
            n_kept_src += n_kept
            n_malformed_src += n_malformed
            n_filtered_src += n_filtered
            logger.info(
                "[dpo:%s] %s -> %s: in=%d kept=%d malformed=%d filtered=%d",
                source, in_path.name, out_path.name,
                n_in, n_kept, n_malformed, n_filtered,
            )

        stats_per_source[source] = {
            "files": files_src,
            "in": n_in_src,
            "kept": n_kept_src,
            "malformed": n_malformed_src,
            "filtered": n_filtered_src,
        }
        files_processed += files_src
        grand_total_in += n_in_src
        grand_total_kept += n_kept_src
        grand_total_malformed += n_malformed_src
        grand_total_filtered += n_filtered_src

    return {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "sources": sources,
        "per_source": stats_per_source,
        "files_processed": files_processed,
        "input_records": grand_total_in,
        "kept_records": grand_total_kept,
        "dropped_malformed": grand_total_malformed,
        "dropped_filter": grand_total_filtered,
    }


def _parse_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {s.strip() for s in value.split(",") if s.strip()}


def _parse_sources(value: str) -> tuple[str, ...]:
    items = tuple(s.strip() for s in value.split(",") if s.strip())
    bad = [s for s in items if s not in _KNOWN_SOURCES]
    if bad:
        raise SystemExit(
            f"unknown DPO source(s) {bad!r}. Known sources: {list(_KNOWN_SOURCES)} "
            f"(the trainer mixes these via mix_ratio_register / mix_ratio_on_policy)"
        )
    return items


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strip filter-pipeline wrappers + diagnostic fields from "
            "filtered DPO data so the trainer sees a minimal, "
            "deterministic record shape. Counterpart of clean_sft_train_data.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dpo-in", default="data/dpo_filtered")
    p.add_argument("--dpo-out", default="data/dpo_clean")
    p.add_argument(
        "--sources", default=",".join(_KNOWN_SOURCES),
        help=(
            "Comma-separated DPO source prefixes to clean. The DPO "
            "trainer loads each source into its own pool: register pairs "
            "come from offline register-rewrite generation, on_policy "
            "pairs come from trained-policy vs teacher judging. Default: both."
        ),
    )
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
        help="Also drop the on-disk system_prompt (which the DPO renderer "
             "ignores anyway). NOTE: breaks strict DPOExample validation "
             "in the existing trainer loader -- use only with a custom "
             "loader that re-renders the prompt from training.yaml.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sources = _parse_sources(args.sources)
    level_set = _parse_csv(args.levels)
    locale_set = _parse_csv(args.locales)

    stats = _convert(
        in_dir=Path(args.dpo_in),
        out_dir=Path(args.dpo_out),
        sources=sources,
        level_set=level_set,
        locale_set=locale_set,
        drop_system_prompt=args.drop_system_prompt,
    )

    print("\n=== clean_dpo_train_data ===")
    print(f"  {stats['in_dir']} -> {stats['out_dir']}")
    print(f"  files processed:     {stats['files_processed']}")
    print(f"  input records:       {stats['input_records']}")
    print(f"  kept:                {stats['kept_records']}")
    print(f"  dropped (malformed): {stats['dropped_malformed']:>4d}")
    print(f"  dropped (filter):    {stats['dropped_filter']:>4d}")
    print()
    print("  per-source breakdown:")
    for source, st in stats["per_source"].items():
        print(
            f"    {source:>10s}: files={st['files']:>2d} "
            f"in={st['in']:>4d} kept={st['kept']:>4d} "
            f"malformed={st['malformed']:>3d} filtered={st['filtered']:>4d}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
