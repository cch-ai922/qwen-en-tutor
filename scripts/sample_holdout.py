"""Sample a held-out evaluation set from ``data/seeds/``.

Picks ``--n-per-level`` scenarios per CEFR level (deterministic by seed
id hash, so re-runs select the same set) and writes them to
``data/holdout/{level}.jsonl``. By default it also REMOVES the sampled
scenarios from the source seeds — guarantees they're never seen during
training.

Run BEFORE training. If you call this after generating training data
you'll need to regenerate or hand-prune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample a holdout evaluation set from data/seeds/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds-dir", default="data/seeds")
    p.add_argument("--holdout-dir", default="data/holdout")
    p.add_argument("--n-per-level", type=int, default=10)
    p.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    p.add_argument(
        "--keep-in-seeds",
        action="store_true",
        help="don't remove sampled records from the source seed files",
    )
    p.add_argument(
        "--salt",
        default="qwen-en-tutor-holdout-v1",
        help="hash salt for deterministic sampling",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _hash_score(seed_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{seed_id}".encode("utf-8")).hexdigest()[:12], 16)


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]
    seeds_dir = Path(args.seeds_dir)
    holdout_dir = Path(args.holdout_dir)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    totals = {"sampled": 0, "remaining_in_seeds": 0}
    for level in levels:
        seeds_path = seeds_dir / f"{level}.jsonl"
        records = _read_records(seeds_path)
        if not records:
            LOGGER.warning("no seeds for level %s at %s", level, seeds_path)
            continue
        # Deterministic by id hash → stable across runs.
        ranked = sorted(
            records,
            key=lambda r: _hash_score(str(r.get("id", "")), args.salt),
        )
        sampled = ranked[: args.n_per_level]
        kept = ranked[args.n_per_level:]
        out_path = holdout_dir / f"{level}.jsonl"
        _write_records(out_path, sampled)
        totals["sampled"] += len(sampled)
        totals["remaining_in_seeds"] += len(kept)
        if not args.keep_in_seeds:
            _write_records(seeds_path, kept)
            LOGGER.info(
                "[%s] sampled %d → %s, kept %d in seeds",
                level, len(sampled), out_path, len(kept),
            )
        else:
            LOGGER.info(
                "[%s] sampled %d → %s (source seeds unchanged)",
                level, len(sampled), out_path,
            )

    print(
        f"\nholdout sampled {totals['sampled']} scenarios; "
        f"{totals['remaining_in_seeds']} remain in seeds/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
