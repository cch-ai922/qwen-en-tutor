"""Semantic deduplication of generated scenario seeds.

Why this exists
---------------
The seed generator's existing dedup (``seeds.py`` ``seen_topics`` set)
catches **exact-string** topic duplicates only. Near-duplicates with
distinct topic strings sneak through:

    "buying apples at the Sanyuanli market"     # batch 3, seed 7
    "purchasing oranges at the Sanyuanli market" # batch 12, seed 4

Both pass the exact-string check, both produce nearly identical SFT
dialogues, and both waste teacher tokens.

This module reads each ``data/seeds/<level>.jsonl`` file, groups seeds
by ``(cefr_level, locale, category)`` and either:

  * **Threshold pruning (default).** Walk seeds in order. Keep a seed
    only if its maximum cosine similarity (TF-IDF over
    ``topic + setting + subtopics``) to any already-kept seed in the
    same cell is below ``similarity_threshold``.
  * **Greedy farthest-point (when ``max_per_cell`` is set).** Anchor at
    the least-central seed in the cell, then repeatedly add the seed
    whose minimum similarity to the kept set is smallest, until the
    cell has ``max_per_cell`` seeds.

Pure stdlib — no sklearn, no sentence-transformers, no model download.
Safe under the project's offline-mode requirement.

Use this when you over-generate seeds (e.g. ``n_per_level=500``) and
want a controlled, semantically-diverse subset to feed downstream
``sft`` / ``redirect`` / ``eval`` stages.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimal English stopword list. TF-IDF's IDF already downweights
# frequent terms, so this only trims the most obvious noise.
_STOPWORDS: frozenset[str] = frozenset(
    "the a an and or but if of to in on at for with by from as is are was "
    "were be been being have has had do does did this that these those it "
    "its his her their our my your i you he she we they them us not no "
    "so than then there here too very up out about into over under "
    "between within while".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _seed_feature_text(seed: dict) -> str:
    """Concatenate the scenario fields that distinguish one seed from another.

    We intentionally include ``setting`` (city / neighborhood / time)
    because two scenarios with the same topic but different settings
    are genuinely different conversations.
    """
    parts = [
        seed.get("topic") or "",
        seed.get("setting") or "",
        " ".join(seed.get("subtopics") or []),
    ]
    return " ".join(parts)


def _tf_idf(docs: list[list[str]]) -> list[dict[str, float]]:
    """Return l2-normalized TF-IDF vectors as ``{term: weight}`` dicts.

    Smoothed IDF: ``log((1+N)/(1+df)) + 1`` so terms appearing in every
    document still get a small non-zero weight rather than being
    dropped entirely.
    """
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log((1 + n) / (1 + dfv)) + 1.0 for t, dfv in df.items()}
    vectors: list[dict[str, float]] = []
    for d in docs:
        tf = Counter(d)
        v = {t: c * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in v.values()))
        if norm > 0:
            v = {t: w / norm for t, w in v.items()}
        vectors.append(v)
    return vectors


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    # Vectors are l2-normalized so cosine == dot product. Iterate the
    # shorter vector to keep the inner loop small.
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _max_sim_to_kept(
    vec: dict[str, float], kept_vecs: list[dict[str, float]]
) -> float:
    if not kept_vecs:
        return 0.0
    return max(_cos(vec, kv) for kv in kept_vecs)


def _dedup_cell(
    seeds: list[dict],
    similarity_threshold: float,
    max_per_cell: int | None,
) -> tuple[list[dict], list[dict]]:
    """Return ``(kept, dropped)`` for one ``(level, locale, category)`` cell."""
    if len(seeds) <= 1:
        return list(seeds), []
    docs = [_tokenize(_seed_feature_text(s)) for s in seeds]
    vecs = _tf_idf(docs)

    kept: list[dict] = []
    kept_vecs: list[dict[str, float]] = []
    dropped: list[dict] = []

    if max_per_cell is None:
        # Linear threshold pass — preserves input order so reruns are
        # deterministic relative to seeds.jsonl line order.
        for seed, vec in zip(seeds, vecs):
            if _max_sim_to_kept(vec, kept_vecs) < similarity_threshold:
                kept.append(seed)
                kept_vecs.append(vec)
            else:
                dropped.append(seed)
        return kept, dropped

    # Greedy farthest-point. Seed the chain at the least-central seed
    # so subsequent picks fan out from an outlier rather than the
    # densest cluster center.
    remaining = list(range(len(seeds)))
    mean_sims: list[float] = []
    for i in remaining:
        sims = [_cos(vecs[i], vecs[j]) for j in remaining if j != i]
        mean_sims.append(sum(sims) / len(sims) if sims else 0.0)
    seed_pos = mean_sims.index(min(mean_sims))
    seed_idx = remaining[seed_pos]
    kept.append(seeds[seed_idx])
    kept_vecs.append(vecs[seed_idx])
    remaining.remove(seed_idx)

    while remaining and len(kept) < max_per_cell:
        best_idx: int | None = None
        best_max_sim = math.inf
        for i in remaining:
            ms = _max_sim_to_kept(vecs[i], kept_vecs)
            if ms < best_max_sim:
                best_max_sim = ms
                best_idx = i
        if best_idx is None:
            break
        # Honor the threshold even in greedy mode: if all remaining
        # seeds are too similar to the kept set, stop early rather
        # than fill the quota with near-duplicates.
        if best_max_sim >= similarity_threshold:
            break
        kept.append(seeds[best_idx])
        kept_vecs.append(vecs[best_idx])
        remaining.remove(best_idx)

    dropped = [seeds[i] for i in remaining]
    return kept, dropped


def _group_by_cell(
    seeds: Iterable[dict],
) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in seeds:
        key = (
            s.get("cefr_level", ""),
            s.get("locale", "china"),
            s.get("category", "general"),
        )
        groups[key].append(s)
    return groups


def dedup_seeds_file(
    path: Path,
    *,
    similarity_threshold: float = 0.75,
    max_per_cell: int | None = None,
    backup: bool = True,
    dropped_log: Path | None = None,
) -> dict:
    """Rebuild one ``seeds/<level>.jsonl`` file with dedup applied.

    Groups seeds by ``(cefr_level, locale, category)`` and either
    threshold-prunes (default) or greedy-keeps ``max_per_cell`` most-distinct
    seeds per cell.

    If ``backup=True`` and ``<path>.bak`` does not already exist, the
    original file is snapshotted to ``<path>.bak`` before being rewritten
    in place. Existing ``.bak`` is NEVER overwritten so the *original*
    pre-dedup snapshot is always preserved across reruns.

    Returns a stats dict with input / kept / dropped counts and per-cell
    breakdown.
    """
    path = Path(path)
    if not path.exists():
        return {"file": str(path), "input": 0, "kept": 0, "dropped": 0, "per_cell": []}

    seeds: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seeds.append(json.loads(line))

    if backup:
        bak = path.parent / (path.name + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    cells = _group_by_cell(seeds)
    kept_all: list[dict] = []
    dropped_all: list[dict] = []
    per_cell_stats: list[tuple[tuple[str, str, str], int, int]] = []
    for key, cell_seeds in cells.items():
        kept, dropped = _dedup_cell(
            cell_seeds,
            similarity_threshold=similarity_threshold,
            max_per_cell=max_per_cell,
        )
        kept_all.extend(kept)
        dropped_all.extend(dropped)
        per_cell_stats.append((key, len(cell_seeds), len(kept)))

    with path.open("w", encoding="utf-8") as fh:
        for s in kept_all:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    if dropped_log is not None and dropped_all:
        dropped_log.parent.mkdir(parents=True, exist_ok=True)
        with dropped_log.open("a", encoding="utf-8") as fh:
            for s in dropped_all:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "file": str(path),
        "input": len(seeds),
        "kept": len(kept_all),
        "dropped": len(dropped_all),
        "per_cell": [
            {
                "level": k[0],
                "locale": k[1],
                "category": k[2],
                "input": ni,
                "kept": nk,
            }
            for (k, ni, nk) in per_cell_stats
        ],
    }
    logger.info(
        "[dedup_seeds] %s: %d -> %d (dropped %d)",
        path.name, stats["input"], stats["kept"], stats["dropped"],
    )
    return stats


def dedup_seeds_dir(
    seeds_dir: Path,
    *,
    levels: list[str],
    similarity_threshold: float = 0.75,
    max_per_cell: int | None = None,
    backup: bool = True,
    dropped_log_dir: Path | None = None,
) -> dict:
    """Run dedup over every ``<level>.jsonl`` in ``seeds_dir``.

    ``dropped_log_dir`` defaults to ``<seeds_dir>/_dropped/`` so you can
    inspect which seeds were pruned and refine the threshold. Pass
    ``False`` to disable the dropped log entirely.
    """
    seeds_dir = Path(seeds_dir)
    out: dict = {"files": [], "input": 0, "kept": 0, "dropped": 0}
    default_log_dir = seeds_dir / "_dropped"
    for level in levels:
        path = seeds_dir / f"{level}.jsonl"
        if dropped_log_dir is False:
            log_path: Path | None = None
        else:
            log_dir = Path(dropped_log_dir) if dropped_log_dir is not None else default_log_dir
            log_path = log_dir / f"{level}.jsonl"
        s = dedup_seeds_file(
            path,
            similarity_threshold=similarity_threshold,
            max_per_cell=max_per_cell,
            backup=backup,
            dropped_log=log_path,
        )
        out["files"].append(s)
        out["input"] += s["input"]
        out["kept"] += s["kept"]
        out["dropped"] += s["dropped"]
    return out
