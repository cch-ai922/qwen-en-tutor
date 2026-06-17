"""Scenario-seed generation.

Calls TOPIC_SEED_PROMPT in parallel batches to produce ScenarioSeed
objects per CEFR level, deduplicates by topic, runs a within-batch
diversity check on cities (>30%) and first names (>15%), and regenerates
overrepresented entries with an explicit AVOID-list addendum to the
prompt. Resumes from existing JSONL output.

NOTE: this module does not load a locale pool file. Locale grounding
comes entirely from prompts.py (which embeds the locale-instruction
block derived from ``config/locale.yaml``) and from the teacher's own
knowledge of the target country named there.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qwen_tutor.generation._prompt_select import (
    render_level_spec,
    validate_prompt_has_locale_instruction,
)
from qwen_tutor.generation.teacher import TeacherClient, build_teacher_from_config
from qwen_tutor.schemas import CATEGORIES, ScenarioSeed
from qwen_tutor.utils.runner import (
    append_failure,
    append_jsonl,
    extract_first_json,
    gather_with_concurrency,
    load_existing_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/seeds")
DEFAULT_FAILURES_PATH = Path("data/seeds_failures.jsonl")
DEFAULT_WARNINGS_PATH = Path("data/seeds_warnings.jsonl")

CITY_OVERREP_THRESHOLD = 0.30
NAME_OVERREP_THRESHOLD = 0.15


# The within-batch city distribution check does not use a country-specific
# static list or external NER. Since the seed prompt strongly instructs
# placing the city name at the start of the "setting" field, this simply
# extracts the first capitalized word or multi-word capitalized phrase from
# the setting string.
#
# Limitation: sentence-initial common words like "Today the weather is..."
# can produce false positives, so a short stopword set is filtered out.
# A more complete diversity check runs again in ``diversity.py`` (also regex-
# based), so cases missed here are caught later.

# Find the first multi-word capitalized phrase such as:
# "Shibuya crossing in Tokyo" → "Tokyo" / "Xuhui District, Shanghai" →
# "Xuhui District".
_CITY_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]{2,}(?:[\s\-][A-Z][a-zA-Z'\-]+){0,3})\b"
)

# Stopwords used to remove sentence-initial common-word false positives.
# Additions are welcome.
_CITY_STOPWORDS: frozenset[str] = frozenset(
    s.lower()
    for s in (
        "Today", "Tomorrow", "Yesterday", "Morning", "Afternoon", "Evening",
        "Night", "Spring", "Summer", "Autumn", "Winter", "Monday", "Tuesday",
        "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "The", "A", "An",
    )
)


def _extract_city(setting: str) -> str | None:
    """Return the first capitalized phrase from a setting string that looks like a city.

    Returns ``None`` if no match is found or only stopwords are matched
    (it is then excluded from the diversity count). See the comment above
    for limitations.
    """
    for m in _CITY_RE.finditer(setting):
        candidate = m.group(1).strip()
        first_word = candidate.split()[0].lower()
        if first_word in _CITY_STOPWORDS:
            continue
        return candidate
    return None


def _format_categories_block(categories: list[str]) -> str:
    """Render the per-scenario category assignment as a numbered list.

    The teacher is asked to honor these category tags when picking the
    topic for each scenario. The category value is stamped on the parsed
    output regardless (so the disk record always matches the request).
    """
    if not categories:
        return ""
    lines = [f"  {i + 1}. {c}" for i, c in enumerate(categories)]
    return "\n".join(lines)


def _parse_seed_batch(
    raw: str, level: str, locale: str, categories: list[str] | None = None
) -> list[ScenarioSeed]:
    data = extract_first_json(raw)
    if isinstance(data, dict):
        # Some models occasionally wrap the array in an outer object.
        for key in ("scenarios", "items", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array, got {type(data).__name__}")
    out: list[ScenarioSeed] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        # Pin the CEFR level, locale, and category from the request — some
        # models echo a different value back. Category in particular is
        # stamped per-position so the on-disk record always matches the
        # quota-balancer's assignment, even if the teacher picked a
        # different one in the JSON.
        stamp: dict[str, Any] = {"cefr_level": level, "locale": locale}
        if categories and i < len(categories):
            stamp["category"] = categories[i]
        item = {**item, **stamp}
        try:
            out.append(ScenarioSeed.model_validate(item))
        except ValidationError as exc:
            logger.warning("seeds: dropping malformed scenario: %s", exc)
    return out


async def _call_teacher_for_batch(
    teacher: TeacherClient,
    level: str,
    locale: str,
    batch_size: int,
    level_spec: str,
    avoid_items: list[str] | None,
    max_tokens: int,
    temperature: float,
    categories: list[str],
) -> list[ScenarioSeed]:
    from qwen_tutor.generation._prompt_select import render_prompt

    # Per-locale topic-seed prompt, localized at call time.
    topic_seed_prompt = render_prompt("topic_seed", locale_name=locale)
    prompt = topic_seed_prompt.format(
        N=batch_size,
        level=level,
        level_spec_with_locale_instruction=level_spec,
        categories_block=_format_categories_block(categories),
    )
    if avoid_items:
        prompt = (
            prompt
            + "\n\nADDITIONAL CONSTRAINT — across this batch, AVOID using any "
            "of the following overrepresented items entirely (do not include "
            "them in any scenario): "
            + ", ".join(sorted(set(avoid_items)))
        )
    validate_prompt_has_locale_instruction(prompt, locale_name=locale)
    raw = await teacher.generate(
        system=prompt,
        messages=[],
        cacheable_prefix=level_spec,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_seed_batch(raw, level, locale, categories=categories)


def _read_existing_seeds(
    output_path: Path, target_locale: str | None = None
) -> tuple[set[str], set[str]]:
    """Return (existing_ids, existing_topics) from a seeds output file.

    Both sets are scoped to ``target_locale`` when provided: a topic from
    a different country is NOT considered a duplicate, so e.g. "Weekend
    Market Visit" can legitimately appear once for china AND once for
    japan as different culturally-grounded scenarios. Only same-locale
    topic collisions are filtered.

    Seeds without a ``locale`` field (older single-locale data) are
    treated as default-locale ("china", matching the schema default).
    """
    ids: set[str] = set()
    topics: set[str] = set()
    if not output_path.exists():
        return ids, topics
    with output_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            row_locale = str(obj.get("locale", "china"))
            if target_locale is not None and row_locale != target_locale:
                continue
            if "topic" in obj:
                topics.add(str(obj["topic"]))
            if "id" in obj:
                ids.add(str(obj["id"]))
    return ids, topics


async def _generate_for_level(
    teacher: TeacherClient,
    level: str,
    locale: str,
    n_target: int,
    per_call: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    output_path: Path,
    failures_path: Path,
    warnings_path: Path,
    category_quota: dict[str, int] | None = None,
) -> int:
    """Generate seeds for a single (level, locale).

    When ``category_quota`` is provided, ``n_target`` is ignored and the
    function generates exactly ``sum(category_quota.values())`` new seeds
    with the requested per-category counts. Used by the top-up stage to
    target deficit categories after filtering. The assignments are
    shuffled deterministically so each batch sees a mix of categories
    rather than 5-in-a-row of the same one.
    """
    existing_ids, existing_topics = _read_existing_seeds(
        output_path, target_locale=locale
    )
    if category_quota is not None:
        remaining = sum(max(0, c) for c in category_quota.values())
        if remaining == 0:
            logger.info(
                "[seeds:%s/%s] top-up quota empty, skipping",
                level, locale,
            )
            return 0
    else:
        remaining = max(0, n_target - len(existing_ids))
        if remaining == 0:
            logger.info(
                "[seeds:%s/%s] already at %d/%d, skipping",
                level, locale, len(existing_ids), n_target,
            )
            return 0

    level_spec = render_level_spec(level, locale_name=locale)
    n_batches = (remaining + per_call - 1) // per_call

    # Round-robin category assignment across the full quota. The start offset
    # is derived from existing on-disk count so resumed runs continue cycling
    # from where they left off instead of always starting at category 0.
    # When ``category_quota`` overrides this, build the list explicitly from
    # the deficits and shuffle so batches contain a mix.
    start_offset = len(existing_ids)
    if category_quota is not None:
        import random as _random
        all_assignments = []
        for cat, n in category_quota.items():
            if n > 0:
                all_assignments.extend([cat] * n)
        # Deterministic shuffle keyed on (level, locale) so reruns of the
        # same top-up request hand the same assignments to the teacher.
        _random.Random(f"topup:{level}:{locale}").shuffle(all_assignments)
    else:
        all_assignments = [
            CATEGORIES[(start_offset + k) % len(CATEGORIES)] for k in range(remaining)
        ]

    def _slice_for_batch(idx: int) -> list[str]:
        return all_assignments[idx * per_call : (idx + 1) * per_call]

    async def _one_batch(idx: int) -> list[ScenarioSeed]:
        try:
            return await _call_teacher_for_batch(
                teacher, level, locale, per_call, level_spec,
                avoid_items=None,
                max_tokens=max_tokens, temperature=temperature,
                categories=_slice_for_batch(idx),
            )
        except Exception as exc:  # noqa: BLE001
            append_failure(
                failures_path,
                f"seed_{locale}_{level}_batch{idx}",
                f"{type(exc).__name__}: {exc}",
                level=level, stage="seeds",
            )
            return []

    batches = await gather_with_concurrency(
        [_one_batch(i) for i in range(n_batches)],
        concurrency=concurrency,
        desc=f"seeds[{locale}/{level}]",
    )

    candidates: list[ScenarioSeed] = []
    seen_topics = set(existing_topics)
    for batch in batches:
        for seed in batch:
            if seed.topic in seen_topics:
                continue
            seen_topics.add(seed.topic)
            candidates.append(seed)

    # within-batch diversity check
    city_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    for seed in candidates:
        city = _extract_city(seed.setting)
        if city:
            city_counts[city] += 1
        name_counts[seed.user_role.name] += 1

    total = len(candidates)
    overrep_cities: list[str] = []
    overrep_names: list[str] = []
    if total > 0:
        overrep_cities = [c for c, n in city_counts.items() if n / total > CITY_OVERREP_THRESHOLD]
        overrep_names = [n for n, c in name_counts.items() if c / total > NAME_OVERREP_THRESHOLD]

    if overrep_cities or overrep_names:
        append_jsonl(
            warnings_path,
            {
                "level": level,
                "locale": locale,
                "total_candidates": total,
                "overrep_cities": overrep_cities,
                "overrep_names": overrep_names,
                "city_distribution": city_counts.most_common(),
                "name_distribution": name_counts.most_common(),
            },
        )
        affected = [
            s for s in candidates
            if (_extract_city(s.setting) in overrep_cities)
            or (s.user_role.name in overrep_names)
        ]
        if affected:
            logger.info(
                "[seeds:%s/%s] low-diversity: dropping %d affected scenarios, regenerating",
                level, locale, len(affected),
            )
            candidates = [s for s in candidates if s not in affected]
            regen_batches = (len(affected) + per_call - 1) // per_call
            avoid = overrep_cities + overrep_names

            # Continue the round-robin from where the main loop left off so
            # regenerated seeds stay close to the original category balance.
            regen_offset = start_offset + len(all_assignments)
            regen_total = regen_batches * per_call
            regen_assignments = [
                CATEGORIES[(regen_offset + k) % len(CATEGORIES)] for k in range(regen_total)
            ]

            async def _regen(idx: int) -> list[ScenarioSeed]:
                try:
                    return await _call_teacher_for_batch(
                        teacher, level, locale, per_call, level_spec,
                        avoid_items=avoid,
                        max_tokens=max_tokens, temperature=temperature,
                        categories=regen_assignments[idx * per_call : (idx + 1) * per_call],
                    )
                except Exception as exc:  # noqa: BLE001
                    append_failure(
                        failures_path,
                        f"seed_{locale}_{level}_regen{idx}",
                        f"{type(exc).__name__}: {exc}",
                        level=level, stage="seeds_regen",
                    )
                    return []

            regen_results = await gather_with_concurrency(
                [_regen(i) for i in range(regen_batches)],
                concurrency=concurrency,
                desc=f"seeds[{locale}/{level}]/regen",
            )
            for batch in regen_results:
                for seed in batch:
                    if seed.topic in seen_topics:
                        continue
                    if _extract_city(seed.setting) in overrep_cities:
                        continue
                    if seed.user_role.name in overrep_names:
                        continue
                    seen_topics.add(seed.topic)
                    candidates.append(seed)

    written = 0
    quota = n_target - len(existing_ids)
    written_categories: Counter[str] = Counter()
    for seed in candidates[:quota]:
        seed_id = uuid.uuid4().hex[:12]
        # ScenarioSeed.model_dump() includes ``locale`` (set by _parse_seed_batch)
        # AND an Optional ``id`` field that defaults to None — we have to
        # exclude it from the unpack so our freshly-generated ``seed_id``
        # isn't shadowed by None.
        record = {"id": seed_id, **seed.model_dump(exclude={"id"})}
        append_jsonl(output_path, record)
        written_categories[seed.category] += 1
        written += 1
    if written:
        logger.info(
            "[seeds:%s/%s] category distribution this run: %s",
            level, locale, dict(written_categories.most_common()),
        )
    return written


async def generate_batch(
    n_per_level: int,
    cefr_levels: list[str],
    config_path: str | Path = "config/generation.yaml",
    concurrency: int = 20,
    per_call_size: int = 10,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failures_path: str | Path = DEFAULT_FAILURES_PATH,
    warnings_path: str | Path = DEFAULT_WARNINGS_PATH,
    teacher: TeacherClient | None = None,
    locales: list[str] | None = None,
    category_quotas: dict[tuple[str, str], dict[str, int]] | None = None,
) -> dict[str, int]:
    """Generate scenario seeds for each (CEFR level, locale) pair.

    If ``locales`` is ``None`` or empty, this runs with the single default
    locale from ``config/locale.yaml`` (single-locale compatible mode).
    If multiple locales are provided, it generates ``n_per_level`` seeds for
    each (level, locale) combination.

    Returns ``{f"{level}/{locale}": n_written_this_run}``. Existing seeds in
    each per-level file are preserved and contribute to the per-(level,locale)
    quota.
    """
    if teacher is None:
        teacher = build_teacher_from_config(config_path, role="teacher")
    if not locales:
        from qwen_tutor.locale import DEFAULT_LOCALE_NAME

        locales = [DEFAULT_LOCALE_NAME]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"{level}.jsonl"
        for locale in locales:
            quota = (category_quotas or {}).get((level, locale))
            written = await _generate_for_level(
                teacher=teacher,
                level=level,
                locale=locale,
                n_target=n_per_level,
                per_call=per_call_size,
                concurrency=concurrency,
                max_tokens=max_tokens,
                temperature=temperature,
                output_path=out_path,
                failures_path=Path(failures_path),
                warnings_path=Path(warnings_path),
                category_quota=quota,
            )
            results[f"{level}/{locale}"] = written
    return results


def iter_seeds(
    seeds_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cefr_levels: list[str] | None = None,
):
    """Iterate ``(id, ScenarioSeed)`` pairs from the per-level seed files.

    Used by downstream stages (SFT, redirect, register pairs, evaluation)
    to walk the seed corpus without re-loading the whole batch into
    memory.
    """
    seeds_dir = Path(seeds_dir)
    levels = cefr_levels or ["A1", "A2", "B1", "B2", "C1", "C2"]
    for level in levels:
        path = seeds_dir / f"{level}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                seed_id = str(obj.pop("id"))
                seed = ScenarioSeed.model_validate(obj)
                yield seed_id, seed
