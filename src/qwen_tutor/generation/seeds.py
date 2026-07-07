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
from qwen_tutor.generation.prompts import NO_THINK_DIRECTIVE
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
    """Render the batch's category assignment for the seed prompt.

    Batches are homogeneous — every scenario in one teacher call shares a
    single category — so the teacher can diversify WITHIN that category in
    one shot (the fix for near-duplicate seeds that arose when each seed of
    a category was generated in a separate, mutually-blind call). For a
    homogeneous batch this emits a single strong "produce N genuinely
    different scenarios all in category X" instruction; a mixed batch (only
    the legacy path) falls back to a numbered per-position list. The
    category value is stamped on the parsed output regardless (so the disk
    record always matches the request).
    """
    if not categories:
        return ""
    n = len(categories)
    if len(set(categories)) == 1:
        cat = categories[0]
        return (
            f"ALL {n} scenario(s) in this batch are in the SINGLE life-domain "
            f"category: {cat}.\n"
            f"Produce {n} genuinely DIFFERENT scenario(s) within this one "
            f"category — vary the specific situation, sub-focus, roles, "
            f"setting, city, and time so no two feel like the same scene. "
            f"Do NOT drift out of the category."
        )
    lines = [f"  {i + 1}. {c}" for i, c in enumerate(categories)]
    return "\n".join(lines)


def _build_homogeneous_batches(
    per_category_counts: dict[str, int], per_call: int
) -> list[list[str]]:
    """Split per-category counts into homogeneous batches of <= ``per_call``.

    Returns a list of batches, each a list like ``["food_and_dining"] * k``
    (k <= per_call). Categories with a non-positive count are skipped. Each
    resulting batch becomes one teacher call, so all seeds of a category are
    requested together (in ``per_call``-sized chunks when the count exceeds
    the per-call ceiling).
    """
    batches: list[list[str]] = []
    for cat, count in per_category_counts.items():
        remaining = max(0, int(count))
        while remaining > 0:
            take = min(per_call, remaining)
            batches.append([cat] * take)
            remaining -= take
    return batches


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
    category_meanings_block: str,
    avoid_topics: list[str] | None = None,
) -> list[ScenarioSeed]:
    from qwen_tutor.generation._prompt_select import render_prompt

    # Per-locale topic-seed prompt, localized at call time.
    topic_seed_prompt = render_prompt("topic_seed", locale_name=locale)
    prompt = topic_seed_prompt.format(
        N=batch_size,
        level=level,
        level_spec_with_locale_instruction=level_spec,
        categories_block=_format_categories_block(categories),
        category_meanings_block=category_meanings_block,
    )
    # Topics already on disk for this category — steer the teacher away from
    # regenerating them (cross-batch / resumed-run duplicate suppression, on
    # top of the within-batch distinctness the homogeneous prompt requests).
    if avoid_topics:
        shown = sorted(set(avoid_topics))[:40]
        prompt = (
            prompt
            + "\n\nDO NOT REUSE these existing topics (already generated for "
            "this category — pick genuinely new situations): "
            + "; ".join(shown)
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


# Subtopic count policy. Small quantized teachers anchor hard at 3 subtopics
# regardless of prompt pressure, so we enrich short seeds with a focused
# post-generation expansion call (one per short seed) rather than fighting
# the count in the main scenario prompt. ``_SUBTOPIC_MAX`` matches the schema
# ``max_length``. Expansion is skipped when the configured target is <= 3.
_SUBTOPIC_MIN = 4
_SUBTOPIC_MAX = 6

_SUBTOPIC_EXPANSION_PROMPT = """\
You are refining ONE English-tutoring conversation scenario by adding more
"subtopic" beats to it.

Scenario topic: {topic}
Setting: {setting}
Existing subtopics:
{existing_block}

Add {n_more} MORE subtopic(s). Each new subtopic MUST:
  - be a concrete conversational beat the dialogue can spend a few turns on;
  - be a short clause of roughly 5-10 words with a specific detail (NOT a
    bare 2-3 word label);
  - be genuinely DIFFERENT from every existing subtopic above — no synonyms
    or rephrasings;
  - fit the same topic and setting.

Return ONLY a JSON array of exactly {n_more} new subtopic string(s). No prose,
no markdown, no commentary.
["...", "..."]"""
# Force /no_think: this is a mechanical extraction task, and a thinking-mode
# teacher otherwise burns the token budget in <think> and never emits the array.
_SUBTOPIC_EXPANSION_PROMPT += NO_THINK_DIRECTIVE


async def _expand_subtopics(
    teacher: TeacherClient,
    seed: ScenarioSeed,
    *,
    target: int,
    max_tokens: int,
    temperature: float,
) -> ScenarioSeed:
    """Top a short seed's subtopics up toward ``target`` with one teacher call.

    Fail-forward: on any error, unparseable output, or empty result the seed
    is returned unchanged (never dropped). Deduplicates against existing
    subtopics and caps the merged list at ``_SUBTOPIC_MAX``.
    """
    have = list(seed.subtopics)
    n_more = min(target, _SUBTOPIC_MAX) - len(have)
    if n_more <= 0:
        return seed
    try:
        existing_block = "\n".join(f"  - {s}" for s in have)
        prompt = _SUBTOPIC_EXPANSION_PROMPT.format(
            n_more=n_more, topic=seed.topic, setting=seed.setting,
            existing_block=existing_block,
        )
        raw = await teacher.generate(
            system=prompt, messages=[],
            max_tokens=min(max_tokens, 512), temperature=temperature,
        )
        data = extract_first_json(raw)
        if isinstance(data, dict):
            for key in ("subtopics", "items", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            return seed
        seen = {s.strip().lower() for s in have}
        additions: list[str] = []
        for item in data:
            if not isinstance(item, str):
                continue
            t = item.strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            additions.append(t)
        if not additions:
            return seed
        merged = (have + additions)[:_SUBTOPIC_MAX]
        return seed.model_copy(update={"subtopics": merged})
    except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
        logger.debug("seeds: subtopic expansion failed for %r: %s", seed.topic, exc)
        return seed


def _read_existing_seeds(
    output_path: Path, target_locale: str | None = None
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Return (existing_ids, existing_topics, topics_by_category).

    All three are scoped to ``target_locale`` when provided: a topic from
    a different country is NOT considered a duplicate, so e.g. "Weekend
    Market Visit" can legitimately appear once for china AND once for
    japan as different culturally-grounded scenarios. Only same-locale
    topic collisions are filtered.

    ``topics_by_category`` maps ``category -> {topics}`` and feeds each
    homogeneous batch's per-category "do not reuse" avoid-list.

    Seeds without a ``locale`` field (older single-locale data) are
    treated as default-locale ("china", matching the schema default);
    seeds without a ``category`` fall under "general".
    """
    ids: set[str] = set()
    topics: set[str] = set()
    topics_by_category: dict[str, set[str]] = {}
    if not output_path.exists():
        return ids, topics, topics_by_category
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
                topic = str(obj["topic"])
                topics.add(topic)
                cat = str(obj.get("category", "general"))
                topics_by_category.setdefault(cat, set()).add(topic)
            if "id" in obj:
                ids.add(str(obj["id"]))
    return ids, topics, topics_by_category


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
    category_names: list[str],
    category_meanings_block: str,
    subtopic_target: int = 5,
    category_quota: dict[str, int] | None = None,
) -> int:
    """Generate seeds for a single (level, locale).

    ``category_names`` is the configured category cycle (see
    ``config/generation.yaml`` -> ``generation.categories``). Each teacher
    call is HOMOGENEOUS — every seed in one call shares a single category —
    so the teacher diversifies within the category in one shot instead of
    each seed being drawn from a separate, mutually-blind call (the cause of
    near-duplicate seeds within a category).

    When ``category_quota`` is provided, ``n_target`` is ignored and the
    function generates exactly ``sum(category_quota.values())`` new seeds
    with the requested per-category counts. Used by the top-up stage to
    target deficit categories after filtering.
    """
    existing_ids, existing_topics, topics_by_category = _read_existing_seeds(
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

    # Per-category target counts. Top-up passes them explicitly; the normal
    # path distributes ``remaining`` across the configured categories
    # round-robin, offset by the existing on-disk count so resumed runs keep
    # the balance even rather than always restarting at category 0.
    start_offset = len(existing_ids)
    if category_quota is not None:
        per_category_counts: dict[str, int] = {
            cat: max(0, n) for cat, n in category_quota.items() if n > 0
        }
    else:
        cycle = category_names or list(CATEGORIES)
        per_category_counts = {}
        for k in range(remaining):
            cat = cycle[(start_offset + k) % len(cycle)]
            per_category_counts[cat] = per_category_counts.get(cat, 0) + 1

    # One homogeneous teacher call per (category, per_call-chunk).
    all_batches = _build_homogeneous_batches(per_category_counts, per_call)
    n_batches = len(all_batches)

    async def _one_batch(idx: int) -> list[ScenarioSeed]:
        cats = all_batches[idx]
        try:
            return await _call_teacher_for_batch(
                teacher, level, locale, len(cats), level_spec,
                avoid_items=None,
                max_tokens=max_tokens, temperature=temperature,
                categories=cats,
                category_meanings_block=category_meanings_block,
                avoid_topics=sorted(topics_by_category.get(cats[0], set())),
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
            avoid = overrep_cities + overrep_names

            # Regenerate the dropped seeds keeping their own categories, in
            # homogeneous per-category batches (same distinctness benefit as
            # the main path), so the category balance is preserved.
            regen_counts: Counter[str] = Counter(s.category for s in affected)
            regen_batches_list = _build_homogeneous_batches(dict(regen_counts), per_call)

            async def _regen(idx: int) -> list[ScenarioSeed]:
                cats = regen_batches_list[idx]
                try:
                    return await _call_teacher_for_batch(
                        teacher, level, locale, len(cats), level_spec,
                        avoid_items=avoid,
                        max_tokens=max_tokens, temperature=temperature,
                        categories=cats,
                        category_meanings_block=category_meanings_block,
                        avoid_topics=sorted(seen_topics),
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
                [_regen(i) for i in range(len(regen_batches_list))],
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

    # Subtopic expansion — the teacher anchors at 3 subtopics no matter the
    # prompt pressure, so enrich short seeds with one focused call each (only
    # those we will actually write, concurrency-bounded). Skipped when the
    # configured target is <= 3.
    if subtopic_target >= _SUBTOPIC_MIN:
        to_expand = [
            (i, s) for i, s in enumerate(candidates[:remaining])
            if len(s.subtopics) < subtopic_target
        ]
        if to_expand:
            expanded = await gather_with_concurrency(
                [
                    _expand_subtopics(
                        teacher, s, target=subtopic_target,
                        max_tokens=max_tokens, temperature=temperature,
                    )
                    for _, s in to_expand
                ],
                concurrency=concurrency,
                desc=f"seeds[{locale}/{level}]/subtopics",
            )
            for (i, _orig), new_seed in zip(to_expand, expanded):
                if new_seed is not None:
                    candidates[i] = new_seed

    written = 0
    # Cap writes at the number requested this run. ``remaining`` already
    # accounts for both paths: top-up = sum of the category quota; normal =
    # n_target minus existing. (Previously recomputed as n_target - existing,
    # which went negative on the top-up path where n_target is passed as 0.)
    written_categories: Counter[str] = Counter()
    for seed in candidates[:remaining]:
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
    subtopic_target: int = 5,
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

    # User-configurable category set (generation.categories in the config).
    # Names drive the round-robin cycle; the meanings block is injected into
    # the seed prompt and is re-rendered per locale so "{country}" resolves.
    from qwen_tutor.generation.categories import (
        load_category_specs,
        render_category_meanings_block,
    )
    from qwen_tutor.locale import get_locale

    category_specs = load_category_specs(config_path)
    category_names = [name for name, _ in category_specs]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for level in cefr_levels:
        out_path = output_dir / f"{level}.jsonl"
        for locale in locales:
            quota = (category_quotas or {}).get((level, locale))
            meanings_block = render_category_meanings_block(
                category_specs, country=get_locale(locale).country
            )
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
                category_names=category_names,
                category_meanings_block=meanings_block,
                subtopic_target=subtopic_target,
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
