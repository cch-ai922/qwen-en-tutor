"""User-configurable life-domain categories for seed generation.

The category set the seed quota-balancer cycles through used to be a
hard-coded ``Literal`` enum in ``schemas.py``. It is now read from
``generation.categories`` in ``config/generation.yaml`` so a deployment can
add / remove / rename categories without touching code. When the config key
is absent, the built-in ``schemas.CATEGORIES`` defaults are used, so existing
setups behave exactly as before.

Config accepts either a plain list of names or a list of ``{name,
description}`` objects (mix-and-match allowed)::

    generation:
      categories:
        - food_and_dining                       # name only -> default/blank description
        - name: neighborhood_gossip
          description: "chatting with neighbors about local happenings."

``description`` is a soft guide injected into the seed prompt's
"Category meaning" block; ``{country}`` inside a description is substituted at
render time. A missing description falls back to the built-in default for that
name, or to the name itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from qwen_tutor.schemas import CATEGORIES, DEFAULT_CATEGORY_DESCRIPTIONS

logger = logging.getLogger(__name__)

# (name, description) pairs.
CategorySpec = tuple[str, str]


def _default_specs() -> list[CategorySpec]:
    return [(name, DEFAULT_CATEGORY_DESCRIPTIONS.get(name, name)) for name in CATEGORIES]


def _coerce_item(item: object) -> CategorySpec | None:
    """Normalize one config entry into a (name, description) pair."""
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return name, DEFAULT_CATEGORY_DESCRIPTIONS.get(name, name)
    if isinstance(item, dict):
        name = str(item.get("name", "")).strip()
        if not name:
            return None
        desc = item.get("description")
        if desc is None or not str(desc).strip():
            desc = DEFAULT_CATEGORY_DESCRIPTIONS.get(name, name)
        return name, str(desc).strip()
    logger.warning("categories: ignoring malformed entry %r", item)
    return None


def load_category_specs(
    config_path: str | Path = "config/generation.yaml",
) -> list[CategorySpec]:
    """Return the configured ``[(name, description), ...]`` category specs.

    Falls back to the built-in defaults when the config file is missing, the
    ``generation.categories`` key is absent/empty, or every entry is malformed.
    Duplicate names are de-duplicated, keeping the first occurrence.
    """
    path = Path(config_path)
    if not path.exists():
        return _default_specs()
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - config typo guard
        logger.warning("categories: could not read %s (%s); using defaults", path, exc)
        return _default_specs()

    raw = ((cfg.get("generation") or {}).get("categories")) or []
    if not isinstance(raw, list):
        logger.warning(
            "categories: generation.categories must be a list, got %s; using defaults",
            type(raw).__name__,
        )
        return _default_specs()

    specs: list[CategorySpec] = []
    seen: set[str] = set()
    for item in raw:
        pair = _coerce_item(item)
        if pair is None:
            continue
        name, desc = pair
        if name in seen:
            logger.warning("categories: duplicate category %r ignored", name)
            continue
        seen.add(name)
        specs.append((name, desc))

    if not specs:
        return _default_specs()
    return specs


def load_category_names(
    config_path: str | Path = "config/generation.yaml",
) -> list[str]:
    """Return just the configured category names (balancer cycle order)."""
    return [name for name, _ in load_category_specs(config_path)]


def render_category_meanings_block(
    specs: list[CategorySpec], country: str
) -> str:
    """Render the "Category meaning (soft guide)" block for the seed prompt.

    ``{country}`` tokens in descriptions are substituted with ``country``.
    Kept as a plain pre-rendered string (not a format template) so arbitrary
    user text with stray braces cannot break ``str.format`` downstream.
    """
    lines = [
        f"  - {name}: {desc.replace('{country}', country)}" for name, desc in specs
    ]
    return "\n".join(lines)
