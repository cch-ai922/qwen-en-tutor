"""Tests for the prompt-template module.

The most important regression guard here is
``validate_prompt_has_locale_instruction``: every rendered prompt MUST
contain the Iranian locale-instruction header. These tests fail loudly if
a refactor accidentally removes it from a template or from the level
spec.
"""

import json

import pytest

from qwen_tutor.generation.prompts import (
    ANTI_FAILURE_MODE_BLOCK,
    DIALOGUE_PROMPT_NORMAL,
    DIALOGUE_PROMPT_REDIRECT,
    EVALUATION_GENERATION_PROMPT,
    LOCALE_INSTRUCTION_BLOCK,
    LOCALE_INSTRUCTION_HEADER,
    REGISTER_REWRITE_PROMPT,
    TEMPLATES,
    TOPIC_SEED_PROMPT,
    render_level_spec,
    validate_prompt_has_locale_instruction,
)


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


# ---------------------------------------------------------------------------
# validate_prompt_has_locale_instruction
# ---------------------------------------------------------------------------


def test_validate_passes_when_header_present():
    validate_prompt_has_locale_instruction(
        "preamble\n" + LOCALE_INSTRUCTION_HEADER + "\nbody"
    )


def test_validate_raises_when_header_missing():
    # The header text comes from LOCALE.locale_instruction_header — currently
    # rendered as "[locale-rules:<country>]" (e.g. "[locale-rules:china]").
    # The error message echoes the header verbatim, so we match on the stable
    # code-marker prefix here so the test stays correct regardless of which
    # locale config/locale.yaml is pointing at.
    with pytest.raises(ValueError, match=r"\[locale-rules:"):
        validate_prompt_has_locale_instruction("some prompt without the header")


# ---------------------------------------------------------------------------
# render_level_spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", CEFR_LEVELS)
def test_render_level_spec_includes_locale_instruction(level):
    rendered = render_level_spec(level)
    validate_prompt_has_locale_instruction(rendered)
    assert f"CEFR LEVEL: {level}" in rendered
    assert "Vocabulary:" in rendered
    assert "Grammar:" in rendered
    assert "Naturalness requirements:" in rendered
    assert "Things to avoid:" in rendered
    assert "Few-shot example dialogues" in rendered


def test_render_level_spec_unknown_level_raises():
    with pytest.raises(KeyError):
        render_level_spec("Z9")


# ---------------------------------------------------------------------------
# Locale guard on every template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,template", list(TEMPLATES.items()))
def test_unrendered_template_already_contains_locale_header(name, template):
    """Each template embeds the locale block literally, so the header is
    present even before any .format() substitution happens."""
    validate_prompt_has_locale_instruction(template)


@pytest.mark.parametrize("name,template", list(TEMPLATES.items()))
def test_template_contains_anti_failure_block_or_inline_anti_failure_language(
    name, template
):
    # All five templates should contain explicit anti-Western-defaults
    # language. The shared block is included in three of them; the rewrite
    # template inlines its own anti-Westernization guidance.
    has_block = ANTI_FAILURE_MODE_BLOCK.splitlines()[0] in template
    has_inline = "Do NOT use" in template or "do NOT use" in template
    assert has_block or has_inline, (
        f"{name} is missing anti-failure-mode language"
    )


# ---------------------------------------------------------------------------
# Rendered templates
# ---------------------------------------------------------------------------


def test_topic_seed_prompt_renders():
    rendered = TOPIC_SEED_PROMPT.format(
        N=5,
        level="B1",
        level_spec_with_locale_instruction=render_level_spec("B1"),
        categories_block="  1. food_and_dining\n  2. family_and_relationships",
    )
    validate_prompt_has_locale_instruction(rendered)
    assert "generate 5 distinct scenarios" in rendered.lower()
    assert "CEFR LEVEL: B1" in rendered


def test_dialogue_prompt_normal_renders():
    scenario = {
        "topic": "asking about a power outage",
        "subtopics": ["when it started", "how long", "what to do for dinner"],
        "user_role": {"name": "Niloofar", "description": "designer in Sa'adat Abad"},
        "model_role": {"name": "neighbor", "description": "Mr. Ahmadi downstairs"},
        "setting": "apartment hallway, Tehran, summer afternoon",
        "cefr_level": "A2",
    }
    rendered = DIALOGUE_PROMPT_NORMAL.format(
        scenario_json=json.dumps(scenario, ensure_ascii=False),
        level="A2",
        level_spec_with_locale_instruction=render_level_spec("A2"),
    )
    validate_prompt_has_locale_instruction(rendered)
    assert "Niloofar" in rendered
    assert "CEFR LEVEL: A2" in rendered


def test_dialogue_prompt_redirect_renders():
    scenario = {
        "topic": "asking about evening plans",
        "subtopics": ["restaurant", "ride home", "schedule"],
        "user_role": {"name": "Sina", "description": "engineering student"},
        "model_role": {"name": "classmate", "description": "Sara from his cohort"},
        "setting": "Sharif campus cafeteria, Tehran, evening",
        "cefr_level": "B2",
    }
    rendered = DIALOGUE_PROMPT_REDIRECT.format(
        scenario_json=json.dumps(scenario, ensure_ascii=False),
        level="B2",
        level_spec_with_locale_instruction=render_level_spec("B2"),
        redirect_axis="alcohol_dating",
    )
    validate_prompt_has_locale_instruction(rendered)
    assert "alcohol_dating" in rendered
    assert "graceful" in rendered.lower() or "redirect" in rendered.lower()


def test_register_rewrite_prompt_renders():
    rendered = REGISTER_REWRITE_PROMPT.format(
        natural_turn="That sounds great. Did you visit Naqsh-e Jahan Square?",
        cefr_level="A2",
        context="User: I went to Isfahan last weekend.",
    )
    validate_prompt_has_locale_instruction(rendered)
    assert "Naqsh-e Jahan" in rendered
    assert "rewritten" in rendered


def test_evaluation_generation_prompt_renders():
    dialogue = {
        "messages": [
            {"role": "user", "content": "I goed to Isfahan last weekend."},
            {"role": "assistant", "content": "That sounds great. Did you visit Naqsh-e Jahan Square?"},
        ]
    }
    rendered = EVALUATION_GENERATION_PROMPT.format(
        full_dialogue_json=json.dumps(dialogue, ensure_ascii=False),
        target_cefr="A2",
        topic="weekend plans in Isfahan",
        subtopics_block="- food\n- transport\n- sightseeing",
        model_role_name="Hassan",
        model_role_description="a friendly Isfahan tour guide",
        user_role_description="an A2 Iranian learner curious about her city",
    )
    validate_prompt_has_locale_instruction(rendered)
    assert "<think>" in rendered
    assert "EvaluationOutput" in rendered or "overall_cefr_estimate" in rendered
    assert "topic_adherence" in rendered
    assert "weekend plans in Isfahan" in rendered
    assert "Hassan" in rendered
    assert "friendly Isfahan tour guide" in rendered
    assert "A2 Iranian learner curious about her city" in rendered


# ---------------------------------------------------------------------------
# Negative tests — confirm guard catches a bad template
# ---------------------------------------------------------------------------


def test_validate_catches_template_with_stripped_locale_block():
    sabotaged = TOPIC_SEED_PROMPT.replace(LOCALE_INSTRUCTION_BLOCK, "")
    sabotaged = sabotaged.replace(LOCALE_INSTRUCTION_HEADER, "")
    # Also strip the placeholder so it does not pull the header back in
    # when formatted.
    sabotaged = sabotaged.replace(
        "{level_spec_with_locale_instruction}", "[level spec omitted]"
    )
    rendered = sabotaged.format(
        N=3,
        level="A1",
        categories_block="",
    )
    with pytest.raises(ValueError):
        validate_prompt_has_locale_instruction(rendered)
