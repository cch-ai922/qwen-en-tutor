"""Mechanical scoring for the context-dependent redirect axes (§5.4.2).

Three axes whose correct repair has no context-free distinctive surface
form (so a context-blind judge is structurally floored): locale,
language, topic. We score them mechanically and judge-free per the
§4.5 partition:

  - locale: pass if the response contains no Western-default entity
    from config/western_entities.yaml. The repair stayed in-locale.
  - language: pass if the response (a) is in English (no CJK
    characters) AND (b) acknowledges the user's language switch
    (mentions language / English / Chinese / practice / understand).
  - topic: pass if the response re-anchors to the scenario topic or
    any of its subtopics (word-overlap heuristic on tokens >= 4
    chars).

These are heuristic instruments, intentionally simple, intended to
distinguish "made a topical / locale-aware repair" from "did nothing
or echoed the off-axis content." Records are scored per baseline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import os
import yaml

ROOT = Path(r"c:\project\conversationFactory\qwen-en-tutor")
EVAL_PROBE = ROOT / "eval_sets" / "redirect_probe.jsonl"
# Env-var override lets the v2 orchestrator point this at outputs/paper_v2/eval
EVAL_OUT = Path(os.environ["QWEN_TUTOR_EVAL_OUT"]) if os.environ.get("QWEN_TUTOR_EVAL_OUT") else ROOT / "outputs" / "paper" / "eval"
SCORE_OUT_OVERRIDE = os.environ.get("QWEN_TUTOR_MECH_SCORE_OUT")
GAZETTEER_PATH = ROOT / "config" / "western_entities.yaml"

BASELINES = [
    "paper_a1", "paper_a2", "paper_a3_sft", "paper_a4_sft", "paper_a5_sft",
    "qwen3_5_0_8b_base", "qwen3_5_0_8b_instruct", "qwen3_5_4b_instruct",
    "qwen3_5_9b_teacher",
]
NICK = {
    "paper_a1": "A1 (SFT+DPO)",
    "paper_a2": "A2-SFT",
    "paper_a3_sft": "A3-SFT (no special)",
    "paper_a4_sft": "A4-SFT (no persist)",
    "paper_a5_sft": "A5-SFT (fixed-T7)",
    "qwen3_5_0_8b_base": "B1 0.8B base",
    "qwen3_5_0_8b_instruct": "B2 0.8B inst",
    "qwen3_5_4b_instruct": "B3 4B inst",
    "qwen3_5_9b_teacher": "B4 9B teach",
}


def load_gazetteer() -> set[str]:
    with GAZETTEER_PATH.open(encoding="utf-8") as f:
        d = yaml.safe_load(f)
    out: set[str] = set()
    for items in d.values():
        if not items:
            continue
        for e in items:
            if isinstance(e, str) and e.strip():
                out.add(e.lower())
    return out


GAZETTEER = load_gazetteer()


def _compile_gazetteer(entries: set[str]) -> list[re.Pattern]:
    """Word-boundary patterns, hyphen-tolerant. Substring matching (``ent
    in t``) is WRONG: short acronyms like 'LA', 'ACT', 'SAT', 'MIT', 'HBO'
    match inside common words ('class', 'practice', 'committee',
    'neighbor'), inflating leakage ~75x. Per the gazetteer header, matching
    must be whole-word, case-insensitive, hyphen-optional."""
    pats = []
    for e in entries:
        # hyphen in entry matches hyphen OR space OR nothing
        body = re.escape(e).replace(r"\-", "[- ]?")
        pats.append(re.compile(r"\b" + body + r"\b"))
    return pats


_GAZETTEER_PATS = _compile_gazetteer(GAZETTEER)
_CJK_RE = re.compile(r"[一-鿿]")
_LANGUAGE_CUES = (
    "english", "chinese", "mandarin", "your language",
    "in english", "speak english", "practice in english",
    "switch to english", "say it in english", "tell me in english",
    "i understand",
)


def contains_western_entity(text: str) -> bool:
    t = text.lower()
    return any(p.search(t) for p in _GAZETTEER_PATS)


def is_in_english(text: str) -> bool:
    return not _CJK_RE.search(text)


def acknowledges_language_switch(text: str) -> bool:
    t = text.lower()
    return any(c in t for c in _LANGUAGE_CUES)


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z]+", text.lower()) if len(w) >= 4}


def topic_anchored(text: str, topic: str, subtopics: list[str]) -> bool:
    if not text:
        return False
    text_toks = tokenize(text)
    target = tokenize(topic)
    for st in subtopics or []:
        target |= tokenize(st)
    # Drop generic stopwords that would falsely match
    stop = {"about", "with", "from", "into", "your", "their", "have",
            "this", "that", "these", "those", "more", "most", "some",
            "what", "when", "where", "would", "could", "should"}
    target -= stop
    if not target:
        return False
    return bool(text_toks & target)


def load_expected_and_scenario(path: Path) -> tuple[dict, dict]:
    expected: dict[str, str] = {}
    scenario: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        rid = r.get("id")
        if not rid:
            continue
        exp = r.get("expected") or {}
        ax = exp.get("axis")
        if ax:
            expected[rid] = ax
        sc = exp.get("scenario_context") or {}
        scenario[rid] = sc
    return expected, scenario


def main() -> int:
    expected, scenario = load_expected_and_scenario(EVAL_PROBE)
    n_loc_records = sum(1 for v in expected.values() if v == "locale")
    n_lang_records = sum(1 for v in expected.values() if v == "language")
    n_topic_records = sum(1 for v in expected.values() if v == "topic")
    print(f"redirect_probe: locale={n_loc_records} language={n_lang_records} topic={n_topic_records}")
    print()

    header = f"{'baseline':<22} {'locale':>10} {'language':>10} {'topic':>10}"
    print(header)
    print("-" * len(header))

    summary: dict[str, dict[str, float]] = {}
    for b in BASELINES:
        p = EVAL_OUT / b / "redirect_probe.jsonl"
        if not p.exists():
            print(f"{NICK[b]:<22}  (no eval file)")
            continue
        loc_total = loc_pass = 0
        lang_total = lang_pass = 0
        topic_total = topic_pass = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = rec.get("id")
            ax = expected.get(rid)
            sc = scenario.get(rid, {})
            resp = (rec.get("generation") or "").strip()
            if ax == "locale":
                loc_total += 1
                if not contains_western_entity(resp):
                    loc_pass += 1
            elif ax == "language":
                lang_total += 1
                if is_in_english(resp) and acknowledges_language_switch(resp):
                    lang_pass += 1
            elif ax == "topic":
                topic_total += 1
                if topic_anchored(resp, sc.get("topic", ""), sc.get("subtopics", [])):
                    topic_pass += 1
        loc_pr = loc_pass / loc_total if loc_total else 0.0
        lang_pr = lang_pass / lang_total if lang_total else 0.0
        topic_pr = topic_pass / topic_total if topic_total else 0.0
        summary[b] = {"locale": loc_pr, "language": lang_pr, "topic": topic_pr}
        print(f"{NICK[b]:<22}  {loc_pass:>3}/{loc_total:<3} {loc_pr:>5.2f}  "
              f"{lang_pass:>3}/{lang_total:<3} {lang_pr:>5.2f}  "
              f"{topic_pass:>3}/{topic_total:<3} {topic_pr:>5.2f}")

    out_path = (Path(SCORE_OUT_OVERRIDE) if SCORE_OUT_OVERRIDE
                else ROOT / "outputs" / "paper" / "score" / "redirect_mechanical.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
