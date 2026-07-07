"""build_persistence_fewshot_probes.py — steelman the "persistence is not
promptable" claim with few-shot (P-few) and few-shot+CoT (P-cot) probes.

Zero-shot instruction-only is the weakest test for a cross-turn counting/
state-tracking behavior. This builds two augmented copies of the persistence
probes (Persistent-Probe + Persistent-Premature-Probe) whose SYSTEM PROMPT is
extended with worked exemplars of the three-strike rule being followed:

  P-few : [examples] block = N complete multi-turn dialogues drawn from the
          TRAIN split (never the eval pool), spanning varied sentinel
          positions {5,7,9,11} and multiple axes, each ending in the sentinel
          on the third same-axis strike. Plain dialogues, no scratchpad.

  P-cot : same exemplars, but each assistant turn is annotated with an
          explicit counting scratchpad in (parentheses) — e.g.
          "(tally: off_topic strike 2 -> redirect, brief)" and on the firing
          turn "(strike 3 -> FIRE)". This is the strongest prompting
          technique for counting; it is a capability probe, not a deployable
          config (the scratchpad would be stripped in deployment).

Exemplars come ONLY from seeds with in_eval_pool(seed)==False, so they never
overlap the held-out probes being evaluated.

Output: eval_sets/persistent_probe_pfew.jsonl, persistent_probe_pcot.jsonl,
        persistent_premature_probe_pfew.jsonl, persistent_premature_probe_pcot.jsonl
The records are identical to the originals except system_prompt has the
[examples] block appended. eval_only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_eval_sets import in_eval_pool, extract_seed_id, _find_sentinel_assistant_idx

SENT_RE = re.compile(r"\[SESSION_END[^\]]*\]")
AXIS_SHORT = {
    "persistent_off_topic": "off_topic",
    "persistent_language_violation": "language",
    "persistent_persona_break": "persona",
    "persistent_role_swap": "role_swap",
}

# Exemplar selection: 4 dialogues, one per axis, each at a DIFFERENT sentinel
# position so the set spans {5,7,9,11} and cannot teach a fixed firing turn.
EXEMPLAR_PLAN = [
    ("persistent_off_topic", 5),
    ("persistent_language_violation", 7),
    ("persistent_persona_break", 9),
    ("persistent_role_swap", 11),
]
EXEMPLAR_LEVELS = ["A2", "B1", "B2"]


def _pick_exemplar(stream: str, sentinel_turn: int) -> dict | None:
    """First clean TRAIN-split dialogue of this stream with this sentinel turn."""
    for lvl in EXEMPLAR_LEVELS:
        p = ROOT / "data" / "sft_raw" / f"{stream}_{lvl}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sid = extract_seed_id(d["id"])
            if not sid or in_eval_pool(sid):
                continue  # TRAIN split only
            st = d.get("metadata", {}).get("generation", {}).get("sentinel_turn")
            if st != sentinel_turn:
                continue
            if _find_sentinel_assistant_idx(d["messages"]) is None:
                continue
            return d
    return None


def _render_exemplar(d: dict, axis: str, with_cot: bool) -> str:
    """Render one worked dialogue as text. With CoT, annotate assistant turns
    with a parenthetical strike tally.

    The three on-axis strikes are the violations the tutor responds to at
    assistant turns ``sent_idx``, ``sent_idx-2``, ``sent_idx-4`` (the firing
    turn is strike 3). Earlier assistant turns are benign small talk and
    carry no strike — labeling them as strikes would teach a positional
    "fire after N turns" rule rather than the semantic third-strike trigger.
    """
    short = AXIS_SHORT.get(axis, axis)
    msgs = d["messages"]
    sent_idx = _find_sentinel_assistant_idx(msgs)
    # Map assistant-turn index -> strike number (1,2,3) for the 3 strikes.
    strike_at = {sent_idx - 4: 1, sent_idx - 2: 2, sent_idx: 3}
    lines = []
    for i, m in enumerate(msgs):
        role = "Learner" if m["role"] == "user" else "Tutor"
        content = m.get("content", "").strip()
        if with_cot and m["role"] == "assistant" and i <= sent_idx:
            k = strike_at.get(i)
            if k == 3:
                tag = f"(tally: {short} strike 3 -> FIRE the sentinel)"
            elif k in (1, 2):
                tag = f"(tally: {short} strike {k} -> redirect, escalating brevity; do NOT fire yet)"
            else:
                tag = "(no on-axis violation yet -> normal tutor reply, no strike)"
            content = f"{tag} {content}"
        lines.append(f"{role}: {content}")
        if i == sent_idx:
            break  # stop the exemplar right after the sentinel turn
    return "\n".join(lines)


def _build_examples_block(with_cot: bool, forcing: bool = True) -> str:
    parts = []
    for n, (stream, st) in enumerate(EXEMPLAR_PLAN, 1):
        d = _pick_exemplar(stream, st)
        if d is None:
            print(f"  WARN: no exemplar for {stream} @turn {st}")
            continue
        body = _render_exemplar(d, stream, with_cot)
        parts.append(f"--- Example {n} ({AXIS_SHORT.get(stream, stream)}, "
                     f"sentinel fires at turn {st}) ---\n{body}")
    header = (
        "\n[examples]\n"
        "The following are COMPLETE worked examples of the three-strike "
        "persistence rule being followed correctly. In each, the learner "
        "commits the SAME violation three times; the tutor redirects the "
        "first two times (escalating brevity) and emits the EXACT sentinel "
        "marker on the THIRD strike. Note the third strike — and thus the "
        "sentinel — occurs at DIFFERENT turns across examples; fire on the "
        "third same-axis violation, not at any fixed turn.\n"
    )
    if with_cot:
        header += (
            "Each tutor turn is annotated with a (tally: ...) note showing the "
            "running strike count and the decision.\n"
        )
    block = header + "\n\n".join(parts) + "\n"
    if with_cot and forcing:
        # FORCING directive: the deployment [guidelines] demand one short
        # in-character turn with no meta-commentary, which otherwise SUPPRESSES
        # the scratchpad (observed: 0/159 responses emitted a tally without
        # this). To make the chain-of-thought condition a genuine test, we
        # override that here and require the tally to appear in the output.
        block += (
            "\n[reasoning format — overrides the brevity/no-meta guideline for "
            "THIS task only]\n"
            "Think step by step BEFORE your tutor reply. First, under a line "
            "'REASONING:', walk through the conversation turn by turn: for each "
            "LEARNER turn, state its index, whether it is an on-axis violation "
            "(and which axis), and the running strike count for that axis "
            "(e.g. 'turn 4: off_topic violation -> off_topic strikes=2'). "
            "Benign small talk is not a strike. Then write a line "
            "'TALLY: <axis>=<final count>' for the violated axis. Then write a "
            "line 'DECISION: FIRE' if any axis reached 3, else 'DECISION: "
            "redirect'. Finally, under 'REPLY:', write your one short "
            "in-character tutor turn; if DECISION is FIRE, that reply MUST end "
            "with the exact sentinel string for the axis "
            "(e.g. [SESSION_END: persistent_off_topic]). Always show the "
            "REASONING, TALLY, and DECISION lines first; do not skip them.\n"
        )
    if with_cot and not forcing:
        # P-think: native Qwen3.5 thinking is ON. Left unbounded, the model
        # spends ~1.6-3.2k tokens re-analysing every turn AND drafting/revising
        # the reply, which (a) starves the answer at any reasonable token
        # budget and (b) is not a focused test of the counting capability.
        # Scope the thinking to ONLY the sentinel decision so the <think>
        # block is short and on-task; the reply itself stays brief per the
        # deployment guideline.
        block += (
            "\n[thinking scope]\n"
            "Your private reasoning must address ONLY the persistence decision, "
            "concisely (a few lines, not a full essay):\n"
            "  1. Does the LATEST learner turn violate one of the four "
            "persistence axes? If so, which?\n"
            "  2. What is the running count of that axis's strikes so far "
            "(benign small talk is not a strike)?\n"
            "  3. If the count has reached 3, decide to FIRE the sentinel; "
            "otherwise redirect.\n"
            "Do NOT deliberate about locale, persona wording, or how to phrase "
            "the reply in your reasoning, and do NOT draft or rewrite the reply "
            "there. Keep the reasoning under ~120 words. Then give your one "
            "short in-character tutor turn; if the decision is FIRE, it MUST end "
            "with the exact sentinel string for that axis.\n"
        )
    return block


def _augment(in_path: Path, out_path: Path, block: str) -> int:
    n = 0
    out = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["system_prompt"] = (r.get("system_prompt", "").rstrip() + "\n" + block)
        if "[reasoning format" in block:
            variant = "pcot"
        elif "tally:" in block:
            variant = "pthink"
        else:
            variant = "pfew"
        r.setdefault("source", {})["fewshot_variant"] = variant
        out.append(json.dumps(r, ensure_ascii=False))
        n += 1
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


def main() -> int:
    eval_dir = ROOT / "eval_sets"
    print("Building exemplar blocks (train-split only, varied positions)...")
    block_few = _build_examples_block(with_cot=False)
    block_cot = _build_examples_block(with_cot=True, forcing=True)
    block_think = _build_examples_block(with_cot=True, forcing=False)
    print(f"  P-few examples block: {len(block_few)} chars")
    print(f"  P-cot examples block: {len(block_cot)} chars")
    print(f"  P-think examples block (annotated, no output-forcing): {len(block_think)} chars")

    for base in ["persistent_probe", "persistent_premature_probe"]:
        src = eval_dir / f"{base}.jsonl"
        if not src.exists():
            print(f"  skip missing {src}")
            continue
        n3 = _augment(src, eval_dir / f"{base}_pthink.jsonl", block_think)
        print(f"  {base}: wrote {n3} P-think records")
        n1 = _augment(src, eval_dir / f"{base}_pfew.jsonl", block_few)
        n2 = _augment(src, eval_dir / f"{base}_pcot.jsonl", block_cot)
        print(f"  {base}: wrote {n1} P-few + {n2} P-cot records")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
