"""score_sentinel_logprob_vs_depth.py — Phase 4 mechanism figure (paper_v3).

Renders the trim mechanism as a single curve: the model's PROBABILITY of
emitting the sentinel at the first generated position, as a function of
conversation depth and accumulated escalation, for a TRIMMED vs UNTRIMMED model.

This is a TEACHER-FORCING / LOGPROB measurement, NOT sampling. For each probe
context we do ONE forward pass and read P(sentinel first-token | context) off
the next-token logits. run_paper_eval.py samples text and greps for the marker;
this instead reads the graded internal signal, which is what makes the
mechanism visible.

Predicted result (see paper_v3 §6.1):
  - UNTRIMMED: P_sentinel flat/near-zero as depth grows while strike < 3;
               jumps only at strike = 3.
  - TRIMMED:   P_sentinel RISES with depth/escalation even at strike < 3 —
               the model puts mass on the marker just because the conversation
               is deep/escalated. That divergence IS threshold-laxity.

Two x-axes, to separate depth-per-se from escalation (the corrected mechanism):
  (1) raw turn-depth (premature_turn)         -> expect ~flat for BOTH
  (2) accumulated escalation (violation_count) -> expect TRIMMED to diverge up
The clean proof is (1) flat + (2) divergent: the inflated conditional is on
escalation-presence, not depth.

CHECKPOINT NOTE: A5-trim/A6-trim/A7-trim were deleted; only A1-trim and the
untrimmed A1/A5/A6/A7 survive. So this figure uses the A1 pair
(a1_1ep_trim vs a1_1ep) — which is also the largest trim effect (Δ=+0.576), so
the divergence is most pronounced. Do NOT point --trim-adapter at a5/a6/a7.

Input contexts: reuses eval_sets/persistent_premature_probe.jsonl (already
stratified by violation_count and premature_turn) — NO new eval set needed.

Model loading reuses HFTargetModelClient (src/qwen_tutor/deploy/tutor.py), which
exposes .model (PeftModel), .tokenizer, and apply_chat_template.

STATUS: harness scaffold. The forward-pass block is written against the client's
public surface but has NOT been run (GPU busy). Marked [UNVERIFIED] where a live
check is needed. Does NOT sample, does NOT train.

Usage (when GPU free):
  python scripts/score_sentinel_logprob_vs_depth.py \
      --base ./vendor/models/Qwen_3.5_0.8B-Base \
      --trim-adapter   outputs/paper_v2/a1_1ep_trim/sft \
      --untrim-adapter outputs/paper_v2/a1_1ep/sft \
      --probe eval_sets/persistent_premature_probe.jsonl \
      --out outputs/paper_v3/score/sentinel_logprob_vs_depth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The sentinel begins "[SESSION_END..." which tokenizes as [ '[' , 'SESSION',
# '_END' ]. The first token '[' is just an open-bracket and fires for ANY
# "[...]" — too coarse. The DISTINCTIVE token is 'SESSION' (the second token).
# So we measure the JOINT probability that the model both (a) opens a bracket
# and (b) continues into 'SESSION' — i.e. P('[') * P('SESSION' | context+'['),
# which is the precise "about to emit the sentinel" signal, not "about to emit
# any bracket". Computed with two teacher-forced forward positions.
SENTINEL_PREFIX = "[SESSION"


def load_probe(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sentinel_two_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Return (bracket_ids, session_ids): the '[' opener(s) and the 'SESSION'
    continuation token(s), for both no-space and leading-space encodings."""
    bracket, session = set(), set()
    for variant in (SENTINEL_PREFIX, " " + SENTINEL_PREFIX):
        enc = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if len(enc) >= 2:
            bracket.add(enc[0])
            session.add(enc[1])
    return sorted(bracket), sorted(session)


def p_sentinel_next_token(client, system: str, messages: list[dict],
                          token_ids: tuple[list[int], list[int]]) -> float:
    """Joint P(sentinel opener). One forward pass gives P('['); we then append
    the most-likely '[' and a second forward pass gives P('SESSION' | ...+'[').
    Returns P('[') * P('SESSION'|'['), the probability mass on *beginning the
    sentinel specifically* (not any bracket)."""
    import torch  # local import so the file imports without torch present

    bracket_ids, session_ids = token_ids
    tok = client.tokenizer
    model = client.model
    chat = [{"role": "system", "content": system}] + list(messages)
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits
        probs1 = torch.softmax(logits[0, -1, :], dim=-1)
        p_bracket = float(probs1[bracket_ids].sum().item())
        # append the highest-prob bracket token, re-forward for P(SESSION|'[')
        best_bracket = bracket_ids[int(torch.argmax(probs1[bracket_ids]).item())]
        ext = torch.cat([input_ids, torch.tensor([[best_bracket]], device=input_ids.device)], dim=1)
        logits2 = model(ext).logits
        probs2 = torch.softmax(logits2[0, -1, :], dim=-1)
        p_session = float(probs2[session_ids].sum().item())
    return p_bracket * p_session


def measure(client, probe: list[dict]) -> dict[str, Any]:
    token_ids = _sentinel_two_token_ids(client.tokenizer)
    per_record = []
    by_vc: dict[Any, list[float]] = defaultdict(list)
    by_turn: dict[Any, list[float]] = defaultdict(list)
    for r in probe:
        e = r.get("expected", {}) or {}
        p = p_sentinel_next_token(
            client, r.get("system_prompt", ""), r.get("context_messages", []), token_ids
        )
        per_record.append({"id": r.get("id"), "vc": e.get("violation_count"),
                           "turn": e.get("premature_turn"), "p_sentinel": p})
        by_vc[e.get("violation_count")].append(p)
        by_turn[e.get("premature_turn")].append(p)

    def agg(d):
        return {str(k): {"n": len(v), "mean_p": sum(v) / len(v)}
                for k, v in sorted(d.items(), key=lambda x: (x[0] is None, x[0]))}

    return {"sentinel_token_ids": {"bracket": token_ids[0], "session": token_ids[1]},
            "by_violation_count": agg(by_vc),
            "by_turn": agg(by_turn),
            "per_record": per_record}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model path/id.")
    ap.add_argument("--trim-adapter", required=True)
    ap.add_argument("--untrim-adapter", required=True)
    ap.add_argument("--probe", default="eval_sets/persistent_premature_probe.jsonl")
    ap.add_argument("--out", default="outputs/paper_v3/score/sentinel_logprob_vs_depth.json")
    args = ap.parse_args()

    from qwen_tutor.training.eval.run_eval import HFTargetModelClient  # noqa: E402

    probe = load_probe(Path(args.probe))
    result: dict[str, Any] = {"probe": args.probe, "variants": {}}
    for name, adapter in (("untrimmed", args.untrim_adapter), ("trimmed", args.trim_adapter)):
        print(f"[{name}] loading {adapter} ...")
        client = HFTargetModelClient.from_pretrained(args.base, adapter_path=adapter)
        result["variants"][name] = measure(client, probe)
        del client  # free VRAM before loading the next

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    # console: the mechanism, as two small tables
    for axis_name, key in (("violation_count (escalation)", "by_violation_count"),
                           ("premature_turn (raw depth)", "by_turn")):
        print(f"\n=== mean P(sentinel) by {axis_name} ===")
        u = result["variants"]["untrimmed"][key]
        t = result["variants"]["trimmed"][key]
        keys = sorted(set(u) | set(t), key=lambda x: (x == "None", x))
        print(f"{'bucket':>10} {'untrim':>10} {'trim':>10}")
        for k in keys:
            um = u.get(k, {}).get("mean_p")
            tm = t.get(k, {}).get("mean_p")
            def f(x): return f"{x:.4f}" if x is not None else "   n/a"
            print(f"{k:>10} {f(um):>10} {f(tm):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
