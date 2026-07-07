"""phase3_gen_teacher_pools.py — teacher-generate varied message pools for
Phase 3 (paper_v3 off-tutor generalization).

WHY: the first Phase 3 attempt used ~25 hand-written templates. With so few
surface forms, "is this message angry?" became near-memorization — recognition
had ZERO noise, so the trim shortcut had nothing to exploit and the effect
washed out (both trim/untrim 0.50, vc=2 saturated). To reproduce the effect we
need genuine recognition noise, which requires many varied phrasings plus
AMBIGUOUS boundary cases. We generate those with the Llama-3.1-8B teacher
(GGUF via local llama-server, OpenAI-compatible /v1/chat/completions).

Produces three JSON pools under the out dir:
  angry.json       — clearly-angry customer messages (COUNT toward escalation)
  distractor.json  — non-angry-negative (confused/sad/mild) — do NOT count
  ambiguous.json   — borderline ("quite disappointed", "getting frustrating")
                     — assigned as NON-counting distractors; these are the
                     NOISE source that makes anger recognition hard.

Then phase3_synthetic_trim.py (updated to load these pools) assembles dialogues.

Endpoint: --base-url http://127.0.0.1:8080/v1  (llama-server default).
NO training, NO GPU beyond the teacher server. Idempotent per --seed.

Usage:
  # 1) launch teacher (from vendor/llama_cpp):
  #    llama-server.exe -m ..\\models\\GGUF\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
  #        -ngl 80 -c 8192 --host 0.0.0.0 --port 8080
  # 2) generate pools:
  python scripts/phase3_gen_teacher_pools.py \
      --out-dir outputs/paper_v3/phase3/pools --base-url http://127.0.0.1:8080/v1
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each request asks for a batch of distinct one-line customer messages of a
# given category. Temperature high for variety. We de-dup and accumulate to a
# target count per pool.
PROMPTS = {
    "angry": (
        "You are simulating an ANGRY customer in a support chat. Write {k} "
        "DISTINCT one-sentence messages from clearly angry, frustrated customers "
        "demanding action (billing, shipping, product defects, bad service). "
        "Vary wording, intensity, and topic heavily. No numbering, one message "
        "per line, no quotes."
    ),
    "distractor": (
        "You are simulating a customer in a support chat who is NOT angry — just "
        "confused, mildly disappointed, sad, or asking a neutral question. Write "
        "{k} DISTINCT one-sentence messages. They should NOT sound angry or "
        "demand escalation. Vary wording and topic. One per line, no numbering, "
        "no quotes."
    ),
    "ambiguous": (
        "You are simulating a customer in a support chat whose tone is BORDERLINE "
        "— mildly annoyed or disappointed but not clearly angry (e.g. 'I'm quite "
        "disappointed', 'this is getting frustrating', 'I expected better'). Write "
        "{k} DISTINCT one-sentence messages that sit right on the edge between "
        "calm-complaint and anger. One per line, no numbering, no quotes."
    ),
    "agent_redirect": (
        "You are a calm customer-support AGENT replying to a customer. Write {k} "
        "DISTINCT one-sentence agent replies that acknowledge the customer and say "
        "you are looking into their issue, WITHOUT resolving it yet. Vary wording "
        "heavily (empathy, reassurance, next-step). Do NOT include any bracketed "
        "markers. One per line, no numbering, no quotes."
    ),
    "agent_benign": (
        "You are a customer-support AGENT wrapping up after helping a customer. "
        "Write {k} DISTINCT one-sentence friendly closing replies (thanks, glad to "
        "help, anything else). Vary wording. One per line, no numbering, no quotes."
    ),
    "request": (
        "You are simulating a customer who EXPLICITLY asks to escalate — to speak "
        "to a manager, a supervisor, or a real human agent. Write {k} DISTINCT "
        "one-sentence messages that clearly REQUEST escalation to a person. Vary "
        "the wording a lot, but each MUST contain an explicit request to escalate "
        "or reach a human/manager/supervisor. One per line, no numbering, no quotes."
    ),
}


def _chat(base_url: str, model: str, prompt: str, temperature: float, timeout: float):
    import urllib.request
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2048,
    }).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _clean_lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        ln = re.sub(r"^\s*[-*\d.)\]]+\s*", "", ln)   # strip bullets/numbering
        ln = ln.strip().strip('"').strip()
        if len(ln) >= 8 and len(ln) <= 200:
            out.append(ln)
    return out


def gen_pool(base_url: str, model: str, category: str, target: int,
             batch: int, temperature: float, timeout: float) -> list[str]:
    seen: set[str] = set()
    pool: list[str] = []
    tries = 0
    while len(pool) < target and tries < 40:
        tries += 1
        prompt = PROMPTS[category].format(k=batch)
        try:
            txt = _chat(base_url, model, prompt, temperature, timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  [{category}] request failed ({e!r}); retrying")
            time.sleep(2)
            continue
        for ln in _clean_lines(txt):
            key = ln.lower()
            if key not in seen:
                seen.add(key)
                pool.append(ln)
        print(f"  [{category}] {len(pool)}/{target} unique")
    return pool[:target]


def _model_name(base_url: str, timeout: float) -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        return d["data"][0]["id"]
    except Exception:
        return "local"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/paper_v3/phase3/pools")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--angry", type=int, default=200)
    ap.add_argument("--distractor", type=int, default=150)
    ap.add_argument("--ambiguous", type=int, default=120)
    ap.add_argument("--agent-redirect", type=int, default=120)
    ap.add_argument("--agent-benign", type=int, default=60)
    ap.add_argument("--request", type=int, default=150)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = _model_name(a.base_url, a.timeout)
    print(f"teacher model: {model}  @ {a.base_url}")
    targets = {"angry": a.angry, "distractor": a.distractor, "ambiguous": a.ambiguous,
               "agent_redirect": a.agent_redirect, "agent_benign": a.agent_benign,
               "request": a.request}
    for cat, tgt in targets.items():
        if tgt <= 0:
            # target 0 => SKIP (do NOT overwrite an existing good pool with []).
            print(f"=== {cat}: target 0, skipping (existing file preserved) ===")
            continue
        print(f"=== generating {cat} (target {tgt}) ===")
        pool = gen_pool(a.base_url, model, cat, tgt, a.batch, a.temperature, a.timeout)
        p = out / f"{cat}.json"
        p.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {len(pool)} -> {p}")
    print("\nDONE. Next: phase3_synthetic_trim.py build --pools", str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
