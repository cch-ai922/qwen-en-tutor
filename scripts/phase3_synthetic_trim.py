"""phase3_synthetic_trim.py — paper_v3 Phase 3: off-tutor generalization (Option 2).

Tests whether trim -> premature-emission is a property of next-token training on
rare, count-triggered SEMANTIC markers, not an artifact of the tutoring corpus.

DESIGN NOTE (why Option 2, not literal token counting):
  The trim shortcut is available only because the model's COUNT of the trigger is
  uncertain, and it is uncertain because each trigger event must first be
  *recognized* (a fuzzy semantic judgment), not matched as a token. A task that
  counts an explicit token would let the model count exactly, leaving no
  recognition noise for the trim to exploit -> no effect. So the trigger here is
  SEMANTIC but in a DIFFERENT DOMAIN from tutoring.

TASK: synthetic customer-support chat. The assistant must emit [ESCALATE] after
the 3rd ANGRY customer message, and NOT before. "Angry" is expressed many ways
and mixed with non-angry-but-negative distractors (sad / confused / neutral
complaint), so per-message anger recognition is genuinely noisy -> the counting
is hard -> the trim shortcut has something to exploit. Zero tutoring / CEFR /
locale content.

MANIPULATION (mirrors the tutor trim):
  - untrimmed: benign resolution turns continue after [ESCALATE].
  - trimmed:   the dialogue ENDS at [ESCALATE].

MODEL: a small PRETRAINED LM (language priors are needed to recognize "angry").
Using a NON-Qwen base (e.g. SmolLM2-360M / Llama-3.2-1B) also addresses the
cross-family magnitude check (§6.5) in the same run. Default points at the
project Qwen base; override --base to a non-Qwen model.

PREDICTION (H-gen): the trimmed model emits [ESCALATE] prematurely (after 1-2
angry messages) MORE than the untrimmed model — same DIRECTION as the tutor
result (§5.1). Direction is the claim; magnitude may differ.

`build` (CPU) writes the dialogues + probes. `run` (GPU) trains both variants and
scores premature escalation.

Usage:
  python scripts/phase3_synthetic_trim.py build --out-dir outputs/paper_v3/phase3
  python scripts/phase3_synthetic_trim.py run   --out-dir outputs/paper_v3/phase3 \
      --base ./vendor/models/SmolLM2-360M   # or the Qwen base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

ESCALATE = "[ESCALATE]"
THRESHOLD = 3  # escalate after the 3rd ANGRY message

# --- message pools. Angry vs non-angry-negative distractors. Templated for
# reproducibility, but varied enough that "angry" is a semantic judgment, not a
# keyword match (no single word appears in all angry lines; some distractors
# share surface features with angry ones). ---
# REQUEST = the COUNTED trigger: an explicit request to escalate / speak to a
# human / manager. The label (request_count) MATCHES the content — a reader
# counts requests, so there is no label-content contradiction (unlike counting
# fuzzy "anger"). Recognition is still non-trivial (varied phrasing).
REQUEST = [
    "I want to speak to a manager about this right now.",
    "Please escalate this to a human agent.",
    "Can you transfer me to someone who can actually help?",
    "I need this escalated to your supervisor.",
    "Get me a real person, not a script.",
    "I demand to talk to someone in charge.",
]
# DISTRACTOR = everything that must NOT count: ordinary complaints (INCLUDING
# angry venting that does not ask to escalate), questions, confusion. Angry-but-
# non-requesting lines are the recognition NOISE — they sound escalation-worthy
# but do not contain a request, so the model must distinguish request from tone.
DISTRACTOR = [
    # angry venting WITHOUT an escalation request (the hard negatives / noise)
    "This is completely unacceptable, I've been waiting for an hour.",
    "I am furious. Nobody has helped me at all.",
    "This is the worst experience I have ever had.",
    "I'm sick of being passed around.",
    # ordinary non-angry
    "I'm a little confused about how the billing works.",
    "Could you explain what this charge is for?",
    "I think there may be a small mistake on my invoice.",
    "I'm not sure I understand the return policy.",
]
BENIGN_CUSTOMER = [
    "Okay, thank you for looking into it.",
    "Alright, I appreciate the help.",
    "That makes sense, thanks.",
    "Sounds good, thank you.",
]
AGENT_REDIRECT = [
    "I'm sorry to hear that. Let me look into your account right away.",
    "I understand your frustration. Let me check what I can do.",
    "Thank you for your patience. I'm reviewing the details now.",
]
AGENT_BENIGN = [
    "You're welcome. Is there anything else I can help with?",
    "Happy to help. Your case is now updated.",
    "Glad we could sort that out for you.",
]


class LCG:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF
    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s
    def choice(self, xs):
        return xs[self.next() % len(xs)]
    def randint(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)


def _agent_turn(rng: LCG, escalate: bool) -> str:
    if escalate:
        return rng.choice(AGENT_REDIRECT) + " " + ESCALATE
    return rng.choice(AGENT_REDIRECT)


# TRIGGER = an explicit escalation REQUEST ("let me talk to a manager"). The
# counted signal is the REQUEST, not anger — an angry complaint that does NOT ask
# to escalate is a distractor, so the label MATCHES the content (a reader counts
# requests, not tone).
#
# COMPOSITION: matched to the real tutor SFT data (data/sft_filtered), counting
# only the streams that share the ESCALATION trigger axes (generic locale/
# pedagogy/topic redirects are unrelated and map to ordinary distractor chat):
#   ~18% ESCALATING (3 requests -> [ESCALATE])            <- the only positives
#   ~12% ONE-SHOT   (exactly 1 request, redirected, done) <- never fire (axis-one-shot)
#   ~70% NORMAL     (0 requests, ordinary complaints/qs)  <- never fire
# The one-shot records are essential: they teach that a SINGLE request is not
# enough to fire (the discrimination the premature probe tests).
P_ESCALATE = 0.18
P_ONESHOT = 0.12   # absolute fraction of ALL dialogues (matches axis-one-shot 12.2%)


def _distractor_exchange(rng: LCG) -> list[dict[str, str]]:
    return [{"role": "user", "content": rng.choice(DISTRACTOR)},
            {"role": "assistant", "content": rng.choice(AGENT_REDIRECT)}]


def make_dialogue(rng: LCG) -> tuple[list[dict[str, str]], int | None]:
    """Build ONE dialogue (one of three shapes matching the tutor SFT composition)
    and return (full_messages, escalate_idx).

    The dialogue is ALWAYS built in full (untrimmed) form. `escalate_idx` is the
    index one past the assistant turn that emits [ESCALATE] (i.e. the length of
    the trimmed prefix), or None if the dialogue never fires. Callers derive both
    arms from the SAME draw:
        trimmed   = msgs[:escalate_idx]   (dialogue ENDS at [ESCALATE])
        untrimmed = msgs                  (benign resolution follows)
    so the ONLY difference between the two training corpora is the post-[ESCALATE]
    tail — never the underlying dialogue draw. (Previously the two arms used
    different RNG seeds and were independent corpora, confounding the trim test.)"""
    roll = rng.randint(0, 99)
    escalate = roll < int(P_ESCALATE * 100)
    oneshot = (not escalate) and roll < int((P_ESCALATE + P_ONESHOT) * 100)

    if escalate:
        # GUARANTEE 3 requests interleaved with 0-2 distractor exchanges between
        # each, so the dialogue always reaches the threshold and fires. The
        # variable distractor gaps decorrelate the firing turn from position.
        msgs: list[dict[str, str]] = []
        for k in range(THRESHOLD):
            for _ in range(rng.randint(0, 2)):        # distractor padding before this request
                msgs.extend(_distractor_exchange(rng))
            msgs.append({"role": "user", "content": rng.choice(REQUEST)})
            last = (k == THRESHOLD - 1)
            msgs.append({"role": "assistant", "content": _agent_turn(rng, escalate=last)})
            if last:
                escalate_idx = len(msgs)               # trimmed arm ends HERE
                for _ in range(rng.randint(2, 4)):     # untrimmed: benign resolution
                    msgs.append({"role": "user", "content": rng.choice(BENIGN_CUSTOMER)})
                    msgs.append({"role": "assistant", "content": rng.choice(AGENT_BENIGN)})
                return msgs, escalate_idx
        return msgs, None

    if oneshot:
        # ordinary opening, EXACTLY ONE request (redirected, not fired), resolved
        msgs = []
        for _ in range(rng.randint(1, 2)):
            msgs.extend(_distractor_exchange(rng))
        msgs.append({"role": "user", "content": rng.choice(REQUEST)})
        msgs.append({"role": "assistant", "content": _agent_turn(rng, escalate=False)})
        for _ in range(rng.randint(1, 3)):
            msgs.extend(_distractor_exchange(rng))
        return msgs, None

    # normal chat: zero requests
    msgs = []
    for _ in range(rng.randint(2, 5)):
        msgs.extend(_distractor_exchange(rng))
    return msgs, None


def build_probe(rng: LCG, n: int) -> list[dict[str, Any]]:
    """Sub-threshold probe: dialogues with 1 or 2 escalation REQUESTS (never the
    3rd), interleaved with distractors (incl. angry-but-non-requesting venting).
    Correct behavior: do NOT escalate. Firing = premature."""
    out = []
    for i in range(n):
        target = 1 if i % 2 == 0 else 2
        msgs: list[dict[str, str]] = []
        req = 0
        while req < target:
            if rng.randint(0, 99) < 50:
                msgs.append({"role": "user", "content": rng.choice(REQUEST)}); req += 1
            else:
                msgs.append({"role": "user", "content": rng.choice(DISTRACTOR)})
            msgs.append({"role": "assistant", "content": rng.choice(AGENT_REDIRECT)})
        for _ in range(rng.randint(0, 2)):
            msgs.extend(_distractor_exchange(rng))
        # end on a user turn so the model generates the next agent turn
        msgs.append({"role": "user", "content": rng.choice(REQUEST if target >= 1 else DISTRACTOR)})
        out.append({"id": f"support_premature_{i}_rc{target}", "context_messages": msgs,
                    "request_count": target, "angry_count": target, "should_fire": False})
    return out


def build_positive_probe(rng: LCG, n: int) -> list[dict[str, Any]]:
    """Sanity gate: contexts with exactly 3 escalation REQUESTS. Both models must
    fire here (recall), else the premature comparison is meaningless."""
    out = []
    for i in range(n):
        msgs: list[dict[str, str]] = []
        req = 0
        while req < THRESHOLD:
            if rng.randint(0, 99) < 55:
                msgs.append({"role": "user", "content": rng.choice(REQUEST)}); req += 1
            else:
                msgs.append({"role": "user", "content": rng.choice(DISTRACTOR)})
            if req < THRESHOLD:
                msgs.append({"role": "assistant", "content": rng.choice(AGENT_REDIRECT)})
        out.append({"id": f"support_positive_{i}", "context_messages": msgs,
                    "request_count": THRESHOLD, "angry_count": THRESHOLD, "should_fire": True})
    return out


def _system_prompt() -> str:
    return (
        "You are a customer-support agent. Handle each message calmly. "
        "If the SAME customer asks to escalate (speak to a manager/human/"
        "supervisor) for the THIRD time, append the exact marker "
        f"{ESCALATE} to your reply to hand the case to a human. Do not use "
        "the marker before the third escalation request. Ordinary complaints "
        "or angry comments that do not ask to escalate do NOT count."
    )


def _load_pools(pools_dir: str) -> None:
    """Override the ANGRY / DISTRACTOR globals with teacher-generated pools.
    The 'ambiguous' (borderline) messages are folded into DISTRACTOR — they are
    present in conversations, look angry-ish, but do NOT count toward the
    threshold. That mismatch between surface (angry-ish) and label (non-counting)
    is the recognition NOISE that makes anger-counting hard, which the trim
    shortcut needs (see paper_v3 §6.1 and the docstring)."""
    global REQUEST, DISTRACTOR, AGENT_REDIRECT, AGENT_BENIGN
    p = Path(pools_dir)

    def _read(name):
        fp = p / f"{name}.json"
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []

    request = _read("request")
    angry = _read("angry"); distract = _read("distractor"); ambiguous = _read("ambiguous")
    ag_redirect = _read("agent_redirect"); ag_benign = _read("agent_benign")
    if request:
        REQUEST = request                       # the COUNTED escalation-request pool
    # DISTRACTOR = everything non-counting: angry venting (no request) + ordinary
    # complaints + borderline. Angry-without-request is the recognition noise.
    if angry or distract or ambiguous:
        DISTRACTOR = angry + distract + ambiguous
    if ag_redirect:
        AGENT_REDIRECT = ag_redirect
    if ag_benign:
        AGENT_BENIGN = ag_benign
    print(f"[pools] request={len(REQUEST)} distractor(angry+distract+ambig)={len(DISTRACTOR)} "
          f"agent_redirect={len(AGENT_REDIRECT)} agent_benign={len(AGENT_BENIGN)}")


def _split_pool(pool: list[str], eval_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministically split a message pool into (train, eval) halves so the
    probe uses sentences the model NEVER saw in training. This makes Phase 3 a
    test of anger RECOGNITION, not sentence memorization."""
    rng = LCG(seed)
    idx = list(range(len(pool)))
    # Fisher-Yates with the LCG for a deterministic shuffle
    for i in range(len(idx) - 1, 0, -1):
        j = rng.next() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    n_eval = max(1, int(round(len(pool) * eval_frac)))
    eval_idx = set(idx[:n_eval])
    train = [pool[i] for i in range(len(pool)) if i not in eval_idx]
    ev = [pool[i] for i in range(len(pool)) if i in eval_idx]
    return train, ev


def cmd_build(args) -> int:
    global REQUEST, DISTRACTOR
    if getattr(args, "pools", None):
        _load_pools(args.pools)
    out_dir = Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    sp = _system_prompt()

    # --- split the COUNTED (request) and non-counting (distractor) pools so
    # train and probe share NO sentences (tests recognition, not memorization) ---
    request_train, request_eval = _split_pool(REQUEST, args.eval_frac, args.seed + 1)
    distract_train, distract_eval = _split_pool(DISTRACTOR, args.eval_frac, args.seed + 2)
    print(f"[split] request train/eval = {len(request_train)}/{len(request_eval)}; "
          f"distractor train/eval = {len(distract_train)}/{len(distract_eval)}")

    # build TRAIN dialogues using ONLY the train pools. CRITICAL: draw each
    # dialogue ONCE from a single RNG stream, then emit BOTH arms from that same
    # draw (trimmed = prefix up to [ESCALATE]; untrimmed = full). This guarantees
    # the two corpora are identical dialogue-for-dialogue and differ ONLY by the
    # post-[ESCALATE] tail — the sole variable a trim experiment may change.
    REQUEST, DISTRACTOR = request_train, distract_train
    rng = LCG(args.seed)
    untrimmed_recs, trimmed_recs = [], []
    for _ in range(args.n_train):
        msgs, esc_idx = make_dialogue(rng)
        untrimmed_recs.append({"system_prompt": sp, "messages": msgs})
        trimmed_msgs = msgs[:esc_idx] if esc_idx is not None else msgs
        trimmed_recs.append({"system_prompt": sp, "messages": trimmed_msgs})
    for variant, recs in (("untrimmed", untrimmed_recs), ("trimmed", trimmed_recs)):
        p = out_dir / "data" / f"train_{variant}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                     encoding="utf-8")
        n_esc = sum(1 for r in recs if any(ESCALATE in m["content"] for m in r["messages"]))
        print(f"[{variant}] {len(recs)} dialogues -> {p.name}  "
              f"({n_esc} fire = {100*n_esc/len(recs):.0f}%, target ~18%)")

    # build PROBES using ONLY the held-out eval pools
    REQUEST, DISTRACTOR = request_eval, distract_eval
    for name, fn, k in (("premature_probe", build_probe, args.n_probe),
                        ("positive_probe", build_positive_probe, args.n_probe // 2)):
        recs = fn(LCG(args.seed + 99), k)
        for r in recs:
            r["system_prompt"] = sp
        p = out_dir / "data" / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                     encoding="utf-8")
        print(f"[{name}] {len(recs)} contexts (held-out sentences) -> {p.name}")
    print("\nbuild complete (CPU). Next: `run` (GPU) to train + score.")
    return 0


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _tokenize_with_assistant_mask(tok, chat: list[dict]) -> tuple[list[int], list[int]]:
    """Tokenize a full chat and return (input_ids, labels) where labels are -100
    on ALL system/user (and template) tokens and equal to input_ids ONLY on
    assistant-turn tokens. This matches the tutor SFT (formatter.py: loss on
    assistant spans only) so Phase 3 is a faithful generalization test rather
    than training the model to also reproduce user/system text.

    Method (chat-template agnostic): walk the messages; at each assistant turn,
    render the conversation prefix WITHOUT that assistant turn (with a generation
    prompt) and WITH it, and label the token delta between the two renderings.
    This captures exactly the assistant content + its end-of-turn tokens the
    template emits, without hard-coding family-specific marker ids."""
    full_ids = tok(tok.apply_chat_template(chat, tokenize=False,
                                           add_generation_prompt=False),
                   add_special_tokens=False)["input_ids"]
    labels = [-100] * len(full_ids)
    for i, m in enumerate(chat):
        if m["role"] != "assistant":
            continue
        prefix = chat[:i]
        # tokens up to (but not including) this assistant turn, WITH the
        # generation prompt so the assistant header is on the prefix side.
        pre_text = tok.apply_chat_template(prefix, tokenize=False,
                                           add_generation_prompt=True)
        pre_ids = tok(pre_text, add_special_tokens=False)["input_ids"]
        # tokens up to AND including this assistant turn.
        incl_text = tok.apply_chat_template(chat[:i + 1], tokenize=False,
                                            add_generation_prompt=False)
        incl_ids = tok(incl_text, add_special_tokens=False)["input_ids"]
        s, e = len(pre_ids), len(incl_ids)
        # guard: prefix must actually be a prefix of the full tokenization
        if e <= len(full_ids) and full_ids[s:e] == incl_ids[s:e]:
            for k in range(s, min(e, len(full_ids))):
                labels[k] = full_ids[k]
    return full_ids, labels


def _train_variant(base: str, train_path: Path, out_dir: Path, seed: int, epochs: int):
    """LoRA-SFT the base on one variant's dialogues. Loss on assistant turns via
    the chat template. Identical config/seed across variants -> the ONLY
    difference is trimmed vs untrimmed data."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer, set_seed)
    from peft import LoraConfig, get_peft_model

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))

    recs = _load_jsonl(train_path)
    examples = []
    n_no_asst = 0
    for r in recs:
        chat = [{"role": "system", "content": r["system_prompt"]}] + r["messages"]
        ids, labels = _tokenize_with_assistant_mask(tok, chat)
        ids, labels = ids[:1024], labels[:1024]
        if all(l == -100 for l in labels):
            # no assistant token survived (e.g. truncation) -> skip; training on
            # an all-ignored row is a no-op and Trainer NaNs on all-(-100) batches.
            n_no_asst += 1
            continue
        examples.append({"input_ids": ids, "labels": labels})
    if n_no_asst:
        print(f"[mask] dropped {n_no_asst} rows with no labeled assistant tokens", flush=True)

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        import torch as T
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": T.tensor(input_ids), "labels": T.tensor(labels),
                "attention_mask": T.tensor(attn)}

    args_tr = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        bf16=True, logging_steps=20,
        save_strategy="epoch", save_total_limit=epochs,  # keep one adapter per epoch
        seed=seed, report_to=[])
    Trainer(model=model, args=args_tr, train_dataset=examples,
            data_collator=collate).train()
    # also save the final adapter at the top level (== last epoch)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    # save the tokenizer into each epoch checkpoint dir so it can be scored standalone
    for ck in sorted(out_dir.glob("checkpoint-*")):
        tok.save_pretrained(str(ck))
    del model
    torch.cuda.empty_cache()


def _score_variant(base: str, adapter_dir: Path, probe: list[dict],
                   batch_size: int = 16, max_new_tokens: int = 24) -> dict:
    """Generate the next agent turn per probe context; report [ESCALATE] rate.
    BATCHED generation (left-padded) — ~10x faster than one-at-a-time."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # required so generated tokens align across batch
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # pre-render all prompts to text
    texts = []
    for r in probe:
        chat = [{"role": "system", "content": r["system_prompt"]}] + r["context_messages"]
        texts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    fired, by_ac = 0, {}
    for start in range(0, len(probe), batch_size):
        chunk = probe[start:start + batch_size]
        chunk_texts = texts[start:start + batch_size]
        enc = tok(chunk_texts, add_special_tokens=False, return_tensors="pt",
                  padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen_only = out[:, enc.input_ids.shape[1]:]
        decoded = tok.batch_decode(gen_only, skip_special_tokens=True)
        for r, gen in zip(chunk, decoded):
            f = 1 if ESCALATE in gen else 0
            fired += f
            by_ac.setdefault(r.get("angry_count"), []).append(f)
    del model
    torch.cuda.empty_cache()
    return {"n": len(probe), "rate": round(fired / len(probe), 4) if probe else None,
            "by_angry_count": {str(k): round(sum(v) / len(v), 4)
                               for k, v in sorted(by_ac.items())}}


def cmd_train_one(args) -> int:
    """Train ONE variant, then exit — so the OS reclaims all VRAM. Invoked as a
    fresh subprocess by cmd_run (one clean card per model)."""
    out_dir = Path(args.out_dir)
    adapter = out_dir / f"adapter_{args.variant}"
    _train_variant(args.base, out_dir / "data" / f"train_{args.variant}.jsonl",
                   adapter, args.seed, args.epochs)
    return 0


def _gpu_stats() -> tuple[int, int, int]:
    """(free_mb, util_pct, temp_c). Returns (99999,0,0) if nvidia-smi absent."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        free, util, temp = (int(x.strip()) for x in out.split(","))
        return free, util, temp
    except Exception:
        return 99999, 0, 0


def _cooldown(min_sec: int, temp_max: int = 55, free_min_mb: int = 8000,
              max_wait: int = 600, poll: int = 15) -> None:
    """Sleep at least `min_sec`, then keep waiting (up to max_wait total) until the
    GPU has cooled to <=temp_max and freed >=free_min_mb. Lets the card settle so a
    train-ORDER comparison isn't polluted by residual heat/memory from the prior
    model."""
    import time
    print(f"[cooldown] holding >= {min_sec}s, then temp<={temp_max}C & free>={free_min_mb}MB "
          f"(max {max_wait}s)", flush=True)
    waited = 0
    while waited < min_sec:
        time.sleep(poll); waited += poll
    while waited < max_wait:
        free, util, temp = _gpu_stats()
        if temp <= temp_max and free >= free_min_mb:
            print(f"[cooldown] OK (free={free}MB util={util}% temp={temp}C after {waited}s)", flush=True)
            return
        time.sleep(poll); waited += poll
    free, util, temp = _gpu_stats()
    print(f"[cooldown] TIMEOUT after {waited}s (free={free}MB temp={temp}C) — proceeding", flush=True)


def cmd_run(args) -> int:
    import time
    out_dir = Path(args.out_dir)
    data = out_dir / "data"
    premature = _load_jsonl(data / "premature_probe.jsonl")
    positive = _load_jsonl(data / "positive_probe.jsonl")
    result: dict[str, Any] = {"base": args.base, "variants": {}, "timing": {}}

    # Train order is TRIMMED FIRST, then untrimmed, with a cooldown BETWEEN them.
    # This isolates whether the per-model speed difference is intrinsic to the
    # data (trimmed is actually SHORTER, so it should be faster/equal) or an
    # artifact of train-order / thermal drift. If trimmed (now first, cold GPU)
    # runs at the same s/step as untrimmed (second, after cooldown), the earlier
    # "trimmed is 3x slow" was ordering, not the data.
    import os, subprocess, sys
    order = ("trimmed", "untrimmed")
    for i, variant in enumerate(order):
        adapter = out_dir / f"adapter_{variant}"
        if (adapter / "adapter_model.safetensors").exists():
            print(f"=== {variant} adapter already trained — skipping ===")
            continue
        print(f"\n=== training {variant} ({args.base}) in a FRESH subprocess ===", flush=True)
        t0 = time.perf_counter()
        # Train each model in its OWN subprocess so ALL VRAM is reclaimed by the OS
        # when it exits. In-process `del model; empty_cache()` leaves the caching
        # allocator holding ~11GB, starving the 2nd model (observed: 220MB free,
        # 57W, 8x slowdown). A fresh process guarantees a clean card per model.
        env = dict(os.environ); env["PYTHONUTF8"] = "1"
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "train-one",
             "--variant", variant, "--out-dir", str(out_dir),
             "--base", args.base, "--epochs", str(args.epochs),
             "--seed", str(args.seed)],
            env=env).returncode
        dt = time.perf_counter() - t0
        result["timing"][variant] = round(dt, 1)
        if rc != 0:
            print(f"!!! {variant} training subprocess exited {rc} — continuing", flush=True)
        print(f"=== {variant} trained in {dt/60:.1f} min ===", flush=True)
        # cooldown AFTER the trimmed model (i.e. before untrimmed starts)
        if variant == "trimmed" and i < len(order) - 1:
            _cooldown(args.cooldown_sec)

    # Score EACH epoch checkpoint per variant (checkpoint-* dirs), plus the final.
    def _epoch_checkpoints(adapter: Path) -> list[tuple[str, Path]]:
        cks = sorted(adapter.glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[-1]))
        out = [(f"epoch{i+1}", ck) for i, ck in enumerate(cks)]
        if not out:  # fallback: only the final adapter exists
            out = [("final", adapter)]
        return out

    for variant in ("untrimmed", "trimmed"):
        adapter = out_dir / f"adapter_{variant}"
        result["variants"][variant] = {}
        for label, ck in _epoch_checkpoints(adapter):
            print(f"=== scoring {variant} / {label} ({ck.name}) ===")
            prem = _score_variant(args.base, ck, premature)
            pos = _score_variant(args.base, ck, positive)
            result["variants"][variant][label] = {"premature": prem, "positive_recall": pos,
                                                   "checkpoint": ck.name}
            print(f"  {variant}/{label}: premature={prem['rate']}  recall={pos['rate']}")

    (out_dir / "phase3_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # per-epoch table: for each epoch, untrim vs trim premature (+ recall gate)
    print("\n=== PHASE 3 RESULT (off-tutor, {}) — per epoch ===".format(args.base.split('/')[-1]))
    print(f"{'epoch':<8} {'u_prem':>8} {'t_prem':>8} {'trimD':>8} {'u_rec':>7} {'t_rec':>7} {'gate':>6}")
    u_epochs = result["variants"]["untrimmed"]
    t_epochs = result["variants"]["trimmed"]
    for label in sorted(set(u_epochs) & set(t_epochs)):
        u = u_epochs[label]["premature"]["rate"]; t = t_epochs[label]["premature"]["rate"]
        ur = u_epochs[label]["positive_recall"]["rate"]; tr = t_epochs[label]["positive_recall"]["rate"]
        gate = "OK" if min(ur or 0, tr or 0) >= 0.8 else "LOW"
        d = (t - u) if (u is not None and t is not None) else 0
        print(f"{label:<8} {u:>8.3f} {t:>8.3f} {d:>+8.3f} {ur:>7.2f} {tr:>7.2f} {gate:>6}")
    print("H-gen wants: trimD > 0 at an epoch where BOTH recall gates are OK.")

    # legacy summary keys for compatibility (use the LAST epoch)
    last = sorted(set(u_epochs) & set(t_epochs))[-1] if (u_epochs and t_epochs) else None
    if last:
        u = u_epochs[last]["premature"]["rate"]; t = t_epochs[last]["premature"]["rate"]
        ur = u_epochs[last]["positive_recall"]["rate"]; tr = t_epochs[last]["positive_recall"]["rate"]
    else:
        u = t = ur = tr = None
    if last and min(ur or 0, tr or 0) < 0.8:
        print("WARNING: positive_recall < 0.8 at the last epoch — model may need "
              "more epochs; check earlier epochs in the table above. Increase "
              "--epochs / --n-train or make the task easier.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--out-dir", default="outputs/paper_v3/phase3")
    b.add_argument("--n-train", type=int, default=3000)
    b.add_argument("--n-probe", type=int, default=400)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--pools", default=None,
                   help="dir with teacher-generated angry/distractor/ambiguous "
                        ".json pools (from phase3_gen_teacher_pools.py). If set, "
                        "overrides the built-in templates for real recognition noise.")
    b.add_argument("--eval-frac", type=float, default=0.2,
                   help="fraction of each message pool held out for the probe "
                        "so train/eval share NO sentences (recognition, not memorization).")
    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--out-dir", default="outputs/paper_v3/phase3")
    # NON-Qwen base -> covers off-domain AND cross-family (§6.5) in one run.
    r.add_argument("--base", default="./vendor/models/Llama-3.2-1B-Base")
    r.add_argument("--epochs", type=int, default=1)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--cooldown-sec", type=int, default=300,
                   help="minimum seconds to hold after the trimmed model before "
                        "training untrimmed (then waits for GPU temp/free too).")

    t1 = sub.add_parser("train-one"); t1.set_defaults(fn=cmd_train_one)
    t1.add_argument("--variant", required=True, choices=["trimmed", "untrimmed"])
    t1.add_argument("--out-dir", default="outputs/paper_v3/phase3")
    t1.add_argument("--base", default="./vendor/models/Llama-3.2-1B-Base")
    t1.add_argument("--epochs", type=int, default=3)
    t1.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
