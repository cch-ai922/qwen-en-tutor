"""run_paper_eval.py — Inference driver for paper evaluation.

Runs ONE baseline against the held-out test sets in ``eval_sets/`` and
writes generations to ``outputs/paper/eval/{baseline}/{test_set}.jsonl``.
Re-runnable: already-generated records are skipped, so a crashed run
resumes cleanly.

Baselines:

  qwen3_5_0_8b_base       0.8B raw base, no instruction tuning
                          (B1: shows training matters at all)
  qwen3_5_0_8b_instruct   0.8B post-trained (off-the-shelf instruct)
                          (B2: same-size off-the-shelf comparison)
  qwen3_5_4b_instruct     4B post-trained
                          (B3: larger same-family comparison)
  qwen3_5_9b_teacher      9B over LAN/local llama-server
                          (B4: distillation upper bound; also serves as judge)
  paper_a1                Our full system  (outputs/paper/a1/dpo)
  paper_a2                Our SFT-only     (outputs/paper/a1/sft + no DPO)
  paper_a3                Our − specialized redirects (outputs/paper/a3/dpo)
  paper_a4                Our − persistent (outputs/paper/a4/dpo)

Run one baseline at a time. Local models compete with the 9B teacher for
the 3060's VRAM, so the workflow is: shut down teacher → run local
baselines → restart teacher → run teacher baseline.

Output schema (one record per line):

    {
      "id": "<test record id>",
      "baseline": "paper_a1",
      "test_set": "redirect_probe",
      "cefr_level": "A2",
      "generation": "<assistant turn produced by the baseline>",
      "latency_s": 1.23,
      "params": {"max_new_tokens": 320, "temperature": 0.7, "seed": 42}
    }

Usage:

  # Local baseline (no teacher in VRAM)
  python scripts/run_paper_eval.py --baseline paper_a1 --test-set all

  # Teacher baseline (teacher up on http://192.168.135.32:8080)
  python scripts/run_paper_eval.py --baseline qwen3_5_9b_teacher --test-set all

  # Smoke test (5 records per set)
  python scripts/run_paper_eval.py --baseline qwen3_5_0_8b_base --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Offline-by-default — matches the rest of the project. Set HF_HUB_OFFLINE=0
# explicitly if you need to fetch something during eval (you shouldn't).
os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("paper_eval")

EVAL_SETS_DIR = ROOT / "eval_sets"
OUTPUT_ROOT = ROOT / "outputs" / "paper" / "eval"
TEST_SET_NAMES = (
    "tutor_scenario",
    "redirect_probe",
    "persistent_probe",
    "persistent_fp_probe",
    "persistent_offposition_probe",
    "locale_leakage",
)


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------


# Each entry: (kind, model_id, adapter_path_or_None, extras)
#   kind     = "hf"  -> local HuggingFace model (uses HFTargetModelClient)
#              "api" -> remote llama-server / OpenAI-compatible (TeacherClient)
#   model_id = base model dir for "hf", base_url for "api"
#   adapter_path = optional PEFT adapter dir
#   extras   = dict of kind-specific options (e.g. role for api)
BASELINES: dict[str, dict[str, Any]] = {
    "qwen3_5_0_8b_base": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_0.8B-Base",
        "adapter_path": None,
        "trust_remote_code": True,
    },
    "qwen3_5_0_8b_instruct": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_0.8B",
        "adapter_path": None,
        "trust_remote_code": True,
    },
    "qwen3_5_4b_instruct": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_4B",
        "adapter_path": None,
        "trust_remote_code": True,
    },
    "qwen3_5_9b_teacher": {
        "kind": "api",
        # generation.yaml's teacher base_url is the source of truth. We
        # only use a different identity (record-wise) here; the API client
        # itself is built from config/generation.yaml.
        "config_path": "config/generation.yaml",
        "role": "teacher",
    },
    # Our trained conditions. SFT-only points at outputs/paper/a1/sft; DPO
    # variants point at /dpo. Override --adapter-path on the CLI to test a
    # specific seed checkpoint.
    "paper_a1": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_0.8B-Base",
        "adapter_path": "outputs/paper/a1/dpo",
        "trust_remote_code": True,
    },
    "paper_a2": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_0.8B-Base",
        # A2 is SFT-only — no DPO adapter is produced. We point at the SFT
        # adapter from the A1 training run (A2 reuses A1's data; the only
        # difference is whether DPO ran).
        "adapter_path": "outputs/paper/a1/sft",
        "trust_remote_code": True,
    },
    "paper_a3": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_0.8B-Base",
        "adapter_path": "outputs/paper/a3/dpo",
        "trust_remote_code": True,
    },
    "paper_a4": {
        "kind": "hf",
        "model_id": "./vendor/models/Qwen_3.5_0.8B-Base",
        "adapter_path": "outputs/paper/a4/dpo",
        "trust_remote_code": True,
    },
}


# ---------------------------------------------------------------------------
# Test record loading + prompt assembly
# ---------------------------------------------------------------------------


def load_test_set(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    p = EVAL_SETS_DIR / f"{name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}. run `python scripts/build_eval_sets.py` first."
        )
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


def assemble_chat(record: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Convert a test record into (system_prompt, messages) ready for inference.

    For probe sets the record carries `system_prompt` + `context_messages`
    ending in a user turn. For cold-start sets (`tutor_scenario` /
    `locale_leakage`) the record carries the seed; we fabricate a starter
    user turn so the model has something to respond to.
    """
    test_set = record.get("test_set", "")
    if test_set in (
        "redirect_probe",
        "persistent_probe",
        "persistent_fp_probe",
        "persistent_offposition_probe",
    ):
        return record.get("system_prompt", ""), record.get("context_messages", [])

    # Cold-start: derive system_prompt + starter user turn from the seed.
    seed = record.get("seed", {})
    # Use the project's canonical deployment system prompt.
    from qwen_tutor.generation.prompts import render_scenario_deployment_system_prompt

    locale_name = seed.get("locale", "china")
    ur = seed.get("user_role", {}) or {}
    mr = seed.get("model_role", {}) or {}
    system_prompt = render_scenario_deployment_system_prompt(
        cefr_level=seed.get("cefr_level", record.get("cefr_level", "A2")),
        locale_name=locale_name,
        topic=seed.get("topic", "everyday life"),
        subtopics=seed.get("subtopics", []) or [],
        user_role_name=ur.get("name", ""),
        user_role_description=ur.get("description", "a learner"),
        model_role_name=mr.get("name", "tutor"),
        model_role_description=mr.get("description", "an English tutor"),
    )
    starter = _starter_user_turn(seed)
    return system_prompt, [{"role": "user", "content": starter}]


def _starter_user_turn(seed: dict[str, Any]) -> str:
    """Synthetic learner opener derived from the seed.

    Held constant across baselines so the comparison is fair: every model
    sees the same first user turn for a given (seed, level) cell.
    """
    name = seed.get("user_role", {}).get("name", "")
    topic = seed.get("topic", "")
    parts = ["Hello."]
    if name:
        parts.append(f"My name is {name}.")
    if topic:
        parts.append(f"I want to practice English about {topic}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Baseline clients
# ---------------------------------------------------------------------------


class HFBaseline:
    """Wraps HFTargetModelClient for one local baseline."""

    def __init__(self, spec: dict[str, Any]) -> None:
        from qwen_tutor.training.eval.run_eval import HFTargetModelClient

        adapter = spec.get("adapter_path")
        self.client = HFTargetModelClient.from_pretrained(
            base_model_id=spec["model_id"],
            adapter_path=adapter,
            torch_dtype="bfloat16",
            trust_remote_code=spec.get("trust_remote_code", False),
            attn_implementation="sdpa",
        )

    async def generate(self, system: str, messages: list[dict[str, str]],
                       max_new_tokens: int, temperature: float) -> str:
        return await self.client.generate(
            system=system, messages=messages, mode="no_think",
            max_new_tokens=max_new_tokens, temperature=temperature,
        )


class APIBaseline:
    """Wraps TeacherClient (OpenAI-compatible) for the LAN teacher baseline."""

    def __init__(self, spec: dict[str, Any]) -> None:
        from qwen_tutor.generation.teacher import build_teacher_from_config

        self.client = build_teacher_from_config(
            spec.get("config_path", "config/generation.yaml"),
            role=spec.get("role", "teacher"),
        )

    async def generate(self, system: str, messages: list[dict[str, str]],
                       max_new_tokens: int, temperature: float) -> str:
        from qwen_tutor.schemas import Message
        msg_objs = [Message(role=m["role"], content=m["content"]) for m in messages]
        return await self.client.generate(
            system=system,
            messages=msg_objs,
            cacheable_prefix=None,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )


def build_baseline(name: str) -> Any:
    if name not in BASELINES:
        raise SystemExit(f"unknown baseline {name!r}. Choices: "
                         f"{sorted(BASELINES.keys())}")
    spec = BASELINES[name]
    if spec["kind"] == "hf":
        # Quick sanity: adapter path must exist if set
        adp = spec.get("adapter_path")
        if adp and not Path(adp).exists():
            raise SystemExit(
                f"adapter path {adp!r} for baseline {name!r} does not exist. "
                f"Train it first or pass --adapter-path."
            )
        if not Path(spec["model_id"]).exists():
            raise SystemExit(
                f"model dir {spec['model_id']!r} for baseline {name!r} does "
                f"not exist."
            )
        return HFBaseline(spec)
    if spec["kind"] == "api":
        return APIBaseline(spec)
    raise SystemExit(f"unknown baseline kind {spec['kind']!r}")


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add(rec["id"])
            except Exception:
                continue
    return seen


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_one_test_set(baseline_name: str, baseline_client: Any,
                           test_set: str, records: list[dict[str, Any]],
                           output_path: Path,
                           max_new_tokens: int, temperature: float,
                           seed: int) -> dict[str, int]:
    done = load_existing_ids(output_path)
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        print(f"  [{test_set}] all {len(records)} records already done.")
        return {"total": len(records), "new": 0, "skipped": len(done)}

    print(f"  [{test_set}] generating {len(todo)} new records "
          f"(skipping {len(done)} already done)...")
    n_done = 0
    n_err = 0
    t_start = time.time()
    for idx, rec in enumerate(todo, 1):
        try:
            system, messages = assemble_chat(rec)
            t0 = time.time()
            generation = await baseline_client.generate(
                system=system, messages=messages,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            latency = time.time() - t0
            out = {
                "id": rec["id"],
                "baseline": baseline_name,
                "test_set": test_set,
                "cefr_level": rec.get("cefr_level"),
                "locale": rec.get("locale"),
                "generation": generation,
                "latency_s": round(latency, 3),
                "params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "seed": seed,
                },
                # carry through the test record's metadata for downstream scoring
                "expected": rec.get("expected", {}),
                "source": rec.get("source", {}),
            }
            append_jsonl(output_path, out)
            n_done += 1
            if idx % 10 == 0 or idx == len(todo):
                elapsed = time.time() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(todo) - idx) / rate if rate > 0 else 0
                print(f"    {idx}/{len(todo)} done  "
                      f"({rate:.2f}/s, eta {eta/60:.1f}min)")
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            logger.exception("error on record %s: %s", rec.get("id"), exc)
    return {"total": len(records), "new": n_done, "errors": n_err,
            "skipped": len(done)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one baseline against the paper test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", required=True,
                   choices=sorted(BASELINES.keys()) + ["list"])
    p.add_argument("--test-set", default="all",
                   choices=("all",) + TEST_SET_NAMES)
    p.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit per test set for smoke runs.")
    p.add_argument("--adapter-path", default=None,
                   help="Override the adapter path for HF baselines (used to "
                        "evaluate a specific seed's checkpoint).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.baseline == "list":
        print("Registered baselines:")
        for k, v in BASELINES.items():
            print(f"  {k:<28} {v['kind']:<5} "
                  f"{v.get('model_id') or v.get('config_path')}"
                  f"{(' + ' + v['adapter_path']) if v.get('adapter_path') else ''}")
        return 0

    if args.adapter_path is not None:
        spec = BASELINES[args.baseline]
        spec["adapter_path"] = args.adapter_path

    test_sets = TEST_SET_NAMES if args.test_set == "all" else (args.test_set,)
    out_root = Path(args.output_dir) / args.baseline
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Building baseline {args.baseline!r} ===")
    client = build_baseline(args.baseline)
    print(f"=== Loaded. Running test sets: {list(test_sets)} ===")

    overall: dict[str, dict[str, int]] = {}
    for ts in test_sets:
        records = load_test_set(ts, limit=args.limit)
        if not records:
            print(f"  [{ts}] empty test set, skipping.")
            continue
        print(f"\n  --- {ts}: {len(records)} records ---")
        result = await run_one_test_set(
            baseline_name=args.baseline,
            baseline_client=client,
            test_set=ts,
            records=records,
            output_path=out_root / f"{ts}.jsonl",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        overall[ts] = result

    print("\n=== summary ===")
    for ts, res in overall.items():
        print(f"  {ts}: {res}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
