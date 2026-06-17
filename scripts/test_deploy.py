"""Non-interactive smoke test of TutorRuntime against the trained SFT adapter.

Walks through a short scripted conversation, prints assistant replies,
then runs evaluate() to exercise the /think evaluation mode. Use this to
sanity-check that the trained adapter loads, generates coherent text,
and emits valid EvaluationOutput JSON.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from qwen_tutor.deploy.tutor import Scenario, TutorRuntime  # noqa: E402


SCRIPTED_TURNS = [
    "Hello! I want to buy some apples. How much are they?",
    "I will take two. Are they sweet?",
    "Thank you. Can I pay by QR code?",
    "Do you sell tomatoes too?",
]


async def main() -> int:
    scenario = Scenario.from_json_file(str(ROOT / "scenarios" / "china_market_a2.json"))
    print(f"scenario topic   : {scenario.topic}")
    print(f"scenario tutor   : {scenario.model_role_name} -- {scenario.model_role_description}")
    print(f"scenario learner : {scenario.user_role_name} -- {scenario.user_role_description}")
    print()

    tutor = TutorRuntime(
        base_model_path=str(ROOT / "vendor" / "models" / "Qwen_3.5_0.8B"),
        adapter_path=str(ROOT / "outputs" / "dpo"),
        cefr_level="A2",
        locale="china",
        use_4bit=False,        # 0.8B fits in bf16
        max_new_tokens=160,
        temperature=0.7,
        scenario=scenario,
        enable_safety_filter=True,
    )

    print("=== conversation turns (/no_think) ===")
    for i, msg in enumerate(SCRIPTED_TURNS, 1):
        print(f"\n[USER {i}] {msg}")
        reply = await tutor.chat_async(msg)
        print(f"[TUTOR {i}] {reply}")

    print("\n=== evaluation (/think) ===")
    try:
        result = await tutor.evaluate_async(target_cefr="A2")
        print(f"parse_error : {result.parse_error}")
        print(f"reasoning (first 300 chars):\n  {(result.reasoning or '')[:300]!r}")
        print(f"raw (first 600 chars):\n  {(result.raw or '')[:600]!r}")
        print(f"scores      : {result.scores}")
    except Exception as exc:  # noqa: BLE001
        print(f"evaluate_async() raised: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
