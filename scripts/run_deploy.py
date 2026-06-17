"""run_deploy.py  -  Interactive CLI for the trained tutor.

Loads the base model + LoRA adapter (defaults read from training.yaml)
and starts a REPL. Commands: ``/reset`` to clear history, ``/quit`` to exit.

Examples:
    # Default: base from training.yaml, adapter outputs/dpo, A2 china
    python -m scripts.run_deploy

    # Override CEFR level and locale
    python -m scripts.run_deploy --cefr B1 --locale japan

    # Hand-crafted single-file scenario
    python -m scripts.run_deploy --scenario-file scenarios/china_market_a2.json

    # Deploy a training seed directly — no conversion step.
    # CEFR / locale auto-default to the seed's cefr_level / locale.
    python -m scripts.run_deploy --seed-jsonl data/seeds/A2.jsonl --seed-id 26a7d984b0a6

    # Pick by row index instead of id
    python -m scripts.run_deploy --seed-jsonl data/seeds/A2.jsonl --seed-index 7

    # Use the SFT adapter (skip DPO output)
    python -m scripts.run_deploy --adapter outputs/sft

    # Full-precision (no 4-bit quant) — needs more VRAM but faster decoding
    python -m scripts.run_deploy --no-4bit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Offline-by-default: keep transformers and the Hub off the network so the
# deployed tutor runs purely from local adapter + vendor/models. Must be set
# before any transformers/peft import. Users can override by exporting these
# vars explicitly before launching the script.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml

DEFAULT_TRAINING_YAML = Path("config/training.yaml")


def _load_defaults(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive tutor REPL.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_YAML)
    parser.add_argument("--base-model", type=str, default=None,
                        help="Path to base model. Default: base_model.model_id from training.yaml.")
    parser.add_argument("--adapter", type=str, default="outputs/dpo",
                        help="Path to LoRA adapter directory. Default: outputs/dpo.")
    parser.add_argument("--cefr", type=str, default=None,
                        choices=["A1", "A2", "B1", "B2", "C1", "C2"],
                        help="CEFR target register. If omitted and a seed is loaded "
                             "via --seed-jsonl / --scenario-file, defaults to the "
                             "seed's cefr_level. Otherwise falls back to A2.")
    parser.add_argument("--locale", type=str, default=None,
                        help="Locale name (must exist in config/locale.yaml). If "
                             "omitted and a seed is loaded, defaults to the seed's "
                             "locale. Otherwise falls back to china.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization.")
    parser.add_argument("--no-safety", action="store_true",
                        help="Disable BannedTermsFilter (debug only).")
    parser.add_argument("--banned-terms", type=str, default=None,
                        help="Override banned-terms YAML path. Default: config/banned_terms_deploy.yaml.")
    parser.add_argument("--scenario-file", type=str, default=None,
                        help="Path to a single-object scenario JSON (topic + subtopics + "
                             "user_role + model_role). Accepts both the hand-crafted "
                             "scenarios/*.json shape AND a single-object seed dump. "
                             "Strongly recommended -- the model was trained with these fields "
                             "in the system prompt. Without one, the runtime falls back to a "
                             "generic prompt and logs a warning.")
    parser.add_argument("--seed-jsonl", type=str, default=None,
                        help="Path to a training seeds JSONL file (data/seeds/<level>.jsonl). "
                             "One row is loaded as the scenario; CEFR / locale auto-default to "
                             "the seed's cefr_level / locale. Mutually exclusive with "
                             "--scenario-file.")
    parser.add_argument("--seed-id", type=str, default=None,
                        help="When loading from --seed-jsonl, pick the row with this 'id' "
                             "(preferred for reproducibility). Mutually exclusive with --seed-index.")
    parser.add_argument("--seed-index", type=int, default=0,
                        help="When loading from --seed-jsonl without --seed-id, pick the Nth "
                             "(0-indexed) non-blank row. Default: 0 (first row).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    defaults = _load_defaults(args.training_config)
    base_model = args.base_model or defaults.get("base_model", {}).get("model_id")
    if not base_model:
        print(
            f"ERROR: no base model. Pass --base-model or set base_model.model_id "
            f"in {args.training_config}.",
            file=sys.stderr,
        )
        return 2
    if not Path(args.adapter).exists():
        print(
            f"ERROR: adapter directory not found: {args.adapter}\n"
            f"Run training first, or pass --adapter <path>.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {base_model} + adapter {args.adapter} ...", flush=True)

    # Import lazily so --help is fast.
    from qwen_tutor.deploy.tutor import Scenario, TutorRuntime

    if args.scenario_file and args.seed_jsonl:
        print(
            "ERROR: --scenario-file and --seed-jsonl are mutually exclusive.",
            file=sys.stderr,
        )
        return 2
    if args.seed_id and args.seed_index != 0:
        print(
            "ERROR: --seed-id and --seed-index are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    scenario = None
    if args.scenario_file:
        scenario_path = Path(args.scenario_file)
        if not scenario_path.exists():
            print(f"ERROR: scenario file not found: {scenario_path}", file=sys.stderr)
            return 2
        try:
            scenario = Scenario.from_json_file(scenario_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: failed to load scenario {scenario_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    elif args.seed_jsonl:
        try:
            scenario = Scenario.from_seed_jsonl(
                args.seed_jsonl,
                seed_id=args.seed_id,
                seed_index=args.seed_index,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: failed to load seed: {exc}", file=sys.stderr)
            return 2

    # Resolve CEFR and locale. Priority: explicit CLI > seed metadata > fallback.
    cefr = args.cefr
    locale = args.locale
    if scenario is not None:
        if cefr is None and scenario.metadata.get("cefr_level"):
            cefr = scenario.metadata["cefr_level"]
        if locale is None and scenario.metadata.get("locale"):
            locale = scenario.metadata["locale"]
    if cefr is None:
        cefr = "A2"
    if locale is None:
        locale = "china"

    tutor = TutorRuntime(
        base_model_path=base_model,
        adapter_path=args.adapter,
        cefr_level=cefr,
        locale=locale,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_safety_filter=not args.no_safety,
        banned_terms_path=args.banned_terms,
        scenario=scenario,
    )

    if scenario is not None:
        seed_id = scenario.metadata.get("id")
        seed_suffix = f" [seed_id={seed_id}]" if seed_id else ""
        scenario_line = (
            f"scenario={scenario.model_role_name!r} talking with "
            f"{scenario.user_role_name!r} about {scenario.topic!r}{seed_suffix}"
        )
    else:
        scenario_line = "scenario=NONE (model running outside trained distribution)"
    print(
        f"\nReady. CEFR={cefr} locale={locale} "
        f"safety={'on' if not args.no_safety else 'OFF'}\n"
        f"{scenario_line}\n"
        f"Type your message. Commands: /reset, /eval [target_cefr], /quit\n",
        flush=True,
    )

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            tutor.reset()
            print("[history cleared]", flush=True)
            continue
        if user.startswith("/eval"):
            parts = user.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else None
            try:
                result = tutor.evaluate(target_cefr=target)
            except RuntimeError as exc:
                print(f"[eval error: {exc}]", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[eval error: {type(exc).__name__}: {exc}]", flush=True)
                continue
            print(f"\n--- EVAL (target CEFR={result.target_cefr}) ---")
            if result.reasoning:
                print(f"reasoning:\n{result.reasoning}\n")
            else:
                print("(no <think>...</think> block found)\n")
            if result.scores is not None:
                import json as _json
                print("scores:")
                print(_json.dumps(result.scores, indent=2, ensure_ascii=False))
            else:
                print(f"!! parse_error: {result.parse_error}")
                print(
                    "!! falling back to RAW model output below — inspect for "
                    "malformed JSON, code fences, or truncation."
                )
                print("---- raw model output ----")
                print(result.raw if result.raw else "(empty)")
                print("---- end raw ----")
            print("--- end EVAL ---\n", flush=True)
            continue
        try:
            reply = tutor.chat(user)
        except Exception as exc:  # noqa: BLE001
            print(f"[generation error: {type(exc).__name__}: {exc}]", flush=True)
            continue
        print(f"tutor> {reply}\n", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
