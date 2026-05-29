"""run_deploy.py  -  Interactive CLI for the trained tutor.

Loads the base model + LoRA adapter (defaults read from training.yaml)
and starts a REPL. Commands: ``/reset`` to clear history, ``/quit`` to exit.

Examples:
    # Default: base from training.yaml, adapter outputs/dpo, A2 china
    python -m scripts.run_deploy

    # Override CEFR level and locale
    python -m scripts.run_deploy --cefr B1 --locale japan

    # Use the SFT adapter (skip DPO output)
    python -m scripts.run_deploy --adapter outputs/sft

    # Full-precision (no 4-bit quant) — needs more VRAM but faster decoding
    python -m scripts.run_deploy --no-4bit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
    parser.add_argument("--cefr", type=str, default="A2",
                        choices=["A1", "A2", "B1", "B2", "C1", "C2"])
    parser.add_argument("--locale", type=str, default="china",
                        help="Locale name (must exist in config/locale.yaml).")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization.")
    parser.add_argument("--no-safety", action="store_true",
                        help="Disable BannedTermsFilter (debug only).")
    parser.add_argument("--banned-terms", type=str, default=None,
                        help="Override banned-terms YAML path. Default: config/banned_terms_deploy.yaml.")
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
    from qwen_tutor.deploy.tutor import TutorRuntime

    tutor = TutorRuntime(
        base_model_path=base_model,
        adapter_path=args.adapter,
        cefr_level=args.cefr,
        locale=args.locale,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_safety_filter=not args.no_safety,
        banned_terms_path=args.banned_terms,
    )

    print(
        f"\nReady. CEFR={args.cefr} locale={args.locale} "
        f"safety={'on' if not args.no_safety else 'OFF'}\n"
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
