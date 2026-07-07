"""rerender_system_prompts.py — Re-render system_prompt in all sft_raw JSONL files.

Run after any change to _SCENARIO_DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE in
prompts.py so all existing training records carry the updated template
before filter_sft bakes them into sft_filtered/.

Directories processed:
  data/sft_raw/                 all generation streams
  data/sft_raw_a5_persistent/   A5 fixed-turn-7 persistent stream

Usage:
  python scripts/rerender_system_prompts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation.prompts import render_scenario_deployment_system_prompt

RAW_DIRS = [
    ROOT / "data" / "sft_raw",
    ROOT / "data" / "sft_raw_a5_persistent",
]


def _rerender_file(path: Path) -> tuple[int, int]:
    """Re-render system_prompt in one JSONL file. Returns (n_updated, n_skipped)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    n_updated = 0
    n_skipped = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            n_skipped += 1
            continue

        meta = record.get("metadata") or {}
        cefr_level = meta.get("cefr_level", "")
        locale_name = meta.get("locale") or None
        topic = meta.get("topic", "")
        subtopics = meta.get("subtopics") or []

        user_role = meta.get("user_role") or {}
        model_role = meta.get("model_role") or {}
        if isinstance(user_role, dict):
            user_role_name = user_role.get("name", "")
            user_role_description = user_role.get("description", "")
        else:
            user_role_name = str(user_role)
            user_role_description = ""
        if isinstance(model_role, dict):
            model_role_name = model_role.get("name", "")
            model_role_description = model_role.get("description", "")
        else:
            model_role_name = str(model_role)
            model_role_description = ""

        if not (cefr_level and topic and model_role_name and model_role_description):
            out_lines.append(line)
            n_skipped += 1
            continue

        try:
            new_prompt = render_scenario_deployment_system_prompt(
                cefr_level=cefr_level,
                locale_name=locale_name,
                topic=topic,
                subtopics=subtopics,
                user_role_name=user_role_name,
                user_role_description=user_role_description,
                model_role_name=model_role_name,
                model_role_description=model_role_description,
            )
        except Exception as exc:
            print(f"  WARNING: render failed for {record.get('id', '?')}: {exc}")
            out_lines.append(line)
            n_skipped += 1
            continue

        if record.get("system_prompt") != new_prompt:
            record["system_prompt"] = new_prompt
            n_updated += 1

        out_lines.append(json.dumps(record, ensure_ascii=False))

    if n_updated > 0:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return n_updated, n_skipped


def main() -> None:
    total_files = 0
    total_updated = 0
    total_skipped = 0

    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            print(f"skipping {raw_dir.name}/ (not found)")
            continue

        jsonl_files = sorted(
            p for p in raw_dir.glob("*.jsonl")
            if not p.name.startswith("_")
        )
        print(f"\n{raw_dir.name}/  ({len(jsonl_files)} files)")
        for path in jsonl_files:
            n_updated, n_skipped = _rerender_file(path)
            total_files += 1
            total_updated += n_updated
            total_skipped += n_skipped
            suffix = f"  (skipped={n_skipped})" if n_skipped else ""
            print(f"  {path.name}: updated={n_updated}{suffix}")

    print(f"\nDone. {total_files} files, {total_updated} records updated, "
          f"{total_skipped} skipped.")


if __name__ == "__main__":
    main()
