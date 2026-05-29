"""run_training.py  -  2단계: 학습 + 평가 단일 진입점.

``config/training.yaml`` 한 파일만 읽고 다음 7 단계를 순서대로 실행합니다.
필요한 raw 데이터는 ``scripts/run_generation.py`` 가 이미 만들어 두었어야
합니다.

    train_sft           SFT LoRA 학습
    eval_intermediate   SFT 직후 모델로 holdout 평가 (점수 베이스라인)
    on_policy_gen       SFT 어댑터로 한 턴 재생성 → 추가 DPO 페어 수집
    filter_dpo          dpo_raw/register + dpo_raw/on_policy 모두 필터링
    train_dpo           SFT 위에 DPO 추가 학습
    eval_final          DPO 직후 모델로 holdout 평가
    compare             intermediate vs final 메트릭 비교 리포트

학습은 GPU 메모리를 많이 씁니다. RTX 3060(12GB) 에서는 동작하지 않으며
RTX 5090(32GB) 이상에서 돌리는 것을 가정합니다.

사용 예:
    # 전체 단계
    python scripts/run_training.py

    # SFT 만 (DPO 와 on-policy 생략)
    python scripts/run_training.py --stages train_sft,eval_intermediate

    # 평가 단계만
    python scripts/run_training.py --stages eval_intermediate,eval_final,compare
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Windows 콘솔(cp1252/cp949)은 기본적으로 한글을 출력하지 못해 argparse 의
# --help 텍스트에서 UnicodeEncodeError 가 발생합니다. stdout/stderr 를
# UTF-8 로 다시 설정해 도움말 + 모든 print 메시지가 깨지지 않게 합니다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# src/ 를 sys.path 에 추가 (editable 설치 안 한 경우 대비)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")

logger = logging.getLogger("run_training")

ALL_STAGES = (
    "train_sft",
    "eval_intermediate",
    "on_policy_gen",
    "filter_dpo",
    "train_dpo",
    "eval_final",
    "compare",
)


# ---------------------------------------------------------------------------
# 설정 로딩
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _generation_levels(gen_cfg_path: Path) -> list[str]:
    """필터/평가에서 사용할 CEFR 레벨 목록을 generation.yaml 에서 가져옵니다."""
    gen_cfg = _load_yaml(gen_cfg_path)
    return gen_cfg.get("generation", {}).get("cefr_levels", ["A1", "A2", "B1", "B2", "C1", "C2"])


# ---------------------------------------------------------------------------
# 1) SFT 학습
# ---------------------------------------------------------------------------


def _stage_train_sft(train_cfg_path: Path) -> None:
    """SFT 학습을 실행합니다. 출력 디렉토리는 training.yaml 의 sft.output_dir."""
    print("\n=== train_sft ===")
    from qwen_tutor.training.sft import run_sft

    out_dir = run_sft(str(train_cfg_path))
    print(f"  SFT 어댑터 저장 위치: {out_dir}")


# ---------------------------------------------------------------------------
# 2) On-policy DPO 페어 생성 (학습된 SFT 모델로 한 턴 재생성)
# ---------------------------------------------------------------------------


async def _stage_on_policy_gen(
    train_cfg: dict[str, Any], train_cfg_path: Path, gen_cfg_path: Path
) -> None:
    """학습된 SFT 어댑터로 정책 출력 한 턴을 만들고, teacher 와 비교해 더 자연스러운
    쪽을 chosen 으로 골라 ``data/dpo_raw/on_policy_<level>.jsonl`` 에 추가합니다.
    """
    op_cfg = train_cfg.get("on_policy", {})
    if not op_cfg.get("enabled", True):
        print("\n=== on_policy_gen (skipped, on_policy.enabled=false) ===")
        return
    print("\n=== on_policy_gen ===")

    from qwen_tutor.generation.on_policy_pairs import generate_batch
    from qwen_tutor.generation.prompts import DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE
    from qwen_tutor.generation.teacher import build_teacher_from_config
    from qwen_tutor.training.eval.run_eval import HFTargetModelClient

    levels = _generation_levels(gen_cfg_path)
    target = HFTargetModelClient.from_pretrained(
        base_model_id=train_cfg["base_model"]["model_id"],
        adapter_path=op_cfg.get("adapter_path", "outputs/sft"),
    )
    judge = build_teacher_from_config(str(gen_cfg_path), role="judge")
    result = await generate_batch(
        target=target,
        deployment_system_prompt_template=DEPLOYMENT_SYSTEM_PROMPT_TEMPLATE,
        cefr_levels=levels,
        sft_filtered_dir="data/sft_filtered",
        output_dir="data/dpo_raw",
        config_path=str(gen_cfg_path),
        judge=judge,
        min_margin=op_cfg.get("min_margin", 2),
        max_per_level=op_cfg.get("max_per_level"),
        concurrency=op_cfg.get("concurrency", 4),
        target_max_tokens=op_cfg.get("target_max_tokens", 320),
        target_temperature=op_cfg.get("target_temperature", 0.7),
    )
    print(f"  on-policy pairs written: {result}")


# ---------------------------------------------------------------------------
# 3) DPO 필터링 (register + on_policy 둘 다)
# ---------------------------------------------------------------------------


async def _stage_filter_dpo(gen_cfg_path: Path) -> None:
    """register_*.jsonl 과 on_policy_*.jsonl 두 가지 raw DPO 입력을 모두 필터링.

    필터 구성은 ``config/generation.yaml`` 의 filtering 블록을 그대로 사용합니다.
    """
    print("\n=== filter_dpo (register + on_policy) ===")
    from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
    from qwen_tutor.generation.filters.mode_consistency import ModeConsistencyFilter
    from qwen_tutor.generation.filters.naturalness import NaturalnessFilter
    from qwen_tutor.generation.filters.pipeline import FilterPipeline
    from qwen_tutor.schemas import DPOExample

    gen_cfg = _load_yaml(gen_cfg_path)
    fcfg = gen_cfg.get("filtering", {})
    filters_list: list = [
        BannedTermsFilter(),
        ModeConsistencyFilter(),
        NaturalnessFilter(),
    ]
    if fcfg.get("enable_locale_judge"):
        from qwen_tutor.generation.filters.locale_judge import LocaleLLMJudge
        from qwen_tutor.generation.teacher import build_teacher_from_config

        judge = build_teacher_from_config(str(gen_cfg_path), role="judge")
        filters_list.append(LocaleLLMJudge(judge=judge))

    pipeline = FilterPipeline(
        filters_list, short_circuit=fcfg.get("short_circuit", True)
    )
    in_dir = Path("data/dpo_raw")
    out_dir = Path("data/dpo_filtered")
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = _generation_levels(gen_cfg_path)
    for level in levels:
        for prefix in ("register", "on_policy"):
            in_path = in_dir / f"{prefix}_{level}.jsonl"
            if not in_path.exists():
                continue
            examples = list(DPOExample.from_jsonl(in_path))
            base = f"{prefix}_{level}"
            passed_path = out_dir / f"{base}_passed.jsonl"
            failed_path = out_dir / f"{base}_failed.jsonl"
            for p in (passed_path, failed_path):
                if p.exists():
                    p.unlink()
            summary = await pipeline.run_stream(
                examples,
                passed_path=passed_path,
                failed_path=failed_path,
                concurrency=fcfg.get("concurrency", 2),
                progress_desc=f"filter_dpo[{base}]",
            )
            print(f"  {base}: {summary}")


# ---------------------------------------------------------------------------
# 4) DPO 학습
# ---------------------------------------------------------------------------


def _stage_train_dpo(train_cfg_path: Path) -> None:
    print("\n=== train_dpo ===")
    from qwen_tutor.training.dpo import run_dpo

    out_dir = run_dpo(str(train_cfg_path))
    print(f"  DPO 어댑터 저장 위치: {out_dir}")


# ---------------------------------------------------------------------------
# 5) Holdout 평가 (intermediate / final)
# ---------------------------------------------------------------------------


async def _stage_eval(
    *,
    train_cfg: dict[str, Any],
    train_cfg_path: Path,
    gen_cfg_path: Path,
    which: str,  # "intermediate" | "final"
    run_id: str,
) -> str:
    print(f"\n=== eval_{which} ===")
    from qwen_tutor.training.eval.run_eval import build_runner

    ecfg = train_cfg.get("evaluation", {})
    adapter = (
        ecfg.get("intermediate_adapter_path", "outputs/sft")
        if which == "intermediate"
        else ecfg.get("final_adapter_path", "outputs/dpo")
    )
    runner = build_runner(
        training_yaml=str(train_cfg_path),
        generation_yaml=str(gen_cfg_path),
        adapter_path=adapter,
    )
    out_root = Path("data/eval_results")
    full_run_id = f"{run_id}_{which}"
    summary = await runner.run_holdout(
        holdout_dir=ecfg.get("holdout_dir", "data/holdout"),
        output_dir=out_root,
        cefr_levels=_generation_levels(gen_cfg_path),
        n_per_level=ecfg.get("n_per_level", 5),
        run_id=full_run_id,
        concurrency=ecfg.get("concurrency", 4),
    )
    print(f"  run_id: {summary['run_id']}, n_examples: {summary['n_examples']}")
    return full_run_id


# ---------------------------------------------------------------------------
# 6) intermediate vs final 비교
# ---------------------------------------------------------------------------


def _stage_compare(train_cfg: dict[str, Any], run_id: str) -> int:
    """eval_intermediate 와 eval_final 결과를 비교해 회귀 여부를 알려줍니다."""
    print("\n=== compare ===")
    from qwen_tutor.training.eval.compare import compare_runs

    base_dir = Path("data/eval_results") / f"{run_id}_intermediate"
    cand_dir = Path("data/eval_results") / f"{run_id}_final"
    if not base_dir.exists() or not cand_dir.exists():
        print(f"  비교에 필요한 디렉토리가 없습니다: {base_dir} / {cand_dir}")
        return 0
    ccfg = train_cfg.get("comparison", {})
    report = compare_runs(
        baseline_run=base_dir,
        candidate_run=cand_dir,
        threshold=ccfg.get("threshold", 0.05),
    )
    out_md = cand_dir / "compare_report.md"
    out_md.write_text(report.to_markdown(), encoding="utf-8")
    print(f"  비교 리포트: {out_md}")
    if ccfg.get("fail_on_regression") and report.has_regression():
        print("  회귀가 감지되어 종료 코드 1 을 반환합니다.")
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="qwen-en-tutor 학습 + 평가 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--training-config", default="config/training.yaml")
    p.add_argument("--generation-config", default="config/generation.yaml")
    p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help="실행할 단계 (콤마로 구분). 기본은 전체.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="평가 결과 디렉토리 이름의 prefix. 기본은 현재 시각.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _validate_stages(stages: list[str]) -> list[str]:
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        raise SystemExit(f"잘못된 stage: {bad!r}.  유효한 값: {list(ALL_STAGES)}")
    return stages


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_cfg_path = Path(args.training_config)
    gen_cfg_path = Path(args.generation_config)
    for p in (train_cfg_path, gen_cfg_path):
        if not p.exists():
            raise SystemExit(f"설정 파일이 없습니다: {p}")
    train_cfg = _load_yaml(train_cfg_path)
    stages = _validate_stages([s.strip() for s in args.stages.split(",") if s.strip()])
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")

    exit_code = 0
    for stage in stages:
        if stage == "train_sft":
            _stage_train_sft(train_cfg_path)
        elif stage == "eval_intermediate":
            await _stage_eval(
                train_cfg=train_cfg,
                train_cfg_path=train_cfg_path,
                gen_cfg_path=gen_cfg_path,
                which="intermediate",
                run_id=run_id,
            )
        elif stage == "on_policy_gen":
            await _stage_on_policy_gen(train_cfg, train_cfg_path, gen_cfg_path)
        elif stage == "filter_dpo":
            await _stage_filter_dpo(gen_cfg_path)
        elif stage == "train_dpo":
            _stage_train_dpo(train_cfg_path)
        elif stage == "eval_final":
            await _stage_eval(
                train_cfg=train_cfg,
                train_cfg_path=train_cfg_path,
                gen_cfg_path=gen_cfg_path,
                which="final",
                run_id=run_id,
            )
        elif stage == "compare":
            rc = _stage_compare(train_cfg, run_id)
            if rc != 0:
                exit_code = rc

    print("\n=== 학습 파이프라인 완료 ===")
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
