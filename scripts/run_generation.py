"""run_generation.py  -  1단계: 대화 데이터 생성 + 필터링 단일 진입점.

이 스크립트는 ``config/generation.yaml`` 한 파일만 보고 다음 8 단계를
순차적으로 실행합니다.

    seeds         시나리오 시드 생성 (CEFR 레벨별)
    sft           일상 대화 SFT 예시 생성
    redirect      리다이렉트 SFT 예시 생성
    register      register 페어 (DPO) 생성
    eval          /think 평가 예시 생성
    filter_sft    sft_raw 필터링 (normal + redirect)
    filter_eval   eval_raw 필터링
    filter_dpo    dpo_raw 필터링 (register)

이전 실행에서 만들어진 결과 파일은 그대로 두며, 각 모듈은 이미 처리한 ID 를
건너뛰므로 중간에 멈췄다 다시 돌려도 안전합니다.

사용 예:
    # 전체 단계, YAML 기본값으로 실행
    python scripts/run_generation.py

    # 빠른 스모크 - A2 레벨 2개만 SFT 까지
    python scripts/run_generation.py --levels A2 --n-per-level 2 --stages seeds,sft

    # 필터만 다시 돌리기 (예: 필터 코드 수정 후)
    python scripts/run_generation.py --stages filter_sft,filter_eval,filter_dpo

전제: ``scripts/start_llama_cpp.ps1`` 로 로컬 llama.cpp 서버가
``http://127.0.0.1:8080/v1`` 에 떠 있어야 합니다 (또는 ``generation.yaml``
의 teacher/judge 블록을 클라우드 API 로 바꿔야 합니다).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
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
# src/ 를 sys.path 에 추가 (editable 설치 안 한 경우를 대비)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 컴팩트 프롬프트가 기본값이 되도록 환경 변수가 비어 있으면 채워 줍니다.
os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")

from qwen_tutor.generation import (  # noqa: E402
    eval_gen as eval_mod,
    language_redirect as language_redirect_mod,
    locale_redirect as locale_redirect_mod,
    pedagogy_redirect as pedagogy_redirect_mod,
    persona_redirect as persona_redirect_mod,
    redirect as redirect_mod,
    register_pairs as register_mod,
    seeds as seeds_mod,
    sft_dialogues as sft_mod,
)
from qwen_tutor.generation.teacher import build_teacher_from_config  # noqa: E402
from qwen_tutor.schemas import DPOExample, EvaluationExample, SFTExample  # noqa: E402

logger = logging.getLogger("run_generation")

ALL_STAGES = (
    "seeds",
    "sft",
    "redirect",
    "locale_redirect",
    "pedagogy_redirect",
    "language_redirect",
    "persona_redirect",
    "register",
    "eval",
    "filter_sft",
    "filter_eval",
    "filter_dpo",
)


# ---------------------------------------------------------------------------
# 설정 로딩
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    """generation.yaml 을 dict 로 읽습니다."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """CLI 인자가 우선, 없으면 YAML 값을 씁니다."""
    from qwen_tutor.locale import DEFAULT_LOCALE_NAME, list_locales

    gen = cfg.get("generation", {})

    # 어떤 locale 로 데이터를 만들지. CLI > YAML > [default_locale] 우선순위.
    if args.locales:
        locales = [s.strip() for s in args.locales.split(",") if s.strip()]
    else:
        locales = list(gen.get("locales") or [])
    if not locales:
        locales = [DEFAULT_LOCALE_NAME]
    # locale 이름 검증 - 잘못된 이름으로 조용히 데이터가 만들어지면 큰 사고.
    known = set(list_locales())
    bad = [name for name in locales if name not in known]
    if bad:
        raise SystemExit(
            f"unknown locale(s) {bad!r}. config/locale.yaml 에 정의된 locale: "
            f"{sorted(known)}"
        )

    return {
        "levels": args.levels.split(",") if args.levels else gen.get("cefr_levels", ["A2"]),
        "locales": locales,
        "n_per_level": args.n_per_level if args.n_per_level is not None else gen.get("n_per_level", 3),
        "concurrency": args.concurrency if args.concurrency is not None else gen.get("concurrency", 2),
        "per_call_size": gen.get("per_call_size", 3),
        "redirect_fraction": float(gen.get("redirect_fraction", 0.20)),
        "eval_fraction": float(gen.get("eval_fraction", 0.25)),
        "locale_redirect_fraction": float(gen.get("locale_redirect_fraction", 0.15)),
        "pedagogy_redirect_fraction": float(gen.get("pedagogy_redirect_fraction", 0.15)),
        "language_redirect_fraction": float(gen.get("language_redirect_fraction", 0.15)),
        "persona_redirect_fraction": float(gen.get("persona_redirect_fraction", 0.15)),
        "dialogues_per_seed": int(gen.get("dialogues_per_seed", 1)),
    }


# ---------------------------------------------------------------------------
# 생성 4단계 (seeds / sft / redirect / register / eval)
# ---------------------------------------------------------------------------


async def _stage_seeds(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== seeds (levels={params['levels']}, locales={params['locales']}, "
        f"n_per_level={params['n_per_level']}) ==="
    )
    res = await seeds_mod.generate_batch(
        n_per_level=params["n_per_level"],
        cefr_levels=params["levels"],
        locales=params["locales"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        per_call_size=params["per_call_size"],
        output_dir=paths["seeds_dir"],
    )
    print(f"  seeds written this run: {res}")


async def _stage_sft(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== sft (normal dialogues, dialogues_per_seed="
        f"{params['dialogues_per_seed']}) ==="
    )
    res = await sft_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  sft examples written this run: {res}")


async def _stage_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== redirect (redirect-moment dialogues, fraction="
        f"{params['redirect_fraction']:.2f}) ==="
    )
    res = await redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        redirect_fraction=params["redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  redirect examples written this run: {res}")


async def _stage_locale_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== locale_redirect (user-side locale_violation handling, fraction="
        f"{params['locale_redirect_fraction']:.2f}) ==="
    )
    res = await locale_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        locale_redirect_fraction=params["locale_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  locale_redirect examples written this run: {res}")


async def _stage_pedagogy_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== pedagogy_redirect (user-side pedagogy_weak handling, fraction="
        f"{params['pedagogy_redirect_fraction']:.2f}) ==="
    )
    res = await pedagogy_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        pedagogy_redirect_fraction=params["pedagogy_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  pedagogy_redirect examples written this run: {res}")


async def _stage_language_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== language_redirect (user-side language_violation handling, fraction="
        f"{params['language_redirect_fraction']:.2f}) ==="
    )
    res = await language_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        language_redirect_fraction=params["language_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  language_redirect examples written this run: {res}")


async def _stage_persona_redirect(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== persona_redirect (user-side persona_break handling, fraction="
        f"{params['persona_redirect_fraction']:.2f}) ==="
    )
    res = await persona_redirect_mod.generate_batch(
        cefr_levels=params["levels"],
        seeds_dir=paths["seeds_dir"],
        output_dir=paths["sft_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        persona_redirect_fraction=params["persona_redirect_fraction"],
        dialogues_per_seed=params["dialogues_per_seed"],
    )
    print(f"  persona_redirect examples written this run: {res}")


async def _stage_register(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== register (multi-axis DPO pairs) ===")
    res = await register_mod.generate_batch(
        cefr_levels=params["levels"],
        sft_dir=paths["sft_raw"],
        output_dir=paths["dpo_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
    )
    print(f"  register pairs written this run: {res}")


async def _stage_eval(cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print(
        f"\n=== eval (/think examiner examples, fraction="
        f"{params['eval_fraction']:.2f}) ==="
    )
    res = await eval_mod.generate_batch(
        cefr_levels=params["levels"],
        sft_dir=paths["sft_raw"],
        output_dir=paths["eval_raw"],
        config_path=str(cfg_path),
        concurrency=params["concurrency"],
        eval_fraction=params["eval_fraction"],
    )
    print(f"  eval examples written this run: {res}")


# ---------------------------------------------------------------------------
# 필터 3단계 (filter_sft / filter_eval / filter_dpo)
# ---------------------------------------------------------------------------


def _build_filters(cfg: dict[str, Any], cfg_path: Path) -> list:
    """generation.yaml 의 filtering 블록을 보고 활성화할 필터 리스트를 만듭니다."""
    from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
    from qwen_tutor.generation.filters.mode_consistency import ModeConsistencyFilter
    from qwen_tutor.generation.filters.naturalness import NaturalnessFilter
    from qwen_tutor.generation.filters.non_latin_script import NonLatinScriptFilter
    from qwen_tutor.generation.filters.speaks_l1_sanity import SpeaksL1SanityFilter

    fcfg = cfg.get("filtering", {})
    # 필터 순서:
    #  1) SpeaksL1SanityFilter - speaks_l1 examples 중 L1 turn 이 빠진
    #     degenerate 케이스를 가장 일찍 잡아 냅니다. speaks_l1 이 아닌
    #     record 는 그대로 통과.
    #  2) NonLatinScriptFilter - 그 외 record 의 비-Latin 글자 leak 차단.
    #     speaks_l1 user turn 은 자체적으로 exempt 되어 있어 1과 충돌하지
    #     않습니다.
    #  3-5) 기존 mechanical 필터들.
    filters_list: list = [
        SpeaksL1SanityFilter(),
        NonLatinScriptFilter(),
        BannedTermsFilter(),
        ModeConsistencyFilter(),
        NaturalnessFilter(),
    ]
    # locale judge 는 teacher 모델 호출이 비싸므로 옵션 처리.
    if fcfg.get("enable_locale_judge"):
        from qwen_tutor.generation.filters.locale_judge import LocaleLLMJudge

        judge = build_teacher_from_config(str(cfg_path), role="judge")
        filters_list.append(LocaleLLMJudge(judge=judge))
    if fcfg.get("enable_naturalness_judge"):
        from qwen_tutor.generation.filters.judge import NaturalnessLLMJudge

        judge2 = build_teacher_from_config(str(cfg_path), role="judge")
        filters_list.append(
            NaturalnessLLMJudge(
                judge=judge2,
                sample_rate=fcfg.get("naturalness_sample_rate", 0.0),
            )
        )
    return filters_list


async def _run_filter(
    *,
    in_dir: Path,
    out_dir: Path,
    schema,
    file_patterns: list[str],
    levels: list[str],
    filters_list: list,
    fcfg: dict[str, Any],
    label: str,
) -> None:
    """파일 패턴별로 raw 입력을 읽어 *_passed / *_failed 로 분리해서 씁니다."""
    from qwen_tutor.generation.filters.pipeline import FilterPipeline

    pipeline = FilterPipeline(
        filters_list, short_circuit=fcfg.get("short_circuit", True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        for pattern in file_patterns:
            in_path = in_dir / pattern.format(level=level)
            if not in_path.exists():
                continue
            examples = list(schema.from_jsonl(in_path))
            base = pattern.format(level=level).replace(".jsonl", "")
            passed_path = out_dir / f"{base}_passed.jsonl"
            failed_path = out_dir / f"{base}_failed.jsonl"
            # 같은 raw 파일을 두 번 돌릴 때 결과가 누적되지 않도록 비웁니다.
            for p in (passed_path, failed_path):
                if p.exists():
                    p.unlink()
            summary = await pipeline.run_stream(
                examples,
                passed_path=passed_path,
                failed_path=failed_path,
                concurrency=fcfg.get("concurrency", 2),
                progress_desc=f"{label}[{base}]",
            )
            print(f"  {base}: {summary}")


async def _stage_filter_sft(cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== filter_sft (normal + redirect + 4 user-side redirect streams) ===")
    await _run_filter(
        in_dir=Path(paths["sft_raw"]),
        out_dir=Path(paths["sft_filtered"]),
        schema=SFTExample,
        file_patterns=[
            "normal_{level}.jsonl",
            "redirect_{level}.jsonl",
            "locale_redirect_{level}.jsonl",
            "pedagogy_redirect_{level}.jsonl",
            "language_redirect_{level}.jsonl",
            "persona_redirect_{level}.jsonl",
        ],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label="filter_sft",
    )


async def _stage_filter_eval(cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== filter_eval ===")
    await _run_filter(
        in_dir=Path(paths["eval_raw"]),
        out_dir=Path(paths["eval_filtered"]),
        schema=EvaluationExample,
        file_patterns=["{level}.jsonl"],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label="filter_eval",
    )


async def _stage_filter_dpo(cfg: dict[str, Any], cfg_path: Path, params: dict[str, Any], paths: dict[str, str]) -> None:
    print("\n=== filter_dpo (register pairs) ===")
    await _run_filter(
        in_dir=Path(paths["dpo_raw"]),
        out_dir=Path(paths["dpo_filtered"]),
        schema=DPOExample,
        file_patterns=["register_{level}.jsonl"],
        levels=params["levels"],
        filters_list=_build_filters(cfg, cfg_path),
        fcfg=cfg.get("filtering", {}),
        label="filter_dpo",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="qwen-en-tutor 데이터 생성 + 필터링 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/generation.yaml")
    p.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help="실행할 단계 (콤마로 구분). 기본은 전체.",
    )
    p.add_argument(
        "--levels",
        default=None,
        help="CEFR 레벨 (콤마 구분). 기본은 YAML 의 generation.cefr_levels.",
    )
    p.add_argument(
        "--locales",
        default=None,
        help="생성할 locale 이름 (콤마 구분, 예: china,japan,italy). 기본은 YAML 의 "
        "generation.locales 또는 config/locale.yaml 의 default_locale.",
    )
    p.add_argument(
        "--n-per-level",
        type=int,
        default=None,
        help="레벨당 시드 수. 기본은 YAML 의 generation.n_per_level.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="동시 호출 수. 기본은 YAML 의 generation.concurrency.",
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
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"설정 파일이 없습니다: {cfg_path}")
    cfg = _load_config(cfg_path)
    paths = cfg.get("paths", {})
    params = _resolve(args, cfg)
    stages = _validate_stages([s.strip() for s in args.stages.split(",") if s.strip()])

    # 각 stage 핸들러 매핑
    gen_handlers = {
        "seeds":              lambda: _stage_seeds(cfg_path, params, paths),
        "sft":                lambda: _stage_sft(cfg_path, params, paths),
        "redirect":           lambda: _stage_redirect(cfg_path, params, paths),
        "locale_redirect":    lambda: _stage_locale_redirect(cfg_path, params, paths),
        "pedagogy_redirect":  lambda: _stage_pedagogy_redirect(cfg_path, params, paths),
        "language_redirect":  lambda: _stage_language_redirect(cfg_path, params, paths),
        "persona_redirect":   lambda: _stage_persona_redirect(cfg_path, params, paths),
        "register":           lambda: _stage_register(cfg_path, params, paths),
        "eval":               lambda: _stage_eval(cfg_path, params, paths),
    }
    filter_handlers = {
        "filter_sft":  lambda: _stage_filter_sft(cfg, cfg_path, params, paths),
        "filter_eval": lambda: _stage_filter_eval(cfg, cfg_path, params, paths),
        "filter_dpo":  lambda: _stage_filter_dpo(cfg, cfg_path, params, paths),
    }

    for stage in stages:
        if stage in gen_handlers:
            await gen_handlers[stage]()
        elif stage in filter_handlers:
            await filter_handlers[stage]()

    print("\n=== 생성 파이프라인 완료 ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
