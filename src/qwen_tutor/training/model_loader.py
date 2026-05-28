"""Arch-aware base-model loader for SFT / DPO / evaluation / merge.

학생 모델은 항상 같은 인터페이스 (causal LM) 로 학습되어야 하지만,
체크포인트가 실제로 어떤 architecture 로 저장되어 있느냐는 다릅니다.

  - Qwen3-8B            → ``Qwen3ForCausalLM``                  (text-only, AutoModelForCausalLM 로 로드)
  - Qwen3-VL 4B (=3.5)  → ``Qwen3_5ForConditionalGeneration``    (multimodal, language tower 가 ``model.language_model.*`` 에 wrap)

전자는 ``AutoModelForCausalLM`` 로 충분하지만 후자는 image-text-to-text
auto-class 가 필요합니다. 이 모듈은 ``config.json`` 의 ``architectures``
필드를 읽어 적절한 class 를 자동으로 선택합니다.

학습 데이터에는 이미지가 없으므로 multimodal 모델이라도 vision encoder
는 사용되지 않고, LoRA target_modules (q_proj/k_proj/v_proj/o_proj/
gate_proj/up_proj/down_proj) 의 suffix 매칭으로 language tower 의 linear
layer 만 어댑터가 붙습니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _looks_multimodal(architectures: list[str]) -> bool:
    """architectures 문자열에 vision / image / VL / ConditionalGeneration
    같은 multimodal 시그널이 있는지 확인."""
    needles = ("Vision", "Image", "VL", "ConditionalGeneration", "Multimodal")
    return any(any(n in a for n in needles) for a in architectures)


def load_base_model_for_training(
    model_id: str | Path,
    *,
    quantization_config: Any = None,
    torch_dtype: Any = None,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
    device_map: Any = None,
):
    """Load a base model with the right HF auto-class.

    Parameters mirror ``AutoModelForCausalLM.from_pretrained`` plus
    ``attn_implementation``. ``device_map`` is forwarded only when set so
    that the trainer's default placement still works when it is ``None``.

    For multimodal checkpoints, falls back through:
        AutoModelForImageTextToText → AutoModelForVision2Seq
        → AutoModelForCausalLM (last resort, may fail).
    """
    from transformers import AutoConfig

    cfg_path = Path(str(model_id))
    config = AutoConfig.from_pretrained(
        str(model_id), trust_remote_code=trust_remote_code
    )
    archs = list(getattr(config, "architectures", None) or [])
    is_multimodal = _looks_multimodal(archs)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    logger.info(
        "loading base model from %s | architectures=%s | multimodal=%s",
        cfg_path, archs, is_multimodal,
    )

    if is_multimodal:
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(
                str(model_id), **load_kwargs
            )
        except (ImportError, AttributeError) as exc:
            logger.info(
                "AutoModelForImageTextToText unavailable (%s); "
                "trying AutoModelForVision2Seq",
                exc,
            )
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(
                str(model_id), **load_kwargs
            )
        except (ImportError, AttributeError, ValueError) as exc:
            logger.warning(
                "AutoModelForVision2Seq did not accept this checkpoint (%s); "
                "falling back to AutoModelForCausalLM",
                exc,
            )

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(model_id), **load_kwargs)
