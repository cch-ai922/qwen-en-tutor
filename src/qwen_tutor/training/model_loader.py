"""Arch-aware base-model loader for SFT / DPO / evaluation / merge.

The student model is always trained against the same interface (causal LM),
but the checkpoint may actually be stored under a different architecture.

  - Qwen3-8B            → ``Qwen3ForCausalLM``                  (text-only, loaded via AutoModelForCausalLM)
  - Qwen3-VL 4B (=3.5)  → ``Qwen3_5ForConditionalGeneration``    (multimodal, language tower wrapped under ``model.language_model.*``)

The former is fine with ``AutoModelForCausalLM``, but the latter requires an
image-text-to-text auto-class. This module reads the ``architectures`` field
from ``config.json`` and chooses the appropriate class automatically.

The training data contains no images, so even for multimodal models the
vision encoder is unused. Only the language tower's linear layers receive adapters
via suffix-matching in LoRA target_modules (q_proj/k_proj/v_proj/o_proj/
gate_proj/up_proj/down_proj).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _looks_multimodal(architectures: list[str]) -> bool:
    """Check whether the architectures list contains multimodal signals such as
    vision/image/VL/ConditionalGeneration."""
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
