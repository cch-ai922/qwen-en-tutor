"""DPO training entry point.

Continues from the SFT-trained LoRA adapter on register-preference
data. Loads ``DPOExample`` records from ``data/dpo_filtered/``, renders
prompts via the deployment system prompt template, and runs
``trl.DPOTrainer`` with the configured beta / loss / schedule.

The base model is loaded under the same QLoRA setup as SFT, then the
SFT adapter is attached via ``PeftModel.from_pretrained(..., is_trainable=True)``.
``ref_model=None`` lets the DPO trainer use the frozen base+adapter as
the reference policy automatically.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Eager-import pyarrow/datasets/trl DPO classes at module load (before any
# torch + bnb CUDA DLLs are initialized), same Windows DLL load-order
# workaround as in sft.py. The lazy import of DPOTrainer otherwise segfaults
# during the access-violation race with bitsandbytes CUDA libs.
import pyarrow.dataset  # noqa: F401
import datasets  # noqa: F401
from trl import DPOConfig, DPOTrainer  # noqa: F401

from qwen_tutor.schemas import DPOExample
from qwen_tutor.training.callbacks import TutorEvalCallback
from qwen_tutor.training.formatter import ChatFormatter, NO_THINK_TAG
from qwen_tutor.training.sft import load_training_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DPO data loading + tokenization
# ---------------------------------------------------------------------------


def _iter_filtered_records_by_prefix(
    directory: Path, prefix: str
) -> Iterable[dict[str, Any]]:
    """``prefix_*_passed.jsonl`` 패턴의 필터 통과 레코드만 순회."""
    if not directory.exists():
        return
    for path in sorted(directory.glob(f"{prefix}_*.jsonl")):
        if path.stem.endswith("_failed"):
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                example = rec.get("example") if isinstance(rec, dict) else None
                if example is None:
                    example = rec
                if isinstance(example, dict):
                    yield example


def _load_dpo_by_source(directory: str | Path) -> tuple[list[DPOExample], list[DPOExample]]:
    """``data/dpo_filtered/`` 에서 register / on_policy 페어를 분리해서 로딩."""
    d = Path(directory)
    register: list[DPOExample] = []
    on_policy: list[DPOExample] = []
    for rec in _iter_filtered_records_by_prefix(d, "register"):
        try:
            register.append(DPOExample.model_validate(rec))
        except Exception as exc:  # noqa: BLE001
            logger.warning("dpo loader: dropping malformed register record: %s", exc)
    for rec in _iter_filtered_records_by_prefix(d, "on_policy"):
        try:
            on_policy.append(DPOExample.model_validate(rec))
        except Exception as exc:  # noqa: BLE001
            logger.warning("dpo loader: dropping malformed on_policy record: %s", exc)
    return register, on_policy


def load_dpo_examples(directory: str | Path) -> list[DPOExample]:
    """후방 호환용: register + on_policy 를 무가공으로 합쳐서 반환."""
    register, on_policy = _load_dpo_by_source(directory)
    return register + on_policy


def _mix_dpo_pools(
    register: list[DPOExample],
    on_policy: list[DPOExample],
    mix_ratio_register: float,
    mix_ratio_on_policy: float,
    seed: int,
) -> list[DPOExample]:
    """register / on_policy 두 풀을 mix ratio 에 맞춰 잘라 합칩니다.

    더 풍부한 쪽이 학습을 독점하지 않도록 작은 쪽 기준으로 큰 쪽을 trim.
    한쪽이 비어 있으면 mix ratio 와 무관하게 다른 한쪽을 그대로 사용합니다.
    """
    rng = random.Random(seed)
    rng.shuffle(register)
    rng.shuffle(on_policy)

    if not register or not on_policy:
        return register + on_policy
    if mix_ratio_register + mix_ratio_on_policy <= 0:
        raise ValueError("dpo mix ratios sum to zero")
    total = mix_ratio_register + mix_ratio_on_policy
    r_target = mix_ratio_register / total
    p_target = mix_ratio_on_policy / total

    r_count = len(register)
    p_count = len(on_policy)
    if (r_count / max(1, p_count)) > r_target / p_target:
        keep_r = int(p_count * r_target / p_target)
        register = register[:keep_r]
    elif (p_count / max(1, r_count)) > p_target / r_target:
        keep_p = int(r_count * p_target / r_target)
        on_policy = on_policy[:keep_p]
    logger.info(
        "dpo mix: register=%d (from %d)  on_policy=%d (from %d)  ratio=%.2f/%.2f",
        len(register), r_count, len(on_policy), p_count, r_target, p_target,
    )
    return register + on_policy


def _stratified_split(
    examples: list[DPOExample], validation_split: float, seed: int
) -> tuple[list[DPOExample], list[DPOExample]]:
    rng = random.Random(seed)
    rng.shuffle(examples)
    buckets: dict[str, list[DPOExample]] = defaultdict(list)
    for ex in examples:
        buckets[ex.metadata.cefr_level].append(ex)
    train: list[DPOExample] = []
    val: list[DPOExample] = []
    for items in buckets.values():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * validation_split))) if items else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _render_dpo_row(
    ex: DPOExample, formatter: ChatFormatter
) -> dict[str, str] | None:
    """Render a DPO example as the ``{prompt, chosen, rejected}`` row TRL expects.

    The ``prompt`` is the chat-template-rendered string up to (but not
    including) the assistant turn. The ``chosen`` and ``rejected``
    strings are JUST the assistant content (TRL concatenates internally).
    """
    tok = formatter.tokenizer
    if tok is None:
        raise RuntimeError("DPO requires a tokenizer on the ChatFormatter")

    # Build the chat messages with the authoritative system prompt and
    # /no_think tag on the first user turn (DPO pairs come from
    # conversation-mode SFT examples). System prompt is rendered for the
    # DPO example's locale so each pair trains with the right country
    # grounding.
    system_content = formatter._render_deployment_system_prompt(
        ex.metadata.cefr_level, locale_name=ex.metadata.locale
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    first_user_seen = False
    for m in ex.prompt_messages:
        if m.role == "system":
            continue
        content = m.content
        if m.role == "user" and not first_user_seen:
            content = formatter._ensure_no_think(content)
            first_user_seen = True
        messages.append({"role": m.role, "content": content})
    if not first_user_seen:
        # DPO pair with no user prompt — skip.
        return None

    try:
        prompt_text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return {
        "prompt": prompt_text,
        "chosen": ex.chosen.content,
        "rejected": ex.rejected.content,
    }


def _build_dpo_dataset(
    examples: list[DPOExample], formatter: ChatFormatter, max_length: int
):
    from datasets import Dataset

    rows: list[dict[str, str]] = []
    for ex in examples:
        row = _render_dpo_row(ex, formatter)
        if row is None:
            continue
        # rough length guard — DPOTrainer will truncate internally too.
        total_len = len(row["prompt"]) + max(
            len(row["chosen"]), len(row["rejected"])
        )
        if total_len > max_length * 6:  # chars-not-tokens, generous slack
            continue
        rows.append(row)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def run_dpo(config_path: str | Path = "config/training.yaml") -> str:
    """Run DPO on top of the SFT LoRA; return the output adapter directory."""

    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    # DPOConfig / DPOTrainer / datasets / pyarrow.dataset are imported at
    # module level above (Windows DLL load-order workaround).
    from qwen_tutor.training.model_loader import load_base_model_for_training

    cfg = load_training_config(config_path)
    base_cfg = cfg["base_model"]
    dpo_cfg = cfg["dpo"]
    cb_cfg = cfg.get("eval_callback") or {}

    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg["tokenizer_id"],
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatter = ChatFormatter(
        tokenizer=tokenizer,
        max_seq_length=dpo_cfg["max_length"],
    )

    # ---- data --------------------------------------------------------
    data_cfg = dpo_cfg["data"]
    register, on_policy = _load_dpo_by_source(data_cfg["dpo_filtered_dir"])
    logger.info(
        "loaded DPO: %d register + %d on_policy examples (pre-mix)",
        len(register), len(on_policy),
    )
    examples = _mix_dpo_pools(
        register=register,
        on_policy=on_policy,
        mix_ratio_register=float(data_cfg.get("mix_ratio_register", 0.70)),
        mix_ratio_on_policy=float(data_cfg.get("mix_ratio_on_policy", 0.30)),
        seed=data_cfg["shuffle_seed"],
    )
    logger.info("DPO post-mix total: %d examples", len(examples))
    train_examples, val_examples = _stratified_split(
        examples,
        validation_split=data_cfg["validation_split"],
        seed=data_cfg["shuffle_seed"],
    )
    train_ds = _build_dpo_dataset(train_examples, formatter, dpo_cfg["max_length"])
    val_ds = _build_dpo_dataset(val_examples, formatter, dpo_cfg["max_length"])
    logger.info("dpo train=%d  val=%d", len(train_ds), len(val_ds))

    # ---- model + SFT adapter ----------------------------------------
    q_cfg = base_cfg.get("quantization") or {}
    bnb_config = None
    if q_cfg.get("enabled"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, q_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
        )

    base_model = load_base_model_for_training(
        base_cfg["model_id"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, base_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=base_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )
    if base_cfg.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    sft_adapter = dpo_cfg["sft_adapter_path"]
    logger.info("attaching SFT adapter from %s with is_trainable=True", sft_adapter)
    model = PeftModel.from_pretrained(base_model, sft_adapter, is_trainable=True)

    # ---- trainer ----------------------------------------------------
    dpo_args = DPOConfig(
        output_dir=dpo_cfg["output_dir"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        max_steps=dpo_cfg.get("max_steps", -1),
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=dpo_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        lr_scheduler_type=dpo_cfg["lr_scheduler_type"],
        weight_decay=dpo_cfg["weight_decay"],
        beta=dpo_cfg["beta"],
        loss_type=dpo_cfg["loss_type"],
        max_length=dpo_cfg["max_length"],
        # trl 1.5 removed max_prompt_length; max_length now governs both.
        bf16=dpo_cfg["bf16"],
        fp16=dpo_cfg["fp16"],
        optim=dpo_cfg["optim"],
        logging_steps=dpo_cfg["logging_steps"],
        eval_strategy=dpo_cfg.get("eval_strategy", "no"),
        eval_steps=dpo_cfg.get("eval_steps", 100),
        save_strategy=dpo_cfg.get("save_strategy", "no"),
        save_steps=dpo_cfg.get("save_steps", 100),
        save_total_limit=dpo_cfg.get("save_total_limit", 1),
        report_to=dpo_cfg.get("report_to", "none"),
        seed=dpo_cfg["seed"],
    )

    callbacks = []
    if cb_cfg.get("enabled", True):
        callbacks.append(
            TutorEvalCallback(
                formatter=formatter,
                prompts_path=cb_cfg.get("held_out_prompts_path"),
                n_conversation=cb_cfg.get("n_conversation_samples", 10),
                n_evaluation=cb_cfg.get("n_evaluation_samples", 10),
                max_new_tokens=cb_cfg.get("max_new_tokens", 512),
                temperature=cb_cfg.get("temperature", 0.7),
                top_p=cb_cfg.get("top_p", 0.95),
                log_samples_table=cb_cfg.get("log_samples_table", True),
                conversation_filters=cb_cfg.get("conversation_filters"),
            )
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=dpo_cfg.get("ref_model"),
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(dpo_cfg["output_dir"])
    tokenizer.save_pretrained(dpo_cfg["output_dir"])
    logger.info("DPO complete; adapter saved to %s", dpo_cfg["output_dir"])
    return dpo_cfg["output_dir"]
