"""SFT training entry point.

Joint conversation + evaluation training for a single Qwen3-8B model.
Loads filtered SFT and Evaluation examples, mixes them at a configurable
ratio (default 80/20), tokenizes via the ChatFormatter (so /no_think and
/think modes interleave cleanly), and trains a QLoRA adapter with
``trl.SFTTrainer``.

Stratified 5% validation split keyed on (example_type, cefr_level).

Hyperparameters live in ``config/training.yaml``; this module reads
them and threads them into ``BitsAndBytesConfig``, ``LoraConfig`` and
``SFTConfig``.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

# Force pyarrow / datasets to load NOW, before any torch + bitsandbytes CUDA
# DLLs are initialized. On Windows there is a load-order conflict where the
# MSVC runtime that pyarrow ships gets clobbered by libs that come in with
# torch+bnb at model-load time, causing an access-violation segfault inside
# trl's lazy import of SFTTrainer (which pulls datasets → pyarrow.dataset).
# Eager-importing here at module-load avoids the conflict.
import pyarrow.dataset  # noqa: F401
import datasets  # noqa: F401
from trl import SFTConfig, SFTTrainer  # noqa: F401

from qwen_tutor.schemas import EvaluationExample, SFTExample
from qwen_tutor.training.callbacks import TutorEvalCallback
from qwen_tutor.training.formatter import (
    LOSS_IGNORE_INDEX,
    ChatFormatter,
    TokenizedExample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_training_config(path: str | Path) -> dict[str, Any]:
    """training.yaml 을 dict 로 읽어 반환합니다.

    배포 system 프롬프트와 평가 system 프롬프트는 더 이상 이 YAML 에서
    오지 않습니다 — ``ChatFormatter`` 가 ``qwen_tutor.generation.prompts``
    에서 직접 가져오고, 그 모듈이 ``config/locale.yaml`` 을 단일 진실 소스
    로 사용합니다. 여기서 추가로 치환할 placeholder 는 없습니다.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _iter_filtered_records(directory: Path) -> Iterable[dict[str, Any]]:
    """Iterate ``example`` dicts from a directory of filter-pipeline JSONL."""
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
        # We only want passed examples; failed files (suffix _failed.jsonl)
        # are skipped explicitly.
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
                    # raw-format fallback
                    example = rec
                if isinstance(example, dict):
                    yield example


def load_sft_examples(directory: str | Path) -> list[SFTExample]:
    out: list[SFTExample] = []
    for rec in _iter_filtered_records(Path(directory)):
        try:
            out.append(SFTExample.model_validate(rec))
        except Exception as exc:  # noqa: BLE001
            logger.warning("sft loader: dropping malformed record: %s", exc)
    return out


def load_evaluation_examples(directory: str | Path) -> list[EvaluationExample]:
    out: list[EvaluationExample] = []
    for rec in _iter_filtered_records(Path(directory)):
        try:
            out.append(EvaluationExample.model_validate(rec))
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval loader: dropping malformed record: %s", exc)
    return out


# ---------------------------------------------------------------------------
# Mixing + stratified split
# ---------------------------------------------------------------------------


def mix_and_split(
    sft_examples: list[SFTExample],
    eval_examples: list[EvaluationExample],
    mix_ratio_sft: float,
    mix_ratio_eval: float,
    validation_split: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Build a single train+val pair stratified by (example_type, cefr_level).

    The mix ratios trim each pool down so the overall distribution
    matches the request; the remaining 5% per stratum is held out for
    validation.
    """
    if mix_ratio_sft + mix_ratio_eval <= 0:
        raise ValueError("mix ratios sum to zero")
    total = mix_ratio_sft + mix_ratio_eval
    mix_ratio_sft /= total
    mix_ratio_eval /= total

    rng = random.Random(seed)
    rng.shuffle(sft_examples)
    rng.shuffle(eval_examples)

    # Trim the larger pool so the (kept_sft, kept_eval) ratio matches.
    if eval_examples and (
        len(sft_examples) / max(1, len(eval_examples))
    ) > mix_ratio_sft / mix_ratio_eval:
        target = int(len(eval_examples) * mix_ratio_sft / mix_ratio_eval)
        sft_examples = sft_examples[:target]
    elif sft_examples and (
        len(eval_examples) / max(1, len(sft_examples))
    ) > mix_ratio_eval / mix_ratio_sft:
        target = int(len(sft_examples) * mix_ratio_eval / mix_ratio_sft)
        eval_examples = eval_examples[:target]

    # Stratify: bucket by (type, cefr_level).
    buckets: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for ex in sft_examples:
        buckets[("sft", ex.metadata.cefr_level)].append(ex)
    for ex in eval_examples:
        buckets[("evaluation", ex.metadata.learner_cefr_target)].append(ex)

    train: list[Any] = []
    val: list[Any] = []
    for key, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * validation_split))) if items else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Tokenization pipeline
# ---------------------------------------------------------------------------


def _build_hf_dataset(
    examples: list[Any], formatter: ChatFormatter, max_seq_length: int
):
    """Materialize a HF ``Dataset`` of pretokenized rows.

    Each row carries ``input_ids``, ``attention_mask``, and ``labels`` so
    the trainer can use a plain ``DataCollatorWithPadding`` (no on-the-fly
    chat-template work).
    """
    from datasets import Dataset

    rows: list[dict[str, list[int]]] = []
    dropped = 0
    for ex in examples:
        try:
            tok = formatter.format_for_training(ex)
        except Exception as exc:  # noqa: BLE001
            dropped += 1
            logger.warning("formatter dropped %s: %s", getattr(ex, "id", "?"), exc)
            continue
        if len(tok.input_ids) > max_seq_length:
            continue
        rows.append(
            {
                "input_ids": tok.input_ids,
                "attention_mask": tok.attention_mask,
                "labels": tok.labels,
            }
        )
    if dropped:
        logger.info("formatter dropped %d examples", dropped)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def run_sft(config_path: str | Path = "config/training.yaml") -> str:
    """Run SFT and return the output directory containing the LoRA adapter."""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
    )
    # SFTConfig / SFTTrainer / datasets / pyarrow.dataset are imported at
    # module level above (Windows DLL load-order workaround).
    from qwen_tutor.training.model_loader import load_base_model_for_training

    cfg = load_training_config(config_path)
    base_cfg = cfg["base_model"]
    sft_cfg = cfg["sft"]
    lora_cfg = cfg["lora"]
    cb_cfg = cfg.get("eval_callback") or {}

    logger.info("loading tokenizer from %s", base_cfg["tokenizer_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg["tokenizer_id"],
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatter = ChatFormatter(
        tokenizer=tokenizer,
        max_seq_length=sft_cfg["max_seq_length"],
    )

    # ---- data --------------------------------------------------------
    data_cfg = sft_cfg["data"]
    sft_examples = load_sft_examples(data_cfg["sft_filtered_dir"])
    eval_examples = load_evaluation_examples(data_cfg["eval_filtered_dir"])
    logger.info(
        "loaded %d SFT + %d Evaluation examples (pre-mix)",
        len(sft_examples), len(eval_examples),
    )
    train_examples, val_examples = mix_and_split(
        sft_examples=sft_examples,
        eval_examples=eval_examples,
        mix_ratio_sft=data_cfg["mix_ratio_sft"],
        mix_ratio_eval=data_cfg["mix_ratio_eval"],
        validation_split=data_cfg["validation_split"],
        seed=data_cfg["shuffle_seed"],
    )
    logger.info("train=%d  val=%d", len(train_examples), len(val_examples))

    train_ds = _build_hf_dataset(train_examples, formatter, sft_cfg["max_seq_length"])
    val_ds = _build_hf_dataset(val_examples, formatter, sft_cfg["max_seq_length"])

    # ---- base model ---------------------------------------------------
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

    model = load_base_model_for_training(
        base_cfg["model_id"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, base_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=base_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )
    if base_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- trainer ------------------------------------------------------
    sft_trainer_cfg = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        num_train_epochs=sft_cfg["num_train_epochs"],
        max_steps=sft_cfg.get("max_steps", -1),
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=sft_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        weight_decay=sft_cfg["weight_decay"],
        max_length=sft_cfg["max_seq_length"],
        packing=sft_cfg["packing"],
        bf16=sft_cfg["bf16"],
        fp16=sft_cfg["fp16"],
        optim=sft_cfg["optim"],
        logging_steps=sft_cfg["logging_steps"],
        eval_strategy=sft_cfg.get("eval_strategy", "no"),
        eval_steps=sft_cfg.get("eval_steps", 200),
        save_strategy=sft_cfg.get("save_strategy", "no"),
        save_steps=sft_cfg.get("save_steps", 200),
        save_total_limit=sft_cfg.get("save_total_limit", 1),
        load_best_model_at_end=sft_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=sft_cfg.get("metric_for_best_model"),
        greater_is_better=sft_cfg.get("greater_is_better", False),
        report_to=sft_cfg.get("report_to", "none"),
        seed=sft_cfg["seed"],
        dataset_text_field=None,  # we pass pre-tokenized rows
    )

    # We pre-tokenize and pre-mask `labels` ourselves (assistant-only spans),
    # so we need a collator that PADS labels too (with -100). The standard
    # DataCollatorForLanguageModeling generates labels from input_ids and
    # cannot pad pre-existing label sequences, leading to "excessive nesting"
    # errors when batch examples have different lengths.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=LOSS_IGNORE_INDEX,
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

    trainer = SFTTrainer(
        model=model,
        args=sft_trainer_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(sft_cfg["output_dir"])
    tokenizer.save_pretrained(sft_cfg["output_dir"])
    logger.info("SFT complete; adapter saved to %s", sft_cfg["output_dir"])
    return sft_cfg["output_dir"]
