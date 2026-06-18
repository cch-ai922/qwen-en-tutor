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
    """Iterate only filtered records matching the ``prefix_*_passed.jsonl`` pattern."""
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


def _load_dpo_by_source(
    directory: str | Path,
) -> tuple[list[DPOExample], list[DPOExample], list[DPOExample]]:
    """Load register, on_policy, and sentinel pairs separately from
    ``data/dpo_filtered/``.

    The sentinel pool covers both offline (strip-marker) and on-policy
    (regen) sentinel pairs emitted by ``sentinel_pairs.py``; they share
    the ``sentinel_<level>.jsonl`` filename and are distinguished only
    by id prefix (``dpo_sentinel_offline_`` vs ``dpo_sentinel_onpolicy_``).
    """
    d = Path(directory)
    register: list[DPOExample] = []
    on_policy: list[DPOExample] = []
    sentinel: list[DPOExample] = []
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
    for rec in _iter_filtered_records_by_prefix(d, "sentinel"):
        try:
            sentinel.append(DPOExample.model_validate(rec))
        except Exception as exc:  # noqa: BLE001
            logger.warning("dpo loader: dropping malformed sentinel record: %s", exc)
    return register, on_policy, sentinel


def load_dpo_examples(directory: str | Path) -> list[DPOExample]:
    """Backward-compatible helper: return all three pools concatenated."""
    register, on_policy, sentinel = _load_dpo_by_source(directory)
    return register + on_policy + sentinel


def _mix_dpo_pools(
    register: list[DPOExample],
    on_policy: list[DPOExample],
    sentinel: list[DPOExample],
    mix_ratio_register: float,
    mix_ratio_on_policy: float,
    mix_ratio_sentinel: float,
    seed: int,
    use_all_data: bool = True,
) -> list[DPOExample]:
    """Combine the register, on_policy, and sentinel pools.

    Default behavior (``use_all_data=True``): skip the ratio-based
    trim entirely and return ``register + on_policy + sentinel``
    unchanged. The mix ratios are still required (sanity-checked, must
    sum > 0) but have no size effect — they exist so a ratio-aware
    trim is available without changing config schema if the operator
    flips ``use_all_data=False``.

    When ``use_all_data=False``: scale every pool so its share of the
    combined dataset matches its target ratio. The smallest pool that
    cannot reach its target via shrinking the others sets the overall
    size; larger pools are subsampled to fit. The sentinel pool is
    typically the smallest (it has at most one pair per filtered
    ``persistent_*`` record), so a low ``mix_ratio_sentinel`` is
    expected — e.g. 0.05–0.10 — so the trimmer does not crater the
    other pools to match it.
    """
    rng = random.Random(seed)
    rng.shuffle(register)
    rng.shuffle(on_policy)
    rng.shuffle(sentinel)

    r_count = len(register)
    p_count = len(on_policy)
    s_count = len(sentinel)
    total_pre = r_count + p_count + s_count

    targets = {
        "register": mix_ratio_register,
        "on_policy": mix_ratio_on_policy,
        "sentinel": mix_ratio_sentinel,
    }
    if sum(targets.values()) <= 0:
        raise ValueError("dpo mix ratios sum to zero")

    if use_all_data:
        logger.info(
            "dpo mix: use_all_data=True -- skipping ratio trim, "
            "keeping all %d register + %d on_policy + %d sentinel examples "
            "(natural ratio %.1f%% / %.1f%% / %.1f%%)",
            r_count, p_count, s_count,
            100.0 * r_count / max(1, total_pre),
            100.0 * p_count / max(1, total_pre),
            100.0 * s_count / max(1, total_pre),
        )
        return register + on_policy + sentinel

    # Ratio-trim mode: scale each pool so its share matches its target.
    # The smallest "binding" pool (where current / target is minimum) sets
    # the overall scale; the others are trimmed down to match.
    pools = {"register": register, "on_policy": on_policy, "sentinel": sentinel}
    counts = {k: len(v) for k, v in pools.items()}
    total_target = sum(targets.values())
    norm_targets = {k: t / total_target for k, t in targets.items()}

    # Each non-empty pool implies a feasible total: counts[k] / norm_targets[k].
    feasible_totals = [
        counts[k] / norm_targets[k]
        for k in pools
        if norm_targets[k] > 0 and counts[k] > 0
    ]
    if not feasible_totals:
        return []
    target_total = min(feasible_totals)
    keeps = {k: int(target_total * norm_targets[k]) for k in pools}

    for k in pools:
        if keeps[k] < counts[k]:
            dropped = counts[k] - keeps[k]
            pools[k] = pools[k][: keeps[k]]
            if dropped > 0:
                logger.warning(
                    "dpo mix: trimmed %d %s examples (%d -> %d) to match "
                    "ratio %.2f. Set dpo.data.use_all_data=true to disable.",
                    dropped, k, counts[k], keeps[k], norm_targets[k],
                )

    logger.info(
        "dpo mix: register=%d on_policy=%d sentinel=%d  "
        "targets=%.2f/%.2f/%.2f",
        len(pools["register"]), len(pools["on_policy"]), len(pools["sentinel"]),
        norm_targets["register"], norm_targets["on_policy"], norm_targets["sentinel"],
    )
    return pools["register"] + pools["on_policy"] + pools["sentinel"]


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

    # Build the chat messages with the SCENARIO-AWARE system prompt and
    # /no_think tag on the first user turn (DPO pairs come from
    # conversation-mode SFT examples). The scenario template includes
    # topic / subtopics / user_role / model_role from metadata so the model
    # sees the same prompt at DPO training as at SFT training and at
    # deploy -- otherwise DPO would train against a generic "patient
    # tutor" frame even though the chosen/rejected contrast is about
    # staying on the scenario topic.
    system_content = formatter._render_scenario_deployment_system_prompt(
        ex.metadata
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
    register, on_policy, sentinel = _load_dpo_by_source(data_cfg["dpo_filtered_dir"])
    logger.info(
        "loaded DPO: %d register + %d on_policy + %d sentinel examples (pre-mix)",
        len(register), len(on_policy), len(sentinel),
    )
    examples = _mix_dpo_pools(
        register=register,
        on_policy=on_policy,
        sentinel=sentinel,
        mix_ratio_register=float(data_cfg.get("mix_ratio_register", 0.65)),
        mix_ratio_on_policy=float(data_cfg.get("mix_ratio_on_policy", 0.25)),
        mix_ratio_sentinel=float(data_cfg.get("mix_ratio_sentinel", 0.10)),
        seed=data_cfg["shuffle_seed"],
        use_all_data=bool(data_cfg.get("use_all_data", True)),
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
        # Force PrinterCallback (vs default ProgressCallback) so the loss
        # dict is written to stdout every `logging_steps` instead of only
        # into a tqdm postfix that gets eaten by carriage-return overwrites.
        # Same fix as sft.py — keeps the training observable.
        disable_tqdm=dpo_cfg.get("disable_tqdm", True),
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
