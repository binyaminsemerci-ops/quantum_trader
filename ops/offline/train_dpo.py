"""train_dpo.py — First experimental DPO training run (PATCH-10A).

Base model : meta-llama/Meta-Llama-3.1-8B-Instruct
Method     : DPO (Direct Preference Optimization) via HuggingFace TRL
LoRA rank  : 8
Beta       : 0.5  (conservative — prevents reward collapse)
Epochs     : 2
Purpose    : Experimental only. Output NOT promoted to production without
             shadow evaluation on Binance testnet (see eval_shadow_checklist.md)

Hardware requirements
---------------------
  GPU  : ≥24 GB VRAM  (e.g. A10G, 3090, 4090, A100-40G)
  RAM  : ≥32 GB system RAM
  Disk : ≥60 GB free

Setup
-----
  pip install -r requirements_train.txt
  python format_dpo_dataset.py          # produces dpo_formatted/
  python train_dpo.py

Outputs (written to ./dpo_run_output/)
-------
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
  trainer_state.json
  eval_results.json
"""
from __future__ import annotations

import json
import os
import pathlib

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_FILE   = pathlib.Path("dpo_formatted/dpo_train.jsonl")
VAL_FILE     = pathlib.Path("dpo_formatted/dpo_val.jsonl")
OUTPUT_DIR   = pathlib.Path("dpo_run_output")
LOG_DIR      = OUTPUT_DIR / "logs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
# First run: 8B proxy model.
# The exit agent uses llama-3.3-70b-versatile via Groq API (not downloadable).
# Training on the 8B instruct validates the pipeline and learns the three
# correction patterns.  If results are good, promote to 70B via Together.ai API.
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

# ── Hyperparameters ────────────────────────────────────────────────────────────
LORA_RANK        = 8          # conservative — low forgetting risk
LORA_ALPHA       = 16         # = 2 * rank  (standard scaling)
LORA_DROPOUT     = 0.05
DPO_BETA         = 0.5        # KL penalty; 0.1 = aggressive, 0.5 = conservative
LEARNING_RATE    = 5e-5       # lower than SFT default; DPO is sensitive
BATCH_SIZE       = 4          # per-device
GRAD_ACCUM       = 4          # effective batch = 16
NUM_EPOCHS       = 2          # small dataset → 2 epochs max before overfit
# System prompt alone is ~300 tokens; keep prompt budget at 700 to avoid
# silent truncation that corrupts the action-constraint instructions.
MAX_SEQ_LEN      = 1024       # full sequence (prompt + response)
MAX_PROMPT_LEN   = 700        # must be > system_prompt tokens (~300)
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
EVAL_STEPS       = 20         # evaluate every 20 steps (~every 12.5% of epoch)
LOGGING_STEPS    = 5
SAVE_STEPS       = 40         # save checkpoint twice per epoch


def load_jsonl(path: pathlib.Path) -> Dataset:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return Dataset.from_list(rows)


def main():
    print(f"[train_dpo] Loading base model: {BASE_MODEL}")
    print(f"[train_dpo] LoRA rank={LORA_RANK}  beta={DPO_BETA}  epochs={NUM_EPOCHS}")

    # ── Quantisation (4-bit QLoRA for <24GB VRAM cards) ──────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for DPO

    # Use flash_attention_2 when available; fall back to eager otherwise.
    _attn_kwargs: dict = {}
    try:
        import importlib
        importlib.import_module("flash_attn")
        _attn_kwargs["attn_implementation"] = "flash_attention_2"
        print("[train_dpo] flash_attn detected — using flash_attention_2")
    except ImportError:
        print("[train_dpo] flash_attn not found — using default attention (eager)")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        **_attn_kwargs,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ── LoRA config ────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # Target the attention + MLP projection layers for Llama3
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_ds = load_jsonl(TRAIN_FILE)
    val_ds   = load_jsonl(VAL_FILE)
    print(f"[train_dpo] train={len(train_ds)} val={len(val_ds)}")

    # ── DPO training config ────────────────────────────────────────────────────
    dpo_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        beta=DPO_BETA,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_dir=str(LOG_DIR),
        logging_steps=LOGGING_STEPS,
        fp16=False,
        bf16=True,
        report_to="none",         # change to "wandb" if wandb is configured
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        # Safety: abort if val_loss > 1.2× step-0 val_loss (collapse guard)
        # (manual monitoring in watch_training.py — TRL lacks built-in abort)
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,        # None = use frozen copy of the initial model
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
    )

    print("[train_dpo] Starting training …")
    trainer.train()

    print(f"[train_dpo] Saving final adapter → {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ── Final eval ─────────────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("[train_dpo] Final eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    with (OUTPUT_DIR / "eval_results.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("[train_dpo] Done.")


if __name__ == "__main__":
    main()
