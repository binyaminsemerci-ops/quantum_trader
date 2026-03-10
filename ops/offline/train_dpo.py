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

# ── HuggingFace authentication ────────────────────────────────────────────────
# Llama-3.2 / 3.1 are gated repos. You must:
#   1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
#   2. Get a token at https://huggingface.co/settings/tokens
#   3. Set the token one of two ways:
#      PowerShell: $env:HF_TOKEN = "hf_xxx..."
#      Once/permanent: huggingface-cli login
def _hf_login() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[train_dpo] HF_TOKEN found — logged in to HuggingFace")
    else:
        # Check if already logged in (from huggingface-cli login)
        try:
            from huggingface_hub import whoami
            user = whoami()
            print(f"[train_dpo] Already authenticated as: {user['name']}")
        except Exception:
            # Qwen2.5 models are Apache 2.0 — no license gate required.
            # HF token is optional but recommended to avoid rate limits.
            print("[train_dpo] No HF token found — continuing without auth (Qwen2.5 is open)")

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_FILE   = pathlib.Path("dpo_formatted/dpo_train.jsonl")
VAL_FILE     = pathlib.Path("dpo_formatted/dpo_val.jsonl")
OUTPUT_DIR   = pathlib.Path("dpo_run_output")
LOG_DIR      = OUTPUT_DIR / "logs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Model — auto-selected by available VRAM ───────────────────────────────────
# The exit agent uses llama-3.3-70b-versatile via Groq API (not downloadable).
# We train a local proxy model on the same DPO dataset to validate the pipeline.
#
#   < 8 GB VRAM  →  Qwen/Qwen2.5-3B-Instruct   (Apache 2.0, no license gate)
#   ≥ 8 GB VRAM  →  Qwen/Qwen2.5-7B-Instruct   (Apache 2.0, no license gate)
#
# Override with:  BASE_MODEL=Qwen/... python train_dpo.py
def _auto_select_model() -> str:
    env = os.getenv("BASE_MODEL")
    if env:
        return env
    try:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 8.0:
            print(f"[train_dpo] VRAM={vram_gb:.1f}GB < 8 GB → using Qwen2.5-3B proxy model")
            return "Qwen/Qwen2.5-3B-Instruct"
    except Exception:
        pass
    return "Qwen/Qwen2.5-7B-Instruct"

BASE_MODEL = _auto_select_model()

# ── Hyperparameters — auto-tuned by VRAM ──────────────────────────────────────
def _auto_config() -> dict:
    try:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        vram_gb = 99.0
    if vram_gb < 8.0:          # 6 GB (RTX 3060 Laptop, etc.)
        return dict(batch=1, accum=16, seq=768, prompt=512)
    if vram_gb < 20.0:         # 8–16 GB (T4, RTX 3080, etc.)
        return dict(batch=1, accum=16, seq=1024, prompt=700)
    return dict(batch=4, accum=4, seq=1024, prompt=700)  # 24 GB+

_cfg             = _auto_config()
LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
DPO_BETA         = 0.5
LEARNING_RATE    = 5e-5
BATCH_SIZE       = _cfg["batch"]
GRAD_ACCUM       = _cfg["accum"]
NUM_EPOCHS       = 2
MAX_SEQ_LEN      = _cfg["seq"]
MAX_PROMPT_LEN   = _cfg["prompt"]   # must be > system_prompt tokens (~300)
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
EVAL_STEPS       = 20
LOGGING_STEPS    = 5
SAVE_STEPS       = 40


def load_jsonl(path: pathlib.Path) -> Dataset:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return Dataset.from_list(rows)


def main():
    _hf_login()
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
    # Gradient checkpointing: required on <24 GB VRAM to trade compute for memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
        processing_class=tokenizer,
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
