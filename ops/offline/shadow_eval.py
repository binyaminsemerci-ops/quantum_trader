"""shadow_eval.py — Post-training shadow evaluation on Binance testnet.

After training produces dpo_run_output/ (LoRA adapter), load the fine-tuned
model and compare its action distribution against:
  A) the original llama-3.3-70b-versatile Groq inference (current live model)
  B) the formula engine baseline

This script does NOT write to Redis, does NOT place any trades, and does NOT
modify the running service.  It is pure offline analysis over the DPO validation
set.

Usage
-----
  python shadow_eval.py \
      --adapter dpo_run_output \
      --val     dpo_formatted/dpo_val.jsonl \
      --base    meta-llama/Meta-Llama-3.1-8B-Instruct

Outputs
-------
  shadow_eval_results.json    — full comparison table
  (prints go/no-go recommendation to stdout)
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Optional

import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def _load_model(base: str, adapter: Optional[str]):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.pad_token = tok.eos_token
    # Use GPU (RTX 3060 / 12 GB VRAM) — 3B float16 ≈ 6 GB, fits comfortably.
    # device_map="auto" sends everything to CUDA if available, else falls back to CPU.
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    gen = pipeline("text-generation", model=model, tokenizer=tok,
                   max_new_tokens=60, do_sample=False,
                   return_full_text=False)
    return gen


def _free_model(gen) -> None:
    """Explicitly del model and free GPU memory between passes."""
    del gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _infer(gen, prompt: str) -> str:
    with torch.no_grad():
        out = gen(prompt, return_full_text=False)
    result = (out[0]["generated_text"] or "").strip()
    gc.collect()          # per-sample cleanup to avoid memory creep
    return result


def _parse_action(raw: str) -> str:
    # Model outputs valid JSON first, then may append <|...|> tokens — strip them.
    clean = raw.split("<|")[0].strip()
    try:
        return json.loads(clean).get("action", "PARSE_ERROR")
    except (json.JSONDecodeError, ValueError):
        return "PARSE_ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="dpo_run_output")
    ap.add_argument("--val",     default="dpo_formatted/dpo_val.jsonl")
    ap.add_argument("--base",    default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    val_pairs = [json.loads(l) for l in pathlib.Path(args.val).open() if l.strip()]
    print(f"Loaded {len(val_pairs)} validation pairs")

    base_ckpt_path  = pathlib.Path("shadow_eval_base_checkpoint.json")
    tuned_ckpt_path = pathlib.Path("shadow_eval_tuned_checkpoint.json")

    # ── Pass 1: base model (no adapter) ──────────────────────────────────────
    if base_ckpt_path.exists():
        print("\n[Pass 1/2] Checkpoint found — skipping base inference …")
        base_ckpt = json.loads(base_ckpt_path.read_text())
        base_raw_outputs = base_ckpt["outputs"]
        preferred_list   = base_ckpt["preferred"]
        prompts          = base_ckpt["prompts"]
    else:
        print("\n[Pass 1/2] Loading BASE model (no adapter) …")
        base_gen = _load_model(args.base, None)
        base_raw_outputs = []
        preferred_list   = []
        prompts          = []
        for i, row in enumerate(val_pairs):
            preferred_list.append(json.loads(row["chosen"]).get("action", ""))
            prompts.append(row["prompt"])
            raw = _infer(base_gen, row["prompt"])
            base_raw_outputs.append(raw)
            if i % 10 == 0:
                print(f"  base [{i:>3}/{len(val_pairs)}]")
        print(f"  base [{len(val_pairs):>3}/{len(val_pairs)}]")
        print("[Pass 1/2] Done — freeing memory …")
        _free_model(base_gen)
        # Save checkpoint so Pass 1 is never re-run if Pass 2 OOMs
        base_ckpt_path.write_text(json.dumps({
            "outputs":   base_raw_outputs,
            "preferred": preferred_list,
            "prompts":   prompts,
        }))
        print(f"[Pass 1/2] Checkpoint saved -> {base_ckpt_path}")

    # ── Pass 2: DPO-tuned model (with adapter) ────────────────────────────────
    # Resume from partial checkpoint if available
    if tuned_ckpt_path.exists():
        tuned_ckpt = json.loads(tuned_ckpt_path.read_text())
        tuned_raw_outputs = tuned_ckpt["outputs"]
        resume_from = len(tuned_raw_outputs)
        print(f"\n[Pass 2/2] Partial checkpoint found — resuming from [{resume_from}/{len(val_pairs)}] …")
    else:
        tuned_raw_outputs = []
        resume_from = 0

    if resume_from < len(val_pairs):
        print("\n[Pass 2/2] Loading DPO-tuned model (with adapter) …")
        tuned_gen = _load_model(args.base, args.adapter)
        for i, row in enumerate(val_pairs):
            if i < resume_from:
                continue
            raw = _infer(tuned_gen, row["prompt"])
            tuned_raw_outputs.append(raw)
            if i % 10 == 0:
                print(f"  tuned [{i:>3}/{len(val_pairs)}]")
                # Checkpoint every 10 samples
                tuned_ckpt_path.write_text(json.dumps({"outputs": tuned_raw_outputs}))
        print(f"  tuned [{len(val_pairs):>3}/{len(val_pairs)}]")
        print("[Pass 2/2] Done — freeing memory …")
        _free_model(tuned_gen)
        tuned_ckpt_path.write_text(json.dumps({"outputs": tuned_raw_outputs}))
        print(f"[Pass 2/2] Checkpoint saved -> {tuned_ckpt_path}")
    else:
        print("\n[Pass 2/2] Checkpoint complete — skipping tuned inference …")

    # ── Collate results ───────────────────────────────────────────────────────
    results = []
    base_correct = tuned_correct = 0
    for i, (preferred, base_raw, tuned_raw) in enumerate(
            zip(preferred_list, base_raw_outputs, tuned_raw_outputs)):
        base_action  = _parse_action(base_raw)
        tuned_action = _parse_action(tuned_raw)
        bc = base_action  == preferred
        tc = tuned_action == preferred
        base_correct  += int(bc)
        tuned_correct += int(tc)
        results.append({
            "idx":           i,
            "preferred":     preferred,
            "base_action":   base_action,
            "tuned_action":  tuned_action,
            "base_correct":  bc,
            "tuned_correct": tc,
        })

    n  = len(val_pairs)
    ba = base_correct  / n
    ta = tuned_correct / n
    lift = ta - ba

    print("\n── Shadow Evaluation Results ─────────────────────────")
    print(f"  n validation pairs   : {n}")
    print(f"  Base model accuracy  : {ba:.3f}  ({base_correct}/{n})")
    print(f"  Tuned model accuracy : {ta:.3f}  ({tuned_correct}/{n})")
    print(f"  Accuracy lift        : {lift:+.3f}")

    base_dist  = Counter(r["base_action"]  for r in results)
    tuned_dist = Counter(r["tuned_action"] for r in results)
    pref_dist  = Counter(r["preferred"]    for r in results)

    print("\n  Action distribution (preferred / base / tuned):")
    all_actions = sorted(set(list(base_dist) + list(tuned_dist) + list(pref_dist)))
    for a in all_actions:
        print(f"    {a:<22} pref={pref_dist[a]:>3}  base={base_dist[a]:>3}  tuned={tuned_dist[a]:>3}")

    # Parse-error rate
    pe_base  = base_dist.get("PARSE_ERROR", 0) / n
    pe_tuned = tuned_dist.get("PARSE_ERROR", 0) / n
    print(f"\n  Parse-error rate  base={pe_base:.3f}  tuned={pe_tuned:.3f}")

    # ── Go / No-Go decision ────────────────────────────────────────────────────
    print("\n── Go / No-Go ────────────────────────────────────────")
    issues = []
    if ta < 0.40:
        issues.append(f"FAIL: tuned accuracy {ta:.3f} < 0.40 minimum")
    if lift < 0.05:
        issues.append(f"FAIL: accuracy lift {lift:+.3f} < +0.05 minimum")
    if pe_tuned > 0.05:
        issues.append(f"FAIL: parse-error rate {pe_tuned:.3f} > 0.05")
    if tuned_dist.get("PARSE_ERROR", 0) > 2:
        issues.append(f"WARN: {tuned_dist['PARSE_ERROR']} parse errors in {n} inferences")

    if not issues:
        verdict = "GO — tuned model improves action accuracy; proceed to testnet shadow deployment"
    else:
        verdict = "NO-GO — do not deploy;\n  " + "\n  ".join(issues)

    print(f"  {verdict}")

    out = {
        "n": n,
        "base_accuracy": ba,
        "tuned_accuracy": ta,
        "lift": lift,
        "base_dist": dict(base_dist),
        "tuned_dist": dict(tuned_dist),
        "preferred_dist": dict(pref_dist),
        "parse_error_base": pe_base,
        "parse_error_tuned": pe_tuned,
        "verdict": verdict,
        "per_pair": results,
    }
    with pathlib.Path("shadow_eval_results.json").open("w") as f:
        json.dump(out, f, indent=2)
    print("\n  Full results -> shadow_eval_results.json")


if __name__ == "__main__":
    main()
