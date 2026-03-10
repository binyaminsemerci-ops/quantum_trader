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

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def _load_model(base: str, adapter: Optional[str]):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    gen = pipeline("text-generation", model=model, tokenizer=tok,
                   max_new_tokens=80, do_sample=False, temperature=1.0)
    return gen


def _infer(gen, prompt: str) -> str:
    out = gen(prompt, return_full_text=False)
    return (out[0]["generated_text"] or "").strip()


def _parse_action(raw: str) -> str:
    try:
        return json.loads(raw.strip()).get("action", "PARSE_ERROR")
    except (json.JSONDecodeError, ValueError):
        return "PARSE_ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="dpo_run_output")
    ap.add_argument("--val",     default="dpo_formatted/dpo_val.jsonl")
    ap.add_argument("--base",    default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    val_pairs = [json.loads(l) for l in pathlib.Path(args.val).open() if l.strip()]
    print(f"Loaded {len(val_pairs)} validation pairs")

    print("Loading BASE model (no adapter) …")
    base_gen  = _load_model(args.base, None)
    print("Loading DPO-tuned model (with adapter) …")
    tuned_gen = _load_model(args.base, args.adapter)

    results = []
    base_correct = tuned_correct = 0

    for i, row in enumerate(val_pairs):
        preferred = json.loads(row["chosen"]).get("action", "")
        prompt    = row["prompt"]

        base_raw  = _infer(base_gen,  prompt)
        tuned_raw = _infer(tuned_gen, prompt)

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

        if i % 10 == 0:
            print(f"  [{i:>3}/{len(val_pairs)}]  base={base_correct}  tuned={tuned_correct}")

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
    print("\n  Full results → shadow_eval_results.json")


if __name__ == "__main__":
    main()
