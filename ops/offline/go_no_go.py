"""go_no_go.py — Go/No-Go gate for keeping a DPO-trained adapter.

Reads trainer_state.json (from HuggingFace Trainer) and
shadow_eval_results.json (from shadow_eval.py), applies all
pass/fail criteria, and exits 0 (GO) or 1 (NO-GO).

Usage
-----
  python go_no_go.py \
      --trainer-state dpo_run_output/trainer_state.json \
      --shadow-eval   shadow_eval_results.json \
      [--strict]        # strict mode: no warnings allowed

Exit codes
----------
  0  — GO: all criteria pass
  1  — NO-GO: one or more criteria fail
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List


# ── Pass/Fail thresholds (edit here only) ─────────────────────────────────────
THRESHOLDS = {
    # Training convergence
    "min_loss_reduction":     0.05,    # final_train_loss must be < initial - this
    "max_val_loss_ratio":     1.25,    # val_loss at end ≤ 1.25× best_val_loss
    "min_grad_norm_decrease": False,   # not enforced in first dry run

    # Shadow eval accuracy
    "min_tuned_accuracy":     0.40,    # DPO model correct on ≥40 % of val pairs
    "min_accuracy_lift":      0.05,    # DPO vs base ≥ +5 pp
    "max_parse_error_rate":   0.05,    # JSON parse errors ≤ 5 %

    # Action diversity (prevents degenerate collapse)
    "max_single_action_share":  0.80,  # no action >80 % of all DPO outputs
    "forbidden_time_stop_share": 0.20, # TIME_STOP_EXIT ≤ 20 % of outputs
}


@dataclass
class Check:
    name:    str
    passed:  bool
    detail:  str
    fatal:   bool = True


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_training(ts: dict) -> List[Check]:
    checks: List[Check] = []
    log = ts.get("log_history", [])

    # --- Extract train loss trajectory ---
    train_steps = [e for e in log if "loss" in e and "eval_loss" not in e]
    if not train_steps:
        checks.append(Check("train_loss_records", False,
                            "no training loss entries in trainer_state.json", fatal=True))
        return checks

    init_loss  = train_steps[0]["loss"]
    final_loss = train_steps[-1]["loss"]
    reduction  = init_loss - final_loss
    thresh     = THRESHOLDS["min_loss_reduction"]
    checks.append(Check(
        "train_loss_reduction",
        passed=reduction >= thresh,
        detail=f"initial={init_loss:.4f} final={final_loss:.4f} "
               f"reduction={reduction:.4f} (min={thresh})",
    ))

    # --- Monotonicity in first epoch ---
    first_epoch = [e for e in train_steps if e.get("epoch", 1) < 1.05]
    if len(first_epoch) >= 3:
        losses = [e["loss"] for e in first_epoch]
        # allow one uptick — but final half must be lower than first half
        mid = len(losses) // 2
        first_half_mean = sum(losses[:mid]) / mid
        second_half_mean = sum(losses[mid:]) / len(losses[mid:])
        checks.append(Check(
            "epoch1_loss_decreasing",
            passed=second_half_mean < first_half_mean,
            detail=f"first_half_mean={first_half_mean:.4f} "
                   f"second_half_mean={second_half_mean:.4f}",
        ))

    # --- Val loss stability ---
    eval_steps = [e for e in log if "eval_loss" in e]
    if eval_steps:
        best_val  = min(e["eval_loss"] for e in eval_steps)
        final_val = eval_steps[-1]["eval_loss"]
        ratio     = final_val / best_val if best_val > 0 else 999
        thresh_r  = THRESHOLDS["max_val_loss_ratio"]
        checks.append(Check(
            "val_loss_stability",
            passed=ratio <= thresh_r,
            detail=f"best_val={best_val:.4f} final_val={final_val:.4f} "
                   f"ratio={ratio:.2f} (max={thresh_r})",
        ))
    else:
        checks.append(Check("val_loss_stability", False,
                            "no eval_loss entries in trainer_state.json (was eval enabled?)",
                            fatal=False))

    return checks


def check_shadow(se: dict) -> List[Check]:
    checks: List[Check] = []

    ta   = se["tuned_accuracy"]
    ba   = se["base_accuracy"]
    lift = se["lift"]
    pe   = se["parse_error_tuned"]
    dist = se["tuned_dist"]
    n    = se["n"]

    # --- Minimum accuracy ---
    mt = THRESHOLDS["min_tuned_accuracy"]
    checks.append(Check(
        "min_tuned_accuracy",
        passed=ta >= mt,
        detail=f"tuned_accuracy={ta:.3f} (min={mt})",
    ))

    # --- Accuracy lift vs base ---
    ml = THRESHOLDS["min_accuracy_lift"]
    checks.append(Check(
        "min_accuracy_lift",
        passed=lift >= ml,
        detail=f"lift={lift:+.3f} (min={ml:+.3f})  base={ba:.3f} tuned={ta:.3f}",
    ))

    # --- Parse error rate ---
    mp = THRESHOLDS["max_parse_error_rate"]
    checks.append(Check(
        "max_parse_error_rate",
        passed=pe <= mp,
        detail=f"parse_error_tuned={pe:.3f} (max={mp})",
    ))

    # --- No single action dominates ---
    total_outputs = sum(dist.values())
    if total_outputs > 0:
        ms = THRESHOLDS["max_single_action_share"]
        worst_action = max(dist, key=dist.get)
        worst_share  = dist[worst_action] / total_outputs
        checks.append(Check(
            "action_diversity",
            passed=worst_share <= ms,
            detail=f"dominant_action={worst_action} share={worst_share:.3f} (max={ms})",
        ))

    # --- TIME_STOP_EXIT not overrepresented ---
    ts_share = dist.get("TIME_STOP_EXIT", 0) / total_outputs if total_outputs else 0
    mt2 = THRESHOLDS["forbidden_time_stop_share"]
    checks.append(Check(
        "time_stop_exit_share",
        passed=ts_share <= mt2,
        detail=f"TIME_STOP_EXIT share={ts_share:.3f} (max={mt2})",
        fatal=False,
    ))

    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-state",
                    default="dpo_run_output/trainer_state.json")
    ap.add_argument("--shadow-eval",
                    default="shadow_eval_results.json")
    ap.add_argument("--strict", action="store_true",
                    help="treat non-fatal warnings as failures")
    args = ap.parse_args()

    all_checks: List[Check] = []

    # --- Training convergence ---
    try:
        ts = load(args.trainer_state)
        print(f"\nLoaded trainer_state: {len(ts.get('log_history', []))} log entries")
        all_checks += check_training(ts)
    except FileNotFoundError:
        print(f"WARNING: {args.trainer_state} not found — skipping training checks")

    # --- Shadow evaluation ---
    try:
        se = load(args.shadow_eval)
        print(f"Loaded shadow_eval: n={se.get('n')} pairs")
        all_checks += check_shadow(se)
    except FileNotFoundError:
        print(f"WARNING: {args.shadow_eval} not found — skipping shadow eval checks")

    if not all_checks:
        print("ERROR: no checks could be run — missing input files")
        sys.exit(1)

    # --- Print results --------------------------------------------------------
    print("\n── Go / No-Go Evaluation ─────────────────────────────")
    failures = 0
    for c in all_checks:
        severity = "FATAL " if c.fatal else "WARN  "
        symbol   = "✓ PASS" if c.passed else f"✗ FAIL ({severity})"
        print(f"  [{symbol}] {c.name:<30}  {c.detail}")
        if not c.passed and (c.fatal or args.strict):
            failures += 1

    print("\n── Verdict ───────────────────────────────────────────")
    if failures == 0:
        print("  ✅  GO — adapter passes all criteria")
        print("  Next step: copy dpo_run_output/ to VPS, load via Ollama or vllm")
        print("  and enable EXIT_AGENT_QWEN3_SHADOW=true on testnet\n")
        sys.exit(0)
    else:
        print(f"  ❌  NO-GO — {failures} fatal check(s) failed")
        print("  Do NOT deploy.  Review training logs and consider:\n"
              "    - increasing NUM_EPOCHS (try 3-4)\n"
              "    - reducing DPO_BETA (try 0.1)\n"
              "    - expanding the dataset (need ≥500 pairs)\n"
              "    - checking the system prompt match in format_dpo_dataset.py\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
