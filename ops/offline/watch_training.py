"""watch_training.py — Monitor training loop, detect loss collapse, and enforce
stop-loss on val_loss.

Run in a separate terminal alongside train_dpo.py:
    python watch_training.py

Reads trainer_state.json from dpo_run_output/ every 30 seconds and:
- prints the current train_loss and eval_loss curve
- flags if eval_loss > 1.25× the step-0 baseline (collapse warning)
- prints go/no-go recommendation after training completes
"""
from __future__ import annotations

import json
import pathlib
import time

STATE_FILE = pathlib.Path("dpo_run_output/trainer_state.json")
POLL_SEC = 30
COLLAPSE_FACTOR = 1.25   # eval_loss > 1.25× initial is a collapse signal

def _read_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def main():
    print("Watching training state… (Ctrl+C to stop)")
    baseline_eval_loss = None
    seen_steps = set()

    while True:
        state = _read_state()
        if state is None:
            print(f"  [{time.strftime('%H:%M:%S')}] waiting for trainer_state.json…")
            time.sleep(POLL_SEC)
            continue

        log_history = state.get("log_history", [])
        new_entries = [e for e in log_history if e.get("step") not in seen_steps]

        for entry in new_entries:
            step  = entry.get("step", "?")
            seen_steps.add(step)

            train_loss = entry.get("loss")
            eval_loss  = entry.get("eval_loss")

            line_parts = [f"step={step:>4}"]
            if train_loss is not None:
                line_parts.append(f"train_loss={train_loss:.4f}")
            if eval_loss is not None:
                if baseline_eval_loss is None:
                    baseline_eval_loss = eval_loss
                    line_parts.append(f"eval_loss={eval_loss:.4f}  [baseline set]")
                else:
                    ratio = eval_loss / baseline_eval_loss
                    flag  = "  ⚠ COLLAPSE RISK" if ratio > COLLAPSE_FACTOR else ""
                    line_parts.append(f"eval_loss={eval_loss:.4f}  ratio={ratio:.3f}{flag}")
            print("  " + "  |  ".join(line_parts))

        # Check if training is done
        if state.get("is_world_process_zero") is not None:
            best_metric = state.get("best_metric")
            best_step   = state.get("best_model_checkpoint", "?")
            if best_metric is not None and not new_entries:
                print(f"\nTraining complete.")
                print(f"  Best eval_loss : {best_metric:.4f}")
                print(f"  Best checkpoint: {best_step}")
                if baseline_eval_loss and best_metric <= baseline_eval_loss:
                    print("  GO: val_loss improved — proceed to shadow evaluation")
                elif baseline_eval_loss and best_metric <= baseline_eval_loss * 1.05:
                    print("  MARGINAL GO: val_loss within 5% of baseline — proceed with caution")
                else:
                    print("  NO-GO: val_loss did not improve — do not deploy adapter")
                break

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
