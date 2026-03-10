#!/usr/bin/env bash
# run_dpo_pipeline.sh — Full DPO dry run on a GPU cloud instance.
#
# Usage:
#   bash run_dpo_pipeline.sh [/path/to/frozen_dataset.jsonl]
#
# If the frozen dataset path is not provided as an argument the script
# expects to already have it at:
#   ./dpo_patch10a_v2_FROZEN.jsonl
#
# Exit codes:
#   0  — pipeline completed and go_no_go.py returned GO
#   2  — pipeline completed but go_no_go.py returned NO-GO
#   1  — pipeline setup or training failed

set -euo pipefail

FROZEN_SHA="5fb175f398e9b87846559d0867351644c25e63ae5f46cdcd118ed311ea5ced80"
DATASET="dpo_patch10a_v2_FROZEN.jsonl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── 0. Locate dataset ─────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    cp "$1" "$DATASET"
    log "Copied dataset from $1"
fi

[[ -f "$DATASET" ]] || die "$DATASET not found. Copy from VPS first (see LAUNCH_GUIDE.md)."

# ── 1. Verify SHA256 ──────────────────────────────────────────────────────────
log "Verifying dataset SHA256 …"
ACTUAL=$(sha256sum "$DATASET" | awk '{print $1}')
if [[ "$ACTUAL" != "$FROZEN_SHA" ]]; then
    die "SHA256 mismatch!  expected=$FROZEN_SHA  actual=$ACTUAL"
fi
log "SHA256 OK: $ACTUAL"

# ── 2. Install requirements ───────────────────────────────────────────────────
log "Installing requirements …"
pip install -q -r requirements_train.txt
log "Requirements installed."

# ── 3. Authenticate with Hugging Face (skip if HF_TOKEN is set) ───────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN found — logging in non-interactively."
    huggingface-cli login --token "$HF_TOKEN"
else
    log "HF_TOKEN not set — assuming already authenticated."
    log "If training fails with 401, run: huggingface-cli login"
fi

# ── 4. Format dataset ─────────────────────────────────────────────────────────
log "Formatting DPO dataset …"
python format_dpo_dataset.py
TRAIN_N=$(wc -l < dpo_formatted/dpo_train.jsonl)
VAL_N=$(wc -l < dpo_formatted/dpo_val.jsonl)
log "Dataset formatted: train=$TRAIN_N  val=$VAL_N"

# ── 5. Training (with side-car collapse monitor) ──────────────────────────────
log "Starting DPO training …"

# Start watch_training.py in the background as collapse monitor
python watch_training.py --state dpo_run_output/trainer_state.json &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT

python train_dpo.py 2>&1 | tee dpo_train_run.log

# Give watch_training.py one final check cycle
sleep 35
kill $WATCH_PID 2>/dev/null || true
trap - EXIT

log "Training complete.  Log → dpo_train_run.log"

# ── 6. Shadow evaluation ──────────────────────────────────────────────────────
log "Running shadow evaluation …"
python shadow_eval.py \
    --adapter dpo_run_output \
    --val     dpo_formatted/dpo_val.jsonl \
    --base    "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    2>&1 | tee dpo_shadow_eval.log
log "Shadow eval complete.  Results → shadow_eval_results.json"

# ── 7. Go / No-Go gate ────────────────────────────────────────────────────────
log "Running go/no-go gate …"
set +e
python go_no_go.py \
    --trainer-state dpo_run_output/trainer_state.json \
    --shadow-eval   shadow_eval_results.json
GATE_CODE=$?
set -e

# ── 8. Print final summary ────────────────────────────────────────────────────
log "──────────────────────────────────────"
log "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
log "Dataset:         $DATASET (SHA256 verified)"
log "Train set:       $TRAIN_N pairs"
log "Val set:         $VAL_N pairs"
log "Adapter output:  dpo_run_output/"

if python - <<'EOF' 2>/dev/null
import json, pathlib
ts = json.loads(pathlib.Path("dpo_run_output/trainer_state.json").read_text())
log = ts.get("log_history", [])
train = [e for e in log if "loss" in e and "eval_loss" not in e]
evals = [e for e in log if "eval_loss" in e]
if train:
    print(f"  Train loss: {train[0]['loss']:.4f} → {train[-1]['loss']:.4f}")
if evals:
    best = min(e['eval_loss'] for e in evals)
    print(f"  Val loss:   {evals[0]['eval_loss']:.4f} → {evals[-1]['eval_loss']:.4f}  (best={best:.4f})")
EOF
then
    : # printed inline
fi

if [[ -f shadow_eval_results.json ]]; then
    python - <<'EOF' 2>/dev/null || true
import json
r = json.loads(open("shadow_eval_results.json").read())
print(f"  Base accuracy:  {r['base_accuracy']:.3f}")
print(f"  Tuned accuracy: {r['tuned_accuracy']:.3f}")
print(f"  Accuracy lift:  {r['lift']:+.3f}")
print(f"  Parse errors:   {r['parse_error_tuned']:.3f}")
print(f"  Tuned dist:     {r['tuned_dist']}")
EOF
fi

log "──────────────────────────────────────"

if [[ $GATE_CODE -eq 0 ]]; then
    log "VERDICT: GO"
    log "Adapter is ready for shadow validation on Binance testnet."
    log "Next: copy dpo_run_output/ to VPS and enable EXIT_AGENT_QWEN3_SHADOW=true"
    log ""
    log "Pack and upload:"
    log "  tar -czf dpo_run_output.tar.gz dpo_run_output/"
    log "  scp dpo_run_output.tar.gz root@46.224.116.254:/home/qt/quantum_trader/ops/offline/"
    exit 0
else
    log "VERDICT: NO-GO — do not deploy this adapter."
    log "See go_no_go.py output above for the specific failure(s)."
    log "Common fixes:"
    log "  - val_loss collapse → reduce DPO_BETA to 0.1 in train_dpo.py"
    log "  - accuracy too low  → add more DPO pairs (target ≥500 pairs)"
    log "  - parse errors      → check system prompt match in format_dpo_dataset.py"
    exit 2
fi
