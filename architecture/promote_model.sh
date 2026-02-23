#!/usr/bin/env bash
# =============================================================================
# promote_model.sh — Manual Model Promotion Script
# =============================================================================
#
# PURPOSE:
#   Promotes a model artifact from staging/ to approved/ in the model registry.
#   This is the ONLY valid path for a model to reach production.
#
# RULES:
#   - Must be run as quantum-admin
#   - Requires a validation report to exist in staging
#   - Atomically archives the current approved model before promoting
#   - Updates .registry_manifest.json with SHA-256 and provenance
#   - NEVER auto-runs — manual invocation only
#
# USAGE:
#   sudo -u quantum-admin ./promote_model.sh \
#       --type rl \
#       --staging-file rl/sizing_agent_v12_candidate.pt \
#       --target-name sizing_agent_v12.pt \
#       --notes "PPO v4 — 500k steps, shadow test passed 2026-02-21"
#
# STATUS: DRAFT — Review before deployment
# =============================================================================

set -euo pipefail

REGISTRY_BASE="/opt/quantum/model_registry"
STAGING_DIR="$REGISTRY_BASE/staging"
APPROVED_DIR="$REGISTRY_BASE/approved"
ARCHIVED_DIR="$REGISTRY_BASE/archived"
MANIFEST="$APPROVED_DIR/.registry_manifest.json"
LOG_FILE="/var/log/quantum/model_promotions.log"

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; exit 1; }

# ─── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --type TYPE --staging-file REL_PATH --target-name FILENAME --notes "NOTE"

Options:
  --type          Model type: signal | rl | clm
  --staging-file  Path relative to staging/  e.g. rl/my_model_candidate.pt
  --target-name   Filename in approved/       e.g. my_model_v12.pt
  --notes         Human-readable promotion notes (required)
  --dry-run       Print what would happen, do not execute
  --help          Show this message
EOF
    exit 0
}

# ─── Defaults ─────────────────────────────────────────────────────────────────
MODEL_TYPE=""
STAGING_FILE=""
TARGET_NAME=""
NOTES=""
DRY_RUN=false
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATE_SLUG=$(date +"%Y-%m-%d")

# ─── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)           MODEL_TYPE="$2";   shift 2 ;;
        --staging-file)   STAGING_FILE="$2"; shift 2 ;;
        --target-name)    TARGET_NAME="$2";  shift 2 ;;
        --notes)          NOTES="$2";        shift 2 ;;
        --dry-run)        DRY_RUN=true;      shift   ;;
        --help|-h)        usage ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ─── Validate arguments ───────────────────────────────────────────────────────
[[ -z "$MODEL_TYPE"    ]] && error "--type is required (signal | rl | clm)"
[[ -z "$STAGING_FILE"  ]] && error "--staging-file is required"
[[ -z "$TARGET_NAME"   ]] && error "--target-name is required"
[[ -z "$NOTES"         ]] && error "--notes is required"

case "$MODEL_TYPE" in
    signal|rl|clm) ;;
    *) error "Invalid --type '$MODEL_TYPE'. Must be: signal | rl | clm" ;;
esac

# ─── Security check ───────────────────────────────────────────────────────────
CURRENT_USER=$(whoami)
if [[ "$CURRENT_USER" != "quantum-admin" ]]; then
    error "Must run as quantum-admin. Current user: $CURRENT_USER"
fi

# ─── Resolve paths ────────────────────────────────────────────────────────────
STAGING_ABS="$STAGING_DIR/$STAGING_FILE"
APPROVED_ABS="$APPROVED_DIR/$MODEL_TYPE/$TARGET_NAME"
VALIDATION_REPORT="$STAGING_DIR/$MODEL_TYPE/.validation_$(basename "$TARGET_NAME" | sed 's/\.[^.]*$//').json"

# ─── Pre-flight checks ────────────────────────────────────────────────────────
info "=== Quantum Model Promotion ==="
info "Timestamp:       $TIMESTAMP"
info "Operator:        $CURRENT_USER"
info "Model type:      $MODEL_TYPE"
info "Staging file:    $STAGING_ABS"
info "Target:          $APPROVED_ABS"
info "Notes:           $NOTES"
echo ""

# Check staging file exists
[[ -f "$STAGING_ABS" ]] || error "Staging file not found: $STAGING_ABS"

# Check validation report exists
if [[ ! -f "$VALIDATION_REPORT" ]]; then
    warn "Validation report not found at expected path: $VALIDATION_REPORT"
    warn "A validation report is REQUIRED for promotion."
    echo ""
    read -r -p "Acknowledge missing validation report and proceed anyway? [type YES to continue]: " CONFIRM
    [[ "$CONFIRM" == "YES" ]] || error "Promotion aborted — validation report required."
fi

# Compute SHA-256 of staging artifact
STAGING_SHA256=$(sha256sum "$STAGING_ABS" | awk '{print $1}')
info "SHA-256 (staging): $STAGING_SHA256"

# ─── Dry run exit ─────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == true ]]; then
    warn "DRY RUN — no changes made."
    echo ""
    info "Would archive:  $APPROVED_ABS → $ARCHIVED_DIR/$MODEL_TYPE/"
    info "Would copy:     $STAGING_ABS → $APPROVED_ABS"
    info "Would update:   $MANIFEST"
    exit 0
fi

# ─── Manual confirmation ──────────────────────────────────────────────────────
echo ""
warn "You are about to promote a model to APPROVED (LIVE) tier."
warn "This will affect live trading services."
echo ""
echo "  Staging:  $STAGING_ABS"
echo "  Target:   $APPROVED_ABS"
echo "  Notes:    $NOTES"
echo ""
read -r -p "Type the model filename to confirm promotion: " CONFIRM_NAME
[[ "$CONFIRM_NAME" == "$TARGET_NAME" ]] || error "Confirmation mismatch. Aborting."

# ─── Step 1: Archive current approved model (if exists) ───────────────────────
ARCHIVE_SLOT="$ARCHIVED_DIR/$MODEL_TYPE/${TARGET_NAME%.*}_demoted_$DATE_SLUG"
if [[ -f "$APPROVED_ABS" ]]; then
    info "Archiving current approved model..."
    mkdir -p "$ARCHIVE_SLOT"
    cp -p "$APPROVED_ABS" "$ARCHIVE_SLOT/"
    ARCHIVED_SHA256=$(sha256sum "$APPROVED_ABS" | awk '{print $1}')
    # Write archive metadata
    cat > "$ARCHIVE_SLOT/.archive_meta.json" <<ARCHIVEMETA
{
  "archived_at": "$TIMESTAMP",
  "archived_by": "$CURRENT_USER",
  "original_path": "$APPROVED_ABS",
  "sha256": "$ARCHIVED_SHA256",
  "reason": "Superseded by $TARGET_NAME on $DATE_SLUG"
}
ARCHIVEMETA
    info "Archived to: $ARCHIVE_SLOT"
else
    info "No existing approved model to archive (first promotion for this slot)."
fi

# ─── Step 2: Copy staging artifact to approved ────────────────────────────────
info "Promoting artifact to approved/..."
mkdir -p "$APPROVED_DIR/$MODEL_TYPE"
cp -p "$STAGING_ABS" "$APPROVED_ABS"
chmod 444 "$APPROVED_ABS"   # Immutable: no writes after promotion
PROMOTED_SHA256=$(sha256sum "$APPROVED_ABS" | awk '{print $1}')

# ─── Step 3: Verify SHA-256 integrity ─────────────────────────────────────────
if [[ "$STAGING_SHA256" != "$PROMOTED_SHA256" ]]; then
    error "SHA-256 MISMATCH after copy! Staging: $STAGING_SHA256 | Approved: $PROMOTED_SHA256. Aborting."
fi
info "SHA-256 verified: $PROMOTED_SHA256"

# ─── Step 4: Update manifest (append entry via Python for JSON safety) ────────
info "Updating registry manifest..."
python3 - <<PYEOF
import json, os, sys
from datetime import datetime

manifest_path = "$MANIFEST"
if not os.path.exists(manifest_path):
    manifest = {"version": "1.0", "models": {}}
else:
    with open(manifest_path) as f:
        manifest = json.load(f)

manifest["last_promoted"] = "$TIMESTAMP"
manifest["promoted_by"]   = "$CURRENT_USER"

key = "$MODEL_TYPE/$TARGET_NAME"
manifest["models"][key] = {
    "sha256": "$PROMOTED_SHA256",
    "promoted_at": "$TIMESTAMP",
    "promoted_from": "$STAGING_FILE",
    "notes": "$NOTES",
    "validation_report": "$VALIDATION_REPORT" if os.path.exists("$VALIDATION_REPORT") else "NOT_PROVIDED"
}

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"  Manifest updated: {key}")
PYEOF

# ─── Step 5: Emit audit log entry ─────────────────────────────────────────────
mkdir -p "$(dirname "$LOG_FILE")"
cat >> "$LOG_FILE" <<AUDITLOG

=== PROMOTION RECORD ===
Timestamp:        $TIMESTAMP
Operator:         $CURRENT_USER
Model type:       $MODEL_TYPE
Staging file:     $STAGING_ABS
Approved target:  $APPROVED_ABS
SHA-256:          $PROMOTED_SHA256
Notes:            $NOTES
Archive slot:     ${ARCHIVE_SLOT:-NONE (first promotion)}
========================
AUDITLOG

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
info "==================================================="
info " PROMOTION COMPLETE"
info "==================================================="
info " Model:   $APPROVED_ABS"
info " SHA-256: $PROMOTED_SHA256"
info " Live services will pick up this model on next restart."
info " To restart a live service:"
info "   systemctl restart quantum-signal.service   # if signal model"
info "   systemctl restart quantum-rl-trainer.service  # if rl model"
info ""
warn " REMINDER: Live services are NOT restarted automatically."
warn " Manual restart required after reviewing this promotion."
