# /opt/quantum — Full Directory Tree & Permission Plan

**Status:** DRAFT — No files moved. No services restarted. Structure pending creation only.
**Date:** 2026-02-21
**Author:** Architectural Refactor (Live/Research Separation)

---

## 1. Target Directory Tree

```
/opt/quantum/
│
├── live_core/                        # Production services — LIVE money at risk
│   ├── signal_engine/                # Signal generation (AI 5-model ensemble)
│   │   ├── config/
│   │   └── logs/
│   ├── allocator/                    # Position sizing & slot allocation
│   │   ├── config/
│   │   └── logs/
│   ├── risk_engine/                  # RiskGuardV2, SafetyGovernor
│   │   ├── config/
│   │   └── logs/
│   ├── execution/                    # BinanceFuturesExecutionAdapter, exec_risk_service
│   │   ├── config/
│   │   └── logs/
│   └── exit_engine/                  # ExitBrainV3.5, robust_exit_engine
│       ├── config/
│       └── logs/
│
├── research_lab/                     # Experimental — ISOLATED from live money
│   ├── rl_trainer/                   # RL calibrator, rl_training microservice
│   │   ├── experiments/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── clm_trainer/                  # CLM v3 training loop
│   │   ├── experiments/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── backtests/                    # Historical simulation runs
│   │   └── results/
│   └── experiments/                  # Ad-hoc research notebooks, scripts
│       └── scratch/
│
└── model_registry/                   # Single source of truth for all models
    ├── approved/                     # LIVE services read ONLY from here
    │   ├── signal/
    │   ├── rl/
    │   ├── clm/
    │   └── .registry_manifest.json  # Signed manifest of approved models
    ├── staging/                      # Research writes ONLY here
    │   ├── signal/
    │   ├── rl/
    │   ├── clm/
    │   └── .staging_manifest.json   # Auto-updated on each research write
    └── archived/                     # Demoted/superseded models
        ├── signal/
        ├── rl/
        └── clm/
```

---

## 2. File Permission Plan

### System Users

| User               | Group              | Purpose                                    |
|--------------------|--------------------|--------------------------------------------|
| `quantum-live`     | `quantum-live`     | Runs all live_core services                |
| `quantum-research` | `quantum-research` | Runs all research_lab services             |
| `quantum-registry` | `quantum-registry` | Owns model_registry tree                   |
| `quantum-admin`    | `quantum`          | Runs promotion script only (manual gate)   |

### Permission Matrix

| Path                                      | Owner              | Group              | Mode   | Notes                                  |
|-------------------------------------------|--------------------|--------------------|--------|----------------------------------------|
| `/opt/quantum/`                           | root               | quantum            | `755`  | Top-level container                    |
| `/opt/quantum/live_core/`                 | quantum-live       | quantum-live       | `750`  | No world access                        |
| `/opt/quantum/live_core/*/config/`        | quantum-live       | quantum-live       | `640`  | Config files read-only after deploy    |
| `/opt/quantum/live_core/*/logs/`          | quantum-live       | quantum-live       | `750`  | Live service log output                |
| `/opt/quantum/research_lab/`              | quantum-research   | quantum-research   | `750`  | No world access                        |
| `/opt/quantum/research_lab/*/checkpoints/`| quantum-research   | quantum-research   | `750`  | Research writes checkpoints here       |
| `/opt/quantum/model_registry/`            | quantum-registry   | quantum-registry   | `755`  | Registry root                          |
| `/opt/quantum/model_registry/approved/`   | quantum-registry   | quantum-live       | `750`  | quantum-live: r-x, quantum-research: — |
| `/opt/quantum/model_registry/staging/`    | quantum-registry   | quantum-research   | `750`  | quantum-research: rwx, quantum-live: — |
| `/opt/quantum/model_registry/archived/`   | quantum-registry   | quantum-registry   | `750`  | quantum-admin promotion step only      |

### ACL Enforcement (setfacl)

```bash
# Live services: read-only on approved/
setfacl -Rm u:quantum-live:r-x    /opt/quantum/model_registry/approved/
setfacl -Rm u:quantum-live:---    /opt/quantum/model_registry/staging/
setfacl -Rm u:quantum-live:---    /opt/quantum/model_registry/archived/

# Research services: write-only on staging/, no access to approved/
setfacl -Rm u:quantum-research:rwx /opt/quantum/model_registry/staging/
setfacl -Rm u:quantum-research:---  /opt/quantum/model_registry/approved/
setfacl -Rm u:quantum-research:---  /opt/quantum/model_registry/archived/

# Admin: full access to registry for promotion
setfacl -Rm u:quantum-admin:rwx   /opt/quantum/model_registry/
```

---

## 3. Creation Script (Run Once — Does NOT Move Existing Files)

```bash
#!/bin/bash
# create_quantum_structure.sh
# SAFE: Creates empty directories only. Does NOT touch existing /opt/quantum content.

set -euo pipefail

BASE="/opt/quantum"

create_dir() {
    mkdir -p "$1"
    echo "  [+] $1"
}

echo "=== Creating /opt/quantum directory structure ==="

# live_core
for svc in signal_engine allocator risk_engine execution exit_engine; do
    create_dir "$BASE/live_core/$svc/config"
    create_dir "$BASE/live_core/$svc/logs"
done

# research_lab
for svc in rl_trainer clm_trainer; do
    create_dir "$BASE/research_lab/$svc/experiments"
    create_dir "$BASE/research_lab/$svc/checkpoints"
    create_dir "$BASE/research_lab/$svc/logs"
done
create_dir "$BASE/research_lab/backtests/results"
create_dir "$BASE/research_lab/experiments/scratch"

# model_registry
for tier in approved staging archived; do
    for model_type in signal rl clm; do
        create_dir "$BASE/model_registry/$tier/$model_type"
    done
done

# Sentinel files to prevent accidental deletion
touch "$BASE/model_registry/approved/.registry_manifest.json"
touch "$BASE/model_registry/staging/.staging_manifest.json"

echo ""
echo "=== Applying ownership ==="
useradd -r -s /sbin/nologin quantum-live    2>/dev/null || true
useradd -r -s /sbin/nologin quantum-research 2>/dev/null || true
useradd -r -s /sbin/nologin quantum-registry 2>/dev/null || true
groupadd quantum 2>/dev/null || true
usermod -aG quantum quantum-live 2>/dev/null || true
usermod -aG quantum quantum-research 2>/dev/null || true
usermod -aG quantum quantum-registry 2>/dev/null || true

chown -R quantum-live:quantum-live       "$BASE/live_core/"
chown -R quantum-research:quantum-research "$BASE/research_lab/"
chown -R quantum-registry:quantum-registry "$BASE/model_registry/"

chmod 750 "$BASE/live_core/"
chmod 750 "$BASE/research_lab/"
chmod 755 "$BASE/model_registry/"

echo ""
echo "=== Applying ACLs ==="
setfacl -Rm u:quantum-live:r-x    "$BASE/model_registry/approved/"
setfacl -Rm u:quantum-live:---    "$BASE/model_registry/staging/"
setfacl -Rm u:quantum-research:rwx "$BASE/model_registry/staging/"
setfacl -Rm u:quantum-research:--- "$BASE/model_registry/approved/"

echo ""
echo "=== DONE. Existing files untouched. ==="
echo "Next step: See architecture/MIGRATION_PLAN.md"
```

---

## 4. Verify Structure (Dry Run)

```bash
# Run this to verify what WILL exist before executing create script
find /opt/quantum -type d | sort
```

---

*No services have been modified. No files have been moved. This document is planning only.*
