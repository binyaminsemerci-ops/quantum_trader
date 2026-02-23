# Model Registry — Governance Rules

**Version:** 1.0  
**Status:** DRAFT — Not enforced until deployment gate is passed  
**Owner:** quantum-admin  
**Last Updated:** 2026-02-21  

---

## Purpose

The Model Registry is the **single, authoritative source of truth** for all AI/ML model artifacts
used by the Quantum Trader system. It enforces hard separation between models under research
(unstable, experimental) and models approved for live trading (validated, versioned, immutable).

**Core principle:** *A model running live must be identical to the model that passed validation.*
There must be no path by which a research training loop can overwrite a production artifact.

---

## Registry Tiers

### 1. `approved/` — Live Tier

- **Read by:** `quantum-live` (signal_engine, allocator, risk_engine, execution, exit_engine)
- **Written by:** `quantum-admin` via `promote_model.sh` ONLY  
- **Modified by:** Nobody — models are **immutable** once promoted  
- **Deleted by:** `quantum-admin` (triggers mandatory archival, never raw deletion)  

```
/opt/quantum/model_registry/approved/
    signal/         # Signal generation models (5-model ensemble, CLM, TFT, etc.)
    rl/             # RL sizing/calibration models (PPO, A2C variants)
    clm/            # CLM v3 continuous learning models
    .registry_manifest.json   # Signed manifest — see schema below
```

**Immutability contract:**
- Every file in `approved/` has an SHA-256 hash recorded in `.registry_manifest.json`
- The live services check the manifest on startup; mismatches cause hard abort
- No process except `quantum-admin` (running `promote_model.sh`) may write here

---

### 2. `staging/` — Research Tier

- **Written by:** `quantum-research` (rl-trainer, clm-trainer, any experiment)
- **Read by:** `quantum-research` ONLY — live services have **zero access** to staging
- **Promoted:** Only by `quantum-admin` running `promote_model.sh` after manual review  

```
/opt/quantum/model_registry/staging/
    signal/         # Signal model candidates from research experiments
    rl/             # RL candidate checkpoints
    clm/            # CLM v3 candidate checkpoints
    .staging_manifest.json     # Auto-updated by research services on write
```

**Staging contract:**
- Research services may freely overwrite staging artifacts during training cycles
- Staging artifacts carry no stability guarantee
- A model in staging is **assumed broken** until validated and promoted

---

### 3. `archived/` — Historical Tier

- **Written by:** `quantum-admin` only (during demotion/rollback)
- **Read by:** `quantum-admin` only (for rollback or audit)
- **Deleted by:** Never — archived models are permanent audit records  

```
/opt/quantum/model_registry/archived/
    signal/
        signal_v1.2.3_demoted_2026-02-21/
    rl/
        rl_ppo_v4_demoted_2026-02-15/
    clm/
        clm_v3_checkpoint_42_demoted_2026-02-10/
```

**Archive naming convention:**
```
{model_type}_{version}_{reason}_{YYYY-MM-DD}/
```

---

## Registry Manifest Schema

### `.registry_manifest.json` (approved/)

```json
{
  "version": "1.0",
  "last_promoted": "2026-02-21T00:00:00Z",
  "promoted_by": "quantum-admin",
  "models": {
    "signal/ensemble_v5.pkl": {
      "sha256": "<hash>",
      "promoted_at": "2026-02-21T00:00:00Z",
      "promoted_from": "staging/signal/ensemble_v5_candidate.pkl",
      "staging_validation_report": "staging/signal/.validation_v5.json",
      "notes": "5-model ensemble — passed 30d backtest, Sharpe > 1.5"
    },
    "rl/sizing_agent_v12.pt": {
      "sha256": "<hash>",
      "promoted_at": "2026-02-10T00:00:00Z",
      "promoted_from": "staging/rl/sizing_agent_v12_candidate.pt",
      "staging_validation_report": "staging/rl/.validation_v12.json",
      "notes": "PPO v4 — 500k steps validation, live shadow test passed"
    }
  }
}
```

### `.staging_manifest.json` (staging/)

```json
{
  "version": "1.0",
  "last_written": "2026-02-21T12:00:00Z",
  "written_by": "quantum-research",
  "candidates": {
    "signal/ensemble_v6_candidate.pkl": {
      "sha256": "<hash>",
      "written_at": "2026-02-21T12:00:00Z",
      "training_job_id": "rl_training_run_20260221",
      "status": "CANDIDATE",
      "promoted": false
    }
  }
}
```

---

## Promotion Rules

### Hard Requirements Before Any Promotion

1. **Validation report exists** in `staging/{model_type}/.validation_{version}.json`
2. **Backtest summary** attached: Sharpe ratio, max drawdown, win rate documented
3. **Shadow test** completed: model ran in shadow mode against live signals for ≥ 24h
4. **SHA-256 match**: artifact hash in staging matches the artifact in validation report
5. **Manual sign-off**: `quantum-admin` reviews report and explicitly invokes `promote_model.sh`
6. **Previous model archived**: demotion of current approved model happens atomically with promotion

### What Is Explicitly Forbidden

| Action                                          | Forbidden To      | Enforcement                        |
|-------------------------------------------------|-------------------|------------------------------------|
| Writing to `approved/` directly                 | ALL              | ACL: `quantum-research`: `---`     |
| Auto-promoting from training callback           | ALL              | No code path — script only         |
| Reading `staging/` from live services           | quantum-live     | ACL: `quantum-live`: `---`         |
| Deleting from `archived/`                       | ALL              | ACL + cron integrity check         |
| Promoting without validation report             | quantum-admin    | promote_model.sh hard check        |
| Promoting without archiving current model       | quantum-admin    | promote_model.sh atomic operation  |

---

## Integrity Monitoring

A cron job (once implemented) will run `verify_registry.sh` every 5 minutes to:

1. Recompute SHA-256 hashes of all files in `approved/`
2. Compare against `.registry_manifest.json`
3. Alert via Redis stream `quantum:stream:registry.alert` if mismatch detected
4. Live services abort on manifest mismatch at startup

---

## Versioning Convention

```
{model_type}_v{MAJOR}.{MINOR}.{PATCH}_{purpose}.{ext}

Examples:
  signal_v5.0.0_ensemble.pkl
  rl_v12.3.0_ppo_sizing.pt
  clm_v3.1.0_continuous.pt
```

- MAJOR: Breaking architecture change  
- MINOR: New training data or hyperparameter regime  
- PATCH: Bug fix or minor checkpoint update  

---

*This document is planning/governance only. No enforcement is active until explicitly deployed.*
