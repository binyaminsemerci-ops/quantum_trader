# QTOS Phase 1: Stabilization Checklist
## Priority: IMMEDIATE — Clean the Battlefield Before Building

### Status: NOT STARTED
### Date: 2026-03-14

---

## 1. Kill Zombie Services (91 of 134)

Services to REMOVE (not running, not needed):
```bash
# These service files should be disabled and removed from /etc/systemd/system/
# Full list in QTOS_COMPLETE_SYSTEM_MAP.md section 2.3
# Keep ONLY the 43 currently running + any needed for maintenance
```

## 2. Eliminate /opt/quantum Ghost

- [ ] Move market-publisher WorkingDirectory from /opt/quantum/ops/market → /home/qt/quantum_trader/ops/market
- [ ] Verify no service reads model files from /opt/quantum/ai_engine/models/
- [ ] Archive /opt/quantum → /opt/quantum.archive.2026-03-14
- [ ] Update all systemd services using /opt/quantum/venvs/ → main venv

## 3. Consolidate to 1 Python Interpreter

- [ ] Identify all packages needed across 3 interpreters
- [ ] Install ALL into /home/qt/quantum_trader_venv/
- [ ] Update 7 services using /opt/quantum/venvs/ai-engine/ → main venv
- [ ] Update ~15 services using /usr/bin/python3 → main venv
- [ ] Remove /opt/quantum/venvs/ and /mnt/HC_Volume_104287969/quantum-venvs/

## 4. Archive Root-Level File Sprawl

- [ ] Create archive/ directory
- [ ] Move all _*.py, fix_*.py, check_*.py, deploy_*.sh → archive/scripts/
- [ ] Move all AI_*.md, *_COMPLETE*.md, *_REPORT*.md → archive/docs/
- [ ] Keep only: README.md, .env, .gitignore, requirements.txt, pyproject.toml

## 5. Kill Dual Execution Path

- [ ] Verify execution/ service is inactive (confirmed dead)
- [ ] Remove quantum-execution.service from systemd
- [ ] Audit: ensure ONLY intent_executor handles trade.intent (via intent_bridge)

## 6. Remove Duplicate Services

Known duplicates in service files:
- quantum-ai-engine.service vs quantum-ai_engine.service
- quantum-portfolio-intelligence.service vs quantum-portfolio_intelligence.service
- quantum-position-monitor.service vs quantum-position_monitor.service
- quantum-risk-safety.service vs quantum-risk_safety.service
- quantum-universe.service appears TWICE

---

## Validation After Phase 1

```bash
# Should show exactly 43 (or fewer) services
systemctl list-units quantum-*.service --all | grep loaded | wc -l

# Should show ONLY /home/qt/quantum_trader_venv paths
systemctl show quantum-*.service -p ExecStart | grep -oP '/\S+python\S*' | sort -u

# Should show NOTHING
ls /opt/quantum/ 2>/dev/null

# Root should be clean
ls *.py *.sh *.md 2>/dev/null | wc -l  # Should be < 10
```
