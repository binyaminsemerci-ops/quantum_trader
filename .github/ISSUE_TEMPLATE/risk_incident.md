---
name: Risk Incident Report
about: Report a risk-related incident (loss, near-miss, policy violation)
title: '[INCIDENT] '
labels: risk-critical, priority: critical
assignees: ''
---

## 🚨 RISK INCIDENT REPORT

**Filed by:** @___
**Incident time:** YYYY-MM-DD HH:MM UTC
**Report time:** YYYY-MM-DD HH:MM UTC

---

## Incident Classification

**Type:**
- [ ] Financial loss
- [ ] Near-miss (loss avoided)
- [ ] Policy violation (no loss)
- [ ] System failure (potential loss)
- [ ] Data integrity issue

**Severity:**
- [ ] CRITICAL – Active loss or system down
- [ ] HIGH – Loss occurred, system stable
- [ ] MEDIUM – Near-miss or minor issue
- [ ] LOW – For documentation only

---

## Incident Summary

**One-line summary:**
<!-- Brief description -->

**Detailed description:**
<!-- Full account of what happened -->

---

## Financial Impact

| Metric | Value |
|--------|-------|
| Realized loss | $_______ |
| Unrealized loss (if position still open) | $_______ |
| Loss as % of equity | _______% |
| Affected position(s) | |
| Affected asset(s) | |

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| | |
| | |
| | |

---

## Root Cause Analysis

**What failed?**
- [ ] Risk Kernel – didn't VETO
- [ ] Exit Brain – didn't exit in time
- [ ] Kill-switch – didn't trigger
- [ ] Human override – manual action
- [ ] External factor – exchange, API
- [ ] Other: ___

**Why did it fail?**
<!-- Technical root cause -->

**Contributing factors:**
<!-- What conditions led to this -->

---

## Policy Violations

**Were any Grunnlov violated?**

| Grunnlov | Violated? | Details |
|----------|-----------|---------|
| §1 – Kapitalvern | [ ] Yes  [ ] No | |
| §2 – Risk VETO | [ ] Yes  [ ] No | |
| §7 – 2% max trade risk | [ ] Yes  [ ] No | |
| §10 – 5% daily limit | [ ] Yes  [ ] No | |
| §11 – 20% drawdown limit | [ ] Yes  [ ] No | |

---

## Immediate Response

**Actions taken:**
- [ ] Position closed
- [ ] Kill-switch activated
- [ ] Manual intervention
- [ ] Service restarted
- [ ] Other: ___

**Current system state:**
- [ ] OPERATIONAL – Back to normal
- [ ] DEGRADED – Limited functionality
- [ ] KILL-SWITCH ACTIVE – All trading halted
- [ ] INVESTIGATING – Not yet resolved

---

## Evidence

**Logs:**
```
[Relevant log excerpts]
```

**Screenshots:**
<!-- Attach screenshots -->

**Data exports:**
<!-- Attach relevant data -->

---

## Corrective Actions

**Immediate fixes:**
1. 
2. 

**Long-term improvements:**
1. 
2. 

**Policy changes needed?**
- [ ] Yes – File separate policy change request
- [ ] No

---

## Lessons Learned

**What should we have done differently?**
<!-- Hindsight analysis -->

**What early warning signs were missed?**
<!-- Signals that could have prevented this -->

---

## Follow-up Required

- [ ] Post-mortem meeting scheduled
- [ ] Code fix PR created
- [ ] Policy review initiated
- [ ] Additional monitoring added
- [ ] Documentation updated

---

**For maintainer use:**
- [ ] Incident acknowledged
- [ ] Root cause confirmed
- [ ] Corrective actions assigned
- [ ] Follow-up scheduled
- [ ] Incident closed
