---
name: Feature Request
about: Propose a new feature for Quantum Trader
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Pre-Submission Checklist

- [ ] I have checked that this feature doesn't already exist
- [ ] I have considered the risk implications
- [ ] This feature aligns with the fund's constitution

---

## Feature Description

**What feature do you want?**
<!-- Clear description of the proposed feature -->

**Why do you need it?**
<!-- Business/technical justification -->

---

## Risk Assessment (REQUIRED)

**Does this feature:**

| Question | Answer |
|----------|--------|
| Modify Risk Kernel logic? | [ ] Yes  [ ] No |
| Change position sizing? | [ ] Yes  [ ] No |
| Affect exit mechanisms? | [ ] Yes  [ ] No |
| Touch execution engine? | [ ] Yes  [ ] No |
| Add new failure modes? | [ ] Yes  [ ] No |
| Require kill-switch changes? | [ ] Yes  [ ] No |

**If any "Yes" above:**
- [ ] I understand this requires extended shadow testing
- [ ] I have considered fail-closed behavior
- [ ] I have consulted constitution/RISK_POLICY.md

---

## Constitutional Alignment

**Does this feature comply with:**

| Grunnlov | Compliant? | Notes |
|----------|------------|-------|
| §1 – Capital preservation first | [ ] Yes  [ ] N/A | |
| §2 – Risk Kernel VETO authority | [ ] Yes  [ ] N/A | |
| §8 – AI is advisor, not decider | [ ] Yes  [ ] N/A | |
| §15 – System CAN stand still | [ ] Yes  [ ] N/A | |

**Policy references (if applicable):**
- Grunnlov §___ : 
- Risk policy section: 

---

## Scope

**Phase:**
- [ ] MVP (required for launch)
- [ ] Post-MVP (nice to have)

**Complexity estimate:**
- [ ] Small (hours)
- [ ] Medium (days)
- [ ] Large (weeks)

**Affected services:**
- [ ] Risk Kernel
- [ ] Exit Brain
- [ ] Signal Advisory
- [ ] Position Sizer
- [ ] Execution Engine
- [ ] Regime Detector
- [ ] Other: ___

---

## Proposed Implementation

**High-level approach:**
<!-- How would you implement this? -->

**Data flow changes:**
<!-- How does data move through the system? -->

**New events (if any):**
```yaml
# New Redis events this feature would publish/consume
```

---

## Testing Strategy

**How will this be validated?**

- [ ] Unit tests
- [ ] Integration tests
- [ ] Shadow mode testing (duration: ___ days)
- [ ] Micro-capital live test

**Failure scenarios to test:**
1. 
2. 
3. 

---

## Rollback Plan

**If this feature causes issues:**
<!-- How do we disable/rollback quickly? -->

---

## Additional Context

<!-- Any other relevant information, mockups, examples -->

---

**For maintainer use:**
- [ ] Triaged
- [ ] Risk assessment reviewed
- [ ] Constitutional alignment verified
- [ ] Added to roadmap
