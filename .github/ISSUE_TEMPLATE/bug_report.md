---
name: Bug Report
about: Report a bug in Quantum Trader
title: '[BUG] '
labels: bug
assignees: ''
---

## Pre-Submission Checklist

- [ ] I have checked existing issues for duplicates
- [ ] I have verified this is not expected behavior
- [ ] I am not exposing sensitive data (API keys, positions, etc.)

---

## Bug Description

**What happened?**
<!-- Clear description of the bug -->

**What should have happened?**
<!-- Expected behavior -->

---

## Risk Assessment

**Does this bug affect:**

- [ ] Risk Kernel (trade approval/rejection)
- [ ] Exit Brain (position exits)
- [ ] Kill-switch mechanism
- [ ] Position sizing
- [ ] Order execution
- [ ] None of the above (non-critical path)

**Has this caused financial loss?**
- [ ] Yes – Amount: $ ___
- [ ] No
- [ ] Unknown

**Is this currently affecting production?**
- [ ] Yes – actively causing issues
- [ ] No – found in testing/shadow
- [ ] Unknown

---

## Environment

- **Environment**: [ ] Production  [ ] Shadow  [ ] Development
- **Service(s) affected**: 
- **Approximate time of occurrence**: 
- **Frequency**: [ ] Always  [ ] Sometimes  [ ] Once

---

## Reproduction Steps

1. 
2. 
3. 

**Minimum reproducible example:**
```python
# Code to reproduce (if applicable)
```

---

## Logs & Evidence

**Relevant logs:**
```
# Paste relevant log output
```

**Screenshots (if applicable):**
<!-- Drag and drop images -->

---

## Policy Check

**Could this bug have violated any Grunnlov?**

| Grunnlov | Possibly Violated? |
|----------|-------------------|
| §1 – Kapitalvern | [ ] Yes  [ ] No  [ ] Unknown |
| §2 – Risk Kernel VETO | [ ] Yes  [ ] No  [ ] Unknown |
| §7 – 2% per trade | [ ] Yes  [ ] No  [ ] Unknown |
| §10 – 5% daily limit | [ ] Yes  [ ] No  [ ] Unknown |

---

## Additional Context

<!-- Any other relevant information -->

---

**For maintainer use:**
- [ ] Triaged
- [ ] Priority assigned
- [ ] Policy violation assessed
- [ ] Root cause identified
