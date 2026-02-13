# Pull Request

## Description

<!-- Brief description of changes -->

## Related Issues

Fixes #___
Related to #___

---

## Pre-Merge Checklist (REQUIRED)

### Risk Assessment

- [ ] **Risk Kernel considered**: I have verified this change does not bypass Risk Kernel VETO authority
- [ ] **Fail-closed verified**: Any new failure modes default to safe state
- [ ] **No silent failures**: All errors are logged and surfaced

### Testing

- [ ] **Unit tests pass**: All existing tests pass
- [ ] **New tests added**: Coverage for new functionality
- [ ] **Shadow-tested**: This has been tested in shadow environment
  - Shadow duration: ___ days
  - Shadow results: [ ] PASS  [ ] FAIL  [ ] N/A (docs only)

### Policy Compliance

- [ ] **Policy reference**: I have linked to relevant constitution/policy documents where applicable
- [ ] **No policy violations**: This change complies with all Grunnlov
- [ ] **Audit trail**: Changes are traceable and documented

### Documentation

- [ ] **Code documented**: Functions/classes have docstrings
- [ ] **README updated**: If user-facing changes
- [ ] **Architecture docs**: If structural changes

---

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Policy/constitution change
- [ ] Refactoring (no functional change)

---

## Service Impact

**Services modified:**
- [ ] Risk Kernel
- [ ] Exit Brain
- [ ] Signal Advisory
- [ ] Position Sizer
- [ ] Execution Engine
- [ ] Regime Detector
- [ ] Other: ___

**Services affected (downstream):**
<!-- List services that consume this service's output -->

---

## Risk-Critical Changes

> Complete this section if modifying: Risk Kernel, Exit Brain, Execution Engine, Kill-switch, Position Sizer

### Risk Kernel Impact
- Does this change how trades are approved/rejected? [ ] Yes  [ ] No
- Does this modify any risk limits? [ ] Yes  [ ] No
- Does this affect kill-switch behavior? [ ] Yes  [ ] No

### If ANY "Yes" above:
- [ ] Reviewed by second engineer
- [ ] Shadow tested for minimum 48 hours
- [ ] Manual review of shadow trades confirms expected behavior
- [ ] Rollback plan documented below

---

## Policy References

**Relevant Grunnlov:**
<!-- Link to specific policies this change relates to -->
- constitution/RISK_POLICY.md: §___
- constitution/FUND_POLICY.md: §___

**Policy compliance notes:**
<!-- Explain how this complies with policy -->

---

## Deployment Plan

### Pre-deployment
- [ ] Database migrations reviewed (if any)
- [ ] Config changes documented
- [ ] Feature flags in place (if applicable)

### Deployment order
<!-- If multiple services, specify order -->
1. 
2. 

### Rollback plan
<!-- How to quickly revert if issues arise -->

---

## Screenshots/Evidence (if applicable)

<!-- For UI changes or behavior demonstration -->

---

## Reviewer Notes

<!-- Anything specific reviewers should look at -->

---

**For reviewers:**
- [ ] Code quality acceptable
- [ ] Tests adequate
- [ ] Risk assessment complete
- [ ] Shadow testing verified (if applicable)
- [ ] Ready to merge
