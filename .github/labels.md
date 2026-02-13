# GitHub Labels – Quantum Trader

> **Purpose**: Standardized labels for issues and PRs  
> **Policy Reference**: constitution/GOVERNANCE.md  
> **Enforcement**: Manual review + CI checks

---

## 1. Required Labels

### Policy & Governance

| Label | Color | Description | When to Use |
|-------|-------|-------------|-------------|
| `policy` | `#7057ff` (purple) | Related to constitution/policy document | Changes to Grunnlov, risk rules |
| `risk-critical` | `#d73a4a` (red) | Affects risk management directly | Risk Kernel, limits, kill-switch |
| `fail-closed` | `#e99695` (light red) | Must fail-closed by design | New failure modes |
| `kill-switch` | `#b60205` (dark red) | Related to kill-switch mechanism | Emergency stop functionality |
| `audit-required` | `#fbca04` (yellow) | Requires explicit audit/review | Production changes, sensitive |
| `no-live-change` | `#000000` (black) | Cannot go directly to production | Needs shadow testing first |

### Development Phase

| Label | Color | Description | When to Use |
|-------|-------|-------------|-------------|
| `mvp` | `#0e8a16` (green) | MVP scope item | Core functionality |
| `post-mvp` | `#c2e0c6` (light green) | After MVP | Future enhancements |
| `shadow-only` | `#1d76db` (blue) | Shadow mode testing only | Pre-production validation |
| `shadow-tested` | `#0052cc` (dark blue) | Has passed shadow testing | Ready for review |

### Service Tags

| Label | Color | Description | When to Use |
|-------|-------|-------------|-------------|
| `risk-kernel` | `#d73a4a` (red) | Risk Kernel service | Changes to risk evaluation |
| `exit-brain` | `#ff7619` (orange) | Exit Brain service | Exit logic changes |
| `signal-advisory` | `#5319e7` (purple) | Signal Advisory service | Signal generation |
| `position-sizer` | `#006b75` (teal) | Position Sizer service | Position sizing logic |
| `execution-engine` | `#0366d6` (blue) | Execution Engine service | Order execution |
| `regime-detector` | `#84b6eb` (light blue) | Regime Detector service | Market regime |

### Issue Type

| Label | Color | Description | When to Use |
|-------|-------|-------------|-------------|
| `bug` | `#d73a4a` (red) | Something isn't working | Bug reports |
| `enhancement` | `#a2eeef` (cyan) | New feature or improvement | Feature requests |
| `documentation` | `#0075ca` (blue) | Documentation only | Docs updates |
| `security` | `#ee0701` (bright red) | Security related | Security issues |
| `breaking-change` | `#b60205` (dark red) | Breaking API/behavior change | Major changes |

### Priority

| Label | Color | Description | When to Use |
|-------|-------|-------------|-------------|
| `priority: critical` | `#b60205` (dark red) | Must fix immediately | Production down, losing money |
| `priority: high` | `#d93f0b` (orange) | Fix this sprint | Important but not emergency |
| `priority: medium` | `#fbca04` (yellow) | Normal priority | Standard work |
| `priority: low` | `#c5def5` (light blue) | Nice to have | When time permits |

---

## 2. Label Combinations

### Automatic Review Triggers

These label combinations trigger automatic review requirements:

```yaml
# Requires CODEOWNERS review + manual approval
high_scrutiny:
  - ["risk-critical", "kill-switch"]
  - ["policy", "risk-critical"]
  - ["fail-closed", "risk-kernel"]
  - ["no-live-change", "execution-engine"]

# Requires shadow testing first
shadow_required:
  - ["risk-kernel"]
  - ["exit-brain"]
  - ["execution-engine"]
  - ["position-sizer"]

# Requires documentation update
docs_required:
  - ["policy"]
  - ["breaking-change"]
  - ["fail-closed"]
```

### Example Issue Labels

```
[Bug in Risk Kernel]
Labels: bug, risk-kernel, risk-critical, priority: critical

[New Exit Strategy]
Labels: enhancement, exit-brain, shadow-only, mvp

[Policy Document Update]
Labels: policy, documentation, audit-required, no-live-change

[Add new trading pair]
Labels: enhancement, post-mvp, execution-engine
```

---

## 3. Label Policies

### Labels That Block PRs

PRs with these labels CANNOT be merged:

| Label | Why Blocked | To Unblock |
|-------|-------------|------------|
| `no-live-change` | Needs shadow testing | Add `shadow-tested` |
| `audit-required` | Needs explicit approval | Approval from CODEOWNERS |
| `wip` | Work in progress | Remove when ready |
| `do-not-merge` | Explicit block | Remove when resolved |

### Labels That Trigger Actions

```yaml
# GitHub Actions triggers
on_label:
  "shadow-only":
    - Deploy to shadow environment
    - Run integration tests
    
  "risk-critical":
    - Notify #risk-alerts Slack channel
    - Require 2x reviewers
    
  "kill-switch":
    - Block auto-merge
    - Require founder approval
    
  "policy":
    - Update policy docs
    - Trigger constitution validation
```

---

## 4. Label YAML for Import

```yaml
# .github/labels.yml
# Use with github-label-sync or similar tool

- name: policy
  color: "7057ff"
  description: "Related to constitution/policy document"

- name: risk-critical
  color: "d73a4a"
  description: "Affects risk management directly"

- name: fail-closed
  color: "e99695"
  description: "Must fail-closed by design"

- name: kill-switch
  color: "b60205"
  description: "Related to kill-switch mechanism"

- name: audit-required
  color: "fbca04"
  description: "Requires explicit audit/review"

- name: no-live-change
  color: "000000"
  description: "Cannot go directly to production"

- name: mvp
  color: "0e8a16"
  description: "MVP scope item"

- name: post-mvp
  color: "c2e0c6"
  description: "After MVP"

- name: shadow-only
  color: "1d76db"
  description: "Shadow mode testing only"

- name: shadow-tested
  color: "0052cc"
  description: "Has passed shadow testing"

- name: risk-kernel
  color: "d73a4a"
  description: "Risk Kernel service"

- name: exit-brain
  color: "ff7619"
  description: "Exit Brain service"

- name: signal-advisory
  color: "5319e7"
  description: "Signal Advisory service"

- name: position-sizer
  color: "006b75"
  description: "Position Sizer service"

- name: execution-engine
  color: "0366d6"
  description: "Execution Engine service"

- name: regime-detector
  color: "84b6eb"
  description: "Regime Detector service"

- name: bug
  color: "d73a4a"
  description: "Something isn't working"

- name: enhancement
  color: "a2eeef"
  description: "New feature or improvement"

- name: documentation
  color: "0075ca"
  description: "Documentation only"

- name: security
  color: "ee0701"
  description: "Security related"

- name: breaking-change
  color: "b60205"
  description: "Breaking API/behavior change"

- name: "priority: critical"
  color: "b60205"
  description: "Must fix immediately"

- name: "priority: high"
  color: "d93f0b"
  description: "Fix this sprint"

- name: "priority: medium"
  color: "fbca04"
  description: "Normal priority"

- name: "priority: low"
  color: "c5def5"
  description: "Nice to have"
```

---

## 5. Label Governance

### Who Can Add/Remove Labels

| Label Type | Add | Remove |
|------------|-----|--------|
| Service tags | Anyone | Anyone |
| Priority | Reviewer, Maintainer | Reviewer, Maintainer |
| Policy/Risk | Maintainer, CODEOWNERS | Maintainer, CODEOWNERS |
| Kill-switch | Founder only | Founder only |

### Label Audit

Monthly audit of label usage and accuracy:
- Are PRs correctly labeled?
- Are risk-critical items caught?
- Are shadow requirements enforced?

---

*Labels are governance. Use them correctly.*
