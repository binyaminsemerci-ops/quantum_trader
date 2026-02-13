# Branch Strategy – Quantum Trader

> **Policy Reference**: constitution/GOVERNANCE.md  
> **Purpose**: Control flow of changes from development to production  
> **Key Principle**: All risk-affecting code must pass shadow testing

---

## 1. Branch Model

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            BRANCH HIERARCHY                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  policy/main ────────────────────────────────────────────────────────────────── │
│       │                                                                         │
│       │  Constitutional documents, rarely changed                               │
│       │  Requires: Founder approval + 7-day review                              │
│       │                                                                         │
│       ▼                                                                         │
│  main ──────────────────────────────────────────────────────────────────────── │
│       │                                                                         │
│       │  Production code, always deployable                                     │
│       │  Requires: PR review + shadow testing                                   │
│       │                                                                         │
│       ├───────▶ shadow/* ────────────────────────────────────────────────────  │
│       │              │                                                          │
│       │              │  Shadow testing environment                              │
│       │              │  Deploys to shadow cluster                               │
│       │              │                                                          │
│       │              └── shadow/risk-kernel-v2                                  │
│       │              └── shadow/exit-brain-improvement                          │
│       │              └── shadow/new-regime-detector                             │
│       │                                                                         │
│       ├───────▶ feature/* ──────────────────────────────────────────────────── │
│       │              │                                                          │
│       │              │  Development work                                        │
│       │              │  No deployment                                           │
│       │              │                                                          │
│       │                                                                         │
│       └───────▶ hotfix/* ───────────────────────────────────────────────────── │
│                      │                                                          │
│                      │  Emergency fixes                                         │
│                      │  Fast-track to production                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Branch Definitions

### `policy/main`

**Purpose:** Constitution and policy documents

**Contents:**
- `constitution/*.md` (Grunnlov, risk policy, etc.)
- `ops/pre_flight/*.md`
- `docs/FOUNDERS_OPERATING_MANUAL.md`

**Rules:**
| Rule | Enforcement |
|------|-------------|
| Minimum 7-day review period | CI check |
| Requires founder approval | CODEOWNERS |
| No code changes allowed | CI check |
| Changes trigger global review | Automated notification |

**Protection rules:**
```yaml
# Branch protection for policy/main
protection:
  required_reviews: 2
  required_reviewers:
    - founders
  dismiss_stale_reviews: true
  require_code_owner_reviews: true
  restrictions:
    - Only founders can push
```

---

### `main`

**Purpose:** Production code, always deployable

**Rules:**
| Rule | Enforcement |
|------|-------------|
| All tests must pass | CI required |
| PR review required | Branch protection |
| Risk-critical changes need shadow testing | Label check |
| No direct pushes | Branch protection |

**Merge requirements:**
```yaml
# Branch protection for main
protection:
  required_reviews: 1
  required_status_checks:
    - tests
    - lint
    - security-scan
  require_linear_history: true
  require_signed_commits: true
```

---

### `shadow/*`

**Purpose:** Shadow testing of new features

**Naming convention:**
```
shadow/<service>-<description>
shadow/risk-kernel-improved-veto
shadow/exit-brain-volatility-exit
shadow/signal-advisory-new-model
```

**Rules:**
| Rule | Enforcement |
|------|-------------|
| Auto-deploys to shadow environment | CI/CD |
| Minimum testing period: 48h (risk-critical: 7d) | CI timestamp check |
| Must produce shadow-test-report.md | PR require |
| Cannot access real capital | Network isolation |

**Lifecycle:**
```
1. Create shadow/<name> from main
2. Develop and push changes
3. CI auto-deploys to shadow cluster
4. Monitor shadow metrics
5. Minimum time period passes
6. Generate shadow test report
7. Create PR to main
8. Add shadow-tested label
9. Review and merge
```

---

### `feature/*`

**Purpose:** Active development

**Naming convention:**
```
feature/<issue-number>-<description>
feature/123-add-volatility-indicator
feature/456-improve-logging
```

**Rules:**
| Rule | Enforcement |
|------|-------------|
| No deployment | CI config |
| Tests must pass | CI required |
| Link to issue | PR template |

---

### `hotfix/*`

**Purpose:** Emergency production fixes

**Naming convention:**
```
hotfix/<incident-id>-<description>
hotfix/INC-001-fix-kill-switch
hotfix/INC-002-api-timeout
```

**Rules:**
| Rule | Enforcement |
|------|-------------|
| Must link to incident report | PR template |
| Fast-track review (can skip shadow) | Manual approval |
| Requires on-call approval | CODEOWNERS |
| Post-merge review required | Process |

**Fast-track criteria:**
```
Hotfix can skip shadow testing ONLY if:
1. Production is actively losing money
2. Kill-switch or safety system is broken
3. Data integrity is compromised

Even fast-track requires:
- At least 1 reviewer
- Tests pass
- Immediate post-deploy monitoring
```

---

## 3. Merge Flow

### Standard Feature Change

```
feature/123-new-feature
       │
       │ Tests pass, PR opened
       ▼
shadow/123-new-feature (if risk-critical)
       │
       │ Shadow testing (48h-7d)
       │ Shadow report generated
       ▼
main (PR merged)
       │
       │ Auto-deploy to production
       ▼
PRODUCTION
```

### Policy Change

```
feature/policy-update-risk-limits
       │
       │ PR opened to policy/main
       │ 7-day review period starts
       ▼
REVIEW PERIOD (7 days)
       │
       │ Founder approval obtained
       │ All concerns addressed
       ▼
policy/main (merged)
       │
       │ Triggers code update PR if needed
       ▼
main (if code changes required)
```

### Hotfix

```
INCIDENT DETECTED
       │
       │ Incident report filed
       ▼
hotfix/INC-xxx-fix-description
       │
       │ Minimal fix developed
       │ Fast-track review
       ▼
main (merged)
       │
       │ Immediate deploy
       │ Post-deploy monitoring
       ▼
PRODUCTION (fixed)
       │
       │ Post-mortem scheduled
       ▼
Follow-up improvements
```

---

## 4. Shadow Testing Requirements

### Test Duration by Change Type

| Change Type | Minimum Shadow Duration |
|-------------|------------------------|
| Risk Kernel changes | 7 days |
| Exit Brain changes | 5 days |
| Execution Engine changes | 5 days |
| Position Sizer changes | 5 days |
| Signal Advisory changes | 3 days |
| Regime Detector changes | 3 days |
| Non-risk code | 48 hours |
| Documentation only | Not required |

### Shadow Test Report Requirements

```markdown
# Shadow Test Report: <branch-name>

## Test Period
- Start: YYYY-MM-DD HH:MM UTC
- End: YYYY-MM-DD HH:MM UTC
- Duration: X days

## Metrics

### Trade Statistics
- Signals generated: X
- Trades approved by Risk Kernel: X
- Trades vetoed: X
- Veto reasons: ...

### Performance
- Shadow P&L (no real money): $X.XX
- Max drawdown: X%
- Win rate: X%

### Comparison to Production
- Production same period: $X.XX
- Difference: $X.XX (X%)

### Errors/Issues
- Errors logged: X
- Warnings: X
- Critical events: X

## Anomalies
<!-- Any unexpected behavior -->

## Conclusion
[ ] PASS - Ready for production
[ ] FAIL - Issues found, addressing in PR
[ ] INCONCLUSIVE - Need more testing
```

---

## 5. CI/CD Configuration

### Branch-Specific Pipelines

```yaml
# .github/workflows/main.yml
name: Main Branch CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: make test
      - name: Run lint
        run: make lint
      - name: Security scan
        run: make security-scan

  check-shadow-tested:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - name: Check shadow label
        uses: actions/github-script@v6
        with:
          script: |
            const labels = context.payload.pull_request.labels;
            const isRiskCritical = labels.some(l => 
              ['risk-kernel', 'exit-brain', 'execution-engine', 'position-sizer'].includes(l.name)
            );
            const isShadowTested = labels.some(l => l.name === 'shadow-tested');
            
            if (isRiskCritical && !isShadowTested) {
              core.setFailed('Risk-critical changes require shadow-tested label');
            }
```

```yaml
# .github/workflows/shadow.yml
name: Shadow Branch Deploy

on:
  push:
    branches: ['shadow/**']

jobs:
  deploy-shadow:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to shadow
        run: make deploy-shadow
      - name: Record start time
        run: |
          echo "SHADOW_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $GITHUB_ENV
```

---

## 6. CODEOWNERS

```
# .github/CODEOWNERS

# Policy documents - founders only
/constitution/                  @founders
/docs/FOUNDERS_OPERATING_MANUAL.md  @founders

# Risk-critical services - senior engineers
/services/risk_kernel/          @risk-team @founders
/services/exit_brain/           @risk-team
/services/execution_engine/     @risk-team

# Standard services
/services/signal_advisory/      @engineering
/services/position_sizer/       @engineering
/services/regime_detector/      @engineering

# Operations
/ops/                           @devops @engineering

# Default
*                               @engineering
```

---

## 7. Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          BRANCH STRATEGY SUMMARY                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  policy/main  → Constitution/policy → Founder approval + 7 days                 │
│  main         → Production code     → PR review + shadow (if risk-critical)     │
│  shadow/*     → Shadow testing      → Auto-deploy, minimum duration             │
│  feature/*    → Development         → Tests pass, no deploy                     │
│  hotfix/*     → Emergency fixes     → Fast-track, post-review required          │
│                                                                                 │
│  RISK-CRITICAL = Risk Kernel, Exit Brain, Execution Engine, Position Sizer     │
│  RISK-CRITICAL MUST shadow test before production merge                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*Kode går ikke til produksjon uten å bevise at den er trygg.*
