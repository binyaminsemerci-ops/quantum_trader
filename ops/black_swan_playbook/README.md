# Black Swan Playbook

**Grunnlov**: §10 (Kill-Switch), §7 (Fail-Closed)  
**Purpose**: Response procedures for extreme events  

## What is a Black Swan?

An event that is:
- Unexpected
- Has extreme impact
- Rationalized after the fact

For our system:
- BTC/ETH > 20% move in 24h
- Flash crash (> 10% in 1h)
- Exchange insolvency
- Major counterparty failure
- Network attack on blockchain

## Response Protocol

### Phase 1: Immediate (0-60 seconds)

```
1. Kill-switch AUTO-ACTIVATED
2. All positions closed at market
3. All orders cancelled
4. Alert all channels
5. Log complete state snapshot
6. System enters lockdown
```

### Phase 2: Assessment (1 minute - 24 hours)

```
1. Verify positions actually closed
2. Calculate realized P&L
3. Check for exchange issues
4. Monitor external news
5. Document everything
6. NO automated decisions
```

### Phase 3: Analysis (24h - 7 days)

```
1. Complete incident report
2. Root cause analysis
3. Risk model review
4. Strategy adjustment if needed
5. System health check
6. Plan recovery
```

### Phase 4: Recovery (7+ days)

```
1. Shadow-mode testing (min 50 trades)
2. Proof Mode restart (Level 1 capital)
3. Gradual scaling over 30 days
4. Weekly reviews
5. Update Black Swan playbook
```

## Decision Tree

```
Event detected
    │
    ├── Price move > 10% in 1h?
    │   └── YES → KILL-SWITCH (Flash crash protocol)
    │
    ├── Price move > 20% in 24h?
    │   └── YES → KILL-SWITCH (Black swan protocol)
    │
    ├── Exchange unresponsive?
    │   └── YES → KILL-SWITCH (Exchange failure protocol)
    │
    └── Unknown extreme event?
        └── YES → KILL-SWITCH (Precautionary)
```

## What NOT To Do

- Do NOT try to "catch the bounce"
- Do NOT add to positions during chaos
- Do NOT trust exchange status immediately
- Do NOT rush back into trading
- Do NOT assume it's over
