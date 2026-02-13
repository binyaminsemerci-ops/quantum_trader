# Multi-Asset Expansion Strategy – Quantum Trader

> **Phase**: Post-MVP (after BTC-only validation)  
> **Timeline**: Q3-Q4 2026  
> **Policy Reference**: constitution/RISK_POLICY.md

---

## 1. Expansion Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MULTI-ASSET EXPANSION PHILOSOPHY                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│     "Prove it on one asset. Scale to many."                                     │
│                                                                                 │
│     We do NOT expand to increase returns.                                       │
│     We expand to:                                                               │
│       1. Diversify uncorrelated opportunities                                   │
│       2. Increase signal frequency (more edges per day)                         │
│       3. Reduce dependence on single-asset regime                               │
│                                                                                 │
│     PREREQUISITES:                                                              │
│       ✓ 3+ months profitable on BTC                                            │
│       ✓ Risk framework battle-tested                                            │
│       ✓ Kill-switch proven in real conditions                                   │
│       ✓ Operations stable and automated                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Asset Selection Criteria

### Inclusion Requirements

```yaml
asset_requirements:
  liquidity:
    min_daily_volume: $500M      # Perpetual futures volume
    min_depth_at_1pct: $5M       # Order book depth
    max_spread: 0.05%            # Bid-ask spread
    
  infrastructure:
    perpetual_futures: true      # Must have perps
    reliable_data_feed: true     # Consistent data
    api_stability: true          # Stable exchange API
    
  correlation:
    btc_correlation_90d: < 0.85  # Not too correlated to BTC
    
  tradability:
    funding_rate_reasonable: true  # < 0.1% per 8h typical
    liquidation_risk: acceptable   # Sufficient liquidity
```

### Asset Tiering

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ASSET TIERS                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TIER 1: Core Assets (Deploy First)                                             │
│  ═══════════════════════════════════                                            │
│    BTC  - Bitcoin     - $50B+ daily volume  - Anchor asset                      │
│    ETH  - Ethereum    - $20B+ daily volume  - Second largest                    │
│                                                                                 │
│  TIER 2: Major Alts (After Tier 1 Stable)                                       │
│  ═════════════════════════════════════════                                      │
│    SOL  - Solana      - $5B+ daily volume   - High beta                         │
│    BNB  - Binance     - $2B+ daily volume   - Exchange token                    │
│    XRP  - Ripple      - $2B+ daily volume   - Different dynamics                │
│                                                                                 │
│  TIER 3: Selective Alts (Opportunistic)                                         │
│  ════════════════════════════════════════                                       │
│    AVAX, MATIC, LINK, etc.                                                      │
│    Added only when clear edge identified                                        │
│    May be seasonal/temporary                                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Risk Management Adjustments

### Portfolio-Level Limits

```python
# Portfolio risk limits for multi-asset
MULTI_ASSET_RISK_LIMITS = {
    # Per-asset limits (same as single-asset)
    "per_asset": {
        "max_risk_per_trade": 0.02,      # 2% per trade
        "max_daily_loss_per_asset": 0.03, # 3% per asset
        "max_position_size": 0.10,        # 10% per position
    },
    
    # Portfolio-level limits (NEW)
    "portfolio": {
        "max_total_exposure": 0.40,       # 40% total capital at risk
        "max_correlated_exposure": 0.25,  # 25% in correlated positions
        "max_daily_loss_portfolio": 0.05, # 5% daily (unchanged)
        "max_drawdown_portfolio": 0.20,   # 20% drawdown (unchanged)
        "max_positions": 5,               # Max 5 simultaneous positions
    },
}
```

### Correlation Monitoring

```python
def check_correlation_limit(
    new_position: Position,
    existing_positions: List[Position],
    correlation_matrix: pd.DataFrame
) -> bool:
    """
    Check if new position would exceed correlation exposure.
    
    Returns False if position would create too much correlated risk.
    """
    
    new_asset = new_position.symbol
    correlated_exposure = 0
    
    for pos in existing_positions:
        correlation = correlation_matrix.loc[new_asset, pos.symbol]
        
        if correlation > 0.70:  # Highly correlated
            correlated_exposure += pos.risk_amount
    
    # Add new position's risk
    total_correlated = correlated_exposure + new_position.risk_amount
    portfolio_equity = get_portfolio_equity()
    
    if total_correlated / portfolio_equity > 0.25:
        return False  # Would exceed correlation limit
    
    return True
```

### Position Slot Allocation

```yaml
position_slots:
  total_slots: 5
  allocation_by_tier:
    tier_1:
      reserved_slots: 2       # BTC, ETH can always have slot
    tier_2:
      max_slots: 2            # Up to 2 major alts
    tier_3:
      max_slots: 1            # At most 1 speculative
      
  priority_on_conflict:
    - Higher edge score
    - Lower correlation to existing
    - Higher liquidity
```

---

## 4. Expansion Timeline

### Phase-by-Phase Rollout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EXPANSION TIMELINE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PHASE 0: BTC ONLY (Current - Month 0-3)                                        │
│  ══════════════════════════════════════                                         │
│   • Single asset, single strategy                                               │
│   • Prove risk framework works                                                  │
│   • Target: 3 months profitable operation                                       │
│                                                                                 │
│  PHASE 1: ADD ETHEREUM (Month 4-6)                                              │
│  ═════════════════════════════════                                              │
│   • Add ETH/USDT perpetual                                                      │
│   • Same strategy, different asset                                              │
│   • Portfolio limits activated                                                  │
│   • Target: Validate multi-asset risk management                                │
│                                                                                 │
│  PHASE 2: MAJOR ALTS (Month 7-9)                                                │
│  ═══════════════════════════════                                                │
│   • Add SOL and/or BNB                                                          │
│   • Full correlation monitoring                                                 │
│   • Position slot management active                                             │
│   • Target: 3-4 assets trading smoothly                                         │
│                                                                                 │
│  PHASE 3: SELECTIVE EXPANSION (Month 10+)                                       │
│  ════════════════════════════════════════                                       │
│   • Opportunistic additions to Tier 3                                           │
│   • Based on identified edges                                                   │
│   • May remove assets if edge disappears                                        │
│   • Target: 4-5 assets optimal mix                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Gate Criteria for Each Phase

```yaml
phase_gates:
  phase_0_to_1:
    required:
      - 3_months_profitable_btc: true
      - max_drawdown_experienced: < -15%
      - kill_switch_tested: true
      - operations_stable: true
    recommended:
      - sharpe_ratio: > 0.8
      - win_rate: > 50%
      
  phase_1_to_2:
    required:
      - 2_months_profitable_multi: true
      - correlation_monitoring_working: true
      - no_portfolio_limit_breaches: true
    recommended:
      - eth_performance_positive: true
      - portfolio_sharpe: > 0.7
      
  phase_2_to_3:
    required:
      - 3_months_stable_3_assets: true
      - slot_management_working: true
      - position_sizing_validated: true
```

---

## 5. Infrastructure Requirements

### Data Feeds

```yaml
data_requirements:
  per_asset:
    - 1m candles (real-time)
    - 5m candles
    - 1h candles
    - Order book snapshots (L2)
    - Trade flow
    - Funding rate
    
  cross_asset:
    - Correlation matrix (24h rolling)
    - BTC dominance
    - Total crypto market cap
    - Sector performance (L1, L2, DeFi, etc.)
```

### Processing Capacity

```yaml
capacity_scaling:
  tier_1_only:
    signals_per_minute: 10
    evaluations_per_minute: 50
    servers_required: 1
    
  tier_1_plus_2:
    signals_per_minute: 30
    evaluations_per_minute: 150
    servers_required: 2
    
  full_portfolio:
    signals_per_minute: 50
    evaluations_per_minute: 250
    servers_required: 3
```

---

## 6. Signal Model Adjustments

### Asset-Specific Models

```python
# Each asset may need model adjustments
ASSET_MODEL_CONFIG = {
    "BTC": {
        "regime_sensitivity": 1.0,         # Baseline
        "trend_lookback_hours": 24,
        "volume_significance": "medium",
    },
    "ETH": {
        "regime_sensitivity": 1.1,         # Slightly more sensitive
        "trend_lookback_hours": 18,        # Faster moves
        "volume_significance": "medium",
        "btc_correlation_factor": 0.8,     # Often follows BTC
    },
    "SOL": {
        "regime_sensitivity": 1.5,         # High beta
        "trend_lookback_hours": 12,        # Much faster
        "volume_significance": "high",
        "btc_correlation_factor": 0.6,
        "narrative_component": true,       # Meme/narrative driven
    },
}
```

### Cross-Asset Signals

```python
# Signals that consider multiple assets
CROSS_ASSET_SIGNALS = {
    "btc_divergence": {
        # When asset diverges from BTC, may revert
        "lookback": "4h",
        "threshold": "2_std",
        "expected_reversion": true,
    },
    "sector_momentum": {
        # Sector showing strength
        "lookback": "24h",
        "assets_in_sector": ["SOL", "AVAX", "MATIC"],
        "signal_type": "momentum",
    },
    "correlation_breakdown": {
        # When correlations break, opportunities arise
        "normal_correlation": 0.7,
        "current_correlation": "real-time",
        "signal_on_breakdown": true,
    },
}
```

---

## 7. Risk Scenarios

### Multi-Asset Risk Scenarios

```yaml
scenarios:
  correlated_crash:
    description: All assets crash together
    probability: 5% per year
    mitigation:
      - Portfolio max exposure 40%
      - Kill-switch still at -5%/-20%
      - Correlation monitoring stops new positions
    worst_case: -20% (kill-switch)
    
  single_asset_anomaly:
    description: One asset moves against position
    probability: Higher (frequent)
    mitigation:
      - Per-asset daily limit 3%
      - Individual stop-losses
      - Position slot limits
    worst_case: -3% from single asset
    
  liquidity_crisis:
    description: Sudden liquidity drops
    probability: Low but catastrophic
    mitigation:
      - Only trade Tier 1-2 assets
      - No illiquid alts
      - Position sizes respect liquidity
    worst_case: Slippage on exits
```

---

## 8. Operational Considerations

### Monitoring Dashboard Updates

```yaml
multi_asset_dashboard:
  portfolio_view:
    - Total equity
    - Total exposure
    - Total P&L (24h)
    - Correlation heatmap
    - Position slot usage
    
  per_asset_view:
    - Current position
    - Entry price
    - Current P&L
    - Risk remaining
    - Regime classification
    
  alerts:
    - Correlation spike detected
    - Portfolio exposure > 35%
    - Asset removed from Tier 1/2
    - Liquidity drop detected
```

### Operational Complexity

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OPERATIONAL COMPLEXITY GROWTH                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1 Asset (BTC):                                                                 │
│   • 1 data feed                                                                 │
│   • 1 regime detector                                                           │
│   • 1 signal generator                                                          │
│   • Simple position management                                                  │
│   Complexity: LOW                                                               │
│                                                                                 │
│  2 Assets (BTC + ETH):                                                          │
│   • 2 data feeds + correlation                                                  │
│   • 2 regime detectors                                                          │
│   • 2 signal generators + cross-asset                                           │
│   • Portfolio position management                                               │
│   Complexity: MEDIUM                                                            │
│                                                                                 │
│  5 Assets (Full):                                                               │
│   • 5 data feeds + 10 pairwise correlations                                     │
│   • 5 regime detectors                                                          │
│   • 5+ signal generators                                                        │
│   • Full portfolio optimization                                                 │
│   Complexity: HIGH                                                              │
│                                                                                 │
│  RECOMMENDATION:                                                                │
│  Stay at 3-4 assets until team/resources grow.                                  │
│  Diminishing returns beyond 5 assets for our size.                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   MULTI-ASSET EXPANSION SUMMARY                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHEN TO EXPAND:                                                                │
│    • After 3+ months profitable on BTC                                          │
│    • Risk framework battle-tested                                               │
│    • Operations stable                                                          │
│                                                                                 │
│  HOW TO EXPAND:                                                                 │
│    1. BTC → ETH (Phase 1)                                                       │
│    2. Add SOL/BNB (Phase 2)                                                     │
│    3. Selective Tier 3 (Phase 3)                                                │
│                                                                                 │
│  RISK LIMITS:                                                                   │
│    • 2% per trade (unchanged)                                                   │
│    • 5% daily loss (unchanged)                                                  │
│    • 20% drawdown (unchanged)                                                   │
│    • 40% max total exposure (NEW)                                               │
│    • 25% max correlated exposure (NEW)                                          │
│    • 5 max positions (NEW)                                                      │
│                                                                                 │
│  KEY PRINCIPLE:                                                                 │
│    Expansion adds diversification, not leverage.                                │
│    Same risk discipline at portfolio level.                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*Multi-asset er diversifisering, ikke multiplikasjon av risiko.*
