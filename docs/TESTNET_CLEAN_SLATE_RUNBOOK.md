# Testnet Clean Slate Runbook

## Purpose
Safely reset testnet to establish clean baseline metrics for P2.9 Capital Allocation gate enforcement.

## When to Use
- Starting fresh P2.9 testing with no legacy positions
- After making significant changes to allocation logic
- When existing positions are far outside allocation targets (e.g., ETHUSDT $31k vs $1.8k target)
- Establishing clear before/after comparison data

## Prerequisites
- ESS controller script available: `/home/qt/quantum_trader/ops/ess_controller.sh`
- Redis running and accessible
- Access to Binance Testnet (for manual position closing if needed)
- Governor service running

## Procedure

### 1. Run the Script
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
cd /home/qt/quantum_trader
bash scripts/testnet_clean_slate.sh
```

### 2. Manual Position Closing Options

**Option A: Binance Testnet UI**
- Login to Binance Futures Testnet
- Navigate to Positions tab
- Click "Close All" or close individual positions

**Option B: Python Script** (if available)
```bash
python3 scripts/close_all_positions.py --testnet
```

**Option C: Natural Close**
- Wait for existing CLOSE signals to execute
- May take longer but requires no manual intervention

### 3. Script Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Activate ESS                                        │
│         → Stop new trade execution                          │
│         → Existing positions remain                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Show Open Positions                                 │
│         → Lists all quantum:position:snapshot:* keys        │
│         → Displays position details (size, PnL, etc)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Wait for Manual Closure                             │
│         → Polls every 10 seconds                            │
│         → Timeout: 10 minutes                               │
│         → Operator closes positions manually                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Deactivate ESS                                      │
│         → Resume normal trading                             │
│         → System ready for clean positions                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Display Baseline Metrics                            │
│         → Show current P2.9 metrics                         │
│         → Display recent P2.9 gate activity                 │
└─────────────────────────────────────────────────────────────┘
```

## Safety Features

### No Destructive Actions
- Script **NEVER** force-closes positions
- Operator maintains full control
- Can abort at any time (Ctrl+C)

### Timeout Handling
- 10-minute wait period
- Graceful timeout with options:
  - Continue waiting
  - Deactivate ESS and exit
  - Proceed with remaining positions

### State Visibility
- Real-time position count updates
- Clear status messages
- Metric snapshots before/after

## Expected Output

### Before Clean Slate
```
gov_p29_checked_total{symbol="ETHUSDT"} 15.0
gov_p29_block_total{symbol="ETHUSDT"} 12.0
gov_p29_checked_total{symbol="BTCUSDT"} 8.0
```

### After Clean Slate (Initial)
```
gov_p29_checked_total{symbol="ETHUSDT"} 15.0  # Historical
gov_p29_block_total{symbol="ETHUSDT"} 12.0    # Historical
```

### After New Positions Open
```
gov_p29_checked_total{symbol="ETHUSDT"} 16.0  # +1 new check
gov_p29_block_total{symbol="ETHUSDT"} 12.0    # No new blocks
gov_p29_checked_total{symbol="NEWUSDT"} 1.0   # Fresh symbol
```

## Post-Clean-Slate Monitoring

### Real-Time P2.9 Activity
```bash
journalctl -u quantum-governor -f | grep --line-buffered -i "testnet.*p2.9"
```

### Metrics Dashboard
```bash
watch -n 5 'curl -s localhost:8044/metrics | grep gov_p29'
```

### Position Status
```bash
redis-cli KEYS "quantum:position:snapshot:*"
redis-cli HGETALL quantum:position:snapshot:BTCUSDT
```

## Troubleshooting

### Issue: ESS Activation Fails
**Cause:** ESS controller script not found or not executable
**Solution:**
```bash
cd /home/qt/quantum_trader
chmod +x ops/ess_controller.sh
bash ops/ess_controller.sh status
```

### Issue: Positions Won't Close
**Cause:** Market conditions or position size
**Solution:**
- Check Binance Testnet for open orders
- Cancel open orders manually
- Use market orders to close positions
- Verify position actually exists on exchange

### Issue: Timeout Reached
**Cause:** Large positions or low liquidity
**Solution:**
- Continue manual closing after timeout
- Run script again after positions close
- Accept legacy positions and track separately

### Issue: Redis Keys Empty
**Cause:** Positions already closed or different key pattern
**Solution:**
```bash
redis-cli KEYS "*position*" | head -20  # Find actual key pattern
redis-cli KEYS "*snapshot*"
```

## Verification Checklist

After running clean slate script:

- [ ] ESS deactivated (trading resumed)
- [ ] No open positions: `redis-cli KEYS "quantum:position:snapshot:*"` returns empty
- [ ] Governor running: `systemctl status quantum-governor`
- [ ] P2.9 metrics baseline captured
- [ ] Monitoring in place for new positions

## Expected Timeline

| Phase | Duration | Actions |
|-------|----------|---------|
| ESS Activation | 5 seconds | Automated |
| Position Display | 10 seconds | Automated |
| Manual Closing | 2-10 minutes | Manual |
| ESS Deactivation | 5 seconds | Automated |
| Metric Display | 10 seconds | Automated |
| **Total** | **3-11 minutes** | **Minimal manual work** |

## Notes

- **Testnet Only**: This script is designed for testnet environments
- **No Data Loss**: All metrics are cumulative (Prometheus counters)
- **Idempotent**: Safe to run multiple times
- **Production**: Do NOT use in production without modification

## Related Documentation

- [P2.9 Capital Allocation Brain](../microservices/capital_allocation/)
- [Governor Gate 0.5 Integration](../microservices/governor/)
- [ESS Controller](../ops/ess_controller.sh)
- [P3.0 Performance Attribution](../microservices/performance_attribution/)

## Quick Reference

```bash
# Run clean slate
bash scripts/testnet_clean_slate.sh

# Check ESS status
bash ops/ess_controller.sh status

# Manual ESS control
bash ops/ess_controller.sh activate
bash ops/ess_controller.sh deactivate

# Monitor P2.9 activity
journalctl -u quantum-governor -f | grep P2.9

# Check metrics
curl localhost:8044/metrics | grep gov_p29
```
