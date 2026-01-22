=== PHASE E0: PREFLIGHT ANALYSIS ===
Date: 2026-01-17 22:50 UTC
Evidence Dir: /tmp/phase_e_20260117_225023

REQUIRED STREAMS FOR HARVESTBRAIN:
1. quantum:stream:execution.result      ✅ FOUND (10000 entries)
2. quantum:stream:position.snapshot     ❌ EMPTY (0 entries)
3. quantum:stream:pnl.unrealized        ❌ EMPTY (0 entries)
4. quantum:stream:pnl.snapshot          ❌ EMPTY (0 entries)
5. quantum:stream:trade.intent          ✅ FOUND (10010 entries)

ACTUAL AVAILABLE STREAMS (for fallback):
- quantum:stream:exitbrain.pnl          ✅ FOUND (1 entry)
- quantum:stream:trade.closed           ✅ FOUND
- quantum:stream:trade.signal           ✅ FOUND
- quantum:stream:trade.position.update  ✅ EXISTS (0 entries - not publishing)

ACTIVE SERVICES (can provide position data):
- quantum-position-monitor              ✅ RUNNING
- quantum-portfolio-intelligence        ✅ RUNNING
- quantum-binance-pnl-tracker           ✅ RUNNING
- quantum-execution.service             ✅ RUNNING

DECISION:
=========
HarvestBrain v1 will use HYBRID approach:

1. Primary input: quantum:stream:execution.result (fills/executions)
2. Fallback: Derive position from execution events (track fills internally)
3. Mode: START IN SHADOW (no live intents until position data reliable)
4. Monitor: Watch for position/pnl streams to populate
5. Future: Upgrade to LIVE when position data confirmed

IMPLEMENTATION STRATEGY:
- Build HarvestBrain with STATE TRACKER (in-memory position per symbol)
- Read from execution.result → update internal position state
- Compute R (return on risk) from internal position tracking
- Publish to harvest.suggestions in SHADOW mode
- When position.snapshot starts publishing: enhance with fresh data
- Switch to LIVE mode when operator confirms

ACTION ITEMS:
✅ PHASE E0 COMPLETE - proceed with HarvestBrain scaffold
⏳ FUTURE: Monitor if position-monitor starts publishing streams
⏳ FUTURE: If not, may need to create minimal position publisher

RISK MITIGATION:
- Fail-closed: No live intents until position data validated
- Dedup: All harvesting actions deduplicated via Redis
- Idempotent: Safe to replay events
- Observable: All decisions logged to journalctl
