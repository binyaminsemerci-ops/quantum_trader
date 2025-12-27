# Dashboard V3.0 Manual Testing Checklist
**POST-DEPLOYMENT VALIDATION**

## Pre-Test Setup
- [ ] Backend running: `http://localhost:8000`
- [ ] Frontend running: `http://localhost:3000`
- [ ] Portfolio Intelligence Service: `http://localhost:8004`
- [ ] Binance testnet credentials configured
- [ ] All Docker containers healthy

---

## 1. OVERVIEW TAB CHECKS

### Environment & Status
- [ ] Environment badge shows correct value (TESTNET/PRODUCTION)
- [ ] GO-LIVE status badge visible (ACTIVE/INACTIVE)
- [ ] Badge colors are appropriate (green=active, gray=inactive)

### Global PnL Display
- [ ] Total equity displays as number (e.g., $816.61)
- [ ] Daily PnL displays with sign (e.g., -$76.33 in red)
- [ ] Daily PnL % displays correctly (e.g., -9.35%)
- [ ] Weekly PnL displays (may be 0.00 initially)
- [ ] Monthly PnL displays (may be 0.00 initially)
- [ ] **NO "NaN" appears anywhere**

### Risk Indicators
- [ ] Global risk state badge visible (OK/WARNING/CRITICAL)
- [ ] Badge color matches state (green=OK, yellow=WARNING, red=CRITICAL)
- [ ] ESS status shows correctly (INACTIVE/ACTIVE)
- [ ] ESS triggers today count displays (should be 0 if inactive)

### Positions Summary
- [ ] Positions count badge displays (e.g., "10 positions")
- [ ] Count matches Binance testnet actual positions

### Exposure Per Exchange
- [ ] Exchange names listed (e.g., "binance_testnet")
- [ ] Exposure values display as currency (e.g., $41,498.50)
- [ ] Multiple exchanges shown if configured

---

## 2. TRADING TAB CHECKS

### Positions Table
- [ ] Table headers visible (Symbol, Side, Size, Entry, Current, PnL, Leverage)
- [ ] Position rows display for each open position
- [ ] Symbols correct (e.g., BTCUSDT, ETHUSDT)
- [ ] Side indicators correct (BUY/LONG in green, SELL/SHORT in red)
- [ ] Position sizes display correctly
- [ ] Entry prices match Binance
- [ ] Current prices update in real-time
- [ ] Unrealized PnL displays with color coding (green=profit, red=loss)
- [ ] Leverage displays correctly (e.g., 20x)
- [ ] **NO "NaN" in any table cell**

### Empty States
- [ ] If no positions: "No open positions" placeholder shows
- [ ] Empty state is clean and informative (not blank screen)

### Recent Orders
- [ ] Recent orders section visible
- [ ] Shows "No recent orders" if empty
- [ ] Order entries show if available (time, symbol, side, status)

### Recent Signals
- [ ] Recent signals section visible
- [ ] Shows "No recent signals" if empty
- [ ] Signal entries show if available (symbol, direction, confidence)

---

## 3. RISK & SAFETY TAB CHECKS

### Risk Gate Decisions
- [ ] "Allow" count displays (e.g., 45)
- [ ] "Block" count displays (e.g., 3)
- [ ] "Scale" count displays (e.g., 12)
- [ ] "Total" count displays (e.g., 60)
- [ ] Total = Allow + Block + Scale (or >= if other decision types)
- [ ] Block count highlighted if > 0

### ESS (Emergency Stop System)
- [ ] ESS status card visible
- [ ] Status shows INACTIVE/ACTIVE
- [ ] Triggers today count displays
- [ ] Daily loss displays (e.g., -1.2%)
- [ ] Threshold displays (e.g., -5.0%)
- [ ] If ACTIVE: warning banner shows

### VaR/ES Metrics
- [ ] VaR 95% displays (e.g., $150.00)
- [ ] VaR 99% displays (e.g., $250.00)
- [ ] ES 95% displays (e.g., $200.00)
- [ ] ES 99% displays (e.g., $350.00)
- [ ] All values are valid numbers (not NaN)

### Drawdown Per Profile
- [ ] Profile names listed (e.g., conservative, aggressive)
- [ ] Current DD values display
- [ ] Max DD values display
- [ ] If empty: "No profiles configured" shows

---

## 4. SYSTEM & STRESS TAB CHECKS

### Microservices Health
- [ ] Service cards visible (AI Engine, Portfolio, Risk, Execution, etc.)
- [ ] Status indicators correct (UP=green, DOWN=red, DEGRADED=yellow)
- [ ] Response time displays for UP services (e.g., 45ms)
- [ ] Last check timestamp displays
- [ ] DOWN services show appropriate error indicator

### Exchange Health
- [ ] Exchange cards visible (Binance, Bybit, etc.)
- [ ] Status indicators correct (UP/DOWN)
- [ ] Latency displays for UP exchanges (e.g., 120ms)
- [ ] High latency highlighted (e.g., >500ms in yellow/red)
- [ ] Last check timestamp displays

### Failover Events
- [ ] "Recent Failover Events" section visible
- [ ] Shows "No recent failovers" if empty
- [ ] If events exist: timestamp, service, reason display

### Stress Test Results
- [ ] "Recent Stress Scenarios" section visible
- [ ] Shows "No recent tests" if empty
- [ ] If tests exist: scenario name, result (PASSED/FAILED), duration display
- [ ] PASSED tests show green indicator
- [ ] FAILED tests show red indicator

---

## 5. REAL-TIME UPDATES

### WebSocket Connection
- [ ] Open browser DevTools > Network > WS tab
- [ ] WebSocket connection established (`ws://localhost:3000/ws/dashboard`)
- [ ] Connection status shows "connected" (if visible in UI)

### Live Updates
- [ ] Watch position PnL change as prices move
- [ ] Equity updates in real-time (refresh every 5-30 seconds)
- [ ] No need to manually refresh page for updates

---

## 6. DATA CONSISTENCY CHECKS

### Cross-Verify with Binance
- [ ] Open Binance Futures Testnet in browser
- [ ] Compare position count: Dashboard == Binance
- [ ] Compare position symbols: Dashboard == Binance
- [ ] Compare position sizes: Dashboard == Binance
- [ ] Compare unrealized PnL: Dashboard ≈ Binance (within small margin for timing)

### Cross-Tab Consistency
- [ ] Positions count in Overview == positions in Trading tab
- [ ] Equity in Overview matches Portfolio data
- [ ] ESS status in Overview == ESS status in Risk tab

---

## 7. ERROR HANDLING

### Service Unavailable
- [ ] Stop Portfolio Intelligence Service
- [ ] Reload dashboard
- [ ] Dashboard still loads (shows defaults or "Service unavailable" message)
- [ ] No white screen of death
- [ ] Restart service → dashboard recovers

### Network Interruption
- [ ] Disconnect internet briefly
- [ ] Dashboard shows error/loading state
- [ ] Reconnect → dashboard recovers automatically

---

## 8. RESPONSIVE DESIGN

### Desktop (1920x1080)
- [ ] All tabs visible
- [ ] Cards arranged in grid layout
- [ ] No horizontal scrolling
- [ ] Text readable

### Tablet (768x1024)
- [ ] Layout adapts (cards stack vertically if needed)
- [ ] Tab navigation works
- [ ] Text remains readable

### Mobile (375x667) - Optional
- [ ] Dashboard accessible
- [ ] Critical data visible
- [ ] Navigation usable

---

## 9. PERFORMANCE

### Load Time
- [ ] Dashboard loads in < 3 seconds
- [ ] No long white screen on initial load

### API Response Time
- [ ] Overview endpoint < 1s (check Network tab)
- [ ] Trading endpoint < 1s
- [ ] Risk endpoint < 500ms
- [ ] System endpoint < 500ms

### Memory Usage
- [ ] Open browser Task Manager
- [ ] Dashboard tab uses < 500MB memory
- [ ] No memory leaks (memory stable after 5 minutes)

---

## 10. EDGE CASES

### Zero Positions
- [ ] Close all positions on Binance testnet
- [ ] Dashboard shows "0 positions"
- [ ] Trading tab shows "No open positions" placeholder
- [ ] No errors or NaN

### Zero Equity
- [ ] (If possible to test) Dashboard shows $0.00
- [ ] No NaN or division errors

### High Loss Day
- [ ] (If possible) Trigger ESS by hitting -5% loss
- [ ] ESS status changes to ACTIVE
- [ ] Warning banner appears
- [ ] Triggers today increments

---

## FINAL SIGN-OFF

### All Tests Passed
- [ ] Backend API tests pass: `pytest tests/api/ -v`
- [ ] Integration tests pass: `pytest tests/integrations/dashboard/ -v`
- [ ] Frontend tests pass: `cd frontend && npm test`
- [ ] Manual checklist items complete

### Known Issues
Document any issues found:
```
Issue #1: [Description]
Severity: [Low/Medium/High]
Workaround: [If any]

Issue #2: ...
```

### Approval
- [ ] QA Engineer Sign-off: _________________ Date: _______
- [ ] Product Owner Sign-off: _________________ Date: _______
- [ ] Tech Lead Sign-off: _________________ Date: _______

---

**NOTES:**
- Complete this checklist AFTER running all automated tests
- Test on fresh browser session (clear cache)
- Test with Binance testnet having active positions (10+ positions recommended)
- Report any NaN, blank screens, or crashes immediately
- Save screenshots of any issues found
