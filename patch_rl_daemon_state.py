#!/usr/bin/env python3
"""
Patch rl_agent_daemon.py: look up quantum:rl:state:{order_id} at trade close
so the RL model trains with the real state that existed when the trade opened.
"""
import shutil, re, json

DAEMON = "/home/qt/quantum_trader/microservices/rl_sizing_agent/rl_agent_daemon.py"
shutil.copy(DAEMON, DAEMON + ".bak_rl_state")

with open(DAEMON) as f:
    src = f.read()

# Use regex with DOTALL to find the inner try-block in process_closed_positions
# Anchor: from pnl_usd line to experiences_processed += 1
PATTERN = re.compile(
    r'(                    try:\n'
    r'                        symbol = fields\.get\("symbol", ""\)\n'
    r'                        pnl = float\(fields\.get\("pnl_usd", 0\)\)\n'
    r'                        entry_price = float\(fields\.get\("entry_price", 0\)\)\n'
    r'                        close_price = float\(fields\.get\("exit_price", 0\)\)\n'
    r'                        leverage = .*?\n'
    r'.*?'
    r'                        if entry_price > 0:.*?'
    r'                            self\.experiences_processed \+= 1\n)',
    re.DOTALL
)

m = PATTERN.search(src)
if not m:
    print("❌ Anchor not found in rl_agent_daemon.py")
    import sys; sys.exit(1)

print(f"Found anchor at chars {m.start()}-{m.end()}")

NEW_BLOCK = '''                    try:
                        symbol = fields.get("symbol", "")
                        pnl = float(fields.get("pnl_usd", 0))
                        entry_price = float(fields.get("entry_price", 0))
                        close_price = float(fields.get("exit_price", 0))
                        order_id = fields.get("order_id", "")

                        # RL STATE LOOKUP: retrieve market context at entry time.
                        # intent_executor writes quantum:rl:state:{order_id} when
                        # an ENTRY order fills. This is the critical link that lets
                        # the RL model learn: "state S at open → action A → reward R"
                        entry_state = {}
                        if order_id and order_id not in ("", "?", "None"):
                            try:
                                raw = self.redis_client.get(f"quantum:rl:state:{order_id}")
                                if raw:
                                    entry_state = json.loads(raw)
                                    logger.debug(
                                        f"[RL-STATE] order={order_id} "
                                        f"conf={entry_state.get('confidence')} "
                                        f"regime={entry_state.get('regime')} "
                                        f"lev={entry_state.get('leverage')}"
                                    )
                            except Exception as _se:
                                logger.warning(f"RL state lookup failed order={order_id}: {_se}")

                        # Use entry-time state if available, fall back to trade.closed fields
                        confidence = float(entry_state.get("confidence") or fields.get("confidence", 0.5))
                        leverage   = float(entry_state.get("leverage")   or fields.get("leverage",    1.0))
                        regime     = entry_state.get("regime") or fields.get("regime", "unknown")

                        # Full state vector for record_experience
                        state = {
                            "confidence":     confidence,
                            "leverage":       leverage,
                            "regime":         regime,
                            "volatility":     1.0,   # not in trade.closed stream
                            "pnl_trend":      0.0,
                            "exch_divergence": 0.0,
                            "funding_rate":   0.0,
                            "margin_util":    0.0,
                        }

                        if entry_price > 0:
                            # PnL% computed with actual leverage used at entry
                            pnl_pct = ((close_price - entry_price) / entry_price) * 100 * leverage
                            reward  = pnl_pct / 100.0  # normalize approximately to [-1,1]

                            logger.info(
                                f"Closed: {symbol} "
                                f"PnL=${pnl:.2f} ({pnl_pct:+.2f}%) "
                                f"reward={reward:+.4f} "
                                f"conf={confidence:.2f} lev={leverage:.1f}x "
                                f"regime={regime} "
                                f"{'[STATE_LINKED]' if entry_state else '[NO_STATE]'}"
                            )

                            # Record experience with real entry-time state context
                            self.agent.record_experience(
                                state=state,
                                action=leverage,          # action = leverage chosen at open
                                pnl_pct=pnl_pct / 100.0,
                                next_state=state.copy(),
                                leverage=leverage,
                                target_leverage=leverage,
                            )

                            # Publish to RL rewards stream for other consumers
                            self.redis_client.xadd(
                                "quantum:stream:rl_rewards",
                                {
                                    "symbol":       symbol,
                                    "reward":       reward,
                                    "pnl":          pnl,
                                    "pnl_pct":      pnl_pct,
                                    "confidence":   confidence,
                                    "leverage":     leverage,
                                    "regime":       regime,
                                    "state_source": "entry_state" if entry_state else "fallback",
                                    "timestamp":    datetime.utcnow().isoformat()
                                }
                            )

                            self.experiences_processed += 1
'''

new_src = src[:m.start()] + NEW_BLOCK + src[m.end():]

with open(DAEMON, "w") as f:
    f.write(new_src)

# Verify
with open(DAEMON) as f:
    check = f.read()

assert "quantum:rl:state:{order_id}" in check, "Patch verification failed"
assert "entry_state" in check, "entry_state not found"
print("✅ rl_agent_daemon patched — looks up entry state by order_id")
print("Restart: systemctl restart quantum-rl-agent")
