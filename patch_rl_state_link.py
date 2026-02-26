#!/usr/bin/env python3
"""
Patch 3 files to link order_id → entry state → RL experience.

Problem: RL agent records every experience with state={}, so the model
trains blind — it can't learn WHICH conditions led to good/bad PnL.

Fix:
  1. intent_bridge: forward confidence + regime from source_payload into apply.plan
  2. intent_executor: after ENTRY fill, store quantum:rl:state:{order_id} in Redis
  3. rl_agent_daemon: look up that state by order_id, pass real state to record_experience
"""
import shutil, re

# ─────────────────────────────────────────────────────────────
# 1. INTENT_BRIDGE — forward confidence + regime to apply.plan
# ─────────────────────────────────────────────────────────────
BRIDGE = "/home/qt/quantum_trader/microservices/intent_bridge/main.py"
shutil.copy(BRIDGE, BRIDGE + ".bak_rl_state")
with open(BRIDGE) as f:
    src = f.read()

# Find the block that adds atr/volatility_factor and insert after it
OLD_BRIDGE = '''        if _atr:
            message_fields[b"atr_value"] = str(_atr).encode()
        if _vol:
            message_fields[b"volatility_factor"] = str(_vol).encode()'''

NEW_BRIDGE = '''        if _atr:
            message_fields[b"atr_value"] = str(_atr).encode()
        if _vol:
            message_fields[b"volatility_factor"] = str(_vol).encode()

        # RL STATE: forward confidence + regime so intent_executor can store
        # order_id → state mapping for correct RL experience attribution
        _conf = _src.get("confidence") or intent.get("confidence")
        _regime = _src.get("regime") or intent.get("regime")
        if _conf is not None:
            message_fields[b"confidence"] = str(_conf).encode()
            logger.debug(f"✓ RL state: confidence={_conf} forwarded to {intent['symbol']}")
        if _regime is not None:
            message_fields[b"regime"] = str(_regime).encode()
            logger.debug(f"✓ RL state: regime={_regime} forwarded to {intent['symbol']}")'''

assert OLD_BRIDGE in src, "BRIDGE patch anchor not found — check intent_bridge/main.py"
src = src.replace(OLD_BRIDGE, NEW_BRIDGE, 1)
with open(BRIDGE, "w") as f:
    f.write(src)
print("✅ 1/3  intent_bridge patched — confidence+regime forwarded to apply.plan")


# ─────────────────────────────────────────────────────────────
# 2. INTENT_EXECUTOR — store RL state after ENTRY fill
# ─────────────────────────────────────────────────────────────
EXECUTOR = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
shutil.copy(EXECUTOR, EXECUTOR + ".bak_rl_state")
with open(EXECUTOR) as f:
    src = f.read()

OLD_EXEC = '''                if final_status == "FILLED":
                    logger.info(f"🔍 DEBUG: Calling ledger commit for {symbol} order_id={order_id}")
                    self._commit_ledger_exactly_once(symbol, order_id, final_filled, side)'''

NEW_EXEC = '''                if final_status == "FILLED":
                    logger.info(f"🔍 DEBUG: Calling ledger commit for {symbol} order_id={order_id}")
                    self._commit_ledger_exactly_once(symbol, order_id, final_filled, side)

                    # RL STATE: store entry context keyed by order_id
                    # rl_agent_daemon looks this up when trade.closed arrives
                    # so the RL model trains with the real state at trade open time
                    if not reduce_only and order_id:
                        try:
                            _rl_state = {
                                "symbol": symbol,
                                "side": side,
                                "leverage": event_data.get(b"leverage", b"1.0").decode() if isinstance(event_data.get(b"leverage"), bytes) else str(event_data.get(b"leverage", 1.0)),
                                "confidence": event_data.get(b"confidence", b"0.5").decode() if isinstance(event_data.get(b"confidence"), bytes) else str(event_data.get(b"confidence", 0.5)),
                                "regime": event_data.get(b"regime", b"unknown").decode() if isinstance(event_data.get(b"regime"), bytes) else str(event_data.get(b"regime", "unknown")),
                                "entry_price": str(mark_price),
                                "qty": str(qty_to_use),
                                "opened_at": str(time.time()),
                            }
                            self.redis.setex(
                                f"quantum:rl:state:{order_id}",
                                7 * 24 * 3600,  # 7-day TTL
                                json.dumps(_rl_state)
                            )
                            logger.info(
                                f"📊 RL state stored: order={order_id} {symbol} {side} "
                                f"conf={_rl_state['confidence']} lev={_rl_state['leverage']} "
                                f"regime={_rl_state['regime']}"
                            )
                        except Exception as _e:
                            logger.warning(f"RL state write failed for order {order_id}: {_e}")'''

assert OLD_EXEC in src, "EXECUTOR patch anchor not found — check indent in intent_executor/main.py"
src = src.replace(OLD_EXEC, NEW_EXEC, 1)
with open(EXECUTOR, "w") as f:
    f.write(src)
print("✅ 2/3  intent_executor patched — stores quantum:rl:state:{order_id} after ENTRY fill")


# ─────────────────────────────────────────────────────────────
# 3. RL_AGENT_DAEMON — look up state by order_id in process_closed_positions
# ─────────────────────────────────────────────────────────────
DAEMON = "/home/qt/quantum_trader/microservices/rl_sizing_agent/rl_agent_daemon.py"
shutil.copy(DAEMON, DAEMON + ".bak_rl_state")
with open(DAEMON) as f:
    src = f.read()

OLD_DAEMON = '''                    try:
                        symbol = fields.get("symbol", "")
                        pnl = float(fields.get("pnl_usd", 0))
                        entry_price = float(fields.get("entry_price", 0))
                        close_price = float(fields.get("exit_price", 0))
                        leverage = float(fields.get("leverage", 1))

                        if entry_price > 0:
                            # Calculate PnL percentage
                            pnl_pct = ((close_price - entry_price) / entry_price
) * 100 * leverage
                            # Simple reward: PnL percentage
                            reward = pnl_pct / 100.0  # Normalize to -1 to +1 ra
nge roughly
                            logger.info(
                                f"Closed position: {symbol} "
                                f"PnL={pnl:.2f} ({pnl_pct:+.2f}%) "
                                f"reward={reward:+.4f}"
                            )

                            # Publish to RL rewards stream for other consumers
                            self.redis_client.xadd(
                                "quantum:stream:rl_rewards",
                                {
                                    "symbol": symbol,
                                    "reward": reward,
                                    "pnl": pnl,
                                    "pnl_pct": pnl_pct,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            )

                            self.experiences_processed += 1'''

# Verify the patch anchor exists
assert OLD_DAEMON in src, "DAEMON patch anchor not found — run: cat -A to see exact whitespace"

NEW_DAEMON = '''                    try:
                        symbol = fields.get("symbol", "")
                        pnl = float(fields.get("pnl_usd", 0))
                        entry_price = float(fields.get("entry_price", 0))
                        close_price = float(fields.get("exit_price", 0))
                        order_id = fields.get("order_id", "")

                        # RL STATE LOOKUP: retrieve market context at entry time
                        # intent_executor writes quantum:rl:state:{order_id} when order fills
                        # This is the key link: order → state → reward → learning
                        entry_state = {}
                        if order_id and order_id not in ("", "?", "None"):
                            try:
                                raw = self.redis_client.get(f"quantum:rl:state:{order_id}")
                                if raw:
                                    entry_state = json.loads(raw)
                                    logger.debug(
                                        f"RL state found for order {order_id}: "
                                        f"conf={entry_state.get('confidence')} "
                                        f"regime={entry_state.get('regime')} "
                                        f"lev={entry_state.get('leverage')}"
                                    )
                            except Exception as _e:
                                logger.warning(f"RL state lookup failed for order {order_id}: {_e}")

                        # Use stored entry state if available, fall back to trade.closed fields
                        confidence = float(entry_state.get("confidence") or fields.get("confidence", 0.5))
                        leverage = float(entry_state.get("leverage") or fields.get("leverage", 1.0))
                        regime = entry_state.get("regime", fields.get("regime", "unknown"))

                        # Build state vector for record_experience
                        state = {
                            "confidence": confidence,
                            "leverage": leverage,
                            "regime": regime,
                            "volatility": 1.0,    # not in trade.closed, use neutral default
                            "pnl_trend": 0.0,
                            "exch_divergence": 0.0,
                            "funding_rate": 0.0,
                            "margin_util": 0.0,
                        }

                        if entry_price > 0:
                            # Calculate PnL percentage using actual leverage at entry
                            pnl_pct = ((close_price - entry_price) / entry_price) * 100 * leverage
                            reward = pnl_pct / 100.0  # normalize to [-1, 1] range roughly

                            logger.info(
                                f"Closed position: {symbol} "
                                f"PnL={pnl:.2f} ({pnl_pct:+.2f}%) "
                                f"reward={reward:+.4f} "
                                f"conf={confidence:.2f} lev={leverage:.1f}x "
                                f"regime={regime} "
                                f"{'[FROM_STATE]' if entry_state else '[FALLBACK]'}"
                            )

                            # Record experience with real state context
                            self.agent.record_experience(
                                state=state,
                                action=leverage,          # action = leverage chosen at entry
                                pnl_pct=pnl_pct / 100.0, # already a fraction
                                next_state=state.copy(),
                                leverage=leverage,
                                target_leverage=leverage,
                            )

                            # Publish to RL rewards stream for other consumers
                            self.redis_client.xadd(
                                "quantum:stream:rl_rewards",
                                {
                                    "symbol": symbol,
                                    "reward": reward,
                                    "pnl": pnl,
                                    "pnl_pct": pnl_pct,
                                    "confidence": confidence,
                                    "leverage": leverage,
                                    "regime": regime,
                                    "state_source": "entry_state" if entry_state else "fallback",
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            )

                            self.experiences_processed += 1'''

with open(DAEMON, "w") as f:
    f.write(src.replace(OLD_DAEMON, NEW_DAEMON, 1))
# verify
with open(DAEMON) as f:
    check = f.read()
assert "quantum:rl:state:{order_id}" in check
print("✅ 3/3  rl_agent_daemon patched — looks up entry state by order_id")

print()
print("All 3 patches applied. Restart services:")
print("  systemctl restart quantum-intent-bridge quantum-intent-executor quantum-rl-agent")
