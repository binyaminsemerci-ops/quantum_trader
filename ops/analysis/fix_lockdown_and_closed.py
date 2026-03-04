#!/usr/bin/env python3
"""
PATCH 1: Add LOCKDOWN gate to intent_bridge/main.py
  - Blocks ENTRY_PROPOSED when quantum:system:mode == LOCKDOWN
  - Allows FULL_CLOSE_PROPOSED through regardless (must be able to exit positions)

PATCH 2: Add trade.closed publish to apply_layer/main.py
  - After every successful close execution, publish to trade.closed stream
  - This fixes the 98% blindspot where monitoring sees 0 of 1822 actual closes
"""
import sys
import os

INTENT_BRIDGE = '/home/qt/quantum_trader/microservices/intent_bridge/main.py'
APPLY_LAYER   = '/opt/quantum/microservices/apply_layer/main.py'

errors = []

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — LOCKDOWN gate in intent_bridge
# ─────────────────────────────────────────────────────────────────────────────
P1_OLD = '''        message_fields[b"action"] = action.encode()
        logger.info(f"📋 Mapped {intent['symbol']} {intent['side']} reduceOnly={intent['reduceOnly']} → action={action}")'''

P1_NEW = '''        message_fields[b"action"] = action.encode()
        logger.info(f"📋 Mapped {intent['symbol']} {intent['side']} reduceOnly={intent['reduceOnly']} → action={action}")

        # LOCKDOWN GATE: Block new entries when system is in LOCKDOWN.
        # Closes (FULL_CLOSE_PROPOSED) are always allowed through so open positions can exit.
        if action == "ENTRY_PROPOSED":
            sys_mode = self.redis.get("quantum:system:mode")
            if sys_mode and sys_mode.decode() if isinstance(sys_mode, bytes) else sys_mode == "LOCKDOWN":
                logger.warning(
                    f"🔒 LOCKDOWN_BLOCKED: {intent['symbol']} {intent['side']} "
                    f"ENTRY_PROPOSED suppressed (system:mode={sys_mode})"
                )
                return'''

content1 = open(INTENT_BRIDGE).read()
if P1_OLD in content1:
    content1 = content1.replace(P1_OLD, P1_NEW, 1)
    open(INTENT_BRIDGE, 'w').write(content1)
    print("PATCH1_INTENT_BRIDGE: OK")
elif 'LOCKDOWN GATE' in content1:
    print("PATCH1_INTENT_BRIDGE: ALREADY_PATCHED")
else:
    print(f"PATCH1_INTENT_BRIDGE: OLD_STRING_NOT_FOUND")
    idx = content1.find("message_fields[b\"action\"]")
    print("  Context:", repr(content1[max(0,idx-20):idx+120]))
    errors.append("patch1")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — trade.closed publish in apply_layer (VPS copy)
# ─────────────────────────────────────────────────────────────────────────────
P2_OLD = '''                                # Publish success result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'True',
                                    'reduceOnly': 'True',
                                    'close_qty': str(close_qty),
                                    'filled_qty': str(filled_qty),
                                    'order_id': str(order_id),
                                    'status': status,
                                    'side': close_side,
                                    'close_pct': str(close_pct),
                                    'timestamp': str(int(time.time()))
                                })'''

P2_NEW = '''                                # Publish success result
                                self.redis.xadd('quantum:stream:apply.result', {
                                    'plan_id': plan_id,
                                    'symbol': symbol,
                                    'action': action,
                                    'executed': 'True',
                                    'reduceOnly': 'True',
                                    'close_qty': str(close_qty),
                                    'filled_qty': str(filled_qty),
                                    'order_id': str(order_id),
                                    'status': status,
                                    'side': close_side,
                                    'close_pct': str(close_pct),
                                    'timestamp': str(int(time.time()))
                                })

                                # FIX: Publish to trade.closed so gate + monitoring can see ALL closes.
                                # Previously 98% of closes were invisible (only autonomous_trader published here).
                                try:
                                    entry_price_raw = self.redis.hget(f"quantum:position:{symbol}", 'entry_price') or b'0'
                                    entry_price = float(entry_price_raw.decode() if isinstance(entry_price_raw, bytes) else entry_price_raw)
                                    avg_fill = float(order_result.get('avgPrice', 0) or order_result.get('price', 0) or 0)
                                    raw_pnl = (avg_fill - entry_price) * filled_qty if position_side == 'LONG' else (entry_price - avg_fill) * filled_qty
                                    self.redis.xadd('trade.closed', {
                                        'symbol': symbol,
                                        'side': position_side,
                                        'close_side': close_side,
                                        'qty': str(filled_qty),
                                        'close_pct': str(close_pct),
                                        'order_id': str(order_id),
                                        'status': status,
                                        'plan_id': plan_id,
                                        'pnl_raw': str(round(raw_pnl, 6)),
                                        'entry_price': str(entry_price),
                                        'exit_price': str(avg_fill),
                                        'source': 'apply_layer',
                                        'timestamp': str(int(time.time()))
                                    }, maxlen=10000)
                                    logger.info(f"[TRADE_CLOSED] Published: {symbol} {position_side} pnl={raw_pnl:.4f} order_id={order_id}")
                                except Exception as _tc_err:
                                    logger.warning(f"[TRADE_CLOSED] Publish failed (non-fatal): {_tc_err}")'''

content2 = open(APPLY_LAYER).read()
if P2_OLD in content2:
    content2 = content2.replace(P2_OLD, P2_NEW, 1)
    open(APPLY_LAYER, 'w').write(content2)
    print("PATCH2_APPLY_LAYER: OK")
elif 'FIX: Publish to trade.closed so gate' in content2:
    print("PATCH2_APPLY_LAYER: ALREADY_PATCHED")
else:
    print("PATCH2_APPLY_LAYER: OLD_STRING_NOT_FOUND")
    idx = content2.find("Publish success result")
    print("  Context:", repr(content2[max(0,idx-20):idx+200]))
    errors.append("patch2")

if errors:
    print(f"\nFAILED: {errors}")
    sys.exit(1)
else:
    print("\nALL_PATCHES_OK")
