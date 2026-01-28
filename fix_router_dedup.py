#!/usr/bin/env python3
"""P0 Hotfix: Fix ai_strategy_router.py dedup to use symbol+side instead of correlation_id"""

import sys

def apply_fix():
    file_path = "/home/qt/quantum_trader/ai_strategy_router.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # OLD: dedup by correlation_id (never matches)
    old_build_dedup = '''    def _build_dedup_key(self, correlation_id: str, trace_id: str, msg_id: str) -> Tuple[str, str, str, str]:
        """Return dedup key components with sane fallbacks."""
        corr = (correlation_id or "").strip()
        trace = (trace_id or "").strip()
        msg = (msg_id or "").strip()

        def _valid(val: str) -> bool:
            return bool(val) and val.lower() not in {"0", "null", "none"}
        
        chosen = corr if _valid(corr) else trace if _valid(trace) else msg
        return chosen, corr, trace, msg'''
    
    # NEW: dedup by symbol+side (matches identical signals)
    new_build_dedup = '''    def _build_dedup_key(self, symbol: str, side: str, correlation_id: str, trace_id: str, msg_id: str) -> Tuple[str, str, str, str]:
        """Return dedup key based on symbol+side for signal fingerprint."""
        corr = (correlation_id or "").strip()
        trace = (trace_id or "").strip()
        msg = (msg_id or "").strip()
        
        # CRITICAL: Use symbol+side as dedup key (not correlation_id)
        # This ensures identical signals are deduplicated even if correlation_id differs
        dedup_key = f"{symbol}:{side}"
        return dedup_key, corr, trace, msg'''
    
    # Replace method
    if old_build_dedup not in content:
        print("ERROR: Could not find old _build_dedup_key method")
        return False
    
    content = content.replace(old_build_dedup, new_build_dedup)
    
    # Update route_decision to pass symbol and side
    old_route_call = '''            dedup_id, corr_id_clean, trace_id_clean, msg_id_clean = self._build_dedup_key(correlation_id, trace_id, msg_id)'''
    
    # Need to move symbol/side extraction BEFORE dedup call
    old_route_block = '''            dedup_id, corr_id_clean, trace_id_clean, msg_id_clean = self._build_dedup_key(correlation_id, trace_id, msg_id)
            was_set = await asyncio.to_thread(
                self.redis.set,
                f"quantum:dedup:trade_intent:{dedup_id}",
                "1",
                nx=True,
                ex=300  # 5 minute TTL to bound cache
            )

            if not was_set:
                logger.warning(
                    f"🔁 DUPLICATE_SKIP key={dedup_id} corr={corr_id_clean} trace={trace_id_clean} msg_id={msg_id_clean}"
                )
                return

            symbol = decision.get("symbol", "").strip() if isinstance(decision, dict) else ""
            side_raw = decision.get("side", decision.get("action", "")).strip() if isinstance(decision, dict) else ""
            side = side_raw.upper()'''
    
    new_route_block = '''            # Extract symbol and side FIRST for dedup key
            symbol = decision.get("symbol", "").strip() if isinstance(decision, dict) else ""
            side_raw = decision.get("side", decision.get("action", "")).strip() if isinstance(decision, dict) else ""
            side = side_raw.upper()
            
            # Validate before dedup (fast fail)
            if not symbol or not side:
                logger.warning(
                    f"⚠️ INVALID_DECISION_DROP symbol={symbol!r} side={side_raw!r} corr={correlation_id}"
                )
                return
            
            # Build dedup key from signal fingerprint (symbol+side)
            dedup_id, corr_id_clean, trace_id_clean, msg_id_clean = self._build_dedup_key(symbol, side, correlation_id, trace_id, msg_id)
            was_set = await asyncio.to_thread(
                self.redis.set,
                f"quantum:dedup:trade_intent:{dedup_id}",
                "1",
                nx=True,
                ex=30  # 30 second TTL (testnet) - prevents rapid-fire duplicates
            )

            if not was_set:
                logger.warning(
                    f"🔁 DUPLICATE_SKIP {symbol} {side} (already published in last 30s) | corr={corr_id_clean}"
                )
                return'''
    
    if old_route_block not in content:
        print("ERROR: Could not find old route_decision dedup block")
        return False
    
    content = content.replace(old_route_block, new_route_block)
    
    # Remove duplicate invalid check
    old_invalid_check = '''            confidence = decision.get("confidence", 0.0)

            if not symbol or not side:
                self._log_invalid_once(
                    f"⚠️ INVALID_DECISION_DROP symbol={symbol!r} side={side_raw!r} corr={corr_id_clean} trace={trace_id_clean} msg_id={msg_id_clean}"
                )
                return

            logger.info(f"📥 AI Decision: {symbol} {side} @ {confidence:.2%} | trace_id={trace_id}")'''
    
    new_invalid_check = '''            confidence = decision.get("confidence", 0.0)

            logger.info(f"📥 AI Decision: {symbol} {side} @ {confidence:.2%} | trace_id={trace_id}")'''
    
    if old_invalid_check in content:
        content = content.replace(old_invalid_check, new_invalid_check)
    
    # Write fixed file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ ai_strategy_router.py fixed")
    print("   - Dedup now uses symbol+side (not correlation_id)")
    print("   - TTL reduced to 30 seconds (from 300)")
    print("   - Symbol/side validated before dedup check")
    return True

if __name__ == "__main__":
    if apply_fix():
        sys.exit(0)
    else:
        sys.exit(1)
