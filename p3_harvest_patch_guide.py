#!/usr/bin/env python3
"""
P3 Harvest Restore - Final Action Handler Patch
Inserts P2 action handlers into harvest_brain.py evaluate() method
"""

import sys

P2_ACTION_HANDLERS = '''
        # Handle P2 harvest actions
        exit_side = 'SELL' if position.side == 'LONG' else 'BUY'
        
        if harvest_action == 'FULL_CLOSE_PROPOSED':
            # Kill score triggered → full close
            intent = HarvestIntent(
                intent_type='FULL_CLOSE_PROPOSED',
                symbol=position.symbol,
                side=exit_side,
                qty=position.qty,
                reason=f'KILL_SCORE={kill_score:.3f} | {" ".join(reason_codes)}',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"kill:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"harvest:{position.symbol}:full:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.warning(
                f"🔴 FULL_CLOSE: {position.symbol} | "
                f"KILL_SCORE={kill_score:.3f} > threshold | "
                f"R={r_net:.2f}R"
            )
        
        elif harvest_action == 'PARTIAL_25':
            intent = HarvestIntent(
                intent_type='PARTIAL_25',
                symbol=position.symbol,
                side=exit_side,
                qty=position.qty * 0.25,
                reason=f'R={r_net:.2f}R >= 2.0R (T1)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"harvest:{position.symbol}:25:{int(position.last_update_ts)}",
                trace_id=f"harvest:{position.symbol}:partial25:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.info(f"🟡 PARTIAL_25: {position.symbol} | R={r_net:.2f}R")
        
        elif harvest_action == 'PARTIAL_50':
            intent = HarvestIntent(
                intent_type='PARTIAL_50',
                symbol=position.symbol,
                side=exit_side,
                qty=position.qty * 0.50,
                reason=f'R={r_net:.2f}R >= 4.0R (T2)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"harvest:{position.symbol}:50:{int(position.last_update_ts)}",
                trace_id=f"harvest:{position.symbol}:partial50:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.info(f"🟠 PARTIAL_50: {position.symbol} | R={r_net:.2f}R")
        
        elif harvest_action == 'PARTIAL_75':
            intent = HarvestIntent(
                intent_type='PARTIAL_75',
                symbol=position.symbol,
                side=exit_side,
                qty=position.qty * 0.75,
                reason=f'R={r_net:.2f}R >= 6.0R (T3)',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"harvest:{position.symbol}:75:{int(position.last_update_ts)}",
                trace_id=f"harvest:{position.symbol}:partial75:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(intent)
            logger.info(f"🔴 PARTIAL_75: {position.symbol} | R={r_net:.2f}R")
        
        # Handle profit lock SL adjustment
        new_sl_proposed = p2_result['new_sl_proposed']
        if new_sl_proposed:
            sl_intent = HarvestIntent(
                intent_type='PROFIT_LOCK_SL',
                symbol=position.symbol,
                side='MOVE_SL',
                qty=position.qty,
                reason=f'Profit lock @ R={r_net:.2f}R → SL={new_sl_proposed:.2f}',
                r_level=r_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"sl_lock:{position.symbol}:{int(position.last_update_ts)}",
                trace_id=f"harvest:{position.symbol}:sl_lock:{int(position.last_update_ts)}",
                dry_run=(self.config.harvest_mode == 'shadow')
            )
            intents.append(sl_intent)
            logger.info(
                f"📍 PROFIT_LOCK: {position.symbol} | "
                f"SL {position.stop_loss:.2f} → {new_sl_proposed:.2f} | "
                f"R={r_net:.2f}R"
            )
        
        return intents
    
    def _get_market_state(self, symbol: str) -> MarketState:
        """Fetch market state from Redis or return defaults"""
        try:
            key = f"quantum:market:{symbol}"
            data = self.redis.hgetall(key)
            
            if data:
                return MarketState(
                    sigma=float(data.get('sigma', 0.01)),
                    ts=float(data.get('ts', 0.35)),
                    p_trend=float(data.get('p_trend', 0.5)),
                    p_mr=float(data.get('p_mr', 0.3)),
                    p_chop=float(data.get('p_chop', 0.2))
                )
        except Exception as e:
            logger.debug(f"Failed to fetch market state for {symbol}: {e}")
        
        # Default market state (neutral)
        return MarketState(
            sigma=0.01,
            ts=0.35,
            p_trend=0.5,
            p_mr=0.3,
            p_chop=0.2
        )
    
    def _get_harvest_theta(self) -> HarvestTheta:
        """Get harvest theta from config or defaults"""
        return HarvestTheta(
            fallback_stop_pct=0.02,
            cost_bps=10.0,
            T1_R=2.0,
            T2_R=4.0,
            T3_R=6.0,
            lock_R=1.5,
            be_plus_pct=0.002,
            kill_threshold=0.6
        )
'''

print(f"""
P3 HARVEST RESTORE - FINAL PATCH
================================

Status: harvest_brain.py already has:
✅ P2 risk_kernel imports (lines 28-33)
✅ Position model extensions (lines 147-163)
✅ Evaluate method with P2 kernel call (lines 480-530)
✅ Stream migration to apply.result (line 61)

Remaining: Insert P2 action handlers (lines 533-638)

ACTION HANDLER CODE TO INSERT:
{P2_ACTION_HANDLERS}

MANUAL STEPS:
1. Open microservices/harvest_brain/harvest_brain.py
2. Find line 533 (after "if r_net < self.config.min_r: return intents")
3. Delete lines 537-638 (old ladder logic)
4. Insert the P2_ACTION_HANDLERS code above
5. Save file

OR use VSCode replace:
- Search for: "# Check for trailing stop opportunity"
- Replace with: "# Handle P2 harvest actions"
- Then paste full P2_ACTION_HANDLERS block
""")
