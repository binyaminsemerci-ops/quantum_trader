"""
Position Monitor Integration for AdaptiveLeverageEngine
Patch to add dynamic TP/SL updates using adaptive leverage levels
"""

# INTEGRATION POINT: backend/services/monitoring/position_monitor.py
# Location: Inside _set_tpsl_for_position method (around line 800-900)

# ==================== STEP 1: Add Import at Top of File ====================
# Add after existing imports (around line 85):

try:
    from backend.domains.exits.exit_brain_v3.v35_integration import get_v35_integration
    EXITBRAIN_V35_AVAILABLE = True
    logger_v35 = logging.getLogger(__name__ + ".exitbrain_v35")
    logger_v35.info("[OK] ExitBrain v3.5 (AdaptiveLeverageEngine) available")
except ImportError as e:
    EXITBRAIN_V35_AVAILABLE = False
    logger_v35 = logging.getLogger(__name__ + ".exitbrain_v35")
    logger_v35.warning(f"[WARNING] ExitBrain v3.5 not available: {e}")


# ==================== STEP 2: Initialize v3.5 Integration in __init__ ====================
# Add in PositionMonitor.__init__ method (around line 130):

        # [NEW] ADAPTIVE LEVERAGE: Initialize ExitBrain v3.5 integration
        self.v35_integration = None
        if EXITBRAIN_V35_AVAILABLE:
            try:
                self.v35_integration = get_v35_integration()
                logger_v35.info("[OK] ExitBrain v3.5 integration initialized")
            except Exception as e:
                logger_v35.error(f"[ERROR] Failed to initialize v3.5: {e}")


# ==================== STEP 3: Add Adaptive Level Calculation Method ====================
# Add as new method in PositionMonitor class (around line 600):

    def _calculate_adaptive_levels(
        self,
        symbol: str,
        side: str,
        leverage: float,
        entry_price: float,
        position_size: float
    ) -> Dict:
        """
        Calculate adaptive TP/SL levels using leverage-aware engine.
        
        Returns dict with:
        - tp1_price, tp2_price, tp3_price
        - sl_price
        - tp1_qty, tp2_qty, tp3_qty
        - harvest_scheme
        - lsf (Leverage Scaling Factor)
        """
        if not EXITBRAIN_V35_AVAILABLE or not self.v35_integration:
            logger_v35.debug(f"[{symbol}] v3.5 not available, using defaults")
            return None
        
        try:
            # Calculate volatility factor from recent price action
            volatility_factor = self._estimate_volatility(symbol)
            
            # Get adaptive levels from v3.5
            adaptive_levels = self.v35_integration.compute_adaptive_levels(
                leverage=leverage,
                volatility_factor=volatility_factor
            )
            
            # Calculate actual prices from percentages
            direction = 1 if side == "LONG" else -1
            
            tp1_price = entry_price * (1 + direction * adaptive_levels['tp1_pct'])
            tp2_price = entry_price * (1 + direction * adaptive_levels['tp2_pct'])
            tp3_price = entry_price * (1 + direction * adaptive_levels['tp3_pct'])
            sl_price = entry_price * (1 - direction * adaptive_levels['sl_pct'])
            
            # Calculate quantities based on harvest scheme
            harvest = adaptive_levels['harvest_scheme']
            tp1_qty = position_size * harvest[0]
            tp2_qty = position_size * harvest[1]
            tp3_qty = position_size * harvest[2]
            
            logger_v35.info(
                f"[{symbol}] Adaptive Levels | {leverage:.1f}x | "
                f"LSF={adaptive_levels['lsf']:.4f} | "
                f"TP1={adaptive_levels['tp1_pct']*100:.2f}% "
                f"TP2={adaptive_levels['tp2_pct']*100:.2f}% "
                f"TP3={adaptive_levels['tp3_pct']*100:.2f}% | "
                f"SL={adaptive_levels['sl_pct']*100:.2f}% | "
                f"Harvest={harvest}"
            )
            
            return {
                'tp1_price': tp1_price,
                'tp2_price': tp2_price,
                'tp3_price': tp3_price,
                'sl_price': sl_price,
                'tp1_qty': tp1_qty,
                'tp2_qty': tp2_qty,
                'tp3_qty': tp3_qty,
                'harvest_scheme': harvest,
                'lsf': adaptive_levels['lsf'],
                'tp1_pct': adaptive_levels['tp1_pct'],
                'tp2_pct': adaptive_levels['tp2_pct'],
                'tp3_pct': adaptive_levels['tp3_pct'],
                'sl_pct': adaptive_levels['sl_pct']
            }
            
        except Exception as e:
            logger_v35.error(f"[{symbol}] Error calculating adaptive levels: {e}")
            return None
    
    def _estimate_volatility(self, symbol: str) -> float:
        """
        Estimate current volatility for symbol.
        Uses recent price changes or defaults to 0.3 (medium volatility).
        """
        try:
            # Get recent klines (last 20 periods)
            klines = self.client.futures_klines(
                symbol=symbol,
                interval='5m',
                limit=20
            )
            
            if not klines or len(klines) < 2:
                return 0.3  # Default medium volatility
            
            # Calculate price changes
            closes = [float(k[4]) for k in klines]
            changes = []
            for i in range(1, len(closes)):
                change = abs(closes[i] - closes[i-1]) / closes[i-1]
                changes.append(change)
            
            # Average absolute change as volatility proxy
            avg_change = sum(changes) / len(changes) if changes else 0
            
            # Scale to 0-1 range (typical crypto moves 0.1%-2% per 5min)
            # 0.5% change = 0.3 volatility (medium)
            # 1.0% change = 0.6 volatility (high)
            # 2.0% change = 1.0 volatility (extreme)
            volatility = min(avg_change * 60, 1.0)  # Cap at 1.0
            
            return volatility
            
        except Exception as e:
            logger_v35.warning(f"[{symbol}] Error estimating volatility: {e}")
            return 0.3  # Default medium volatility


# ==================== STEP 4: Use Adaptive Levels in TP/SL Setting ====================
# Modify _set_tpsl_for_position method (around line 850-900)
# Replace hardcoded TP/SL calculation with adaptive levels:

    async def _set_tpsl_for_position(self, position: Dict):
        """Set TP/SL orders for a position using adaptive leverage engine"""
        symbol = position['symbol']
        side = position['positionSide']
        entry_price = float(position['entryPrice'])
        position_amt = abs(float(position['positionAmt']))
        leverage = float(position.get('leverage', 10))
        
        # [NEW] ADAPTIVE LEVERAGE: Calculate adaptive levels
        adaptive = self._calculate_adaptive_levels(
            symbol=symbol,
            side=side,
            leverage=leverage,
            entry_price=entry_price,
            position_size=position_amt
        )
        
        if adaptive:
            # Use adaptive levels
            logger_v35.info(
                f"[{symbol}] Using adaptive levels | "
                f"LSF={adaptive['lsf']:.4f} | "
                f"TP1=${adaptive['tp1_price']:.2f} "
                f"TP2=${adaptive['tp2_price']:.2f} "
                f"TP3=${adaptive['tp3_price']:.2f} | "
                f"SL=${adaptive['sl_price']:.2f}"
            )
            
            # Place TP orders (multi-stage)
            try:
                # TP1 (first harvest)
                await self._place_take_profit_order(
                    symbol=symbol,
                    side='SELL' if side == 'LONG' else 'BUY',
                    quantity=adaptive['tp1_qty'],
                    price=adaptive['tp1_price'],
                    stage='TP1'
                )
                
                # TP2 (second harvest)
                await self._place_take_profit_order(
                    symbol=symbol,
                    side='SELL' if side == 'LONG' else 'BUY',
                    quantity=adaptive['tp2_qty'],
                    price=adaptive['tp2_price'],
                    stage='TP2'
                )
                
                # TP3 (final harvest)
                await self._place_take_profit_order(
                    symbol=symbol,
                    side='SELL' if side == 'LONG' else 'BUY',
                    quantity=adaptive['tp3_qty'],
                    price=adaptive['tp3_price'],
                    stage='TP3'
                )
                
            except Exception as e:
                logger_v35.error(f"[{symbol}] Error placing TP orders: {e}")
            
            # Place SL order
            try:
                await self._place_stop_loss_order(
                    symbol=symbol,
                    side='SELL' if side == 'LONG' else 'BUY',
                    quantity=position_amt,
                    stop_price=adaptive['sl_price']
                )
            except Exception as e:
                logger_v35.error(f"[{symbol}] Error placing SL order: {e}")
            
        else:
            # Fallback to default TP/SL logic
            logger_v35.warning(f"[{symbol}] Adaptive levels unavailable, using defaults")
            # ... existing default TP/SL logic ...


# ==================== STEP 5: Add Dynamic TP/SL Update ====================
# Add new method to periodically update TP/SL based on changing conditions:

    async def _update_dynamic_tpsl(self, position: Dict):
        """
        Update TP/SL levels dynamically based on current conditions.
        Called periodically (every check_interval) for open positions.
        """
        if not EXITBRAIN_V35_AVAILABLE or not self.v35_integration:
            return
        
        symbol = position['symbol']
        side = position['positionSide']
        leverage = float(position.get('leverage', 10))
        entry_price = float(position['entryPrice'])
        current_price = float(position['markPrice'])
        position_amt = abs(float(position['positionAmt']))
        
        # Calculate current PnL percentage
        direction = 1 if side == 'LONG' else -1
        pnl_pct = direction * (current_price - entry_price) / entry_price
        
        # Only update if position is in profit >0.5%
        if pnl_pct > 0.005:
            try:
                # Recalculate adaptive levels with current volatility
                adaptive = self._calculate_adaptive_levels(
                    symbol=symbol,
                    side=side,
                    leverage=leverage,
                    entry_price=current_price,  # Use current price as new baseline
                    position_size=position_amt
                )
                
                if adaptive:
                    logger_v35.info(
                        f"[{symbol}] Dynamic TP/SL update | "
                        f"PnL={pnl_pct*100:.2f}% | "
                        f"New TP1=${adaptive['tp1_price']:.2f} "
                        f"SL=${adaptive['sl_price']:.2f}"
                    )
                    
                    # Cancel existing orders and place new ones
                    # (Implementation depends on existing order management)
                    
            except Exception as e:
                logger_v35.error(f"[{symbol}] Error updating dynamic TP/SL: {e}")


# ==================== STEP 6: Call Dynamic Update in Monitor Loop ====================
# In monitor_loop method, add after checking positions (around line 300):

            # [NEW] ADAPTIVE LEVERAGE: Dynamic TP/SL update for profitable positions
            if EXITBRAIN_V35_AVAILABLE and self.v35_integration:
                for position in positions:
                    await self._update_dynamic_tpsl(position)


# ==================== USAGE NOTES ====================
"""
After integration:

1. ExitBrain v3.5 will provide leverage-aware TP/SL levels
2. Position Monitor will use adaptive levels when setting protection
3. Multi-stage harvesting (TP1/TP2/TP3) automatically configured
4. Dynamic updates adjust TP/SL as position moves into profit
5. Volatility-based adjustments happen automatically

Monitoring:
- Watch logs for "[OK] ExitBrain v3.5 integration initialized"
- Check for "Adaptive Levels" log entries with LSF values
- Monitor harvest scheme distribution per leverage tier
- Verify multi-stage TP orders placed correctly

Configuration:
- Tune adaptive_leverage_config.py parameters
- Adjust check_interval in PositionMonitor (default 10s)
- Set minimum PnL threshold for dynamic updates (default 0.5%)
"""
