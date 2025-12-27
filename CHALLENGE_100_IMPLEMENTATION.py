# CHALLENGE_100 Methods to add at end of dynamic_executor.py (before last line)

    async def _apply_challenge_100_logic(self, state: PositionExitState, pos_data: Dict, mark_price: float):
        """
        Apply $100 Challenge Mode exit logic.
        
        Rules:
        1. Initial SL: entry ± 2*ATR (clamped to max 1.5R loss)
        2. TP1 at +1R: Take 30% partial
        3. After TP1: Move SL to BE+fees
        4. Runner: 70% with 2*ATR trailing
        5. Time stop: Close if not TP1 within 7200s
        6. Liq safety: SL must be before liq price
        """
        import time
        
        state_key = f"{state.symbol}:{state.side}"
        entry_price = state.entry_price or float(pos_data.get('entryPrice', 0))
        
        if entry_price <= 0:
            self.logger.warning(f"[CHALLENGE_100] {state_key}: Invalid entry_price, skipping")
            return
        
        # Get account equity for 1R calculation
        equity_usdt = await self._get_account_equity()
        r_usdt = equity_usdt * self.challenge_risk_pct
        
        # Calculate ATR for this symbol
        atr = await self._calculate_atr(state.symbol)
        
        # 1. INITIAL SL: entry ± 2*ATR (clamped to 1.5R max loss)
        if state.active_sl is None:
            initial_sl = self._calculate_challenge_initial_sl(entry_price, state.side, atr, r_usdt, state.position_size)
            state.active_sl = initial_sl
            self.logger.warning(
                f"[CHALLENGE_100] {state_key}: Initial SL set @ ${initial_sl:.4f} "
                f"(2*ATR distance, clamped to {self.challenge_max_risk_r}R)"
            )
        
        # 2. LIQ SAFETY: Ensure SL is before liquidation price
        liq_price = float(pos_data.get('liquidationPrice', 0))
        if liq_price > 0:
            safe_sl = self._apply_liq_safety(state.side, state.active_sl, liq_price, entry_price)
            if safe_sl != state.active_sl:
                self.logger.error(
                    f"[CHALLENGE_100] {state_key}: SL adjusted for liq safety: "
                    f"${state.active_sl:.4f} → ${safe_sl:.4f} (liq=${liq_price:.4f})"
                )
                state.active_sl = safe_sl
        
        # 3. TP1 TRACKING: Check if +1R profit reached
        unrealized_pnl_usdt = (mark_price - entry_price) * state.position_size if state.side == "LONG" else (entry_price - mark_price) * state.position_size
        tp1_target_usdt = self.challenge_tp1_r * r_usdt
        
        if not state.tp1_taken and unrealized_pnl_usdt >= tp1_target_usdt:
            # TP1 triggered! Set up partial close
            tp1_price = mark_price  # Use current price as TP1 trigger
            tp1_qty_pct = self.challenge_tp1_qty_pct
            
            # Override tp_levels to trigger TP1 in next cycle
            state.tp_levels = [(tp1_price, tp1_qty_pct)]
            state.tp1_taken = True  # Mark as triggered (will execute in _check_and_execute_levels)
            
            self.logger.warning(
                f"[CHALLENGE_100] 🎯 {state_key}: TP1 REACHED @ ${mark_price:.4f} "
                f"(+{unrealized_pnl_usdt:.2f} USDT >= +{tp1_target_usdt:.2f} USDT = +{self.challenge_tp1_r}R) "
                f"- Will take {tp1_qty_pct:.0%} partial"
            )
        
        # 4. AFTER TP1: Move SL to BE+fees
        if state.tp1_taken and len(state.triggered_legs) > 0:
            # TP1 has been executed, move SL to breakeven + fees
            fee_rate = 0.0004  # 0.04% taker fee (approximate)
            notional = entry_price * state.initial_size
            total_fees = notional * fee_rate * 2  # Open + close fees
            fees_per_unit = total_fees / state.initial_size
            
            be_plus_fees = entry_price + fees_per_unit if state.side == "LONG" else entry_price - fees_per_unit
            
            # Only move SL up (LONG) or down (SHORT), never loosen
            should_update = False
            if state.side == "LONG" and (state.active_sl is None or be_plus_fees > state.active_sl):
                should_update = True
            elif state.side == "SHORT" and (state.active_sl is None or be_plus_fees < state.active_sl):
                should_update = True
            
            if should_update:
                state.active_sl = be_plus_fees
                self.logger.warning(
                    f"[CHALLENGE_100] 🛡️ {state_key}: SL moved to BE+fees @ ${be_plus_fees:.4f} "
                    f"(entry=${entry_price:.4f}, fees_per_unit=${fees_per_unit:.6f})"
                )
            
            # 5. RUNNER TRAILING: After TP1, trail runner with 2*ATR
            self._apply_challenge_trailing(state, mark_price, atr)
        
        # 6. TIME STOP: Close if not TP1 within time_stop_sec
        if not state.tp1_taken and state.opened_at_ts is not None:
            elapsed = time.time() - state.opened_at_ts
            if elapsed >= self.challenge_time_stop_sec:
                self.logger.error(
                    f"[CHALLENGE_100] ⏰ {state_key}: TIME STOP triggered "
                    f"(elapsed={elapsed:.0f}s >= {self.challenge_time_stop_sec:.0f}s) - "
                    f"TP1 not reached, closing position"
                )
                # Force full close via setting active_sl to current price
                state.active_sl = mark_price if state.side == "LONG" else mark_price
    
    def _calculate_challenge_initial_sl(self, entry_price: float, side: str, atr: float, r_usdt: float, position_size: float) -> float:
        """Calculate initial SL for CHALLENGE_100 based on 2*ATR, clamped to 1.5R max loss."""
        # Baseline: 2*ATR distance from entry
        if side == "LONG":
            sl_price = entry_price - (2 * atr)
        else:  # SHORT
            sl_price = entry_price + (2 * atr)
        
        # Clamp to max 1.5R loss
        max_loss_usdt = self.challenge_max_risk_r * r_usdt
        max_move_pct = max_loss_usdt / (entry_price * position_size)
        
        if side == "LONG":
            min_sl_price = entry_price * (1 - max_move_pct)
            sl_price = max(sl_price, min_sl_price)  # SL can't be lower than max loss
        else:  # SHORT
            max_sl_price = entry_price * (1 + max_move_pct)
            sl_price = min(sl_price, max_sl_price)  # SL can't be higher than max loss
        
        return sl_price
    
    def _apply_liq_safety(self, side: str, current_sl: float, liq_price: float, entry_price: float) -> float:
        """Ensure SL is before liquidation price with buffer."""
        if liq_price <= 0:
            return current_sl
        
        buffer = entry_price * self.challenge_liq_buffer_pct
        
        if side == "LONG":
            # Liq is below entry, SL must be above liq
            safe_sl = liq_price + buffer
            if current_sl < safe_sl:
                return safe_sl
        else:  # SHORT
            # Liq is above entry, SL must be below liq
            safe_sl = liq_price - buffer
            if current_sl > safe_sl:
                return safe_sl
        
        return current_sl
    
    def _apply_challenge_trailing(self, state: PositionExitState, current_price: float, atr: float):
        """Apply 2*ATR trailing to runner position after TP1."""
        # Track highest/lowest favorable price
        if state.side == "LONG":
            if state.highest_favorable_price is None or current_price > state.highest_favorable_price:
                state.highest_favorable_price = current_price
            
            # Trail SL: HFP - 2*ATR
            new_sl = state.highest_favorable_price - (2 * atr)
            
            # Only move SL up, never down
            if state.active_sl is None or new_sl > state.active_sl:
                old_sl = state.active_sl
                state.active_sl = new_sl
                self.logger.info(
                    f"[CHALLENGE_100] 📈 {state.symbol} LONG: Runner trailing updated - "
                    f"HFP=${state.highest_favorable_price:.4f}, SL ${old_sl:.4f if old_sl else 'None'} → ${new_sl:.4f} "
                    f"(2*ATR=${2*atr:.4f})"
                )
        else:  # SHORT
            if state.highest_favorable_price is None or current_price < state.highest_favorable_price:
                state.highest_favorable_price = current_price
            
            # Trail SL: LFP + 2*ATR
            new_sl = state.highest_favorable_price + (2 * atr)
            
            # Only move SL down, never up
            if state.active_sl is None or new_sl < state.active_sl:
                old_sl = state.active_sl
                state.active_sl = new_sl
                self.logger.info(
                    f"[CHALLENGE_100] 📉 {state.symbol} SHORT: Runner trailing updated - "
                    f"LFP=${state.highest_favorable_price:.4f}, SL ${old_sl:.4f if old_sl else 'None'} → ${new_sl:.4f} "
                    f"(2*ATR=${2*atr:.4f})"
                )
    
    async def _get_account_equity(self) -> float:
        """Get account equity in USDT for 1R calculation."""
        try:
            from backend.integrations.binance.client_wrapper import BinanceClientWrapper
            wrapper = BinanceClientWrapper()
            
            account_info = await wrapper.call_async(
                self.position_source.futures_account
            )
            
            # Get total wallet balance (USDT equity)
            equity = float(account_info.get('totalWalletBalance', 100.0))
            
            self.logger.debug(f"[CHALLENGE_100] Account equity: ${equity:.2f} USDT")
            return equity
        except Exception as e:
            self.logger.error(f"[CHALLENGE_100] Failed to get account equity: {e}, using default 100 USDT")
            return 100.0  # Fallback for $100 challenge
    
    async def _calculate_atr(self, symbol: str, periods: int = 14) -> float:
        """Calculate ATR (Average True Range) for symbol."""
        try:
            # Check cache first
            cached = self._market_data_cache.get(symbol)
            if cached and 'atr' in cached:
                cache_age = datetime.now(timezone.utc).timestamp() - cached['timestamp']
                if cache_age < self._cache_ttl_sec:
                    return cached['atr']
            
            # Fetch recent candles
            from backend.integrations.binance.client_wrapper import BinanceClientWrapper
            wrapper = BinanceClientWrapper()
            
            klines = await wrapper.call_async(
                self.position_source.futures_klines,
                symbol=symbol,
                interval="5m",
                limit=periods + 1
            )
            
            if not klines or len(klines) < 2:
                self.logger.warning(f"[CHALLENGE_100] No klines for {symbol}, using default ATR")
                return 0.01  # 1% default
            
            # Calculate True Range for each period
            trs = []
            for i in range(1, len(klines)):
                high = float(klines[i][2])
                low = float(klines[i][3])
                prev_close = float(klines[i-1][4])
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                trs.append(tr)
            
            # ATR = average of true ranges
            atr = sum(trs) / len(trs) if trs else 0.01
            
            # Cache result
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {}
            self._market_data_cache[symbol]['atr'] = atr
            self._market_data_cache[symbol]['timestamp'] = datetime.now(timezone.utc).timestamp()
            
            self.logger.debug(f"[CHALLENGE_100] {symbol} ATR (14x5m): ${atr:.4f}")
            return atr
        
        except Exception as e:
            self.logger.error(f"[CHALLENGE_100] ATR calculation failed for {symbol}: {e}")
            return 0.01  # Fallback
    
    async def _place_hard_sl_challenge(self, state: PositionExitState, entry_price: float):
        """
        Place hard SL safety net for CHALLENGE_100 mode.
        
        Hard SL is placed 0.3R "behind" soft SL as last-resort protection.
        Only active in CHALLENGE_100 LIVE mode when gateway blocks legacy modules.
        """
        if state.active_sl is None:
            self.logger.warning(
                f"[CHALLENGE_100] {state.symbol} {state.side}: Cannot place hard SL - "
                f"soft SL not set yet"
            )
            return
        
        # Hard SL = soft SL + 0.3R margin (behind soft SL)
        equity = await self._get_account_equity()
        r_usdt = equity * self.challenge_risk_pct
        extra_r_distance = 0.3 * r_usdt / state.position_size  # Convert to price
        
        if state.side == "LONG":
            hard_sl_price = state.active_sl - extra_r_distance
        else:  # SHORT
            hard_sl_price = state.active_sl + extra_r_distance
        
        # Quantize price
        from .precision import quantize_to_tick, get_binance_precision
        tick_size, step_size = get_binance_precision(state.symbol)
        hard_sl_price = quantize_to_tick(hard_sl_price, tick_size)
        
        # Build STOP_MARKET order
        order_side = "SELL" if state.side == "LONG" else "BUY"
        order_params = {
            "symbol": state.symbol,
            "side": order_side,
            "type": "STOP_MARKET",
            "stopPrice": hard_sl_price,
            "closePosition": True,  # Close full remaining position
            "positionSide": state.side
        }
        
        self.logger.warning(
            f"[CHALLENGE_100] 🛡️ {state.symbol} {state.side}: Placing HARD SL safety net "
            f"@ ${hard_sl_price:.4f} (soft_sl=${state.active_sl:.4f}, margin=0.3R)"
        )
        
        try:
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_brain_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="hard_sl_challenge",
                explanation=f"CHALLENGE_100 hard SL safety net @ {hard_sl_price}"
            )
            
            if resp and resp.get('orderId'):
                state.hard_sl_price = hard_sl_price
                state.hard_sl_order_id = str(resp['orderId'])
                self.logger.warning(
                    f"[CHALLENGE_100] ✅ Hard SL placed - orderId={state.hard_sl_order_id}"
                )
        except Exception as e:
            self.logger.error(f"[CHALLENGE_100] Hard SL placement failed: {e}", exc_info=True)
