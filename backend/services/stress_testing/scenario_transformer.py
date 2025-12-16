"""
Scenario Transformer - Applies stress conditions to market data
"""

import logging
import pandas as pd
import numpy as np
from datetime import timedelta

from .scenario_models import Scenario, ScenarioType

logger = logging.getLogger(__name__)


class ScenarioTransformer:
    """
    Applies transformations to market data to create stress conditions.
    
    Supports:
    - Flash crashes
    - Volatility spikes
    - Liquidity drops
    - Spread explosions
    - Trend reversals
    - Data corruption
    - Correlation breakdowns
    """
    
    def __init__(self):
        """Initialize transformer"""
        logger.info("[SST] ScenarioTransformer initialized")
    
    def apply(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Apply stress transformation to data.
        
        Args:
            df: Original market data
            scenario: Scenario definition
            
        Returns:
            Transformed DataFrame with stress conditions applied
        """
        logger.info(f"[SST] Applying transformation: {scenario.type.value}")
        
        # Copy to avoid mutating original
        df = df.copy()
        
        # Add stress metadata columns
        df["stressed"] = False
        df["stress_type"] = ""
        df["original_close"] = df["close"]
        
        # Apply transformation based on scenario type
        if scenario.type == ScenarioType.FLASH_CRASH:
            df = self._apply_flash_crash(df, scenario)
        
        elif scenario.type == ScenarioType.VOLATILITY_SPIKE:
            df = self._apply_volatility_spike(df, scenario)
        
        elif scenario.type == ScenarioType.TREND_SHIFT:
            df = self._apply_trend_shift(df, scenario)
        
        elif scenario.type == ScenarioType.LIQUIDITY_DROP:
            df = self._apply_liquidity_drop(df, scenario)
        
        elif scenario.type == ScenarioType.SPREAD_EXPLOSION:
            df = self._apply_spread_explosion(df, scenario)
        
        elif scenario.type == ScenarioType.DATA_CORRUPTION:
            df = self._apply_data_corruption(df, scenario)
        
        elif scenario.type == ScenarioType.CORRELATION_BREAKDOWN:
            df = self._apply_correlation_breakdown(df, scenario)
        
        elif scenario.type == ScenarioType.PUMP_DUMP:
            df = self._apply_pump_dump(df, scenario)
        
        elif scenario.type == ScenarioType.MIXED_CUSTOM:
            df = self._apply_mixed_custom(df, scenario)
        
        elif scenario.type == ScenarioType.HISTORIC_REPLAY:
            # No transformation for historical replay
            pass
        
        logger.info(f"[SST] Transformation complete: {df['stressed'].sum()} bars affected")
        return df
    
    def _apply_flash_crash(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Simulate a flash crash.
        
        Parameters:
            drop_pct: Percentage drop (default 0.15 = 15%)
            duration_bars: How many bars crash lasts (default 3)
            recovery_bars: Bars to recover (default 10)
        """
        drop_pct = scenario.parameters.get("drop_pct", 0.15)
        duration = scenario.parameters.get("duration_bars", 3)
        recovery = scenario.parameters.get("recovery_bars", 10)
        
        # Find crash point (middle 50% of data)
        n = len(df)
        crash_start = n // 4 + np.random.randint(0, n // 2)
        crash_end = min(crash_start + duration, n)
        recovery_end = min(crash_end + recovery, n)
        
        logger.info(
            f"[SST] Flash crash: {drop_pct*100:.1f}% drop "
            f"from bar {crash_start} to {crash_end}"
        )
        
        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            symbol_df = df[mask].copy()
            
            # Get pre-crash price
            pre_crash_close = symbol_df.iloc[crash_start-1]["close"]
            crash_low = pre_crash_close * (1 - drop_pct)
            
            # Apply crash
            for i in range(crash_start, crash_end):
                if i >= len(symbol_df):
                    break
                
                idx = symbol_df.index[i]
                progress = (i - crash_start) / max(duration - 1, 1)
                
                # Price drops
                crash_close = pre_crash_close * (1 - drop_pct * progress)
                df.at[idx, "close"] = crash_close
                df.at[idx, "low"] = min(crash_close * 0.98, crash_low)
                df.at[idx, "high"] = crash_close * 1.01
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "flash_crash"
            
            # Recovery phase
            for i in range(crash_end, recovery_end):
                if i >= len(symbol_df):
                    break
                
                idx = symbol_df.index[i]
                progress = (i - crash_end) / max(recovery - 1, 1)
                
                # Price recovers
                recovered_close = crash_low + (pre_crash_close - crash_low) * progress
                df.at[idx, "close"] = recovered_close
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "flash_crash_recovery"
        
        return df
    
    def _apply_volatility_spike(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Multiply volatility by a factor.
        
        Parameters:
            multiplier: Volatility multiplier (default 3.0)
            duration_bars: How long spike lasts (default 20)
        """
        multiplier = scenario.parameters.get("multiplier", 3.0)
        duration = scenario.parameters.get("duration_bars", 20)
        
        # Find spike point
        n = len(df)
        spike_start = n // 4 + np.random.randint(0, n // 2)
        spike_end = min(spike_start + duration, n)
        
        logger.info(
            f"[SST] Volatility spike: {multiplier}x from bar {spike_start} to {spike_end}"
        )
        
        for symbol in df["symbol"].unique():
            mask = (df["symbol"] == symbol) & (df.index >= spike_start) & (df.index < spike_end)
            
            # Calculate typical range
            typical_range = df[df["symbol"] == symbol]["close"].pct_change().std()
            
            # Apply increased volatility
            for idx in df[mask].index:
                # Increase high-low spread
                mid = df.at[idx, "close"]
                current_range = abs(df.at[idx, "high"] - df.at[idx, "low"])
                new_range = current_range * multiplier
                
                df.at[idx, "high"] = mid + new_range / 2
                df.at[idx, "low"] = mid - new_range / 2
                
                # Add noise to close
                noise = np.random.normal(0, typical_range * multiplier)
                df.at[idx, "close"] = mid * (1 + noise)
                
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "volatility_spike"
        
        return df
    
    def _apply_trend_shift(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Reverse price trend direction.
        
        Parameters:
            shift_magnitude: How strong the reversal (default 0.20 = 20%)
            duration_bars: Length of new trend (default 30)
        """
        magnitude = scenario.parameters.get("shift_magnitude", 0.20)
        duration = scenario.parameters.get("duration_bars", 30)
        
        # Find shift point
        n = len(df)
        shift_start = n // 3
        shift_end = min(shift_start + duration, n)
        
        logger.info(f"[SST] Trend shift: {magnitude*100:.1f}% reversal at bar {shift_start}")
        
        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            symbol_df = df[mask].copy()
            
            if shift_start >= len(symbol_df):
                continue
            
            # Get pre-shift price and trend
            pre_shift_close = symbol_df.iloc[shift_start]["close"]
            
            # Calculate historical trend
            if shift_start >= 10:
                recent_prices = symbol_df.iloc[shift_start-10:shift_start]["close"]
                trend_direction = 1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1
            else:
                trend_direction = 1
            
            # Apply reversed trend
            for i in range(shift_start, shift_end):
                if i >= len(symbol_df):
                    break
                
                idx = symbol_df.index[i]
                progress = (i - shift_start) / max(duration - 1, 1)
                
                # Move price in opposite direction
                new_close = pre_shift_close * (1 - trend_direction * magnitude * progress)
                df.at[idx, "close"] = new_close
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "trend_shift"
        
        return df
    
    def _apply_liquidity_drop(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Reduce trading volume dramatically.
        
        Parameters:
            drop_pct: Volume reduction (default 0.90 = 90% drop)
            duration_bars: How long (default 15)
        """
        drop_pct = scenario.parameters.get("drop_pct", 0.90)
        duration = scenario.parameters.get("duration_bars", 15)
        
        # Find drop point
        n = len(df)
        drop_start = n // 3 + np.random.randint(0, n // 3)
        drop_end = min(drop_start + duration, n)
        
        logger.info(f"[SST] Liquidity drop: {drop_pct*100:.1f}% from bar {drop_start}")
        
        mask = (df.index >= drop_start) & (df.index < drop_end)
        df.loc[mask, "volume"] *= (1 - drop_pct)
        df.loc[mask, "stressed"] = True
        df.loc[mask, "stress_type"] = "liquidity_drop"
        
        return df
    
    def _apply_spread_explosion(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Widen bid-ask spreads.
        
        Parameters:
            spread_mult: Spread multiplier (default 5.0)
            duration_bars: How long (default 10)
        """
        spread_mult = scenario.parameters.get("spread_mult", 5.0)
        duration = scenario.parameters.get("duration_bars", 10)
        
        n = len(df)
        spread_start = n // 3
        spread_end = min(spread_start + duration, n)
        
        logger.info(f"[SST] Spread explosion: {spread_mult}x at bar {spread_start}")
        
        # Store spread multiplier in metadata (used by ExchangeSimulator)
        mask = (df.index >= spread_start) & (df.index < spread_end)
        df.loc[mask, "stressed"] = True
        df.loc[mask, "stress_type"] = "spread_explosion"
        df["spread_multiplier"] = 1.0
        df.loc[mask, "spread_multiplier"] = spread_mult
        
        return df
    
    def _apply_data_corruption(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Inject data quality issues.
        
        Parameters:
            corruption_pct: Percentage of bars to corrupt (default 0.05 = 5%)
            corruption_types: Types to inject (default: ["nan", "spike", "duplicate"])
        """
        corruption_pct = scenario.parameters.get("corruption_pct", 0.05)
        corruption_types = scenario.parameters.get(
            "corruption_types",
            ["nan", "spike", "duplicate"]
        )
        
        n_corrupt = int(len(df) * corruption_pct)
        corrupt_indices = np.random.choice(df.index, size=n_corrupt, replace=False)
        
        logger.info(f"[SST] Data corruption: {n_corrupt} bars affected")
        
        for idx in corrupt_indices:
            corruption_type = np.random.choice(corruption_types)
            
            if corruption_type == "nan":
                # Inject NaN values
                df.at[idx, "close"] = np.nan
            
            elif corruption_type == "spike":
                # Price spike
                df.at[idx, "close"] *= np.random.choice([0.5, 2.0])
            
            elif corruption_type == "duplicate":
                # Duplicate previous bar
                if idx > 0:
                    prev_idx = df.index[df.index.get_loc(idx) - 1]
                    df.at[idx, "close"] = df.at[prev_idx, "close"]
            
            df.at[idx, "stressed"] = True
            df.at[idx, "stress_type"] = f"data_corruption_{corruption_type}"
        
        return df
    
    def _apply_correlation_breakdown(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Break correlation between symbols.
        
        Parameters:
            duration_bars: How long breakdown lasts (default 20)
        """
        duration = scenario.parameters.get("duration_bars", 20)
        
        n = len(df)
        breakdown_start = n // 3
        breakdown_end = min(breakdown_start + duration, n)
        
        logger.info(f"[SST] Correlation breakdown at bar {breakdown_start}")
        
        symbols = df["symbol"].unique()
        if len(symbols) < 2:
            logger.warning("[SST] Need multiple symbols for correlation breakdown")
            return df
        
        # Make one symbol move opposite to others
        target_symbol = symbols[0]
        
        for i in range(breakdown_start, breakdown_end):
            for symbol in symbols:
                mask = (df["symbol"] == symbol) & (df.index == i)
                if not mask.any():
                    continue
                
                idx = df[mask].index[0]
                
                if symbol == target_symbol:
                    # Invert movement
                    if i > 0:
                        prev_mask = (df["symbol"] == symbol) & (df.index == i-1)
                        if prev_mask.any():
                            prev_idx = df[prev_mask].index[0]
                            prev_close = df.at[prev_idx, "close"]
                            current_close = df.at[idx, "close"]
                            change = current_close - prev_close
                            df.at[idx, "close"] = prev_close - change
                
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "correlation_breakdown"
        
        return df
    
    def _apply_pump_dump(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Simulate pump-and-dump pattern.
        
        Parameters:
            pump_pct: Pump magnitude (default 0.50 = 50%)
            pump_duration: Pump phase bars (default 10)
            dump_duration: Dump phase bars (default 5)
        """
        pump_pct = scenario.parameters.get("pump_pct", 0.50)
        pump_duration = scenario.parameters.get("pump_duration", 10)
        dump_duration = scenario.parameters.get("dump_duration", 5)
        
        n = len(df)
        pump_start = n // 3
        pump_end = min(pump_start + pump_duration, n)
        dump_end = min(pump_end + dump_duration, n)
        
        logger.info(f"[SST] Pump & dump: +{pump_pct*100:.1f}% at bar {pump_start}")
        
        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            symbol_df = df[mask].copy()
            
            if pump_start >= len(symbol_df):
                continue
            
            base_price = symbol_df.iloc[pump_start]["close"]
            
            # Pump phase
            for i in range(pump_start, pump_end):
                if i >= len(symbol_df):
                    break
                
                idx = symbol_df.index[i]
                progress = (i - pump_start) / max(pump_duration - 1, 1)
                pumped_price = base_price * (1 + pump_pct * progress)
                df.at[idx, "close"] = pumped_price
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "pump"
            
            # Dump phase
            peak_price = base_price * (1 + pump_pct)
            for i in range(pump_end, dump_end):
                if i >= len(symbol_df):
                    break
                
                idx = symbol_df.index[i]
                progress = (i - pump_end) / max(dump_duration - 1, 1)
                dumped_price = peak_price * (1 - pump_pct * progress)
                df.at[idx, "close"] = dumped_price
                df.at[idx, "stressed"] = True
                df.at[idx, "stress_type"] = "dump"
        
        return df
    
    def _apply_mixed_custom(self, df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
        """
        Apply multiple stress conditions simultaneously.
        
        Parameters can include any combination of:
            flash_crash_drop_pct
            volatility_multiplier
            spread_mult
            liquidity_drop_pct
            noise_level
        """
        logger.info("[SST] Applying mixed custom stress conditions")
        
        # Flash crash component
        if "flash_crash_drop_pct" in scenario.parameters:
            flash_scenario = Scenario(
                name="flash_component",
                type=ScenarioType.FLASH_CRASH,
                parameters={"drop_pct": scenario.parameters["flash_crash_drop_pct"]}
            )
            df = self._apply_flash_crash(df, flash_scenario)
        
        # Volatility component
        if "volatility_multiplier" in scenario.parameters:
            vol_scenario = Scenario(
                name="vol_component",
                type=ScenarioType.VOLATILITY_SPIKE,
                parameters={"multiplier": scenario.parameters["volatility_multiplier"]}
            )
            df = self._apply_volatility_spike(df, vol_scenario)
        
        # Spread component
        if "spread_mult" in scenario.parameters:
            spread_scenario = Scenario(
                name="spread_component",
                type=ScenarioType.SPREAD_EXPLOSION,
                parameters={"spread_mult": scenario.parameters["spread_mult"]}
            )
            df = self._apply_spread_explosion(df, spread_scenario)
        
        # Liquidity component
        if "liquidity_drop_pct" in scenario.parameters:
            liq_scenario = Scenario(
                name="liq_component",
                type=ScenarioType.LIQUIDITY_DROP,
                parameters={"drop_pct": scenario.parameters["liquidity_drop_pct"]}
            )
            df = self._apply_liquidity_drop(df, liq_scenario)
        
        return df
