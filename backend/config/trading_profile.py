"""
Trading Profile Configuration

Configuration classes and loaders for trading profile system.

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import json

from backend.services.ai.trading_profile import (
    LiquidityConfig,
    RiskConfig,
    TpslConfig,
    FundingConfig,
    UniverseTier
)


@dataclass
class TradingProfileConfig:
    """Complete trading profile configuration."""
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    tpsl: TpslConfig = field(default_factory=TpslConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    
    # Global settings
    enabled: bool = True
    auto_universe_update_seconds: int = 300  # Update universe every 5min
    
    @classmethod
    def from_env(cls) -> TradingProfileConfig:
        """Load configuration from environment variables."""
        
        # Liquidity config
        liquidity = LiquidityConfig(
            min_quote_volume_24h=float(os.getenv('TP_MIN_VOLUME_24H', '5000000')),
            max_spread_bps=float(os.getenv('TP_MAX_SPREAD_BPS', '3.0')),
            min_depth_notional=float(os.getenv('TP_MIN_DEPTH', '200000')),
            w_volume=float(os.getenv('TP_W_VOLUME', '0.5')),
            w_spread=float(os.getenv('TP_W_SPREAD', '0.3')),
            w_depth=float(os.getenv('TP_W_DEPTH', '0.2')),
            min_liquidity_score=float(os.getenv('TP_MIN_LIQUIDITY_SCORE', '0.0')),
            max_universe_size=int(os.getenv('TP_MAX_UNIVERSE_SIZE', '20'))
        )
        
        # Parse allowed tiers
        allowed_tiers_str = os.getenv('TP_ALLOWED_TIERS', 'MAIN,L1,L2')
        allowed_tiers = [
            UniverseTier(tier.strip().lower())
            for tier in allowed_tiers_str.split(',')
        ]
        liquidity.allowed_tiers = allowed_tiers
        
        # Risk config
        risk = RiskConfig(
            base_risk_frac=float(os.getenv('TP_BASE_RISK_FRAC', '0.01')),
            max_risk_frac=float(os.getenv('TP_MAX_RISK_FRAC', '0.03')),
            min_margin=float(os.getenv('TP_MIN_MARGIN', '10.0')),
            max_margin=float(os.getenv('TP_MAX_MARGIN', '1000.0')),
            max_total_risk_frac=float(os.getenv('TP_MAX_TOTAL_RISK', '0.15')),
            max_positions=int(os.getenv('TP_MAX_POSITIONS', '8')),
            min_ai_risk_factor=float(os.getenv('TP_MIN_AI_RISK_FACTOR', '0.5')),
            max_ai_risk_factor=float(os.getenv('TP_MAX_AI_RISK_FACTOR', '1.5')),
            default_leverage=int(os.getenv('TP_DEFAULT_LEVERAGE', '30')),
            effective_leverage_main=float(os.getenv('TP_LEVERAGE_MAIN', '15.0')),
            effective_leverage_l1=float(os.getenv('TP_LEVERAGE_L1', '12.0')),
            effective_leverage_l2=float(os.getenv('TP_LEVERAGE_L2', '10.0')),
            effective_leverage_min=float(os.getenv('TP_LEVERAGE_MIN', '8.0'))
        )
        
        # TP/SL config
        tpsl = TpslConfig(
            atr_mult_base=float(os.getenv('TP_ATR_MULT_BASE', '1.0')),
            atr_mult_sl=float(os.getenv('TP_ATR_MULT_SL', '1.0')),
            atr_mult_tp1=float(os.getenv('TP_ATR_MULT_TP1', '1.5')),
            atr_mult_tp2=float(os.getenv('TP_ATR_MULT_TP2', '2.5')),
            atr_mult_tp3=float(os.getenv('TP_ATR_MULT_TP3', '4.0')),
            atr_mult_be=float(os.getenv('TP_ATR_MULT_BE', '1.0')),
            be_buffer_bps=float(os.getenv('TP_BE_BUFFER_BPS', '5.0')),
            trail_dist_mult=float(os.getenv('TP_TRAIL_DIST_MULT', '0.8')),
            trail_activation_mult=float(os.getenv('TP_TRAIL_ACTIVATION', '2.5')),
            partial_close_tp1=float(os.getenv('TP_PARTIAL_CLOSE_TP1', '0.5')),
            partial_close_tp2=float(os.getenv('TP_PARTIAL_CLOSE_TP2', '0.3')),
            atr_period=int(os.getenv('TP_ATR_PERIOD', '14')),
            atr_timeframe=os.getenv('TP_ATR_TIMEFRAME', '15m')
        )
        
        # Funding config
        funding = FundingConfig(
            pre_window_minutes=int(os.getenv('TP_FUNDING_PRE_WINDOW', '40')),
            post_window_minutes=int(os.getenv('TP_FUNDING_POST_WINDOW', '20')),
            min_long_funding=float(os.getenv('TP_MIN_LONG_FUNDING', '-0.0003')),
            max_short_funding=float(os.getenv('TP_MAX_SHORT_FUNDING', '0.0003')),
            extreme_funding_threshold=float(os.getenv('TP_EXTREME_FUNDING', '0.001')),
            high_funding_threshold=float(os.getenv('TP_HIGH_FUNDING', '0.0005'))
        )
        
        # Global settings
        enabled = os.getenv('TP_ENABLED', 'true').lower() == 'true'
        auto_update = int(os.getenv('TP_AUTO_UPDATE_SECONDS', '300'))
        
        return cls(
            liquidity=liquidity,
            risk=risk,
            tpsl=tpsl,
            funding=funding,
            enabled=enabled,
            auto_universe_update_seconds=auto_update
        )
    
    @classmethod
    def from_json_file(cls, path: Path) -> TradingProfileConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            liquidity=LiquidityConfig(**data.get('liquidity', {})),
            risk=RiskConfig(**data.get('risk', {})),
            tpsl=TpslConfig(**data.get('tpsl', {})),
            funding=FundingConfig(**data.get('funding', {})),
            enabled=data.get('enabled', True),
            auto_universe_update_seconds=data.get('auto_universe_update_seconds', 300)
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'liquidity': {
                'min_quote_volume_24h': self.liquidity.min_quote_volume_24h,
                'max_spread_bps': self.liquidity.max_spread_bps,
                'min_depth_notional': self.liquidity.min_depth_notional,
                'w_volume': self.liquidity.w_volume,
                'w_spread': self.liquidity.w_spread,
                'w_depth': self.liquidity.w_depth,
                'min_liquidity_score': self.liquidity.min_liquidity_score,
                'max_universe_size': self.liquidity.max_universe_size,
                'allowed_tiers': [t.value for t in self.liquidity.allowed_tiers]
            },
            'risk': {
                'base_risk_frac': self.risk.base_risk_frac,
                'max_risk_frac': self.risk.max_risk_frac,
                'min_margin': self.risk.min_margin,
                'max_margin': self.risk.max_margin,
                'max_total_risk_frac': self.risk.max_total_risk_frac,
                'max_positions': self.risk.max_positions,
                'min_ai_risk_factor': self.risk.min_ai_risk_factor,
                'max_ai_risk_factor': self.risk.max_ai_risk_factor,
                'default_leverage': self.risk.default_leverage,
                'effective_leverage_main': self.risk.effective_leverage_main,
                'effective_leverage_l1': self.risk.effective_leverage_l1,
                'effective_leverage_l2': self.risk.effective_leverage_l2,
                'effective_leverage_min': self.risk.effective_leverage_min
            },
            'tpsl': {
                'atr_mult_base': self.tpsl.atr_mult_base,
                'atr_mult_sl': self.tpsl.atr_mult_sl,
                'atr_mult_tp1': self.tpsl.atr_mult_tp1,
                'atr_mult_tp2': self.tpsl.atr_mult_tp2,
                'atr_mult_tp3': self.tpsl.atr_mult_tp3,
                'atr_mult_be': self.tpsl.atr_mult_be,
                'be_buffer_bps': self.tpsl.be_buffer_bps,
                'trail_dist_mult': self.tpsl.trail_dist_mult,
                'trail_activation_mult': self.tpsl.trail_activation_mult,
                'partial_close_tp1': self.tpsl.partial_close_tp1,
                'partial_close_tp2': self.tpsl.partial_close_tp2,
                'atr_period': self.tpsl.atr_period,
                'atr_timeframe': self.tpsl.atr_timeframe
            },
            'funding': {
                'pre_window_minutes': self.funding.pre_window_minutes,
                'post_window_minutes': self.funding.post_window_minutes,
                'min_long_funding': self.funding.min_long_funding,
                'max_short_funding': self.funding.max_short_funding,
                'extreme_funding_threshold': self.funding.extreme_funding_threshold,
                'high_funding_threshold': self.funding.high_funding_threshold
            },
            'enabled': self.enabled,
            'auto_universe_update_seconds': self.auto_universe_update_seconds
        }
    
    def save_to_json(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global instance
_global_config: Optional[TradingProfileConfig] = None


def load_trading_profile_config(
    from_env: bool = True,
    json_path: Optional[Path] = None
) -> TradingProfileConfig:
    """
    Load trading profile configuration.
    
    Args:
        from_env: Load from environment variables
        json_path: Load from JSON file (overrides env)
    
    Returns:
        TradingProfileConfig instance
    """
    global _global_config
    
    if json_path:
        _global_config = TradingProfileConfig.from_json_file(json_path)
    elif from_env:
        _global_config = TradingProfileConfig.from_env()
    else:
        _global_config = TradingProfileConfig()
    
    return _global_config


def get_trading_profile_config() -> TradingProfileConfig:
    """
    Get global trading profile configuration.
    
    Loads from environment if not already loaded.
    """
    global _global_config
    
    if _global_config is None:
        _global_config = TradingProfileConfig.from_env()
    
    return _global_config
