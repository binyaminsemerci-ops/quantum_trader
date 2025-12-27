"""Trading Profile API endpoints.

RESTful API for the Trading Profile System:
- Universe filtering and ranking
- Symbol validation
- Trade validation (liquidity + funding)
- Dynamic TP/SL calculation
- Position sizing
- Configuration management

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

try:
    from services.ai.trading_profile import (
        SymbolMetrics,
        UniverseTier,
        validate_trade,
        filter_and_rank_universe,
        compute_dynamic_tpsl_long,
        compute_dynamic_tpsl_short,
        compute_position_margin,
        compute_effective_leverage,
        compute_position_size,
        compute_spread,
        compute_spread_bps,
        compute_liquidity_score,
        is_symbol_tradeable,
        classify_symbol_tier,
        calculate_risk_reward_ratio,
    )
    from services.binance_market_data import (
        create_market_data_fetcher,
        calculate_atr,
        calculate_atr_percentage,
    )
    from config.trading_profile import (
        get_trading_profile_config,
        load_trading_profile_config,
    )
except ImportError:
    from backend.services.ai.trading_profile import (
        SymbolMetrics,
        UniverseTier,
        validate_trade,
        filter_and_rank_universe,
        compute_dynamic_tpsl_long,
        compute_dynamic_tpsl_short,
        compute_position_margin,
        compute_effective_leverage,
        compute_position_size,
        compute_spread,
        compute_spread_bps,
        compute_liquidity_score,
        is_symbol_tradeable,
        classify_symbol_tier,
        calculate_risk_reward_ratio,
    )
    from backend.services.binance_market_data import (
        create_market_data_fetcher,
        calculate_atr,
        calculate_atr_percentage,
    )
    from backend.config.trading_profile import (
        get_trading_profile_config,
        load_trading_profile_config,
    )

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trading-profile",
    tags=["Trading Profile"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"},
    },
)

# Initialize market data fetcher globally
try:
    market_data_fetcher = create_market_data_fetcher()
    logger.info("✅ Trading Profile: Market data fetcher initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize market data fetcher: {e}")
    market_data_fetcher = None


# ==================== Pydantic Models ====================

class SymbolInfo(BaseModel):
    """Symbol information response."""
    symbol: str
    quote_volume_24h: float
    spread_bps: float
    depth_notional: float
    funding_rate: float
    next_funding_time: datetime
    mark_price: float
    universe_tier: str
    liquidity_score: float
    tradeable: bool
    rejection_reason: Optional[str] = None


class UniverseResponse(BaseModel):
    """Universe listing response."""
    symbols: List[SymbolInfo]
    total_count: int
    filtered_count: int
    timestamp: datetime


class ValidateTradeRequest(BaseModel):
    """Request to validate a trade."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: str = Field(..., description="Trade side: 'LONG' or 'SHORT'")


class ValidateTradeResponse(BaseModel):
    """Trade validation response."""
    valid: bool
    reason: str
    symbol_info: Optional[SymbolInfo] = None


class TpslRequest(BaseModel):
    """Request to calculate TP/SL levels."""
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="'LONG' or 'SHORT'")
    entry_price: float = Field(..., gt=0, description="Entry price")
    atr: Optional[float] = Field(None, gt=0, description="ATR value (auto-calculated if not provided)")


class TpslLevels(BaseModel):
    """TP/SL levels response."""
    sl_init: float
    tp1: float
    tp2: float
    tp3: Optional[float] = None
    be_trigger: float
    be_price: float
    trail_activation: float
    trail_distance: float
    partial_close_frac_tp1: float
    partial_close_frac_tp2: float
    risk_r: float
    reward_r_tp1: float
    reward_r_tp2: float
    atr_used: float


class TpslResponse(BaseModel):
    """TP/SL calculation response."""
    symbol: str
    side: str
    entry_price: float
    levels: TpslLevels


class PositionSizeRequest(BaseModel):
    """Request to calculate position size."""
    symbol: str = Field(..., description="Trading symbol")
    equity: float = Field(..., gt=0, description="Account equity in USDT")
    ai_risk_factor: float = Field(default=1.0, ge=0.5, le=1.5, description="AI risk factor (0.5-1.5)")
    entry_price: Optional[float] = Field(None, gt=0, description="Entry price (fetched if not provided)")


class PositionSizeResponse(BaseModel):
    """Position size calculation response."""
    symbol: str
    equity: float
    ai_risk_factor: float
    margin: float
    margin_pct: float
    effective_leverage: float
    quantity: float
    notional: float
    entry_price: float


class ConfigResponse(BaseModel):
    """Trading profile configuration response."""
    enabled: bool
    auto_universe_update_seconds: int
    liquidity: Dict
    risk: Dict
    tpsl: Dict
    funding: Dict


# ==================== API Endpoints ====================

@router.get("/universe", response_model=UniverseResponse)
async def get_universe(
    max_symbols: int = Query(default=20, ge=1, le=100, description="Max symbols to return"),
    include_metrics: bool = Query(default=True, description="Include detailed metrics"),
) -> UniverseResponse:
    """
    Get current tradeable universe (filtered and ranked by liquidity).
    
    Returns top symbols that pass all liquidity filters, ranked by liquidity score.
    """
    if not market_data_fetcher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data fetcher not available"
        )
    
    try:
        config = get_trading_profile_config()
        
        # Fetch all symbols
        logger.info(f"Fetching universe (max {max_symbols} symbols)...")
        all_metrics = market_data_fetcher.fetch_universe_metrics(max_symbols=max_symbols * 3)
        
        if not all_metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch market data"
            )
        
        # Filter and rank
        tradeable = filter_and_rank_universe(all_metrics, config.liquidity)
        
        # Limit to max_symbols
        tradeable = tradeable[:max_symbols]
        
        # Build response
        symbols = []
        for metrics in tradeable:
            spread_bps = compute_spread_bps(metrics)
            liq_score = compute_liquidity_score(metrics, config.liquidity)
            valid, reason = is_symbol_tradeable(metrics, config.liquidity)
            
            symbols.append(SymbolInfo(
                symbol=metrics.symbol,
                quote_volume_24h=metrics.quote_volume_24h,
                spread_bps=spread_bps,
                depth_notional=metrics.depth_notional_5bps,
                funding_rate=metrics.funding_rate,
                next_funding_time=metrics.next_funding_time,
                mark_price=metrics.mark_price,
                universe_tier=metrics.universe_tier.value,
                liquidity_score=liq_score,
                tradeable=valid,
                rejection_reason=reason if not valid else None
            ))
        
        return UniverseResponse(
            symbols=symbols,
            total_count=len(all_metrics),
            filtered_count=len(tradeable),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error fetching universe: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch universe: {str(e)}"
        )


@router.get("/symbol/{symbol}", response_model=SymbolInfo)
async def get_symbol_info(symbol: str) -> SymbolInfo:
    """
    Get detailed information about a specific symbol.
    
    Includes liquidity metrics, funding info, and tradeability status.
    """
    if not market_data_fetcher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data fetcher not available"
        )
    
    try:
        config = get_trading_profile_config()
        
        # Fetch symbol metrics
        metrics = market_data_fetcher.fetch_symbol_metrics(symbol)
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol {symbol} not found or no data available"
            )
        
        # Calculate metrics
        spread_bps = compute_spread_bps(metrics)
        liq_score = compute_liquidity_score(metrics, config.liquidity)
        valid, reason = is_symbol_tradeable(metrics, config.liquidity)
        
        return SymbolInfo(
            symbol=metrics.symbol,
            quote_volume_24h=metrics.quote_volume_24h,
            spread_bps=spread_bps,
            depth_notional=metrics.depth_notional_5bps,
            funding_rate=metrics.funding_rate,
            next_funding_time=metrics.next_funding_time,
            mark_price=metrics.mark_price,
            universe_tier=metrics.universe_tier.value,
            liquidity_score=liq_score,
            tradeable=valid,
            rejection_reason=reason if not valid else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching symbol info for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch symbol info: {str(e)}"
        )


@router.post("/validate", response_model=ValidateTradeResponse)
async def validate_trade_endpoint(request: ValidateTradeRequest) -> ValidateTradeResponse:
    """
    Validate if a trade is allowed based on trading profile filters.
    
    Checks:
    - Liquidity (volume, spread, depth)
    - Funding window (time-based blocking)
    - Funding rate (side-specific thresholds)
    """
    if not market_data_fetcher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data fetcher not available"
        )
    
    try:
        config = get_trading_profile_config()
        
        # Fetch symbol metrics
        metrics = market_data_fetcher.fetch_symbol_metrics(request.symbol)
        
        if not metrics:
            return ValidateTradeResponse(
                valid=False,
                reason=f"Failed to fetch data for {request.symbol}",
                symbol_info=None
            )
        
        # Validate trade
        valid, reason = validate_trade(
            metrics,
            request.side,
            config.liquidity,
            config.funding
        )
        
        # Build symbol info
        spread_bps = compute_spread_bps(metrics)
        liq_score = compute_liquidity_score(metrics, config.liquidity)
        
        symbol_info = SymbolInfo(
            symbol=metrics.symbol,
            quote_volume_24h=metrics.quote_volume_24h,
            spread_bps=spread_bps,
            depth_notional=metrics.depth_notional_5bps,
            funding_rate=metrics.funding_rate,
            next_funding_time=metrics.next_funding_time,
            mark_price=metrics.mark_price,
            universe_tier=metrics.universe_tier.value,
            liquidity_score=liq_score,
            tradeable=valid,
            rejection_reason=reason if not valid else None
        )
        
        return ValidateTradeResponse(
            valid=valid,
            reason=reason,
            symbol_info=symbol_info
        )
        
    except Exception as e:
        logger.error(f"Error validating trade {request.symbol} {request.side}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate trade: {str(e)}"
        )


@router.post("/tpsl", response_model=TpslResponse)
async def calculate_tpsl(request: TpslRequest) -> TpslResponse:
    """
    Calculate dynamic TP/SL levels based on ATR.
    
    Returns multi-target setup:
    - SL: 1R (1x ATR)
    - TP1: 1.5R (partial close 50%)
    - TP2: 2.5R (partial close 30%, activate trailing)
    - Break-even: Triggered at 1R
    - Trailing: 0.8R distance from price
    """
    try:
        config = get_trading_profile_config()
        
        # Get ATR if not provided
        if request.atr is None:
            atr = calculate_atr(
                request.symbol,
                period=config.tpsl.atr_period,
                timeframe=config.tpsl.atr_timeframe
            )
            if not atr:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to calculate ATR for {request.symbol}"
                )
        else:
            atr = request.atr
        
        # Calculate TP/SL levels
        if request.side.upper() == 'LONG':
            levels = compute_dynamic_tpsl_long(
                request.entry_price,
                atr,
                config.tpsl
            )
        elif request.side.upper() == 'SHORT':
            levels = compute_dynamic_tpsl_short(
                request.entry_price,
                atr,
                config.tpsl
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid side: {request.side}. Must be 'LONG' or 'SHORT'"
            )
        
        # Build response
        tpsl_levels = TpslLevels(
            sl_init=levels.sl_init,
            tp1=levels.tp1,
            tp2=levels.tp2,
            tp3=levels.tp3,
            be_trigger=levels.be_trigger,
            be_price=levels.be_price,
            trail_activation=levels.trail_activation,
            trail_distance=levels.trail_distance,
            partial_close_frac_tp1=levels.partial_close_frac_tp1,
            partial_close_frac_tp2=levels.partial_close_frac_tp2,
            risk_r=levels.risk_r,
            reward_r_tp1=levels.reward_r_tp1,
            reward_r_tp2=levels.reward_r_tp2,
            atr_used=atr
        )
        
        return TpslResponse(
            symbol=request.symbol,
            side=request.side,
            entry_price=request.entry_price,
            levels=tpsl_levels
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating TP/SL for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate TP/SL: {str(e)}"
        )


@router.post("/position-size", response_model=PositionSizeResponse)
async def calculate_position_size_endpoint(request: PositionSizeRequest) -> PositionSizeResponse:
    """
    Calculate optimal position size based on trading profile.
    
    Factors in:
    - Account equity
    - AI risk factor (0.5-1.5x)
    - Symbol tier (determines leverage)
    - Risk limits (min/max margin)
    """
    if not market_data_fetcher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data fetcher not available"
        )
    
    try:
        config = get_trading_profile_config()
        
        # Fetch symbol metrics
        metrics = market_data_fetcher.fetch_symbol_metrics(request.symbol)
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol {request.symbol} not found"
            )
        
        # Get entry price
        entry_price = request.entry_price if request.entry_price else metrics.mark_price
        
        # Calculate margin
        margin = compute_position_margin(
            request.equity,
            request.ai_risk_factor,
            config.risk
        )
        
        # Calculate effective leverage
        effective_leverage = compute_effective_leverage(metrics, config.risk)
        
        # Calculate position size
        quantity = compute_position_size(margin, effective_leverage, entry_price)
        
        notional = quantity * entry_price
        margin_pct = (margin / request.equity) * 100
        
        return PositionSizeResponse(
            symbol=request.symbol,
            equity=request.equity,
            ai_risk_factor=request.ai_risk_factor,
            margin=margin,
            margin_pct=margin_pct,
            effective_leverage=effective_leverage,
            quantity=quantity,
            notional=notional,
            entry_price=entry_price
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating position size for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate position size: {str(e)}"
        )


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get current trading profile configuration.
    
    Returns all settings: liquidity, risk, TP/SL, funding.
    """
    try:
        config = get_trading_profile_config()
        
        return ConfigResponse(
            enabled=config.enabled,
            auto_universe_update_seconds=config.auto_universe_update_seconds,
            liquidity={
                "min_quote_volume_24h": config.liquidity.min_quote_volume_24h,
                "max_spread_bps": config.liquidity.max_spread_bps,
                "min_depth_notional": config.liquidity.min_depth_notional,
                "w_volume": config.liquidity.w_volume,
                "w_spread": config.liquidity.w_spread,
                "w_depth": config.liquidity.w_depth,
                "min_liquidity_score": config.liquidity.min_liquidity_score,
                "max_universe_size": config.liquidity.max_universe_size,
                "allowed_tiers": [t.value for t in config.liquidity.allowed_tiers]
            },
            risk={
                "base_risk_frac": config.risk.base_risk_frac,
                "max_risk_frac": config.risk.max_risk_frac,
                "min_margin": config.risk.min_margin,
                "max_margin": config.risk.max_margin,
                "max_total_risk_frac": config.risk.max_total_risk_frac,
                "max_positions": config.risk.max_positions,
                "min_ai_risk_factor": config.risk.min_ai_risk_factor,
                "max_ai_risk_factor": config.risk.max_ai_risk_factor,
                "default_leverage": config.risk.default_leverage,
                "effective_leverage_main": config.risk.effective_leverage_main,
                "effective_leverage_l1": config.risk.effective_leverage_l1,
                "effective_leverage_l2": config.risk.effective_leverage_l2,
                "effective_leverage_min": config.risk.effective_leverage_min
            },
            tpsl={
                "atr_period": config.tpsl.atr_period,
                "atr_timeframe": config.tpsl.atr_timeframe,
                "atr_mult_base": config.tpsl.atr_mult_base,
                "atr_mult_sl": config.tpsl.atr_mult_sl,
                "atr_mult_tp1": config.tpsl.atr_mult_tp1,
                "atr_mult_tp2": config.tpsl.atr_mult_tp2,
                "atr_mult_tp3": config.tpsl.atr_mult_tp3,
                "atr_mult_be": config.tpsl.atr_mult_be,
                "be_buffer_bps": config.tpsl.be_buffer_bps,
                "trail_dist_mult": config.tpsl.trail_dist_mult,
                "trail_activation_mult": config.tpsl.trail_activation_mult,
                "partial_close_tp1": config.tpsl.partial_close_tp1,
                "partial_close_tp2": config.tpsl.partial_close_tp2
            },
            funding={
                "pre_window_minutes": config.funding.pre_window_minutes,
                "post_window_minutes": config.funding.post_window_minutes,
                "min_long_funding": config.funding.min_long_funding,
                "max_short_funding": config.funding.max_short_funding,
                "extreme_funding_threshold": config.funding.extreme_funding_threshold,
                "high_funding_threshold": config.funding.high_funding_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch config: {str(e)}"
        )


@router.put("/config/reload")
async def reload_config(from_env: bool = Query(default=True, description="Reload from environment variables")) -> Dict:
    """
    Reload trading profile configuration.
    
    Useful for applying .env changes without restarting the service.
    """
    try:
        if from_env:
            config = load_trading_profile_config(from_env=True)
            source = "environment variables"
        else:
            config = load_trading_profile_config(from_env=False)
            source = "JSON profile"
        
        logger.info(f"✅ Trading profile config reloaded from {source}")
        
        return {
            "success": True,
            "message": f"Configuration reloaded from {source}",
            "enabled": config.enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload config: {str(e)}"
        )
