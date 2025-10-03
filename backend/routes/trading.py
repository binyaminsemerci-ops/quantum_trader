"""
API routes for controlling the Binance trading engine
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
import asyncio

# Use fully-qualified import to avoid ModuleNotFoundError when working directory differs
from backend.services.binance_trading import get_trading_engine

logger = logging.getLogger(__name__)
router = APIRouter()

# Global reference to the trading task
trading_task = None


@router.get("/status")
async def get_trading_status() -> Dict[str, Any]:
    """Get current trading engine status"""
    try:
        engine = get_trading_engine()
        return engine.get_trading_status()
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_trading(
    background_tasks: BackgroundTasks, interval_minutes: int = 5
) -> Dict[str, str]:
    """Start automated trading"""
    global trading_task

    try:
        engine = get_trading_engine()

        if engine.is_running:
            return {"message": "Trading is already running"}

        # Start trading in background
        async def run_trading():
            await engine.start_trading(interval_minutes)

        trading_task = asyncio.create_task(run_trading())

        return {"message": f"Trading started with {interval_minutes} minute intervals"}

    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_trading() -> Dict[str, str]:
    """Stop automated trading"""
    global trading_task

    try:
        engine = get_trading_engine()
        engine.stop_trading()

        if trading_task:
            trading_task.cancel()
            trading_task = None

        return {"message": "Trading stopped"}

    except Exception as e:
        logger.error(f"Error stopping trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{symbol}")
async def analyze_symbol(symbol: str) -> Dict[str, Any]:
    """Analyze a single symbol with AI (no trading)"""
    try:
        engine = get_trading_engine()

        # Get market data
        ohlcv_data = await engine.get_market_data(symbol)
        if not ohlcv_data:
            raise HTTPException(
                status_code=404, detail=f"No market data found for {symbol}"
            )

        # Get AI prediction
        prediction = engine.ai_agent.predict_for_symbol(ohlcv_data)

        return {
            "symbol": symbol,
            "prediction": prediction,
            "current_price": ohlcv_data[-1]["close"],
            "market_data_points": len(ohlcv_data),
        }

    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manual-trade")
async def manual_trade(
    symbol: str, action: str, quantity: Optional[float] = None, force: bool = False
) -> Dict[str, Any]:
    """Execute a manual trade (bypassing AI if force=True)"""
    try:
        engine = get_trading_engine()

        if action not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Action must be BUY or SELL")

        # Get current market data for price
        ohlcv_data = await engine.get_market_data(symbol)
        if not ohlcv_data:
            raise HTTPException(
                status_code=404, detail=f"No market data found for {symbol}"
            )

        current_price = ohlcv_data[-1]["close"]

        # If quantity not provided or force=False, get AI recommendation
        if quantity is None or not force:
            prediction = engine.ai_agent.predict_for_symbol(ohlcv_data)

            if not force and prediction.get("action") != action:
                return {
                    "message": "AI recommendation differs from manual action",
                    "ai_recommendation": prediction,
                    "suggested_action": prediction.get("action"),
                    "confidence": prediction.get("score"),
                    "note": "use force=true to override",
                }

            if quantity is None:
                confidence = prediction.get("score", 0.5)
                quantity = engine.get_position_size(symbol, current_price, confidence)

        # Execute the trade
        result = engine.execute_trade(
            symbol, action, quantity, 1.0 if force else prediction.get("score", 0.5)
        )

        return result

    except Exception as e:
        logger.error(f"Error executing manual trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balances")
async def get_balances() -> Dict[str, float]:
    """Get current account balances"""
    try:
        engine = get_trading_engine()
        return engine.get_account_balance()
    except Exception as e:
        logger.error(f"Error getting balances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_open_positions() -> List[Dict[str, Any]]:
    """Get all open positions"""
    try:
        engine = get_trading_engine()

        # Get futures positions
        positions = []

        # Futures positions
        try:
            futures_positions = engine.client.futures_position_information()
            for pos in futures_positions:
                if float(pos["positionAmt"]) != 0:  # Only open positions
                    positions.append(
                        {
                            "symbol": pos["symbol"],
                            "side": (
                                "LONG" if float(pos["positionAmt"]) > 0 else "SHORT"
                            ),
                            "size": abs(float(pos["positionAmt"])),
                            "entry_price": float(pos["entryPrice"]),
                            "mark_price": float(pos["markPrice"]),
                            "pnl": float(pos["unRealizedProfit"]),
                            "type": "futures",
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not get futures positions: {e}")

        # TODO: Add spot positions logic if needed

        return positions

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-cycle")
async def run_single_cycle() -> Dict[str, Any]:
    """Run a single trading cycle manually"""
    try:
        engine = get_trading_engine()
        results = await engine.run_trading_cycle()

        return {
            "cycle_completed": True,
            "symbols_analyzed": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error running trading cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_trading_symbols() -> List[str]:
    """Get list of symbols being traded"""
    try:
        engine = get_trading_engine()
        return engine.get_trading_symbols()
    except Exception as e:
        logger.error(f"Error getting trading symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-config")
async def update_trading_config(
    max_position_size_usdc: Optional[float] = None,
    min_confidence_threshold: Optional[float] = None,
    risk_per_trade: Optional[float] = None,
) -> Dict[str, str]:
    """Update trading configuration parameters"""
    try:
        engine = get_trading_engine()

        if max_position_size_usdc is not None:
            engine.max_position_size_usdc = max_position_size_usdc

        if min_confidence_threshold is not None:
            if not 0 <= min_confidence_threshold <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Confidence threshold must be between 0 and 1",
                )
            engine.min_confidence_threshold = min_confidence_threshold

        if risk_per_trade is not None:
            if not 0 <= risk_per_trade <= 0.1:  # Max 10% risk per trade
                raise HTTPException(
                    status_code=400,
                    detail="Risk per trade must be between 0 and 0.1 (10%)",
                )
            engine.risk_per_trade = risk_per_trade

        return {"message": "Trading configuration updated successfully"}

    except Exception as e:
        logger.error(f"Error updating trading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai-model-info")
async def get_ai_model_info() -> Dict[str, Any]:
    """Get information about the loaded AI model"""
    try:
        engine = get_trading_engine()
        metadata = engine.ai_agent.get_metadata()

        return {
            "model_loaded": engine.ai_agent.model is not None,
            "scaler_loaded": engine.ai_agent.scaler is not None,
            "model_path": engine.ai_agent.model_path,
            "scaler_path": engine.ai_agent.scaler_path,
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Error getting AI model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
