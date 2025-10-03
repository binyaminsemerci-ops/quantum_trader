"""
Enhanced Data API Routes
Provides endpoints for multi-source market data integration
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime, timezone

from backend.routes.external_data import (
    enhanced_market_data,
    fear_greed_index,
    reddit_sentiment,
    comprehensive_crypto_news,
    on_chain_metrics,
    market_indicators,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enhanced", tags=["enhanced-data"])


# WebSocket connections manager
class EnhancedDataManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.is_broadcasting = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Enhanced data WebSocket connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Enhanced data WebSocket disconnected. Total: {len(self.active_connections)}"
        )

    async def broadcast_enhanced_data(self, data: Dict[str, Any]):
        """Broadcast enhanced data to all connected clients."""
        if not self.active_connections:
            return

        message = {
            "type": "enhanced_data_update",
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send to all connections, remove failed ones
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send enhanced data to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def start_enhanced_broadcast(self, symbols: List[str], interval: int = 300):
        """Start broadcasting enhanced data every 5 minutes."""
        if self.is_broadcasting:
            return

        self.is_broadcasting = True
        logger.info(
            f"🔄 Starting enhanced data broadcast for {symbols} every {interval}s"
        )

        while self.is_broadcasting:
            try:
                # Fetch all enhanced data
                enhanced_data = await enhanced_market_data(symbols)

                if enhanced_data and self.active_connections:
                    await self.broadcast_enhanced_data(enhanced_data)
                    logger.debug(
                        f"📡 Broadcasted enhanced data to {len(self.active_connections)} clients"
                    )

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in enhanced data broadcast: {e}")
                await asyncio.sleep(30)  # Wait before retry

    def stop_enhanced_broadcast(self):
        """Stop broadcasting enhanced data."""
        self.is_broadcasting = False
        logger.info("🛑 Enhanced data broadcast stopped")


# Global manager instance
enhanced_manager = EnhancedDataManager()


@router.get("/market-data")
async def get_enhanced_market_data(symbols: str = "BTC,ETH,ADA,SOL"):
    """Get enhanced market data from multiple sources."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        data = await enhanced_market_data(symbol_list)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Enhanced market data endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fear-greed")
async def get_fear_greed():
    """Get Fear & Greed Index."""
    try:
        data = await fear_greed_index()
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Fear & Greed endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reddit-sentiment")
async def get_reddit_sentiment_endpoint(symbols: str = "BTC,ETH,ADA"):
    """Get Reddit sentiment analysis."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        data = await reddit_sentiment(symbol_list)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Reddit sentiment endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news")
async def get_crypto_news():
    """Get comprehensive crypto news."""
    try:
        data = await comprehensive_crypto_news()
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Crypto news endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/on-chain")
async def get_on_chain_metrics_endpoint(symbols: str = "BTC,ETH"):
    """Get on-chain metrics."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        data = await on_chain_metrics(symbol_list)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"On-chain metrics endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators")
async def get_market_indicators_endpoint():
    """Get global market indicators."""
    try:
        data = await market_indicators()
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Market indicators endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_enhanced_data(symbols: str = "BTC,ETH,ADA,SOL,DOT,LINK"):
    """Get all enhanced data from multiple sources."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Fetch all data concurrently
        tasks = [
            enhanced_market_data(symbol_list),
            fear_greed_index(),
            reddit_sentiment(symbol_list),
            comprehensive_crypto_news(),
            on_chain_metrics(symbol_list),
            market_indicators(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        enhanced_data = {
            "multi_source": results[0] if not isinstance(results[0], Exception) else {},
            "fear_greed": results[1] if not isinstance(results[1], Exception) else {},
            "reddit": results[2] if not isinstance(results[2], Exception) else {},
            "news": results[3] if not isinstance(results[3], Exception) else {},
            "on_chain": results[4] if not isinstance(results[4], Exception) else {},
            "indicators": results[5] if not isinstance(results[5], Exception) else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols_requested": symbol_list,
        }

        return {"success": True, "data": enhanced_data}

    except Exception as e:
        logger.error(f"All enhanced data endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast/start")
async def start_enhanced_broadcast(
    symbols: str = "BTC,ETH,ADA,SOL", interval: int = 300
):
    """Start broadcasting enhanced data via WebSocket."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        if not enhanced_manager.is_broadcasting:
            # Start broadcast in background
            asyncio.create_task(
                enhanced_manager.start_enhanced_broadcast(symbol_list, interval)
            )
            return {
                "success": True,
                "message": f"Enhanced data broadcast started for {symbol_list}",
            }
        else:
            return {
                "success": True,
                "message": "Enhanced data broadcast already running",
            }

    except Exception as e:
        logger.error(f"Start broadcast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast/stop")
async def stop_enhanced_broadcast():
    """Stop broadcasting enhanced data."""
    try:
        enhanced_manager.stop_enhanced_broadcast()
        return {"success": True, "message": "Enhanced data broadcast stopped"}
    except Exception as e:
        logger.error(f"Stop broadcast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def enhanced_data_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time enhanced data."""
    await enhanced_manager.connect(websocket)

    try:
        # Send initial enhanced data
        initial_data = await enhanced_market_data(["BTC", "ETH", "ADA", "SOL"])
        await websocket.send_text(
            json.dumps(
                {
                    "type": "enhanced_data_update",
                    "data": initial_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        enhanced_manager.disconnect(websocket)


# Health check
@router.get("/health")
async def enhanced_data_health():
    """Health check for enhanced data services."""
    try:
        # Test a simple API call
        fear_greed_data = await fear_greed_index()

        return {
            "success": True,
            "status": "healthy",
            "active_connections": len(enhanced_manager.active_connections),
            "broadcasting": enhanced_manager.is_broadcasting,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "test_data_available": bool(fear_greed_data),
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
