"""
Quantum Services Client — Redis-first data access for all Quantum Trader services.

OP 8 rewrite: Reads live data directly from Redis streams/keys.
Only uses HTTP for services that actually listen (ai_engine:8001, risk_kernel:8070).
"""
import aiohttp
import asyncio
import os
import json
import redis
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class QuantumServicesClient:
    """Redis-first client for aggregating live data from Quantum Trader"""

    def __init__(self, timeout: int = 5):
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Redis — primary data source
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

        # HTTP — only for services with actual HTTP endpoints
        service_host = os.getenv('QUANTUM_SERVICES_HOST', 'localhost')
        self.SERVICES = {
            'ai_engine': f'http://{service_host}:8001',
            'risk': f'http://{service_host}:8070',
            'model_supervisor': f'http://{service_host}:8007',
        }

    # ── HTTP helper (only for ai_engine / risk-kernel / model_supervisor) ──

    async def _get(self, service: str, endpoint: str) -> Optional[Dict[Any, Any]]:
        base_url = self.SERVICES.get(service)
        if not base_url:
            return None
        url = f"{base_url}{endpoint}"
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception:
            return None

    # ── Portfolio (Redis) ──

    async def get_portfolio_summary(self) -> Optional[Dict]:
        """Account balance + position summary from Redis streams"""
        if not self.redis_client:
            return None
        try:
            # Latest account balance from stream
            entries = self.redis_client.xrevrange(
                "quantum:stream:account.balance", count=1
            )
            if not entries:
                return None
            _, data = entries[0]
            return {
                "balance": float(data.get("balance", 0)),
                "available": float(data.get("available", 0)),
                "margin_used": float(data.get("margin_used", 0)),
                "unrealized_pnl": float(data.get("unrealized_pnl", 0)),
                "positions_count": int(data.get("positions", 0)),
                "margin_ratio": float(data.get("margin_ratio", 0)),
                "timestamp": data.get("timestamp", ""),
            }
        except Exception as e:
            logger.warning(f"get_portfolio_summary error: {e}")
            return None

    async def get_portfolio_positions(self) -> Optional[Dict]:
        """Active positions from Redis canonical keys"""
        if not self.redis_client:
            return None
        try:
            positions = []
            keys = self.redis_client.keys("quantum:state:positions:*")
            for key in keys:
                info = self.redis_client.hgetall(key)
                amt = float(info.get("positionAmt", "0"))
                if amt == 0:
                    continue
                symbol = key.replace("quantum:state:positions:", "")
                positions.append({
                    "symbol": symbol,
                    "side": "LONG" if amt > 0 else "SHORT",
                    "size": abs(amt),
                    "entry_price": float(info.get("entryPrice", 0)),
                    "mark_price": float(info.get("markPrice", 0)),
                    "unrealized_pnl": float(info.get("unrealizedProfit", "0")),
                    "leverage": int(float(info.get("leverage", "1"))),
                })
            # Also check ledger for positions P3.3 may not reflect yet
            ledger_keys = self.redis_client.keys("quantum:position:ledger:*")
            seen_symbols = {p["symbol"] for p in positions}
            for key in ledger_keys:
                info = self.redis_client.hgetall(key)
                amt = float(info.get("ledger_amt", info.get("position_amt", "0")))
                if amt == 0:
                    continue
                symbol = key.replace("quantum:position:ledger:", "")
                if symbol in seen_symbols:
                    continue
                positions.append({
                    "symbol": symbol,
                    "side": info.get("ledger_side", "LONG" if amt > 0 else "SHORT"),
                    "size": abs(amt),
                    "entry_price": float(info.get("entry_price", info.get("avg_entry_price", 0))),
                    "mark_price": 0,
                    "unrealized_pnl": float(info.get("unrealized_pnl", 0)),
                    "leverage": 1,
                })
            return {"positions": positions, "count": len(positions)}
        except Exception as e:
            logger.warning(f"get_portfolio_positions error: {e}")
            return None

    async def get_portfolio_performance(self) -> Optional[Dict]:
        return await self.get_portfolio_summary()

    # ── Trades (Redis) ──

    async def get_live_trades(self) -> Optional[Dict]:
        """Active trades = positions with non-zero size from ledger"""
        if not self.redis_client:
            return None
        try:
            trades = []
            ledger_keys = self.redis_client.keys("quantum:position:ledger:*")
            for key in ledger_keys:
                info = self.redis_client.hgetall(key)
                amt = float(info.get("ledger_amt", info.get("position_amt", "0")))
                if amt == 0:
                    continue
                symbol = key.replace("quantum:position:ledger:", "")
                trades.append({
                    "symbol": symbol,
                    "side": info.get("ledger_side", "LONG" if amt > 0 else "SHORT"),
                    "qty": abs(amt),
                    "entry_price": float(info.get("entry_price", info.get("avg_entry_price", 0))),
                    "unrealized_pnl": float(info.get("unrealized_pnl", 0)),
                })
            return {"active_trades": trades, "count": len(trades)}
        except Exception as e:
            logger.warning(f"get_live_trades error: {e}")
            return None

    async def get_trade_history(self, limit: int = 100) -> Optional[Dict]:
        """Completed trades from trade.closed stream"""
        if not self.redis_client:
            return None
        try:
            entries = self.redis_client.xrevrange(
                "quantum:stream:trade.closed", count=limit
            )
            trades = []
            for stream_id, data in entries:
                trades.append({
                    "stream_id": stream_id,
                    "symbol": data.get("symbol", ""),
                    "side": data.get("side", ""),
                    "entry_price": data.get("entry_price", "0"),
                    "exit_price": data.get("exit_price", "0"),
                    "pnl_percent": data.get("pnl_percent", "0"),
                    "pnl_usd": data.get("pnl_usd", "0"),
                    "reason": data.get("reason", ""),
                    "timestamp": data.get("timestamp", ""),
                    "source": data.get("source", ""),
                })
            return {"trades": trades, "count": len(trades)}
        except Exception as e:
            logger.warning(f"get_trade_history error: {e}")
            return None

    async def get_trade_signals(self) -> Optional[Dict]:
        """Latest AI signals from trade.intent stream"""
        if not self.redis_client:
            return None
        try:
            entries = self.redis_client.xrevrange(
                "quantum:stream:trade.intent", count=10
            )
            signals = []
            for stream_id, data in entries:
                payload_str = data.get("payload", "{}")
                try:
                    payload = json.loads(payload_str)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                signals.append({
                    "stream_id": stream_id,
                    "symbol": payload.get("symbol", ""),
                    "side": payload.get("side", ""),
                    "confidence": payload.get("confidence", 0),
                    "model": payload.get("model", ""),
                    "timestamp": payload.get("timestamp", ""),
                })
            return {"signals": signals, "count": len(signals)}
        except Exception as e:
            logger.warning(f"get_trade_signals error: {e}")
            return None

    # ── AI Engine (HTTP — port 8001 is live) ──

    async def get_ai_predictions(self) -> Optional[Dict]:
        return await self._get('ai_engine', '/predict')

    async def get_model_performance(self) -> Optional[Dict]:
        return await self._get('ai_engine', '/models/performance')

    async def get_confidence_scores(self) -> Optional[Dict]:
        return await self._get('ai_engine', '/confidence/scores')

    # ── Risk (HTTP — port 8070 is live) ──

    async def get_risk_metrics(self) -> Optional[Dict]:
        return await self._get('risk', '/health')

    async def get_risk_var(self) -> Optional[Dict]:
        """Approximate VaR from account balance and margin data"""
        summary = await self.get_portfolio_summary()
        if not summary:
            return None
        balance = summary.get("balance", 0)
        margin = summary.get("margin_used", 0)
        return {
            "value_at_risk": round(margin * 0.1, 2),
            "balance": balance,
            "margin_used": margin,
            "margin_ratio": summary.get("margin_ratio", 0),
        }

    async def get_risk_exposure(self) -> Optional[Dict]:
        """Exposure = total margin used across positions"""
        summary = await self.get_portfolio_summary()
        if not summary:
            return None
        return {
            "total_exposure": summary.get("margin_used", 0),
            "margin_ratio": summary.get("margin_ratio", 0),
            "positions_count": summary.get("positions_count", 0),
        }

    # ── Strategy (Redis) ──

    async def get_strategy_performance(self) -> Optional[Dict]:
        """Aggregate strategy performance from trade ledgers"""
        if not self.redis_client:
            return None
        try:
            ledger_keys = self.redis_client.keys("quantum:ledger:*")
            total_pnl = 0.0
            total_trades = 0
            wins = 0
            losses = 0
            for key in ledger_keys:
                if "seen_orders" in key:
                    continue
                info = self.redis_client.hgetall(key)
                total_pnl += float(info.get("total_pnl_usdt", 0))
                total_trades += int(info.get("total_trades", 0))
                wins += int(info.get("winning_trades", 0))
                losses += int(info.get("losing_trades", 0))
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            return {
                "total_pnl_usdt": round(total_pnl, 2),
                "total_trades": total_trades,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate": round(win_rate, 1),
            }
        except Exception as e:
            logger.warning(f"get_strategy_performance error: {e}")
            return None

    # ── Model Supervisor (HTTP — port 8007 is live) ──

    async def get_model_health(self) -> Optional[Dict]:
        return await self._get('model_supervisor', '/health')

    # ── Universe (Redis) ──

    async def get_market_universe(self) -> Optional[Dict]:
        """Universe symbol list from Redis"""
        if not self.redis_client:
            return None
        try:
            data = self.redis_client.get("quantum:universe:symbols")
            if data:
                symbols = json.loads(data)
                return {"symbols": symbols, "count": len(symbols)}
            return None
        except Exception:
            return None

    # ── Health Checks ──

    async def health_check(self, service: str) -> bool:
        if service in self.SERVICES:
            result = await self._get(service, '/health')
            return result is not None
        # For Redis-backed "services", check if Redis is alive
        if self.redis_client:
            try:
                return self.redis_client.ping()
            except Exception:
                return False
        return False

    def get_portfolio_status(self) -> Optional[str]:
        """Get portfolio status from Redis"""
        if not self.redis_client:
            return None
        try:
            data = self.redis_client.get('quantum:portfolio:realtime')
            return data
        except Exception as e:
            logger.error(f"Redis fetch error: {e}")
            return None


# Global client instance
quantum_client = QuantumServicesClient()
