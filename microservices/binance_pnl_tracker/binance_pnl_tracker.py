"""
Binance PnL Tracker - Real-time PnL monitoring from Binance positions
Polls Binance /fapi/v2/account for unrealized PnL and /fapi/v1/income for realized PnL
Publishes both to Redis for dashboard and RL monitoring
"""
import os
import asyncio
import logging
import time
import hmac
import hashlib
from urllib.parse import urlencode
from datetime import datetime, timedelta
import aiohttp
import redis.asyncio as redis_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinancePnLTracker:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
        
        if self.use_testnet:
            self.base_url = "https://testnet.binancefuture.com"
            logger.info("🧪 [TESTNET MODE] Using Binance Futures Testnet")
        else:
            self.base_url = "https://fapi.binance.com"
            logger.info("🚀 [LIVE MODE] Using Binance Futures Mainnet")
        
        # Redis
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Polling interval
        self.poll_interval = int(os.getenv("PNL_POLL_INTERVAL", "15"))  # 15 seconds default
        
        # Track last values to avoid unnecessary writes
        self.last_pnl = {}
        
        # Track realized PnL (fetch once per minute to avoid rate limits)
        self.last_realized_fetch = 0
        self.realized_pnl_cache = {}
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = redis_async.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        await self.redis.ping()
        logger.info(f"✅ Connected to Redis at {self.redis_host}:{self.redis_port}")
        
    def _sign_request(self, params: dict) -> str:
        """Sign Binance API request"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def get_account_data(self) -> dict:
        """Fetch account data from Binance with proper authentication"""
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp, "recvWindow": 10000}
        params["signature"] = self._sign_request(params)
        
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}/fapi/v2/account"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"❌ Binance API error: {response.status} - {text}")
                    return None
                return await response.json()
    
    async def get_income_history(self, hours_back: int = 24) -> dict:
        """Fetch realized PnL from Binance income history"""
        timestamp = int(time.time() * 1000)
        start_time = int((time.time() - hours_back * 3600) * 1000)
        
        params = {
            "timestamp": timestamp,
            "recvWindow": 10000,
            "startTime": start_time,
            "incomeType": "REALIZED_PNL"  # Only get realized PnL
        }
        params["signature"] = self._sign_request(params)
        
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}/fapi/v1/income"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"❌ Binance income API error: {response.status} - {text}")
                    return {}
                
                income_list = await response.json()
                
                # Aggregate by symbol
                realized_by_symbol = {}
                total_realized = 0.0
                
                for income in income_list:
                    symbol = income["symbol"]
                    amount = float(income["income"])
                    
                    if symbol not in realized_by_symbol:
                        realized_by_symbol[symbol] = {"total": 0.0, "count": 0, "latest_time": 0}
                    
                    realized_by_symbol[symbol]["total"] += amount
                    realized_by_symbol[symbol]["count"] += 1
                    realized_by_symbol[symbol]["latest_time"] = max(
                        realized_by_symbol[symbol]["latest_time"],
                        income["time"]
                    )
                    total_realized += amount
                
                logger.info(f"💵 Realized PnL (last {hours_back}h): ${total_realized:+.2f} USDT across {len(realized_by_symbol)} symbols")
                return realized_by_symbol
    
    async def process_positions(self, account_data: dict, realized_pnl_data: dict = None):
        """Process active positions and update Redis with PnL data (both unrealized and realized)"""
        if not account_data or "positions" not in account_data:
            return
        
        if realized_pnl_data is None:
            realized_pnl_data = {}
        
        active_count = 0
        total_unrealized_pnl = 0.0
        total_realized_pnl = sum(data["total"] for data in realized_pnl_data.values())
        
        for pos in account_data["positions"]:
            amt = float(pos.get("positionAmt", 0))
            if amt == 0:
                continue
            
            symbol = pos["symbol"]
            side = pos["positionSide"].lower()
            unrealized_pnl = float(pos.get("unrealizedProfit", 0))
            entry_price = float(pos.get("entryPrice", 0))
            notional = abs(float(pos.get("notional", 0)))
            leverage = int(pos.get("leverage", 1))
            
            # Get realized PnL for this symbol
            realized_pnl = realized_pnl_data.get(symbol, {}).get("total", 0.0)
            realized_trades = realized_pnl_data.get(symbol, {}).get("count", 0)
            
            # Calculate PnL percentage
            if notional > 0:
                unrealized_pct = (unrealized_pnl / notional) * 100 * leverage
                # For realized, use the cached notional or current notional
                realized_pct = (realized_pnl / notional) * 100 * leverage if realized_pnl != 0 else 0.0
            else:
                unrealized_pct = 0.0
                realized_pct = 0.0
            
            # Write to Redis RL reward key
            redis_key = f"quantum:rl:reward:{symbol}"
            reward_data = {
                "symbol": symbol,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pct": round(unrealized_pct, 4),
                "realized_pnl": realized_pnl,
                "realized_pct": round(realized_pct, 4),
                "total_pnl": unrealized_pnl + realized_pnl,
                "realized_trades": realized_trades,
                "position_size": abs(amt),
                "side": side,
                "entry_price": entry_price,
                "notional": notional,
                "leverage": leverage,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "binance_pnl_tracker"
            }
            
            # Only update if PnL changed significantly (avoid spam)
            last_pnl = self.last_pnl.get(symbol, 0)
            if abs(unrealized_pnl - last_pnl) > 0.01 or symbol not in self.last_pnl:
                await self.redis.setex(redis_key, 3600, str(reward_data))  # 1 hour TTL
                self.last_pnl[symbol] = unrealized_pnl
                
                realized_str = f" | Realized=${realized_pnl:+.2f} ({realized_trades} trades)" if realized_pnl != 0 else ""
                logger.info(f"💰 {symbol} {side.upper()}: Unrealized=${unrealized_pnl:+.2f} ({unrealized_pct:+.2f}%){realized_str}")
            
            active_count += 1
            total_unrealized_pnl += unrealized_pnl
        
        # Update portfolio summary
        portfolio_key = "quantum:portfolio:realtime"
        portfolio_data = {
            "total_equity": float(account_data.get("totalWalletBalance", 0)),
            "available_balance": float(account_data.get("availableBalance", 0)),
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_realized_pnl": total_realized_pnl,
            "total_pnl": total_unrealized_pnl + total_realized_pnl,
            "total_margin": float(account_data.get("totalPositionInitialMargin", 0)),
            "num_positions": active_count,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "binance_pnl_tracker"
        }
        await self.redis.setex(portfolio_key, 300, str(portfolio_data))  # 5 min TTL
        
        logger.info(f"📊 Portfolio: {active_count} positions | Unrealized=${total_unrealized_pnl:+.2f} | Realized=${total_realized_pnl:+.2f} | Total=${total_unrealized_pnl + total_realized_pnl:+.2f} USDT")
        
    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info(f"🚀 Starting Binance PnL Tracker (polling every {self.poll_interval}s)")
        
        while True:
            try:
                # Fetch unrealized PnL every cycle
                account_data = await self.get_account_data()
                
                # Fetch realized PnL once per minute to avoid rate limits
                current_time = time.time()
                if current_time - self.last_realized_fetch > 60:
                    self.realized_pnl_cache = await self.get_income_history(hours_back=24)
                    self.last_realized_fetch = current_time
                
                if account_data:
                    await self.process_positions(account_data, self.realized_pnl_cache)
                else:
                    logger.warning("⚠️  No account data received")
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(5)


async def main():
    """Entry point"""
    tracker = BinancePnLTracker()
    await tracker.initialize()
    await tracker.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
