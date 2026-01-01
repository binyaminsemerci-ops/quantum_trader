"""
Binance PnL Tracker - Real-time PnL monitoring from Binance positions
Polls Binance /fapi/v2/account and publishes unrealized PnL to Redis for dashboard and RL monitoring
"""
import os
import asyncio
import logging
import time
import hmac
import hashlib
from urllib.parse import urlencode
from datetime import datetime
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
    
    async def process_positions(self, account_data: dict):
        """Process active positions and update Redis with PnL data"""
        if not account_data or "positions" not in account_data:
            return
        
        active_count = 0
        total_unrealized_pnl = 0.0
        
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
            
            # Calculate PnL percentage
            if notional > 0:
                pnl_pct = (unrealized_pnl / notional) * 100 * leverage
            else:
                pnl_pct = 0.0
            
            # Write to Redis RL reward key
            redis_key = f"quantum:rl:reward:{symbol}"
            reward_data = {
                "symbol": symbol,
                "pnl": unrealized_pnl,
                "pnl_pct": round(pnl_pct, 4),
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
                
                logger.info(f"💰 {symbol} {side.upper()}: PnL=${unrealized_pnl:.2f} ({pnl_pct:+.2f}%) | Entry=${entry_price} | Leverage={leverage}x")
            
            active_count += 1
            total_unrealized_pnl += unrealized_pnl
        
        # Update portfolio summary
        portfolio_key = "quantum:portfolio:realtime"
        portfolio_data = {
            "total_equity": float(account_data.get("totalWalletBalance", 0)),
            "available_balance": float(account_data.get("availableBalance", 0)),
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_margin": float(account_data.get("totalPositionInitialMargin", 0)),
            "num_positions": active_count,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "binance_pnl_tracker"
        }
        await self.redis.setex(portfolio_key, 300, str(portfolio_data))  # 5 min TTL
        
        logger.info(f"📊 Portfolio: {active_count} positions | Total PnL: ${total_unrealized_pnl:+.2f} USDT")
        
    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info(f"🚀 Starting Binance PnL Tracker (polling every {self.poll_interval}s)")
        
        while True:
            try:
                account_data = await self.get_account_data()
                if account_data:
                    await self.process_positions(account_data)
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
