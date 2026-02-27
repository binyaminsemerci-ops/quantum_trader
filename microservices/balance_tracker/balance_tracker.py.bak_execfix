#!/usr/bin/env python3
"""
BALANCE TRACKER MICROSERVICE

Poller Binance Futures account balance hver 30 sekunder:
- Henter balance, available_balance, margin_used fra Binance API
- Skriver til Redis: quantum:account:balance 
- Publiserer events: quantum:stream:account.balance

Dette gir andre services real-time account data for positionsstørrelse beregninger.
"""

import os
import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import logging
from datetime import datetime
import redis.asyncio as redis

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BalanceTracker:
    """Tracks Binance Futures account balance and publishes to Redis"""
    
    def __init__(self):
        # Respect environment configuration - no hardcoded overrides
        use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true"
        
        if use_testnet:
            # Testnet configuration
            self.api_key = os.getenv("BINANCE_TESTNET_API_KEY", 
                                    os.getenv("BINANCE_API_KEY", 
                                            "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"))
            self.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET",
                                       os.getenv("BINANCE_API_SECRET",
                                               "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"))
            self.base_url = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
            self.testnet_mode = True
            logger.info("[BALANCE-TRACKER] 🧪 Using TESTNET for balance tracking")
        else:
            # Production configuration
            self.api_key = os.getenv("BINANCE_API_KEY")
            self.api_secret = os.getenv("BINANCE_API_SECRET")
            self.base_url = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
            self.testnet_mode = False
            logger.info("[BALANCE-TRACKER] 🔐 Using PRODUCTION for balance tracking")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not configured")
        
        self.balance_endpoint = "/fapi/v2/balance"
        self.account_endpoint = "/fapi/v2/account"
        
        # Redis connection
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = None
        
        # Polling interval
        self.poll_interval = int(os.getenv("BALANCE_POLL_INTERVAL", "30"))  # 30 seconds
        
        # Running state
        self.running = False
        
        logger.info(f"[BALANCE-TRACKER] Initialized: polling every {self.poll_interval}s")
    
    async def start(self):
        """Start the balance tracking service"""
        logger.info("[BALANCE-TRACKER] Starting balance tracker...")
        
        # Initialize Redis
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        logger.info(f"[BALANCE-TRACKER] ✅ Redis connected: {self.redis_url}")
        
        self.running = True
        
        # Run initial balance fetch
        await self.fetch_and_publish_balance()
        
        # Start polling loop
        while self.running:
            try:
                await asyncio.sleep(self.poll_interval)
                if self.running:
                    await self.fetch_and_publish_balance()
            except Exception as e:
                logger.error(f"[BALANCE-TRACKER] Error in polling loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait 5s before retry
    
    async def stop(self):
        """Stop the balance tracking service"""
        logger.info("[BALANCE-TRACKER] Stopping balance tracker...")
        self.running = False
        if self.redis:
            await self.redis.close()
        logger.info("[BALANCE-TRACKER] ✅ Balance tracker stopped")
    
    async def fetch_and_publish_balance(self):
        """Fetch balance from Binance and publish to Redis"""
        try:
            # Fetch real account info from Binance mainnet
            account_info = await self._fetch_account()
            
            if not account_info:
                logger.error("[BALANCE-TRACKER] Failed to fetch account info from Binance mainnet")
                return
            
            # Extract USDT balance
            usdt_asset = None
            for asset in account_info.get("assets", []):
                if asset["asset"] == "USDT":
                    usdt_asset = asset
                    break
            
            if not usdt_asset:
                logger.error("[BALANCE-TRACKER] USDT asset not found in account")
                return
            
            # Extract balance data
            balance_data = {
                "balance": float(usdt_asset["walletBalance"]),
                "available_balance": float(usdt_asset["availableBalance"]), 
                "margin_used": float(usdt_asset["initialMargin"]),
                "unrealized_pnl": float(usdt_asset["unrealizedProfit"]),
                "total_positions": len([p for p in account_info.get("positions", []) if float(p["positionAmt"]) != 0]),
                "timestamp": int(time.time()),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Calculate derived metrics
            if balance_data["balance"] > 0:
                balance_data["margin_ratio"] = balance_data["margin_used"] / balance_data["balance"]
                balance_data["available_ratio"] = balance_data["available_balance"] / balance_data["balance"]
            else:
                balance_data["margin_ratio"] = 0.0
                balance_data["available_ratio"] = 0.0
            
            # Store in Redis hash
            await self.redis.hset("quantum:account:balance", mapping=balance_data)
            
            # Publish event to stream
            event_data = {
                "event_type": "account.balance",
                "balance": balance_data["balance"],
                "available": balance_data["available_balance"],
                "margin_used": balance_data["margin_used"],
                "unrealized_pnl": balance_data["unrealized_pnl"],
                "positions": balance_data["total_positions"],
                "margin_ratio": balance_data["margin_ratio"],
                "timestamp": balance_data["timestamp"]
            }
            
            await self.redis.xadd(
                "quantum:stream:account.balance",
                event_data,
                maxlen=100  # Keep last 100 balance updates
            )
            
            # Publish active positions to position.snapshot stream
            await self._publish_positions(account_info.get("positions", []))
            
            logger.info(
                f"[BALANCE-TRACKER] ✅ Balance updated: "
                f"${balance_data['balance']:.2f} total, "
                f"${balance_data['available_balance']:.2f} available, "
                f"${balance_data['margin_used']:.2f} margin used, "
                f"{balance_data['total_positions']} positions"
            )
            
        except Exception as e:
            logger.error(f"[BALANCE-TRACKER] Error fetching balance: {e}", exc_info=True)
    
    async def _publish_positions(self, positions):
        """Publish active positions to position.snapshot stream"""
        try:
            active_positions = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
            
            if not active_positions:
                return
            
            # Log first position to see API structure
            if active_positions:
                logger.info(f"[BALANCE-TRACKER] Sample position fields: {list(active_positions[0].keys())}")
            
            # Publish each position
            for position in active_positions:
                # Calculate current price from entry + PnL
                entry_price = float(position.get("entryPrice", 0))
                position_amt = float(position.get("positionAmt", 0))  # Signed: +LONG, -SHORT
                unrealized = float(position.get("unrealizedProfit", 0))
                
                # Calculate mark price from PnL
                # For linear contracts: PnL = (mark - entry) * qty for LONG
                #                       PnL = (entry - mark) * abs(qty) for SHORT
                # Rearranged: mark = entry + (PnL / signed_qty)
                if position_amt != 0 and entry_price > 0:
                    mark_price = entry_price + (unrealized / position_amt)  # Uses signed qty!
                else:
                    mark_price = entry_price
                
                position_data = {
                    "event_type": "position.snapshot",
                    "symbol": position.get("symbol", ""),
                    "side": "LONG" if position_amt > 0 else "SHORT",
                    "position_qty": str(abs(position_amt)),  # PositionTracker expects position_qty
                    "entry_price": str(entry_price),
                    "mark_price": str(mark_price),
                    "unrealized_pnl": str(unrealized),
                    "leverage": str(position.get("leverage", 1)),
                    "isolated": str(position.get("isolated", False)),
                    "liquidation_price": str(position.get("liquidationPrice", 0)),
                    "margin_type": "isolated" if position.get("isolated", False) else "cross",
                    "entry_timestamp": str(int(time.time())),  # Add entry_timestamp
                    "timestamp": str(int(time.time())),
                    "source": "balance-tracker"
                }
                
                await self.redis.xadd(
                    "quantum:stream:position.snapshot",
                    position_data,
                    maxlen=1000  # Keep last 1000 position snapshots
                )
            
            logger.debug(f"[BALANCE-TRACKER] Published {len(active_positions)} position snapshots")
            
        except Exception as e:
            logger.error(f"[BALANCE-TRACKER] Error publishing positions: {e}", exc_info=True)
    
    async def _fetch_account(self):
        """Fetch account info from Binance Futures API"""
        try:
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                self.api_secret.encode('utf-8'), 
                query_string.encode('utf-8'), 
                hashlib.sha256
            ).hexdigest()
            
            url = f"{self.base_url}{self.account_endpoint}?{query_string}&signature={signature}"
            headers = {
                "X-MBX-APIKEY": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"[BALANCE-TRACKER] Binance API error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"[BALANCE-TRACKER] Error calling Binance API: {e}", exc_info=True)
            return None


async def main():
    """Main entry point"""
    try:
        tracker = BalanceTracker()
        await tracker.start()
    except KeyboardInterrupt:
        logger.info("[BALANCE-TRACKER] Received interrupt signal")
    except Exception as e:
        logger.error(f"[BALANCE-TRACKER] Fatal error: {e}", exc_info=True)
    finally:
        if 'tracker' in locals():
            await tracker.stop()


if __name__ == "__main__":
    asyncio.run(main())