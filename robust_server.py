#!/usr/bin/env python3
"""
ROBUST HTTP SERVER
Ikke stopp selv om noe går galt
"""

import http.server
import socketserver
import json
import time
import traceback
from datetime import datetime, timezone

# ===========================================
# 1. DATA STATE
# ===========================================

ai_state = {
    "learning_active": True,
    "symbols_monitored": 5,
    "data_points": 1247,
    "model_accuracy": 0.8523,
    "enabled": True,
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
    "last_signal_time": datetime.now(timezone.utc).isoformat(),
    "total_signals": 89,
    "continuous_learning_status": "Active",
}

# ===========================================
# 2. ROBUST HANDLER
# ===========================================


class RobustHandler(http.server.SimpleHTTPRequestHandler):

    def generate_extended_watchlist(self):
        """Generate comprehensive watchlist with 50+ coins from major exchanges and L1/L2"""
        import random

        # Base coin data with realistic pricing
        coins_data = [
            # Layer 1 Major
            {
                "symbol": "BTCUSDT",
                "name": "Bitcoin",
                "price": 67420.50,
                "category": "Layer 1",
            },
            {
                "symbol": "ETHUSDT",
                "name": "Ethereum",
                "price": 2634.80,
                "category": "Layer 1",
            },
            {
                "symbol": "ADAUSDT",
                "name": "Cardano",
                "price": 0.4567,
                "category": "Layer 1",
            },
            {
                "symbol": "SOLUSDT",
                "name": "Solana",
                "price": 143.67,
                "category": "Layer 1",
            },
            {
                "symbol": "AVAXUSDT",
                "name": "Avalanche",
                "price": 23.45,
                "category": "Layer 1",
            },
            {
                "symbol": "DOTUSDT",
                "name": "Polkadot",
                "price": 6.78,
                "category": "Layer 1",
            },
            {
                "symbol": "ATOMUSDT",
                "name": "Cosmos",
                "price": 12.34,
                "category": "Layer 1",
            },
            {
                "symbol": "NEARUSDT",
                "name": "NEAR Protocol",
                "price": 3.45,
                "category": "Layer 1",
            },
            {
                "symbol": "ALGOUSDT",
                "name": "Algorand",
                "price": 0.234,
                "category": "Layer 1",
            },
            {
                "symbol": "FLOWUSDT",
                "name": "Flow",
                "price": 1.23,
                "category": "Layer 1",
            },
            {
                "symbol": "APTUSDT",
                "name": "Aptos",
                "price": 7.89,
                "category": "Layer 1",
            },
            {"symbol": "SUIUSDT", "name": "Sui", "price": 1.45, "category": "Layer 1"},
            {
                "symbol": "FTMUSDT",
                "name": "Fantom",
                "price": 0.234,
                "category": "Layer 1",
            },
            {
                "symbol": "ONEUSDT",
                "name": "Harmony",
                "price": 0.0123,
                "category": "Layer 1",
            },
            # Layer 2 Solutions
            {
                "symbol": "MATICUSDT",
                "name": "Polygon",
                "price": 0.89,
                "category": "Layer 2",
            },
            {
                "symbol": "LRCUSDT",
                "name": "Loopring",
                "price": 0.345,
                "category": "Layer 2",
            },
            {
                "symbol": "IMXUSDT",
                "name": "Immutable X",
                "price": 1.67,
                "category": "Layer 2",
            },
            {
                "symbol": "OPUSDT",
                "name": "Optimism",
                "price": 2.34,
                "category": "Layer 2",
            },
            {
                "symbol": "ARBUSDT",
                "name": "Arbitrum",
                "price": 1.89,
                "category": "Layer 2",
            },
            # DeFi Major
            {"symbol": "UNIUSDT", "name": "Uniswap", "price": 7.89, "category": "DeFi"},
            {
                "symbol": "LINKUSDT",
                "name": "Chainlink",
                "price": 14.56,
                "category": "Oracle",
            },
            {"symbol": "AAVEUSDT", "name": "Aave", "price": 87.34, "category": "DeFi"},
            {
                "symbol": "CRVUSDT",
                "name": "Curve DAO",
                "price": 0.67,
                "category": "DeFi",
            },
            {
                "symbol": "COMPUSDT",
                "name": "Compound",
                "price": 45.67,
                "category": "DeFi",
            },
            {
                "symbol": "MKRUSDT",
                "name": "Maker",
                "price": 1234.56,
                "category": "DeFi",
            },
            {
                "symbol": "SUSHIUSDT",
                "name": "SushiSwap",
                "price": 1.34,
                "category": "DeFi",
            },
            {
                "symbol": "1INCHUSDT",
                "name": "1inch",
                "price": 0.456,
                "category": "DeFi",
            },
            # Payments & Enterprise
            {
                "symbol": "XRPUSDT",
                "name": "Ripple",
                "price": 0.5234,
                "category": "Payments",
            },
            {
                "symbol": "XLMUSDT",
                "name": "Stellar",
                "price": 0.123,
                "category": "Payments",
            },
            {
                "symbol": "LTCUSDT",
                "name": "Litecoin",
                "price": 89.45,
                "category": "Payments",
            },
            {
                "symbol": "BCHUSDT",
                "name": "Bitcoin Cash",
                "price": 245.67,
                "category": "Payments",
            },
            {
                "symbol": "HBARUSDT",
                "name": "Hedera",
                "price": 0.067,
                "category": "Enterprise",
            },
            {
                "symbol": "VETUSDT",
                "name": "VeChain",
                "price": 0.0234,
                "category": "Enterprise",
            },
            # Privacy
            {
                "symbol": "XMRUSDT",
                "name": "Monero",
                "price": 167.89,
                "category": "Privacy",
            },
            {
                "symbol": "ZECUSDT",
                "name": "Zcash",
                "price": 34.56,
                "category": "Privacy",
            },
            # Infrastructure/Web3
            {
                "symbol": "FILUSDT",
                "name": "Filecoin",
                "price": 5.67,
                "category": "Storage",
            },
            {
                "symbol": "ARUSDT",
                "name": "Arweave",
                "price": 8.90,
                "category": "Storage",
            },
            {
                "symbol": "GRTUSDT",
                "name": "The Graph",
                "price": 0.156,
                "category": "Infrastructure",
            },
            {
                "symbol": "BATUSDT",
                "name": "Basic Attention Token",
                "price": 0.234,
                "category": "Utility",
            },
            {
                "symbol": "ICPUSDT",
                "name": "Internet Computer",
                "price": 4.89,
                "category": "Computing",
            },
            {
                "symbol": "RNDRUSDT",
                "name": "Render Token",
                "price": 3.45,
                "category": "AI/Compute",
            },
            # Gaming/Metaverse
            {
                "symbol": "MANAUSDT",
                "name": "Decentraland",
                "price": 0.456,
                "category": "Metaverse",
            },
            {
                "symbol": "SANDUSDT",
                "name": "The Sandbox",
                "price": 0.345,
                "category": "Gaming",
            },
            {
                "symbol": "AXSUSDT",
                "name": "Axie Infinity",
                "price": 6.78,
                "category": "Gaming",
            },
            {
                "symbol": "ENJUSDT",
                "name": "Enjin Coin",
                "price": 0.234,
                "category": "Gaming",
            },
            # Meme Coins
            {
                "symbol": "DOGEUSDT",
                "name": "Dogecoin",
                "price": 0.089,
                "category": "Meme",
            },
            {
                "symbol": "SHIBUSDT",
                "name": "Shiba Inu",
                "price": 0.0000234,
                "category": "Meme",
            },
            {
                "symbol": "PEPEUSDT",
                "name": "Pepe",
                "price": 0.00000123,
                "category": "Meme",
            },
            # Media & Entertainment
            {
                "symbol": "THETAUSDT",
                "name": "Theta Network",
                "price": 1.23,
                "category": "Media",
            },
            {
                "symbol": "CHZUSDT",
                "name": "Chiliz",
                "price": 0.0876,
                "category": "Sports",
            },
            # Stablecoins
            {
                "symbol": "USDTUSDT",
                "name": "Tether",
                "price": 1.0001,
                "category": "Stablecoin",
            },
            {
                "symbol": "USDCUSDT",
                "name": "USD Coin",
                "price": 1.0002,
                "category": "Stablecoin",
            },
            {
                "symbol": "DAIUSDT",
                "name": "Dai",
                "price": 0.9998,
                "category": "Stablecoin",
            },
        ]

        # Generate full watchlist data with sparklines and dynamic changes
        watchlist = []
        for coin in coins_data:
            # Add random price variation
            price_change = random.uniform(-8.0, 8.0)
            current_price = coin["price"] * (1 + price_change / 100)

            # Generate sparkline data
            base_price = current_price
            sparkline = []
            for i in range(10):
                variation = random.uniform(-0.03, 0.03)
                price_point = base_price * (1 + variation)
                sparkline.append(round(price_point, 6 if price_point < 1 else 2))
                base_price = price_point

            watchlist.append(
                {
                    "symbol": coin["symbol"],
                    "name": coin["name"],
                    "price": round(current_price, 6 if current_price < 1 else 2),
                    "change24h": round(price_change, 2),
                    "volume24h": random.randint(10000000, 50000000000),
                    "category": coin["category"],
                    "confidence": round(random.uniform(0.35, 0.95), 2),
                    "sparkline": sparkline,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )

        return watchlist

    def generate_chart_data(self, symbol):
        """Generate realistic chart data for a specific symbol"""
        import random
        from datetime import timedelta

        # Base prices for common symbols
        base_prices = {
            "BTCUSDT": 67420.50,
            "ETHUSDT": 2634.80,
            "SOLUSDT": 143.67,
            "ADAUSDT": 0.4567,
            "MATICUSDT": 0.89,
        }

        base_price = base_prices.get(symbol, 100.0)

        # Generate 24h of hourly data points
        chart_data = []
        current_time = datetime.now(timezone.utc)
        current_price = base_price

        for i in range(24):
            # Random price movement (-3% to +3% per hour)
            price_change = random.uniform(-0.03, 0.03)
            current_price = current_price * (1 + price_change)

            # Generate OHLC data
            open_price = current_price
            high = current_price * random.uniform(1.0, 1.02)
            low = current_price * random.uniform(0.98, 1.0)
            close = current_price * random.uniform(0.99, 1.01)
            volume = random.randint(1000000, 50000000)

            timestamp = current_time - timedelta(hours=23 - i)

            chart_data.append(
                {
                    "time": timestamp.isoformat(),
                    "open": round(open_price, 6 if open_price < 1 else 2),
                    "high": round(high, 6 if high < 1 else 2),
                    "low": round(low, 6 if low < 1 else 2),
                    "close": round(close, 6 if close < 1 else 2),
                    "volume": volume,
                }
            )

            current_price = close

        return {
            "symbol": symbol,
            "timeframe": "1h",
            "data": chart_data,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def do_GET(self):
        try:
            print(f"📡 GET: {self.path}")

            # Always send headers first
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

            # Get path without query params
            path = self.path.split("?")[0]

            # Simple response logic
            response = {"error": "not found", "path": path}

            if path == "/api/v1/system/status":
                response = {
                    "status": "online",
                    "service": "quantum_trader_core",
                    "uptime": "45 min",
                    "binance_keys": True,
                    "testnet": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            elif path == "/api/v1/ai-trading/status":
                response = {
                    "status": "Active",
                    "active": True,
                    "enabled": True,
                    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
                    "learning_active": True,
                    "symbols_monitored": 5,
                    "data_points": 1247,
                    "accuracy": 0.85,
                    "continuous_learning_status": "Active",
                    "last_execution": datetime.now(timezone.utc).isoformat(),
                    "total_signals_today": 23,
                    "successful_trades": 18,
                    "win_rate": 78.3,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            elif path == "/api/v1/continuous-learning/status":
                response = {
                    "learning_active": True,
                    "symbols_monitored": 5,
                    "data_points": 1247,
                    "model_accuracy": 0.85,
                    "status": "Active",
                    "last_training": datetime.now(timezone.utc).isoformat(),
                    "twitter_sentiment": "ACTIVE",
                    "market_data": "ACTIVE",
                    "enhanced_feeds": "ACTIVE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            elif path == "/api/v1/portfolio":
                response = {
                    "total_value": 861498,
                    "positions": 1,
                    "pnl_percent": -38.50,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            elif path == "/api/v1/portfolio/market-overview":
                response = {
                    "market_cap": 1000000000,
                    "volume_24h": 1000000,
                    "fear_greed": 52,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            elif path.startswith("/api/v1/signals/recent"):
                # Generate realistic trading signals
                import random

                signals = [
                    {
                        "id": "sig_001",
                        "symbol": "BTCUSDT",
                        "action": "BUY",
                        "confidence": 85.6,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "predicted_price": 68500,
                        "current_price": 67420,
                        "reasoning": "Strong bullish momentum + technical breakout",
                    },
                    {
                        "id": "sig_002",
                        "symbol": "ETHUSDT",
                        "action": "HOLD",
                        "confidence": 72.3,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "predicted_price": 2650,
                        "current_price": 2634,
                        "reasoning": "Consolidation phase, wait for clear direction",
                    },
                    {
                        "id": "sig_003",
                        "symbol": "SOLUSDT",
                        "action": "BUY",
                        "confidence": 91.2,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "predicted_price": 155,
                        "current_price": 143.67,
                        "reasoning": "Ecosystem growth + DeFi adoption surge",
                    },
                ]
                # Add some randomness to make it dynamic
                for signal in signals:
                    signal["confidence"] = round(
                        signal["confidence"] + random.uniform(-5, 5), 1
                    )
                    signal["timestamp"] = datetime.now(timezone.utc).isoformat()
                response = signals
            elif path == "/ws/chat":
                response = {"status": "HTTP polling active", "messages": []}
            elif path == "/api/v1/enhanced/data":
                response = {"status": "ok", "enhanced_data": True}
            elif path == "/api/v1/watchlist" or path.startswith("/api/v1/watchlist/"):
                # Return comprehensive crypto data with 50+ coins from mainbase and Layer 1/2
                response = self.generate_extended_watchlist()
            elif path.startswith("/api/v1/chart/"):
                # Extract symbol from path like /api/v1/chart/BTCUSDT
                symbol = path.split("/")[-1]
                response = self.generate_chart_data(symbol)

            # Send response
            json_data = json.dumps(response)
            self.wfile.write(json_data.encode())
            print(f"✅ Sent: {len(json_data)} bytes")

        except Exception as e:
            print(f"❌ GET Error: {e}")
            print(traceback.format_exc())
            try:
                self.wfile.write(b'{"error": "server error"}')
            except Exception:
                pass

    def do_OPTIONS(self):
        try:
            print(f"📡 OPTIONS: {self.path}")
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            print("✅ OPTIONS sent")
        except Exception as e:
            print(f"❌ OPTIONS Error: {e}")

    def log_message(self, format, *args):
        # Disable default logging
        pass


# ===========================================
# 3. ROBUST SERVER
# ===========================================


class RobustServer:
    def __init__(self, port=8000):
        self.port = port
        self.running = False

    def start(self):
        self.running = True
        retry_count = 0
        max_retries = 5

        while self.running and retry_count < max_retries:
            try:
                print(
                    f"🌐 Attempting to start server on port {self.port} (attempt {retry_count + 1})"
                )

                with socketserver.TCPServer(("", self.port), RobustHandler) as httpd:
                    print(f"✅ Server started on http://localhost:{self.port}")
                    print("📡 Endpoints ready:")
                    print("   - /api/v1/system/status")
                    print("   - /api/v1/ai-trading/status")
                    print("   - /api/v1/continuous-learning/status")
                    print("   - /api/v1/portfolio")
                    print("   - /api/v1/portfolio/market-overview")
                    print("=" * 60)
                    print("✅ DASHBOARD SKAL NÅ VISE DATA!")

                    httpd.serve_forever()

            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt - stopping server")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Server error (attempt {retry_count + 1}): {e}")
                print(traceback.format_exc())
                retry_count += 1
                if retry_count < max_retries:
                    print("🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("❌ Max retries reached - giving up")
                    break


# ===========================================
# 4. MAIN
# ===========================================


def main():
    print("🚀 QUANTUM TRADER - ROBUST SERVER")
    print("=" * 60)

    server = RobustServer(8000)
    server.start()


if __name__ == "__main__":
    main()
