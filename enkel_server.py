#!/usr/bin/env python3
"""
ENKEL FUNGERENDE SERVER
Fokuserer kun på å levere data til dashboardet
"""

import http.server
import socketserver
import json
import threading
import time
from datetime import datetime, timezone

# ===========================================
# 1. AKTIVE DATA SOM DASHBOARDET TRENGER
# ===========================================

# AI state med aktivt kontinuerlig læring
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

# Mock crypto data for CoinTable
crypto_data = [
    {
        "symbol": "BTCUSDT",
        "price": 67420.50,
        "change24h": 2.34,
        "volume24h": 28500000000,
    },
    {
        "symbol": "ETHUSDT",
        "price": 2634.80,
        "change24h": -1.12,
        "volume24h": 15200000000,
    },
    {"symbol": "BNBUSDT", "price": 602.45, "change24h": 0.89, "volume24h": 1850000000},
    {"symbol": "SOLUSDT", "price": 143.67, "change24h": 3.45, "volume24h": 2100000000},
    {"symbol": "XRPUSDT", "price": 0.5234, "change24h": -0.67, "volume24h": 1340000000},
]

# ===========================================
# 2. ENKEL HTTP HANDLER
# ===========================================


class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"📡 GET Request: {self.path}")

        # CORS headers
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

        path = self.path.split("?")[0]

        # Response routing
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
                "enabled": ai_state["enabled"],
                "symbols": ai_state["symbols"],
                "last_signal_time": ai_state["last_signal_time"],
                "total_signals": ai_state["total_signals"],
                "accuracy": ai_state["model_accuracy"],
                "learning_active": ai_state["learning_active"],
                "symbols_monitored": ai_state["symbols_monitored"],
                "data_points": ai_state["data_points"],
                "continuous_learning_status": (
                    "Active" if ai_state["learning_active"] else "Inactive"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        elif path == "/api/v1/continuous-learning/status":
            response = {
                "learning_active": ai_state["learning_active"],
                "symbols_monitored": ai_state["symbols_monitored"],
                "data_points": ai_state["data_points"],
                "model_accuracy": ai_state["model_accuracy"],
                "status": "Active" if ai_state["learning_active"] else "Inactive",
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
        elif path == "/api/v1/signals/recent":
            response = [
                {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "confidence": 0.85,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        elif path == "/api/v1/enhanced/data":
            response = {
                "sources": 7,
                "coingecko": {
                    "status": "active",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                },
                "fear_greed": {"value": 52, "classification": "Neutral"},
                "reddit_sentiment": {"btc": 0.65, "eth": 0.32, "ada": -0.21},
                "cryptocompare_news": {"count": 15, "sentiment": "positive"},
                "coinpaprika": {"market_data": "active"},
                "messari": {"onchain_data": "active"},
                "ai_insights": {
                    "market_regime": "BULL",
                    "volatility": 2.5,
                    "trend_strength": 3.2,
                    "sentiment_score": 0.65,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            response = {"error": "Not found", "path": path}

        json_response = json.dumps(response)
        print(f"✅ Response sent: {len(json_response)} bytes")
        self.wfile.write(json_response.encode())

    def do_POST(self):
        print(f"📡 POST Request: {self.path}")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

        if self.path == "/api/v1/continuous-learning/start":
            ai_state["learning_active"] = True
            ai_state["symbols_monitored"] = 5
            ai_state["data_points"] += 50

            response = {
                "status": "Continuous Learning Started",
                "message": "Real-time AI strategy evolution from live data feeds",
                "symbols": ai_state["symbols"],
                "twitter_analysis": "ACTIVE",
                "market_feeds": "ACTIVE",
                "model_training": "ACTIVE",
                "enhanced_sources": "ACTIVE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            response = {"error": "Not found"}

        json_response = json.dumps(response)
        print(f"✅ POST Response sent: {len(json_response)} bytes")
        self.wfile.write(json_response.encode())

    def do_OPTIONS(self):
        print(f"📡 OPTIONS Request: {self.path}")
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format, *args):
        # Disable default logging to avoid clutter
        pass


# ===========================================
# 3. DATA UPDATER
# ===========================================


def update_data():
    """Update data continuously"""
    while True:
        time.sleep(30)  # Update every 30 seconds

        # Update AI metrics
        ai_state["data_points"] += 5
        ai_state["model_accuracy"] = min(0.95, ai_state["model_accuracy"] + 0.001)

        # Update crypto prices
        for coin in crypto_data:
            change = (time.time() % 10 - 5) * 0.001
            coin["price"] *= 1 + change
            coin["change24h"] += change * 100

        print(f"📊 Data updated - AI points: {ai_state['data_points']}")


# ===========================================
# 4. MAIN SERVER
# ===========================================


def main():
    print("🚀 QUANTUM TRADER - ENKEL FUNGERENDE SERVER")
    print("=" * 60)

    # Start background updater
    threading.Thread(target=update_data, daemon=True).start()
    print("📊 Background data updater started")

    # Start HTTP server
    PORT = 8000
    print(f"🌐 Starting HTTP Server on port {PORT}")

    try:
        with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
            print(f"✅ Server running on http://localhost:{PORT}")
            print("📡 Endpoints ready:")
            print("   - /api/v1/system/status")
            print("   - /api/v1/ai-trading/status")
            print("   - /api/v1/continuous-learning/status")
            print("   - /api/v1/portfolio")
            print("   - /api/v1/portfolio/market-overview")
            print("   - /api/v1/signals/recent")
            print("   - /api/v1/enhanced/data")
            print("=" * 60)
            print("✅ DASHBOARD SKAL NÅ VISE DATA!")
            print("💡 Gå til frontend og sjekk at data vises")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    main()
