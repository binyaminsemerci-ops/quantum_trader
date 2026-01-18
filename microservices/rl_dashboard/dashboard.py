#!/usr/bin/env python3
"""
RL Dashboard v2.0 - Enhanced with HarvestBrain Integration

Features:
- RL reward tracking (original)
- HarvestBrain live positions with R-levels
- Harvest history timeline
- Cumulative profit metrics
- High-profit alerts
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import redis
import json
import threading
import os
import time
from datetime import datetime, timedelta

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Data caches
data_cache = {}  # RL rewards
harvest_positions = {}  # Current positions with R-levels
harvest_history_cache = {}  # Recent harvests per symbol


def listen_rl_rewards():
    """Original RL reward listener"""
    pub = r.pubsub()
    pub.subscribe(["quantum:signal:strategy"])
    for msg in pub.listen():
        if msg["type"] != "message":
            continue
        try:
            d = json.loads(msg["data"])
            s = d["symbol"]
            rwd = float(d.get("reward", 0))
            data_cache.setdefault(s, []).append(rwd)
            if len(data_cache[s]) > 100:
                data_cache[s].pop(0)
            socketio.emit("rl_update", {s: rwd})
        except Exception as e:
            print(f"Error in RL listener: {e}")


def listen_harvest_brain():
    """Listen to HarvestBrain execution.result stream for position updates"""
    last_id = "0"
    while True:
        try:
            # Read from execution.result stream
            messages = r.xread(
                {"quantum:stream:execution.result": last_id},
                count=10,
                block=1000
            )
            
            if messages:
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        last_id = message_id
                        
                        # Parse execution event
                        symbol = data.get("symbol", "")
                        status = data.get("status", "").upper()
                        
                        if status == "FILLED":
                            # New position created
                            qty = float(data.get("qty", 0))
                            entry_price = float(data.get("entry_price", 0))
                            stop_loss = float(data.get("stop_loss", 0))
                            entry_risk = abs(entry_price - stop_loss) if stop_loss > 0 else entry_price * 0.02
                            
                            harvest_positions[symbol] = {
                                "symbol": symbol,
                                "qty": qty,
                                "entry_price": entry_price,
                                "current_price": entry_price,
                                "stop_loss": stop_loss,
                                "entry_risk": entry_risk,
                                "unrealized_pnl": 0.0,
                                "r_level": 0.0,
                                "last_update": time.time()
                            }
                            
                            # Emit position update
                            socketio.emit("position_update", harvest_positions[symbol])
                        
                        elif status == "PRICE_UPDATE" and symbol in harvest_positions:
                            # Update price and calculate new R
                            price = float(data.get("price", harvest_positions[symbol]["current_price"]))
                            pos = harvest_positions[symbol]
                            pos["current_price"] = price
                            pos["unrealized_pnl"] = (price - pos["entry_price"]) * pos["qty"]
                            pos["r_level"] = pos["unrealized_pnl"] / pos["entry_risk"] if pos["entry_risk"] > 0 else 0
                            pos["last_update"] = time.time()
                            
                            # Emit position update
                            socketio.emit("position_update", pos)
                            
                            # Check for high-profit alert
                            if pos["r_level"] >= 2.0:
                                socketio.emit("high_profit_alert", {
                                    "symbol": symbol,
                                    "r_level": pos["r_level"],
                                    "pnl": pos["unrealized_pnl"]
                                })
            
        except Exception as e:
            print(f"Error in HarvestBrain listener: {e}")
            time.sleep(1)


def fetch_harvest_history():
    """Periodically fetch harvest history from Redis"""
    while True:
        try:
            # Get all position symbols
            symbols = list(harvest_positions.keys())
            
            for symbol in symbols:
                # Fetch harvest history from sorted set
                history_key = f"quantum:harvest:history:{symbol}"
                entries = r.zrange(history_key, 0, -1, withscores=False)
                
                history_list = []
                for entry_json in entries:
                    try:
                        entry = json.loads(entry_json)
                        history_list.append(entry)
                    except:
                        pass
                
                if history_list:
                    harvest_history_cache[symbol] = history_list
            
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error fetching harvest history: {e}")
            time.sleep(5)


# Start background threads
threading.Thread(target=listen_rl_rewards, daemon=True).start()
threading.Thread(target=listen_harvest_brain, daemon=True).start()
threading.Thread(target=fetch_harvest_history, daemon=True).start()


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def data():
    """Original RL data endpoint"""
    return jsonify({
        "symbols": list(data_cache.keys()),
        "rewards": data_cache
    })


@app.route("/harvest/positions")
def harvest_positions_route():
    """Get all current positions with R-levels"""
    return jsonify({
        "positions": list(harvest_positions.values()),
        "count": len(harvest_positions)
    })


@app.route("/harvest/history/<symbol>")
def harvest_history_route(symbol):
    """Get harvest history for a symbol"""
    history = harvest_history_cache.get(symbol, [])
    return jsonify({
        "symbol": symbol,
        "history": history,
        "count": len(history)
    })


@app.route("/harvest/metrics")
def harvest_metrics_route():
    """Get cumulative harvest metrics"""
    total_harvests = sum(len(h) for h in harvest_history_cache.values())
    total_profit = 0.0
    
    for history in harvest_history_cache.values():
        for entry in history:
            total_profit += entry.get("pnl", 0)
    
    active_positions = len(harvest_positions)
    avg_r_level = sum(p["r_level"] for p in harvest_positions.values()) / active_positions if active_positions > 0 else 0
    
    return jsonify({
        "total_harvests": total_harvests,
        "total_profit": total_profit,
        "active_positions": active_positions,
        "avg_r_level": avg_r_level
    })


@app.route("/harvest/config/<symbol>", methods=["GET", "POST"])
def symbol_config_route(symbol):
    """Get or update symbol-specific config"""
    config_key = f"quantum:config:harvest:{symbol}"
    
    if request.method == "POST":
        # Update config
        config_data = request.get_json()
        if config_data:
            r.hset(config_key, mapping=config_data)
            return jsonify({"status": "success", "config": config_data})
    
    # Get config
    config = r.hgetall(config_key)
    return jsonify({"symbol": symbol, "config": config})


if __name__ == "__main__":
    print(f"🚀 Starting RL Dashboard v2.0 with HarvestBrain integration")
    print(f"📊 Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"🌐 Dashboard: http://0.0.0.0:{os.getenv('DASHBOARD_PORT', 8027)}")
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DASHBOARD_PORT", 8027)),
        allow_unsafe_werkzeug=True
    )
