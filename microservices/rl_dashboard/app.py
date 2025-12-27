from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import redis, threading, time, json, os, subprocess

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
PORT = int(os.getenv("DASHBOARD_PORT", "8025"))
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

latest_data = {"rewards": [], "events": [], "system": {}}

def get_system_metrics():
    """Collect system metrics from host via docker stats and docker ps"""
    try:
        # Get container count
        result = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=5)
        container_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        
        # Get docker stats (CPU and Memory)
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}},{{.MemPerc}}"],
            capture_output=True, text=True, timeout=10
        )
        
        cpu_usage, mem_usage = 0.0, 0.0
        if result.stdout:
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip()]
            for line in lines:
                parts = line.split(',')
                if len(parts) == 2:
                    cpu = float(parts[0].replace('%', ''))
                    mem = float(parts[1].replace('%', ''))
                    cpu_usage += cpu
                    mem_usage = max(mem_usage, mem)
        
        # Cap CPU at 100% for display
        cpu_usage = min(cpu_usage, 100.0)
        
        return {
            "cpu_usage": round(cpu_usage, 1),
            "ram_usage": round(mem_usage, 1),
            "disk_usage": 59.5,  # Static for now
            "containers": container_count
        }
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        return {
            "cpu_usage": 0.0,
            "ram_usage": 0.0,
            "disk_usage": 0.0,
            "containers": 0
        }

def system_monitor():
    """Background thread to collect system metrics every 5 seconds"""
    while True:
        latest_data["system"] = get_system_metrics()
        socketio.emit("system_update", latest_data["system"])
        time.sleep(5)

def redis_listener():
    pubsub = r.pubsub()
    pubsub.psubscribe("__keyspace@0__:quantum:stream:exitbrain.pnl")
    while True:
        msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=2)
        if msg:
            events = r.xrevrange("quantum:stream:exitbrain.pnl", count=10)
            data = []
            for _, e in events:
                d = {k.decode(): v.decode() for k,v in e.items()}
                d["timestamp"] = time.strftime("%H:%M:%S")
                data.append(d)
            latest_data["events"] = data
            rewards = [float(ev.get("pnl", 0))*float(ev.get("confidence", 0.8)) for ev in data]
            latest_data["rewards"] = rewards
            socketio.emit("update", latest_data)
        time.sleep(2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/latest")
def latest():
    return jsonify(latest_data)

@app.route("/api/system")
def system_metrics():
    return jsonify(latest_data.get("system", {}))

if __name__ == "__main__":
    threading.Thread(target=redis_listener, daemon=True).start()
    threading.Thread(target=system_monitor, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=PORT, allow_unsafe_werkzeug=True)
