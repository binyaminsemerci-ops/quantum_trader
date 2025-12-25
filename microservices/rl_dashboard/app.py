from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import redis, threading, time, json, os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
PORT = int(os.getenv("DASHBOARD_PORT", "8025"))
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

latest_data = {"rewards": [], "events": []}

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

if __name__ == "__main__":
    threading.Thread(target=redis_listener, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=PORT, allow_unsafe_werkzeug=True)
