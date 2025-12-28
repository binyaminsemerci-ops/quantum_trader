from flask import Flask,render_template,jsonify
from flask_socketio import SocketIO
import redis,json,threading,os
app=Flask(__name__); socketio=SocketIO(app,cors_allowed_origins="*")
r=redis.Redis(host=os.getenv("REDIS_HOST","redis"),port=6379)
data_cache={}
def listen():
  pub=r.pubsub(); pub.subscribe(["quantum:signal:strategy"])
  for msg in pub.listen():
    if msg["type"]!="message":continue
    d=json.loads(msg["data"]); s=d["symbol"]; rwd=float(d["reward"])
    data_cache.setdefault(s,[]).append(rwd)
    if len(data_cache[s])>100:data_cache[s].pop(0)
    socketio.emit("update",{s:rwd})
threading.Thread(target=listen,daemon=True).start()
@app.route("/") 
def index(): return render_template("index.html")
@app.route("/data")
def data(): return jsonify({"symbols":list(data_cache.keys()),"rewards":data_cache})
if __name__=="__main__": socketio.run(app,host="0.0.0.0",port=int(os.getenv("DASHBOARD_PORT",8027)),allow_unsafe_werkzeug=True)
