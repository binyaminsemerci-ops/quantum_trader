"""
Governance Dashboard - Real-time monitoring for AI Hedge Fund OS
Displays live model weights, validation events, retraining history, and system metrics.
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import time
import redis
import asyncio
from datetime import datetime
from typing import Dict, List

app = FastAPI(
    title="AI Governance Dashboard",
    version="1.0.0",
    description="Real-time monitoring for Phase 4D+4E+4F+4G"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Redis connection
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)


def read_validation_log() -> List[Dict]:
    """Read validation log from AI Engine container"""
    path = "/app/logs/model_validation.log"
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, 'r') as f:
            lines = f.read().strip().split("\n")[-50:]  # Last 50 entries
        
        records = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                records.append({
                    "timestamp": parts[0],
                    "message": parts[1]
                })
        return records
    except Exception as e:
        return [{"error": f"Failed to read validation log: {e}"}]


@app.get("/", response_class=HTMLResponse)
def home():
    """Main dashboard HTML page"""
    return """<!DOCTYPE html>
<html>
<head>
    <title>AI Governance Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff00;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(0, 20, 40, 0.8);
            border: 2px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
        }
        .card h2 {
            color: #00ffff;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid #00ff00;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #333;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #00ff00; }
        .metric-value {
            color: #ffff00;
            font-weight: bold;
        }
        .status-ok { color: #00ff00; }
        .status-warning { color: #ffaa00; }
        .status-error { color: #ff0000; }
        .log-entry {
            padding: 8px;
            margin: 5px 0;
            background: rgba(0, 0, 0, 0.5);
            border-left: 3px solid #00ff00;
            font-size: 0.85em;
            word-wrap: break-word;
        }
        .log-accept { border-left-color: #00ff00; }
        .log-reject { border-left-color: #ff6600; }
        .timestamp {
            color: #888;
            font-size: 0.8em;
        }
        .loading {
            text-align: center;
            color: #888;
            padding: 20px;
        }
        pre {
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.85em;
        }
        .update-time {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI GOVERNANCE DASHBOARD</h1>
        <div class="subtitle">Real-time monitoring: Phase 4D+4E+4F+4G</div>
        
        <div class="grid">
            <!-- Model Weights Card -->
            <div class="card">
                <h2>📊 Model Weights (Governance)</h2>
                <div id="weights" class="loading">Loading weights...</div>
            </div>
            
            <!-- System Status Card -->
            <div class="card">
                <h2>⚙️ System Status</h2>
                <div id="status" class="loading">Loading status...</div>
            </div>
        </div>
        
        <!-- Validation Events Card (Full Width) -->
        <div class="card">
            <h2>🧪 Recent Validation Events (Phase 4G)</h2>
            <div id="events" class="loading">Loading events...</div>
        </div>
        
        <div class="update-time" id="updateTime">Last updated: Never</div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                // Update model weights
                const weightsResp = await fetch('/weights');
                const weights = await weightsResp.json();
                const weightsDiv = document.getElementById('weights');
                
                if (weights.error) {
                    weightsDiv.innerHTML = `<div class="status-error">Error: ${weights.error}</div>`;
                } else if (Object.keys(weights).length === 0) {
                    weightsDiv.innerHTML = '<div class="status-warning">No weights data yet</div>';
                } else {
                    let html = '';
                    for (const [model, weight] of Object.entries(weights)) {
                        html += `<div class="metric">
                            <span class="metric-label">${model}</span>
                            <span class="metric-value">${parseFloat(weight).toFixed(3)}</span>
                        </div>`;
                    }
                    weightsDiv.innerHTML = html;
                }

                // Update system status
                const statusResp = await fetch('/status');
                const status = await statusResp.json();
                const statusDiv = document.getElementById('status');
                
                let statusHtml = '';
                for (const [key, value] of Object.entries(status)) {
                    const statusClass = value === true || value === 'healthy' ? 'status-ok' : 
                                      value === false ? 'status-error' : 'status-warning';
                    statusHtml += `<div class="metric">
                        <span class="metric-label">${key}</span>
                        <span class="metric-value ${statusClass}">${value}</span>
                    </div>`;
                }
                statusDiv.innerHTML = statusHtml;

                // Update validation events
                const eventsResp = await fetch('/events');
                const events = await eventsResp.json();
                const eventsDiv = document.getElementById('events');
                
                if (events.length === 0) {
                    eventsDiv.innerHTML = '<div class="status-warning">No validation events yet (waiting for first retraining cycle)</div>';
                } else {
                    let eventsHtml = '';
                    events.reverse().forEach(event => {
                        if (event.error) {
                            eventsHtml += `<div class="log-entry status-error">${event.error}</div>`;
                        } else {
                            const isAccept = event.message.includes('ACCEPT') || event.message.includes('✅');
                            const logClass = isAccept ? 'log-accept' : 'log-reject';
                            eventsHtml += `<div class="log-entry ${logClass}">
                                <span class="timestamp">${event.timestamp}</span> ${event.message}
                            </div>`;
                        }
                    });
                    eventsDiv.innerHTML = eventsHtml;
                }

                // Update timestamp
                document.getElementById('updateTime').textContent = 
                    `Last updated: ${new Date().toLocaleString()}`;

            } catch (err) {
                console.error('Dashboard update error:', err);
            }
        }

        // Update every 2 seconds
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>"""


@app.get("/weights")
def get_weights() -> JSONResponse:
    """Get current model weights from governance system"""
    try:
        # Try to get from Redis first
        weights = r.hgetall("governance_weights")
        if not weights:
            # Fallback: query AI Engine health endpoint
            import httpx
            try:
                # Try quantum_ai_engine container name first
                try:
                    resp = httpx.get("http://quantum_ai_engine:8001/health", timeout=2.0)
                except:
                    # Fallback to localhost
                    resp = httpx.get("http://localhost:8001/health", timeout=2.0)
                
                data = resp.json()
                if "metrics" in data and "governance" in data["metrics"]:
                    models = data["metrics"]["governance"].get("models", {})
                    weights = {name: str(model_data["weight"]) for name, model_data in models.items()}
            except Exception as e:
                return JSONResponse({"error": f"Failed to fetch from AI Engine: {e}"})
        
        return JSONResponse(weights)
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/events")
def get_events() -> JSONResponse:
    """Get recent validation events from log"""
    return JSONResponse(read_validation_log())


@app.get("/status")
def get_status() -> JSONResponse:
    """Get system status from AI Engine"""
    try:
        import httpx
        # Try quantum_ai_engine container name first
        try:
            resp = httpx.get("http://quantum_ai_engine:8001/health", timeout=2.0)
        except:
            # Fallback to localhost (if on same host network)
            resp = httpx.get("http://localhost:8001/health", timeout=2.0)
        
        data = resp.json()
        
        metrics = data.get("metrics", {})
        status = {
            "models_loaded": metrics.get("models_loaded", 0),
            "governance_active": metrics.get("governance_active", False),
            "retrainer_enabled": metrics.get("adaptive_retrainer", {}).get("enabled", False),
            "validator_enabled": metrics.get("model_validator", {}).get("enabled", False),
            "ai_engine_health": data.get("status", "unknown")
        }
        
        return JSONResponse(status)
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "ai_engine_health": "unreachable"
        })


@app.get("/metrics")
def get_metrics() -> JSONResponse:
    """Get system resource metrics"""
    try:
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "redis_connected": r.ping() if r else False
        }
        
        # Try to get system metrics (will work on Linux)
        try:
            cpu = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2+$4}'").read().strip()
            data["cpu_usage"] = cpu if cpu else "N/A"
        except:
            data["cpu_usage"] = "N/A"
        
        try:
            memory = os.popen("free -m | awk '/Mem:/ {print $3\"/\"$2\" MB\"}'").read().strip()
            data["memory"] = memory if memory else "N/A"
        except:
            data["memory"] = "N/A"
        
        try:
            uptime = os.popen("uptime -p").read().strip()
            data["uptime"] = uptime if uptime else "N/A"
        except:
            data["uptime"] = "N/A"
        
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/report")
def get_latest_report():
    """Get latest performance report from Trade Journal (Phase 7)"""
    try:
        # Try to get latest report from Redis first
        report_json = r.get("latest_report")
        if report_json:
            report = json.loads(report_json)
            return JSONResponse(report)
        
        # Fallback: read from file system
        report_dir = "/app/reports"
        if not os.path.exists(report_dir):
            return JSONResponse({
                "status": "no_data",
                "message": "Trade Journal not yet active or no reports generated"
            })
        
        # Find latest report file
        files = [f for f in os.listdir(report_dir) if f.startswith("daily_report_") and f.endswith(".json")]
        if not files:
            return JSONResponse({
                "status": "no_data",
                "message": "No reports found"
            })
        
        latest_file = max(files)
        report_path = os.path.join(report_dir, latest_file)
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return JSONResponse(report)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Failed to retrieve report: {str(e)}"
        })


@app.get("/reports/history")
def get_report_history():
    """Get historical performance reports"""
    try:
        report_dir = "/app/reports"
        if not os.path.exists(report_dir):
            return JSONResponse({"reports": []})
        
        files = sorted([f for f in os.listdir(report_dir) if f.startswith("daily_report_") and f.endswith(".json")])
        reports = []
        
        for filename in files[-30:]:  # Last 30 reports
            try:
                with open(os.path.join(report_dir, filename), 'r') as f:
                    report = json.load(f)
                    reports.append({
                        "filename": filename,
                        "date": report.get("date"),
                        "total_trades": report.get("total_trades"),
                        "win_rate_%": report.get("win_rate_%"),
                        "total_pnl_%": report.get("total_pnl_%"),
                        "sharpe_ratio": report.get("sharpe_ratio"),
                        "max_drawdown_%": report.get("max_drawdown_%")
                    })
            except:
                continue
        
        return JSONResponse({"reports": reports, "count": len(reports)})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "governance_dashboard",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
