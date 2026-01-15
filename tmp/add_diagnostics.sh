#!/bin/bash
set -e

cd /home/qt/quantum_trader/dashboard_v4/backend

# Backup
cp main.py main.py.bak

# Add datetime import at top if not present
if ! grep -q "from datetime import datetime" main.py; then
    sed -i '1a from datetime import datetime' main.py
fi

# Add diagnostics endpoint before the last router registration
cat >> main.py << 'EOF'

# Quantum Core Diagnostics endpoint - receives diagnostics from systemd service
@app.post("/grafana/api/quantum/core/diagnostics")
async def quantum_diagnostics(payload: dict):
    """Receive and log quantum system diagnostics from systemd timer"""
    try:
        # Log to file for Grafana/Loki to pick up
        log_file = "/opt/quantum/audit/diagnostics_api.log"
        timestamp = datetime.utcnow().isoformat()
        
        with open(log_file, "a") as f:
            f.write(f"{timestamp} | DIAGNOSTICS | {payload}\n")
        
        return {"status": "received", "timestamp": timestamp}
    except Exception as e:
        return {"status": "error", "message": str(e)}
EOF

echo "✅ Diagnostics endpoint added"
tail -25 main.py
