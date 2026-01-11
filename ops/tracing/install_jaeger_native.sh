#!/bin/bash
# Install Jaeger All-in-One natively on Ubuntu (systemd)
# Replaces Docker-based Jaeger from copilot/add-tracing-to-workspace-again

set -e

echo "=== Installing Jaeger All-in-One (Native systemd) ==="

# Download Jaeger binary
JAEGER_VERSION="1.52.0"
JAEGER_DIR="/opt/jaeger"
JAEGER_BIN="$JAEGER_DIR/jaeger-all-in-one"

echo "1. Downloading Jaeger v$JAEGER_VERSION..."
mkdir -p $JAEGER_DIR
cd /tmp
wget -q "https://github.com/jaegertracing/jaeger/releases/download/v${JAEGER_VERSION}/jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz"
tar -xzf "jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz"
mv "jaeger-${JAEGER_VERSION}-linux-amd64/jaeger-all-in-one" "$JAEGER_BIN"
chmod +x "$JAEGER_BIN"
rm -rf /tmp/jaeger-*

echo "2. Creating jaeger systemd service..."
cat > /etc/systemd/system/jaeger.service <<'EOF'
[Unit]
Description=Jaeger All-in-One (OpenTelemetry Tracing)
Documentation=https://www.jaegertracing.io/docs/
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/jaeger/jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --collector.otlp.grpc.host-port=:4317 \
  --collector.otlp.http.host-port=:4318 \
  --query.ui-config=/etc/jaeger/ui-config.json
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

echo "3. Creating UI config..."
mkdir -p /etc/jaeger
cat > /etc/jaeger/ui-config.json <<'EOF'
{
  "monitor": {
    "menuEnabled": true
  },
  "dependencies": {
    "menuEnabled": true
  }
}
EOF

echo "4. Enabling and starting jaeger.service..."
systemctl daemon-reload
systemctl enable jaeger.service
systemctl start jaeger.service

echo ""
echo "✅ Jaeger installed successfully!"
echo ""
echo "🌐 Jaeger UI:     http://localhost:16686"
echo "📊 OTLP gRPC:     localhost:4317"
echo "📊 OTLP HTTP:     localhost:4318"
echo ""
echo "Check status: systemctl status jaeger"
echo "View logs:    journalctl -u jaeger -f"
