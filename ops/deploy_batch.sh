#!/bin/bash
set -e

echo "=== DEPLOY BATCH: Kelly + Gate6 + PaperTrade + L2Promoter ==="

# Strip CRLF
for f in /tmp/apply_layer_main.py /tmp/paper_trade_controller.py /tmp/layer2_signal_promoter.py /tmp/layer4_portfolio_optimizer.py /tmp/quantum-paper-trade-controller.service /tmp/quantum-layer2-signal-promoter.service; do
  sed -i 's/\r//' "$f" && echo "LF ok: $f"
done

# Create service dirs
mkdir -p /opt/quantum/microservices/paper_trade_controller
mkdir -p /opt/quantum/microservices/layer2_signal_promoter
echo "=== DIRS CREATED ==="

# Place files
cp /tmp/layer4_portfolio_optimizer.py /opt/quantum/microservices/layer4_portfolio_optimizer/layer4_portfolio_optimizer.py && echo "OK: layer4"
cp /tmp/apply_layer_main.py           /opt/quantum/microservices/apply_layer/main.py                                      && echo "OK: apply_layer"
cp /tmp/paper_trade_controller.py     /opt/quantum/microservices/paper_trade_controller/paper_trade_controller.py         && echo "OK: paper_trade_controller"
cp /tmp/layer2_signal_promoter.py     /opt/quantum/microservices/layer2_signal_promoter/layer2_signal_promoter.py         && echo "OK: layer2_signal_promoter"
cp /tmp/quantum-paper-trade-controller.service /etc/systemd/system/ && echo "OK: svc paper"
cp /tmp/quantum-layer2-signal-promoter.service /etc/systemd/system/ && echo "OK: svc promoter"
sed -i 's/\r//' /etc/systemd/system/quantum-paper-trade-controller.service
sed -i 's/\r//' /etc/systemd/system/quantum-layer2-signal-promoter.service

# Reload systemd
systemctl daemon-reload && echo "=== DAEMON RELOADED ==="

# Restart updated services
systemctl restart quantum-layer4-portfolio-optimizer && echo "RESTARTED: layer4"
systemctl restart quantum-apply-layer                && echo "RESTARTED: apply_layer"

# Enable + start new services
systemctl enable quantum-paper-trade-controller quantum-layer2-signal-promoter
systemctl start quantum-paper-trade-controller  && echo "STARTED: paper_trade_controller"
systemctl start quantum-layer2-signal-promoter  && echo "STARTED: layer2_signal_promoter"

# Quick status check
echo "=== STATUS ==="
systemctl is-active quantum-layer4-portfolio-optimizer quantum-apply-layer quantum-paper-trade-controller quantum-layer2-signal-promoter
echo "=== DEPLOY COMPLETE ==="
