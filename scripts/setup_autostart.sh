#!/bin/bash
# Quantum Trader RL Auto-Startup Setup
set -e

echo "🚀 Setting up Quantum Trader RL Auto-Startup..."

# Copy startup script
cp ~/quantum_trader/scripts/start_quantum_rl.sh ~/quantum_trader/
chmod +x ~/quantum_trader/start_quantum_rl.sh

# Install systemd service
sudo cp ~/quantum_trader/scripts/quantum-rl.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantum-rl.service

# Add status alias
if ! grep -q "alias qstatus=" ~/.bashrc; then
  echo "alias qstatus='docker ps --format \"table {{.Names}}\t{{.Status}}\t{{.Ports}}\"'" >> ~/.bashrc
  echo "✅ Added qstatus alias to ~/.bashrc"
fi

echo ""
echo "✅ Auto-startup system installed successfully!"
echo "🧠 On reboot, Quantum Trader RL will start automatically."
echo ""
echo "Commands:"
echo "  sudo systemctl status quantum-rl   # Check service status"
echo "  sudo systemctl start quantum-rl    # Start services manually"
echo "  qstatus                            # Quick container status"
echo ""
echo "Testing startup now..."
bash ~/quantum_trader/start_quantum_rl.sh
