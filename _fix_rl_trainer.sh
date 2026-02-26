#!/bin/bash
# Fix rl_trainer startup script - add PYTHONPATH
cp /opt/quantum/bin/start_rl_trainer.sh /opt/quantum/bin/start_rl_trainer.sh.bak.$(date +%s)

cat > /opt/quantum/bin/start_rl_trainer.sh << 'SCRIPT'
#!/bin/bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/runtime/bin/activate
export PYTHONPATH=/home/qt/quantum_trader
exec python3 -u microservices/rl_training/main.py
SCRIPT

chmod +x /opt/quantum/bin/start_rl_trainer.sh
echo "=== Updated script ==="
cat /opt/quantum/bin/start_rl_trainer.sh

# Restart service
systemctl restart quantum-rl-trainer.service
sleep 6
echo "=== Service status ==="
systemctl status quantum-rl-trainer.service --no-pager | head -20
echo "=== Last error log ==="
tail -10 /opt/quantum/logs/rl_trainer.err
