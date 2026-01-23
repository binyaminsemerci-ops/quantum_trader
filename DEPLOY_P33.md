# P3.3 Deployment Command

## One-Command VPS Deployment

```powershell
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'bash /home/qt/quantum_trader/ops/p33_deploy_and_proof.sh'
```

**This will**:
- Pull latest code from GitHub
- Install P3.3 service
- Restart Apply Layer
- Run proof pack
- Save results to docs/P3_3_VPS_PROOF.txt

---

## Alternative: Manual Steps

```powershell
# 1. SSH to VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# 2. Deploy
cd /root/quantum_trader && git pull origin main
bash /home/qt/quantum_trader/ops/p33_deploy_and_proof.sh

# 3. View proof
cat /home/qt/quantum_trader/docs/P3_3_VPS_PROOF.txt
```

---

## Quick Health Check

After deployment, verify:

```powershell
# Check services
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl status quantum-position-state-brain quantum-apply-layer --no-pager | head -30'

# Check metrics
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'curl -s http://localhost:8045/metrics | grep "^p33_" | head -10'

# Check snapshots
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'redis-cli KEYS "quantum:position:snapshot:*"'
```

---

## Expected Output

If successful, you'll see:
- ✅ P3.3 service: **active**
- ✅ Apply Layer: **active**
- ✅ Metrics responding on port 8045
- ✅ Snapshots fresh (< 10s age)
- ✅ Proof pack: 8/8 checks passed

---

## Troubleshooting

**If deployment fails**:
```powershell
# Check logs
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-position-state-brain -n 50'

# Check config
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cat /etc/quantum/position-state-brain.env'

# Restart service
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl restart quantum-position-state-brain'
```

---

## Next: Monitor

After deployment, monitor for 24 hours:

```powershell
# Watch logs
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-position-state-brain -f'

# Check permit issuance
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'redis-cli KEYS "quantum:permit:p33:*"'

# Monitor Apply Layer integration
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-apply-layer --since "10 minutes ago" | grep "P3.3"'
```
