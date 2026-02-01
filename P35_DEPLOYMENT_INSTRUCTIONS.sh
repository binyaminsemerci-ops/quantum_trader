#!/bin/bash
# P3.5 DEPLOYMENT INSTRUCTIONS
# Run this as a guide for deploying P3.5 to production VPS

cat << 'EOF'
╔═════════════════════════════════════════════════════════════════════════════╗
║                   P3.5 DEPLOYMENT INSTRUCTIONS                             ║
║                     Date: 2026-02-01                                        ║
║                  Status: READY FOR DEPLOYMENT                              ║
╚═════════════════════════════════════════════════════════════════════════════╝

📋 PRE-DEPLOYMENT CHECKLIST
═════════════════════════════════════════════════════════════════════════════

[ ] Git repo up to date locally
[ ] VPS SSH key available (~/.ssh/hetzner_fresh)
[ ] Redis accessible on VPS
[ ] Internet connectivity to VPS

═════════════════════════════════════════════════════════════════════════════

🚀 DEPLOYMENT OPTIONS
═════════════════════════════════════════════════════════════════════════════

OPTION 1 - One-Command Deployment (Recommended)
──────────────────────────────────────────────────────────────────────────────

1. From Windows development machine:
   $ cd c:\quantum_trader
   $ git status                    # Ensure clean working directory
   $ git push                      # Push any changes

2. SSH to VPS:
   $ ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

3. Deploy on VPS:
   $ cd /home/qt/quantum_trader
   $ git pull                      # Get latest code
   $ bash deploy_p35.sh            # One-command deployment

4. Verify:
   $ redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10 WITHSCORES
   → Should show analytics data

OPTION 2 - Manual Step-by-Step Deployment
──────────────────────────────────────────────────────────────────────────────

1. SSH to VPS:
   $ ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

2. Navigate to repository:
   $ cd /home/qt/quantum_trader
   $ git pull                      # Get latest code

3. Copy configuration:
   $ sudo cp etc/quantum/p35-decision-intelligence.env /etc/quantum/
   $ sudo chown qt:qt /etc/quantum/p35-decision-intelligence.env

4. Copy systemd unit:
   $ sudo cp etc/systemd/system/quantum-p35-decision-intelligence.service /etc/systemd/system/

5. Reload systemd:
   $ sudo systemctl daemon-reload

6. Start service:
   $ sudo systemctl enable quantum-p35-decision-intelligence
   $ sudo systemctl start quantum-p35-decision-intelligence

7. Verify:
   $ bash scripts/proof_p35_decision_intelligence.sh

═════════════════════════════════════════════════════════════════════════════

✅ VERIFICATION (What to Check)
═════════════════════════════════════════════════════════════════════════════

After deployment, run these commands to verify:

1. Service Status:
   $ systemctl is-active quantum-p35-decision-intelligence
   Expected: active

2. Consumer Group Created:
   $ redis-cli XINFO GROUPS quantum:stream:apply.result
   Expected: Group "p35_decision_intel" with consumers

3. No Pending Messages:
   $ redis-cli XPENDING quantum:stream:apply.result p35_decision_intel
   Expected: 0 (all ACKed)

4. Processed Messages Increasing:
   $ redis-cli HGET quantum:p35:status processed_total
   Expected: > 0 and increasing every minute

5. Analytics Available (after 1+ minute):
   $ redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10 WITHSCORES
   Expected: Top skip reasons visible (no_position, not_in_allowlist, etc.)

6. Decision Distribution:
   $ redis-cli HGETALL quantum:p35:decision:counts:5m
   Expected: EXECUTE, SKIP, BLOCKED, ERROR counts visible

7. Service Logs:
   $ journalctl -u quantum-p35-decision-intelligence -n 20
   Expected: "Processed N messages" messages appearing

═════════════════════════════════════════════════════════════════════════════

📊 FIRST-TIME ANALYTICS TIMELINE
═════════════════════════════════════════════════════════════════════════════

1 minute:     Per-minute buckets appear (quantum:p35:bucket:*)
1-2 minutes:  1-minute window available
5 minutes:    5-minute window available
15 minutes:   15-minute window available
1 hour:       1-hour window available (optional)

During first run:
- Service starts processing apply.result stream
- Creates buckets as decisions arrive
- Recomputes snapshots every 60 seconds
- Status key updated every 100 messages

═════════════════════════════════════════════════════════════════════════════

🔍 TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════

Issue: Service won't start
→ Check: systemctl status quantum-p35-decision-intelligence
→ Check: journalctl -u quantum-p35-decision-intelligence -n 50
→ Check: /etc/quantum/p35-decision-intelligence.env exists
→ Fix: sudo systemctl start quantum-p35-decision-intelligence

Issue: High pending messages
→ Check: redis-cli XPENDING quantum:stream:apply.result p35_decision_intel
→ Check: systemctl status quantum-p35-decision-intelligence
→ Fix: systemctl restart quantum-p35-decision-intelligence

Issue: No analytics data after 5 minutes
→ Check: redis-cli KEYS "quantum:p35:*"
→ Check: redis-cli HGET quantum:p35:status processed_total
→ Check: journalctl -u quantum-p35-decision-intelligence -f
→ Note: May need more time if low apply.result throughput

Issue: Service keeps restarting
→ Check: journalctl -u quantum-p35-decision-intelligence -n 100
→ Check: Redis connectivity (redis-cli ping)
→ Check: Disk space available

═════════════════════════════════════════════════════════════════════════════

📞 SUPPORT & RESOURCES
═════════════════════════════════════════════════════════════════════════════

Documentation:
  - Quick Reference:    P35_QUICK_REFERENCE.md
  - Deployment Guide:   AI_P35_DEPLOYMENT_GUIDE.md
  - Full Docs:          ops/README.md (P3.5 section)

Configuration:
  - File:               /etc/quantum/p35-decision-intelligence.env
  - Can edit and restart service to apply changes

Monitoring:
  - Live logs:          journalctl -u quantum-p35-decision-intelligence -f
  - Status:             redis-cli HGETALL quantum:p35:status
  - Analytics:          redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10

═════════════════════════════════════════════════════════════════════════════

🎯 COMMON QUERIES AFTER DEPLOYMENT
═════════════════════════════════════════════════════════════════════════════

Monitor skip reasons:
  $ watch -n 2 'redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10'

Monitor decision distribution:
  $ watch -n 2 'redis-cli HGETALL quantum:p35:decision:counts:5m'

Monitor service health:
  $ watch -n 5 'redis-cli HGETALL quantum:p35:status && echo && redis-cli XPENDING quantum:stream:apply.result p35_decision_intel'

Check for "no_position" blocking trades:
  $ redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 100 | grep no_position

Check for "not_in_allowlist" filtering:
  $ redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 100 | grep not_in_allowlist

═════════════════════════════════════════════════════════════════════════════

✨ SUCCESS INDICATORS
═════════════════════════════════════════════════════════════════════════════

You'll know deployment is successful when:

✅ Service running:
   systemctl is-active quantum-p35-decision-intelligence → active

✅ Processing messages:
   redis-cli HGET quantum:p35:status processed_total → increasing

✅ ACKing working:
   redis-cli XPENDING quantum:stream:apply.result p35_decision_intel → 0

✅ Analytics appearing:
   redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10 → shows reasons

✅ Logs appearing:
   journalctl -u quantum-p35-decision-intelligence → shows processing

✅ All windows available:
   redis-cli KEYS "quantum:p35:decision:counts:*" → shows 1m,5m,15m,1h

═════════════════════════════════════════════════════════════════════════════

⏱️ DEPLOYMENT TIME ESTIMATES
═════════════════════════════════════════════════════════════════════════════

One-command deployment (deploy_p35.sh):  ~2 minutes
  - Git pull:                            ~10 seconds
  - Copy files:                          ~5 seconds
  - Systemd reload:                      ~2 seconds
  - Service start:                       ~5 seconds
  - Proof script:                        ~1.5 minutes

Manual step-by-step:                     ~3 minutes
  - Each step takes 10-30 seconds

First analytics:                         ~1 minute
All windows ready:                       ~5 minutes

Total time to production:                ~5-7 minutes

═════════════════════════════════════════════════════════════════════════════

🔐 SECURITY NOTES
═════════════════════════════════════════════════════════════════════════════

✅ Service runs as 'qt' user (non-root)
✅ Systemd unit has security hardening:
   - NoNewPrivileges=true
   - PrivateTmp=true
   - ProtectSystem=strict
✅ No passwords/secrets in code or logs
✅ Resource limits enforced (CPU, memory)
✅ Journal logging (audit trail available)

═════════════════════════════════════════════════════════════════════════════

📝 ROLLBACK PROCEDURE (if needed)
═════════════════════════════════════════════════════════════════════════════

1. Stop service:
   $ sudo systemctl stop quantum-p35-decision-intelligence

2. Remove from startup:
   $ sudo systemctl disable quantum-p35-decision-intelligence

3. Remove systemd unit:
   $ sudo rm /etc/systemd/system/quantum-p35-decision-intelligence.service
   $ sudo systemctl daemon-reload

4. (Optional) Remove config:
   $ sudo rm /etc/quantum/p35-decision-intelligence.env

5. Redis data persists (can recover if redeployed)

═════════════════════════════════════════════════════════════════════════════

🎯 NEXT STEPS AFTER DEPLOYMENT
═════════════════════════════════════════════════════════════════════════════

1. Verify analytics appearing:
   $ redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 20 WITHSCORES

2. Set up monitoring alerts (optional):
   - Alert if pending_estimate > 100
   - Alert if service not running
   - Alert if processed_total not increasing

3. Integrate with dashboards (Grafana):
   - Display top reasons
   - Show decision distribution
   - Monitor service status

4. Review skip reasons:
   - Adjust allowlist if too many "not_in_allowlist"
   - Review kill_score thresholds if too many blocks
   - Monitor for unexpected reason codes

═════════════════════════════════════════════════════════════════════════════

💡 TIPS FOR SUCCESS
═════════════════════════════════════════════════════════════════════════════

✓ Use one-command deployment (bash deploy_p35.sh) - simpler, less error-prone
✓ Run proof script after deployment - validates everything in one go
✓ Monitor logs for first 5 minutes - catch any issues early
✓ Check analytics after ~5 minutes - confirm data collection working
✓ Set up follow-up monitoring - track health over time
✓ Document any customizations - for future reference

═════════════════════════════════════════════════════════════════════════════

Ready to deploy? 🚀

$ bash deploy_p35.sh

═════════════════════════════════════════════════════════════════════════════
EOF
