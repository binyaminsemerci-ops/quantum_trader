# P1.2 ROLLBACK PROCEDURES
## 2026-01-19 02:46 UTC

---

## ROLLBACK SEQUENCE

### 1. ROLLBACK PROMETHEUS SCRAPE JOB

**Restore backup config:**
```bash
ssh root@46.224.116.254
cp /etc/prometheus/prometheus.yml.backup_p12_20260119_034418 /etc/prometheus/prometheus.yml
```

**Validate and reload:**
```bash
promtool check config /etc/prometheus/prometheus.yml
systemctl reload prometheus
# OR if reload fails:
systemctl restart prometheus
```

**Verify:**
```bash
curl -s http://localhost:9091/api/v1/targets | grep quantum_safety_telemetry
# Should return nothing if rollback successful
```

---

### 2. ROLLBACK GRAFANA DASHBOARD PROVISIONING

**Remove provisioning config:**
```bash
ssh root@46.224.116.254
rm /etc/grafana/provisioning/dashboards/quantum_safety.yaml
```

**Remove dashboard JSON:**
```bash
rm -rf /var/lib/grafana/dashboards/quantum/
```

**Restart Grafana:**
```bash
systemctl restart grafana-server
systemctl is-active grafana-server
```

---

### 3. ROLLBACK ALERT RULES (if needed)

**Note:** Alert rules already existed before P1.2, so typically don't rollback.

**If needed to remove:**
```bash
ssh root@46.224.116.254
mv /etc/prometheus/rules/quantum_safety_alerts.yml /tmp/quantum_safety_alerts.yml.disabled
```

**Remove rule_files from config:**
```bash
# Edit /etc/prometheus/prometheus.yml
# Remove or comment out:
# rule_files:
#   - "/etc/prometheus/rules/*.yml"
```

**Reload Prometheus:**
```bash
systemctl reload prometheus
```

---

## BACKUP FILES CREATED

### Prometheus
- **Location:** `/etc/prometheus/prometheus.yml.backup_p12_20260119_034418`
- **Size:** 1.1K
- **Created:** 2026-01-19 02:44:18 UTC

### Grafana
- **Original state:** Sample.yaml commented out (no active provisioning)
- **No backup needed** - new files created

---

## VERIFICATION AFTER ROLLBACK

**Check Prometheus targets:**
```bash
curl -s http://localhost:9091/api/v1/targets | \
  python3 -c "import sys,json; print([t['labels']['job'] for t in json.load(sys.stdin)['data']['activeTargets']])"
```

**Expected:** `quantum_safety_telemetry` should NOT be in list

**Check Grafana provisioning:**
```bash
ls /etc/grafana/provisioning/dashboards/quantum_safety.yaml
# Should show: No such file or directory
```

**Check Prometheus rules:**
```bash
curl -s http://localhost:9091/api/v1/rules | \
  python3 -c "import sys,json; print([g['name'] for g in json.load(sys.stdin)['data']['groups']])"
```

**Expected:** Either empty or `quantum_safety` not in list

---

## EMERGENCY FULL ROLLBACK

**Single command to revert everything:**
```bash
ssh root@46.224.116.254 << 'EOF'
# Prometheus
cp /etc/prometheus/prometheus.yml.backup_p12_20260119_034418 /etc/prometheus/prometheus.yml
systemctl restart prometheus

# Grafana
rm -f /etc/grafana/provisioning/dashboards/quantum_safety.yaml
rm -rf /var/lib/grafana/dashboards/quantum/
systemctl restart grafana-server

# Verify
echo "=== STATUS ==="
systemctl is-active prometheus
systemctl is-active grafana-server
EOF
```

---

## PARTIAL ROLLBACK OPTIONS

### Keep Prometheus scraping but remove dashboard:
```bash
# Remove only Grafana provisioning
rm /etc/grafana/provisioning/dashboards/quantum_safety.yaml
rm -rf /var/lib/grafana/dashboards/quantum/
systemctl restart grafana-server
```

### Keep dashboard but remove Prometheus scraping:
```bash
# Restore Prometheus config only
cp /etc/prometheus/prometheus.yml.backup_p12_20260119_034418 /etc/prometheus/prometheus.yml
systemctl reload prometheus
```

---

## CONTACT & SUPPORT

**Exporter service:**
```bash
journalctl -u quantum-safety-telemetry.service -n 50 --no-pager
systemctl status quantum-safety-telemetry.service
```

**Prometheus logs:**
```bash
journalctl -u prometheus -n 50 --no-pager
```

**Grafana logs:**
```bash
journalctl -u grafana-server -n 50 --no-pager
tail -50 /var/log/grafana/grafana.log
```

---

**Rollback procedures verified:** 2026-01-19 02:46 UTC  
**Backup integrity:** ✅ Verified  
**Tested:** No (dry-run documentation only)
