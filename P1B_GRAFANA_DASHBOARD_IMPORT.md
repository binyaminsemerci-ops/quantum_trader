# P1-B: Grafana Dashboard Import Guide

**Date:** January 3, 2026  
**Objective:** Import P1-B Log Aggregation Dashboard to Grafana UI

---

## 🎯 Dashboard Overview

**Dashboard:** P1-B Log Aggregation  
**File:** `observability/grafana/dashboards/p1b_log_aggregation.json`  
**Datasource:** Loki (configured at http://loki:3100)

### Dashboard Panels:
1. **Errors by Level (15m)** - Bar chart showing ERROR/WARNING/CRITICAL counts
2. **Order Flow Logs** - Table view of order submission/response logs
3. **Errors by Service** - Pie chart of error distribution across services
4. **Order Events Timeline** - Time series of order events

---

## 📋 Import Steps

### Method 1: SSH Tunnel + Local Browser

#### Step 1: Create SSH Tunnel
```powershell
# Windows PowerShell
wsl ssh -i ~/.ssh/hetzner_fresh -L 3000:localhost:3000 root@46.224.116.254 -N
```

Keep this terminal open while accessing Grafana.

#### Step 2: Access Grafana
Open browser: **http://localhost:3000**

Default credentials:
- Username: `admin`
- Password: `admin` (or custom password if changed)

#### Step 3: Import Dashboard

1. **Navigate to Dashboards** → Click "New" → "Import"

2. **Upload JSON File:**
   - Click "Upload JSON file"
   - Select: `c:\quantum_trader\observability\grafana\dashboards\p1b_log_aggregation.json`
   
   OR

3. **Paste JSON:**
   - Copy contents of `p1b_log_aggregation.json`
   - Paste into "Import via panel json" field

4. **Configure Settings:**
   - **Name:** P1-B Log Aggregation (or keep default)
   - **Folder:** General (or create "P1-B" folder)
   - **UID:** Auto-generated (or keep from JSON)
   - **Datasource:** Select "Loki" from dropdown

5. **Click "Import"**

6. **Verify Dashboard:**
   - Dashboard should load with 4 panels
   - Check that Loki datasource queries are working
   - Verify logs are flowing (may take 1-2 minutes for data)

---

### Method 2: Direct VPS Access (If VPS has Desktop/Browser)

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Access Grafana locally on VPS
curl http://localhost:3000
# Open browser on VPS (if available) to http://localhost:3000
```

---

### Method 3: Grafana Provisioning (Automated)

Dashboard is already in provisioning directory:
```
observability/grafana/dashboards/p1b_log_aggregation.json
```

**Check if Grafana is configured for provisioning:**
```bash
# On VPS
docker exec quantum_grafana cat /etc/grafana/provisioning/dashboards/dashboard.yml
```

If provisioning is enabled, dashboard should auto-load on Grafana restart:
```bash
docker restart quantum_grafana
# Wait 30 seconds
curl http://localhost:3000/api/dashboards/db/p1b-log-aggregation
```

---

## 🔍 Verification Checklist

After import, verify:

### 1. Datasource Connection
- [ ] Loki datasource shows "Connected" status
- [ ] Test query returns results: `{container=~"quantum_.*"}`

### 2. Panel Queries
- [ ] **Errors by Level:** Shows bars for ERROR/WARNING/CRITICAL
- [ ] **Order Flow Logs:** Shows logs with `order_submit` or `order_response` in message
- [ ] **Errors by Service:** Pie chart with auto_executor/ai_engine slices
- [ ] **Order Events Timeline:** Time series with order event markers

### 3. Filters
- [ ] Time range selector works (Last 1h, Last 6h, etc.)
- [ ] Refresh button updates data
- [ ] Panel-specific filters (if any) work

### 4. Clickable Links
- [ ] correlation_id links navigate to filtered logs
- [ ] order_id links navigate to order-specific logs

---

## 🐛 Troubleshooting

### Issue: No Data in Panels
**Causes:**
1. Loki not ingesting logs yet (wait 2-3 minutes)
2. Promtail not scraping containers
3. Services not emitting JSON logs

**Diagnostics:**
```bash
# Check Loki ingestion
curl -s "http://localhost:3100/loki/api/v1/label/service/values"
# Should return: ["auto_executor", "ai_engine"]

# Check Promtail scraping
journalctl -u quantum_promtail.service --tail 50 | grep "Sending batch"

# Check JSON logs
journalctl -u quantum_auto_executor.service 2>&1 | grep '"service":' | tail -5
```

### Issue: Datasource Not Found
**Solution:**
1. Go to Configuration → Data sources
2. Verify "Loki" datasource exists
3. URL should be: `http://loki:3100`
4. Click "Save & test" to verify connection

### Issue: Dashboard Import Fails
**Solution:**
1. Check JSON syntax (paste into jsonlint.com)
2. Verify Loki datasource exists before import
3. Try importing with different UID (edit JSON and change `"uid"` field)

### Issue: Panels Show "No data"
**Solution:**
1. Adjust time range to "Last 24 hours"
2. Check LogQL query syntax (click "Edit" on panel)
3. Test query in Explore view: `/explore?orgId=1&left=...`

---

## 📊 Dashboard Queries (Reference)

### Panel 1: Errors by Level (15m)
```logql
sum by (level) (count_over_time({container=~"quantum_.*"} | json | level=~"ERROR|WARNING|CRITICAL" [15m]))
```

### Panel 2: Order Flow Logs
```logql
{container=~"quantum_.*"} | json | msg=~"order_submit|order_response" | line_format "{{.ts}} [{{.level}}] {{.msg}}"
```

### Panel 3: Errors by Service
```logql
sum by (service) (count_over_time({container=~"quantum_.*"} | json | level="ERROR" [1h]))
```

### Panel 4: Order Events Timeline
```logql
count_over_time({container=~"quantum_.*"} | json | event=~"order_submit|order_response" [1m])
```

---

## 🎓 Next Steps

After successful import:

1. **Create Custom Views:**
   - Clone P1-B dashboard for service-specific views
   - Add panels for specific error patterns
   - Create alerts based on log patterns

2. **Set Up Alerts:**
   - Add alert rules to panels (e.g., "Errors > 100 in 5m")
   - Configure notification channels (Slack, Email)

3. **Optimize Queries:**
   - Add filters for specific symbols/strategies
   - Create variables for dynamic filtering
   - Add correlation_id drill-down links

4. **Share Dashboard:**
   - Export dashboard JSON for backup
   - Share link with team: `http://46.224.116.254:3000/d/<dashboard-uid>`

---

## ✅ Success Criteria

Dashboard import is successful when:
- ✅ All 4 panels display data
- ✅ Loki datasource queries execute without errors
- ✅ Time range filtering works
- ✅ correlation_id links navigate correctly
- ✅ Dashboard auto-refreshes every 30s (configurable)

---

## 📚 Additional Resources

- **Grafana Loki Documentation:** https://grafana.com/docs/loki/latest/
- **LogQL Query Language:** https://grafana.com/docs/loki/latest/logql/
- **Grafana Dashboards:** https://grafana.com/docs/grafana/latest/dashboards/
- **P1-B Runbook:** `RUNBOOKS/P1B_logging_stack.md`

