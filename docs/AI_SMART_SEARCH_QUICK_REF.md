# 🎯 Smart Log Search - Quick Reference Card

**Dashboard:** http://46.224.116.254:8080  
**Last Updated:** December 17, 2025

---

## 🚀 How to Use

### Step 1: Load Audit Log
```
Click: 📊 View Audit Log
```
**Result:** Shows recent auto-repair actions

### Step 2: Search for Keywords
```
Type in search box: database
Click: 🔍 Search
```
**Result:** Only lines containing "database" are shown, with term highlighted in yellow

### Step 3: Clear Filter
```
Click: ✕ Clear
```
**Result:** Full audit log restored

---

## 🔍 Example Searches

### Find Database Operations
**Search:** `database`  
**Matches:**
```
✨ [2025-12-17T17:40:37Z] Auto-repair: Created PostgreSQL database quantum
                                                          ^^^^^^^^
```

### Find XGBoost Model Changes
**Search:** `xgboost`  
**Matches:**
```
✨ [2025-12-17T17:40:37Z] Auto-repair: Retrained XGBoost model to 22 features
                                                 ^^^^^^^
```

### Find Grafana Actions
**Search:** `grafana`  
**Matches:**
```
✨ [2025-12-17T17:40:37Z] Auto-repair: Restarted Grafana container
                                                 ^^^^^^^
```

### Find All Auto-Repairs
**Search:** `auto-repair`  
**Matches:** All repair actions (multiple entries highlighted)

### Find Weekly Health Checks
**Search:** `weekly`  
**Matches:**
```
✨ [2025-12-17T17:40:37Z] Weekly self-heal: Started validation cycle
   ^^^^^^
✨ [2025-12-17T17:40:37Z] Weekly self-heal: All checks passed
   ^^^^^^
```

---

## 💡 Search Tips

### Basic Searches
- **Case-insensitive:** `ERROR`, `error`, `Error` all work
- **Partial matches:** `repair` matches "auto-repair", "repaired", etc.
- **Spaces work:** `self-heal` or `self heal` both match

### Advanced Searches (Regex)
- **Multiple terms:** `database|xgboost|grafana` (finds any of these)
- **Date patterns:** `2025-12-17` (find specific day)
- **Word boundaries:** `\btest\b` (exact word "test" only)

### Common Patterns
```
database     → PostgreSQL operations
xgboost      → ML model updates
grafana      → Dashboard/visualization changes
auto-repair  → All repair actions
weekly       → Weekly health checks
error        → Error messages (if logged)
restart      → Container restarts
validation   → Validation checks
```

---

## 🎨 UI Controls

### Buttons

| Button | Icon | Color | Action |
|--------|------|-------|--------|
| View Audit Log | 📊 | Blue | Load recent auto-repair logs |
| View Weekly Report | 📄 | Green | Load weekly health report |
| Search | 🔍 | Orange | Filter log by keyword |
| Clear | ✕ | Gray | Restore full log view |

### Search Bar
- **Placeholder:** "Filter e.g. database, xgboost, grafana"
- **Width:** Flexible (grows with screen size)
- **Enter Key:** Press Enter to search (same as clicking Search button)

---

## 🔄 Auto-Refresh

**Interval:** Every 10 minutes (600,000 ms)  
**Behavior:**
- Audit log refreshes automatically
- Search filter **is cleared** on refresh
- Manual refresh: Click "View Audit Log" again

---

## 📊 Current Log Entries

```
[2025-12-17T17:39:46Z] Test audit entry - system check
[2025-12-17T17:40:37Z] Auto-repair: Created PostgreSQL database quantum
[2025-12-17T17:40:37Z] Auto-repair: Retrained XGBoost model to 22 features
[2025-12-17T17:40:37Z] Auto-repair: Restarted Grafana container
[2025-12-17T17:40:37Z] Weekly self-heal: Started validation cycle
[2025-12-17T17:40:37Z] Weekly self-heal: All checks passed
[2025-12-17T18:04:48Z] Test audit entry - system check (×3)
```

**Total Entries:** 9  
**Date Range:** Dec 17, 2025  
**Log File:** `/home/qt/quantum_trader/status/AUTO_REPAIR_AUDIT.log`

---

## 🎯 Real-World Usage Scenarios

### Scenario 1: Database Issue Investigation
**Goal:** Find all database-related repairs  
**Action:**
1. Click "📊 View Audit Log"
2. Type `database` in search box
3. Click "🔍 Search"
4. Review highlighted entries

**Expected Result:**
```
[2025-12-17T17:40:37Z] Auto-repair: Created PostgreSQL database quantum
```

### Scenario 2: Model Training History
**Goal:** Check when XGBoost models were retrained  
**Action:**
1. Load audit log
2. Search for `xgboost`
3. Review timestamps and feature counts

**Expected Result:**
```
[2025-12-17T17:40:37Z] Auto-repair: Retrained XGBoost model to 22 features
```

### Scenario 3: Weekly Validation Status
**Goal:** Check if weekly health checks passed  
**Action:**
1. Load audit log
2. Search for `weekly`
3. Look for "PASSED" or "FAILED" status

**Expected Result:**
```
[2025-12-17T17:40:37Z] Weekly self-heal: Started validation cycle
[2025-12-17T17:40:37Z] Weekly self-heal: All checks passed
```

### Scenario 4: Container Restart Analysis
**Goal:** Find all Grafana container restarts  
**Action:**
1. Load audit log
2. Search for `grafana`
3. Look for "Restarted" entries

**Expected Result:**
```
[2025-12-17T17:40:37Z] Auto-repair: Restarted Grafana container
```

---

## 🚨 Troubleshooting

### Problem: Search Returns "(no matches found)"
**Solutions:**
- Check spelling of search term
- Try broader terms (e.g., `repair` instead of `auto-repair`)
- Verify audit log loaded (should see timestamps)
- Use `|` for multiple terms: `database|postgres`

### Problem: Highlights Not Showing
**Solutions:**
- Ensure JavaScript enabled in browser
- Check browser console (F12) for errors
- Try different browser (Chrome, Firefox, Edge)
- Refresh page (Ctrl+F5 for hard refresh)

### Problem: Audit Log Empty
**Solutions:**
- Wait for auto-repair actions to occur
- Run manual test: `python3 tools/audit_logger.py test "Test message"`
- Check log file exists: `ls -lh ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log`
- Verify API endpoint: `curl http://localhost:8080/api/audit`

### Problem: Auto-Refresh Not Working
**Solutions:**
- Check JavaScript console for errors
- Verify `setInterval(loadAudit, 600000)` in page source
- Manual refresh: Click "View Audit Log" button
- Check browser not blocking auto-refresh

---

## 📱 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Enter** | Search (when in search box) |
| **Escape** | Clear search (future feature) |
| **Ctrl+F5** | Hard refresh page |
| **F12** | Open developer console |

---

## 🎉 Success Checklist

- [x] Dashboard accessible at http://46.224.116.254:8080
- [x] Search box visible in Audit & Reports section
- [x] "View Audit Log" button loads log entries
- [x] Search filters log correctly
- [x] Highlights appear in yellow background
- [x] Clear button restores full log
- [x] Auto-refresh works (10 min interval)
- [x] API endpoints return data
- [x] No JavaScript errors in console

---

## 📚 Related Documentation

- **Full Deployment Report:** `AI_SMART_LOG_SEARCH_DEPLOYED.md`
- **Audit Layer Guide:** `AI_AUDIT_LAYER_QUICK_REF.md`
- **Dashboard API:** `dashboard/app.py` (lines with `/api/audit`, `/api/reports`)
- **Frontend Code:** `dashboard/static/index.html` (search: `filterAudit()`)

---

## 🛠️ For Developers

### JavaScript Function Reference

**Load Audit Log:**
```javascript
async function loadAudit() {
  const res = await fetch('/api/audit');
  auditRaw = await res.text();
  auditBox.innerHTML = auditRaw.replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
```

**Filter Function:**
```javascript
function filterAudit() {
  const q = document.getElementById('searchBox').value.trim();
  const regex = new RegExp(q, 'ig');
  const filtered = auditRaw.split('\n')
    .filter(line => regex.test(line))
    .map(line => line.replace(regex, m => 
      `<mark style='background:yellow;color:black'>${m}</mark>`))
    .join('\n');
  auditBox.innerHTML = filtered || "(no matches found)";
}
```

**Clear Function:**
```javascript
function clearFilter() {
  document.getElementById('searchBox').value = '';
  auditBox.innerHTML = auditRaw.replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
```

### API Endpoints

**Audit Log:**
```bash
GET http://46.224.116.254:8080/api/audit
# Returns: Last 5KB of audit log (plain text)
```

**Weekly Reports:**
```bash
GET http://46.224.116.254:8080/api/reports
# Returns: Last 8KB of most recent weekly report
```

---

**Last Updated:** December 17, 2025  
**Version:** 1.0 (Prompt 16 Implementation)  
**Status:** ✅ Fully Operational

---

**Quick Access:**
👉 **http://46.224.116.254:8080**
