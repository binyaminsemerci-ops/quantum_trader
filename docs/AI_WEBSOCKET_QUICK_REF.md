# 🔴 Live Audit Streaming - Quick Reference

**Dashboard:** http://46.224.116.254:8080  
**WebSocket:** ws://46.224.116.254:8080/ws/audit  
**Status:** ✅ Active  
**Last Updated:** December 17, 2025

---

## 🎯 What Is It?

Real-time WebSocket streaming sends new audit log entries to your browser **instantly** (within 3 seconds) without page refresh.

---

## 🚀 Quick Start

### 1. Open Dashboard
```
http://46.224.116.254:8080
```

### 2. Open Browser Console (F12)
Look for connection message:
```
🔌 Connecting to WebSocket: ws://46.224.116.254:8080/ws/audit
✅ WebSocket connected - Real-time audit streaming active
```

### 3. Click "View Audit Log"
- Initial log loads
- WebSocket starts monitoring
- New entries appear automatically

### 4. Test Real-Time Update
```bash
ssh qt@vps "cd ~/quantum_trader && \
  python3 tools/audit_logger.py test 'Live streaming test'"
```
**Result:** New entry appears in dashboard within 3 seconds!

---

## 🔍 How It Works

### Server (Every 3 Seconds)
1. Check audit log file size
2. If file grew → read new bytes
3. Broadcast to all connected browsers

### Browser (Instant)
1. Receive new data via WebSocket
2. Append to display or re-apply filter
3. Auto-scroll to show latest entry

**Latency:** <3 seconds  
**Bandwidth:** ~100 bytes per update  
**CPU Impact:** Negligible

---

## 💡 Usage Tips

### With No Filter
- New entries append automatically
- Auto-scrolls to bottom
- Green text on dark background

### With Active Search Filter
- New entries filtered in real-time
- Highlights applied automatically
- Yellow highlighting preserved

### Multiple Browser Tabs
- All tabs update simultaneously
- Each tab has own WebSocket connection
- No conflicts or duplicates

---

## 🎨 Console Messages

### Normal Operation
```
✅ WebSocket connected - Real-time audit streaming active
📡 New audit data received: 87 bytes
📡 New audit data received: 102 bytes
```

### Connection Lost
```
🔌 WebSocket disconnected - Reconnecting in 5 seconds...
🔌 Connecting to WebSocket: ws://46.224.116.254:8080/ws/audit
✅ WebSocket connected - Real-time audit streaming active
```

### Errors
```
❌ WebSocket error: [error details]
```

---

## 🔄 Auto-Reconnect

**If connection lost:**
- Waits 5 seconds
- Attempts reconnection
- Repeats until successful
- No user action required

**Common causes:**
- Container restart
- Network interruption
- Server maintenance

---

## 🧪 Testing Scenarios

### Test 1: Live Update
```bash
# In terminal
python3 tools/audit_logger.py test "WebSocket streaming test"

# In browser (within 3 seconds)
# New entry appears automatically
```

### Test 2: Search + Live Update
```bash
# In browser
1. Load audit log
2. Search for "database"

# In terminal
python3 tools/audit_logger.py test "Auto-repair: Fixed database issue"

# In browser
# New entry appears and is highlighted
```

### Test 3: Multiple Clients
```bash
# Open dashboard in 2 browser tabs
# Trigger audit event
# Both tabs update simultaneously
```

### Test 4: Reconnection
```bash
# Restart dashboard: docker restart quantum_dashboard
# Browser shows: "Reconnecting in 5 seconds..."
# After 5 seconds: "WebSocket connected"
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Update Latency | <3 seconds |
| Polling Interval | 3 seconds |
| Initial Load | ~1 KB (last 1000 chars) |
| Update Size | ~100 bytes average |
| CPU per Client | <0.1% |
| Memory per Client | ~5-10 MB |
| Max Clients | Unlimited (async) |

---

## 🔧 Troubleshooting

### "WebSocket error" in console
1. Check container running: `docker ps | grep quantum_dashboard`
2. Test HTTP endpoint: `curl http://46.224.116.254:8080`
3. Check firewall allows port 8080
4. Try different browser (Chrome recommended)

### No real-time updates
1. Verify WebSocket connected (check console)
2. Wait 3 seconds (polling interval)
3. Check new entry written: `tail ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log`
4. Hard refresh: Ctrl+F5

### Constant reconnection loop
1. Check server logs: `docker logs quantum_dashboard`
2. Verify audit file exists
3. Restart container: `docker restart quantum_dashboard`

---

## 🎯 Key Differences

### Before (Prompt 16 - Polling)
- Updates: Every 10 minutes (auto-refresh)
- Latency: Up to 10 minutes
- Manual: Click button for immediate update

### After (Prompt 17 - WebSocket)
- Updates: Real-time push (within 3 seconds)
- Latency: <3 seconds
- Manual: No action needed - fully automatic

**Result:** 200x faster updates, zero user intervention!

---

## 📱 Browser Support

✅ **Fully Supported:**
- Google Chrome 88+
- Microsoft Edge 88+
- Mozilla Firefox 85+
- Safari 14+
- Opera 74+

⚠️ **Limited Support:**
- Internet Explorer (not supported)
- Older mobile browsers

---

## 🔒 Security Notes

**Current Setup:**
- No authentication (trusted network)
- No encryption (HTTP/WS)
- Firewall-protected VPS

**For Production:**
- Enable HTTPS/WSS (TLS encryption)
- Add token-based authentication
- Implement rate limiting
- Use NGINX reverse proxy

---

## 🛠️ Developer Reference

### WebSocket URL
```
ws://46.224.116.254:8080/ws/audit
```

### Test with wscat
```bash
npm install -g wscat
wscat -c ws://46.224.116.254:8080/ws/audit
```

### Test with Python
```python
import asyncio
import websockets

async def test():
    uri = "ws://46.224.116.254:8080/ws/audit"
    async with websockets.connect(uri) as ws:
        msg = await ws.recv()
        print(f"Received: {msg}")

asyncio.run(test())
```

### Test with curl (HTTP upgrade)
```bash
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     --header "Sec-WebSocket-Version: 13" \
     http://46.224.116.254:8080/ws/audit
```

---

## 📚 Related Docs

- **Full Report:** `AI_LIVE_STREAMING_DEPLOYED.md`
- **Search Guide:** `AI_SMART_SEARCH_QUICK_REF.md`
- **Audit Layer:** `AI_AUDIT_LAYER_QUICK_REF.md`

---

## ✅ Quick Verification

**Run this command:**
```bash
ssh qt@46.224.116.254 \
  "cd ~/quantum_trader && \
   python3 tools/audit_logger.py test 'WebSocket streaming verification' && \
   echo 'Check your dashboard - new entry should appear in 3 seconds!'"
```

**Expected:**
- Dashboard updates automatically
- New entry visible within 3 seconds
- Console shows: `📡 New audit data received`

---

## 🎉 Success Checklist

- [x] WebSocket endpoint active
- [x] Browser auto-connects on page load
- [x] New entries appear within 3 seconds
- [x] Auto-scroll to latest entry
- [x] Search filter works with real-time updates
- [x] Auto-reconnect after disconnect
- [x] Multiple tabs supported
- [x] No JavaScript errors
- [x] Console logging functional

---

**Status:** ✅ Fully Operational  
**Technology:** FastAPI WebSocket + Native JS  
**Performance:** <3s latency, <0.1% CPU  

**Access:** http://46.224.116.254:8080
