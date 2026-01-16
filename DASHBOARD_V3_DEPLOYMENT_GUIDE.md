# Dashboard V3.0 Deployment Guide

## Overview

Dashboard V3.0 is a modern, real-time trading dashboard with WebSocket-powered live updates, comprehensive system monitoring, and responsive design. This guide covers deployment, configuration, and troubleshooting.

---

## Architecture

### Frontend (Next.js + TypeScript)
- **Location**: `frontend/src/pages/dashboard/`
- **Framework**: Next.js 14 with App Router
- **State Management**: React hooks + WebSocket
- **Styling**: Global CSS with CSS modules

### Backend (FastAPI + Python)
- **Location**: `backend/api/dashboard/`
- **Framework**: FastAPI with async/await
- **WebSocket**: Real-time event streaming
- **Data Sources**: 
  - Existing: `/api/dashboard/snapshot` (working in testnet)
  - Future: BFF endpoints for microservices (production)

---

## Deployment Steps

### 1. Prerequisites

**System Requirements:**
- Node.js 18+ (for frontend)
- Python 3.11+ (for backend)
- Docker & Docker Compose (recommended)
- 4GB RAM minimum
- Modern browser (Chrome/Firefox/Edge)

**Environment Variables:**
```bash
# Backend (.env)
BINANCE_TESTNET=true
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
REDIS_URL=redis://localhost:6379  # Optional

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### 2. Backend Deployment

#### Option A: Docker (Recommended)
```bash
# Build and start backend
systemctl build backend
systemctl up -d backend

# Verify health
curl http://localhost:8000/health

# Check logs
journalctl -u quantum_backend.service --tail 50
```

#### Option B: Local Python
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify Backend:**
```bash
# Test REST endpoint
curl http://localhost:8000/api/dashboard/snapshot

# Test WebSocket (requires wscat: npm install -g wscat)
wscat -c ws://localhost:8000/ws/dashboard
```

### 3. Frontend Deployment

#### Development Mode
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:3000`

#### Production Build
```bash
cd frontend
npm run build
npm start
```

#### Production Deployment (Vercel/Netlify)
```bash
# Build settings
Build Command: npm run build
Output Directory: .next
Install Command: npm install

# Environment variables (set in platform)
NEXT_PUBLIC_API_URL=https://your-backend.com
NEXT_PUBLIC_WS_URL=wss://your-backend.com
```

### 4. Verify Installation

**Check Frontend Components:**
```bash
# Run validation script
python tests/validate_dashboard_v3_frontend.py
```

Expected output: `✓ 27 successes, ⚠ 0 warnings, ✗ 0 errors`

**Check Backend Endpoints:**
```bash
# Health check
curl http://localhost:8000/health | jq

# Dashboard data
curl http://localhost:8000/api/dashboard/snapshot | jq '.portfolio'

# WebSocket connection
wscat -c ws://localhost:8000/ws/dashboard
# Expected: Real-time events every 5-10 seconds
```

**Access Dashboard:**
1. Open browser: `http://localhost:3000/dashboard`
2. Verify all 4 tabs load: Overview, Trading, Risk, System
3. Check real-time updates appear (timestamp changes)
4. Test tab navigation works smoothly

---

## Configuration

### Backend Configuration

**WebSocket Settings** (`backend/api/dashboard/websocket.py`):
```python
BATCH_SIZE = 10  # Events per batch
MAX_RATE_PER_SEC = 50  # Max events/second
HEARTBEAT_INTERVAL = 30  # Seconds between heartbeats
```

**Dashboard Snapshot** (`backend/api/dashboard/routes.py`):
- Timeout: 5 seconds per service
- Retry: No automatic retry
- Caching: None (real-time data)

### Frontend Configuration

**WebSocket Settings** (`frontend/src/hooks/useDashboardStream.ts`):
```typescript
RECONNECT_DELAY = 3000  // 3 seconds
MAX_RECONNECT_ATTEMPTS = 5
HEARTBEAT_TIMEOUT = 60000  // 60 seconds
```

**Polling Fallback:**
- Disabled by default
- Enable if WebSocket unavailable
- Interval: 10 seconds

---

## API Reference

### REST Endpoints

#### GET `/api/dashboard/snapshot`
**Description**: Complete dashboard data snapshot  
**Response Time**: < 5 seconds  
**Response**:
```json
{
  "schema_version": 1,
  "timestamp": "2025-12-05T00:00:00Z",
  "portfolio": {
    "equity": 15000.0,
    "daily_pnl": 245.50,
    "daily_pnl_pct": 1.64,
    "position_count": 10
  },
  "positions": [...],
  "signals": [...],
  "risk": {...},
  "system": {...}
}
```

#### GET `/health`
**Description**: Backend health check  
**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-05T00:00:00Z",
  "uptime_seconds": 3600,
  "binance_testnet": true
}
```

### WebSocket Events

#### Connection: `ws://localhost:8000/ws/dashboard`

**Event Types:**
- `signal.generated` - New trading signal
- `position.opened` - Position entered
- `position.closed` - Position exited
- `risk.state_changed` - Risk level changed
- `system.health_update` - System status update

**Event Format:**
```json
{
  "event_type": "position.opened",
  "timestamp": "2025-12-05T00:00:00Z",
  "data": {
    "symbol": "BTCUSDT",
    "side": "LONG",
    "quantity": 0.1,
    "entry_price": 42000.0
  }
}
```

---

## Frontend Components

### File Structure
```
frontend/src/
├── pages/dashboard/
│   └── index.tsx                 # Main dashboard page (275 lines)
├── components/dashboard/
│   ├── OverviewTab.tsx          # Global metrics (281 lines)
│   ├── TradingTab.tsx           # Positions & orders (262 lines)
│   ├── RiskTab.tsx              # Risk metrics (288 lines)
│   ├── SystemTab.tsx            # System health (358 lines)
│   └── DashboardCard.tsx        # Reusable card (65 lines)
├── hooks/
│   └── useDashboardStream.ts    # WebSocket hook (237 lines)
└── styles/
    └── globals.css              # Global styles (1700+ lines)
```

### Component Props

**DashboardCard**:
```typescript
interface DashboardCardProps {
  title: string;
  value?: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  status?: 'success' | 'warning' | 'error' | 'info';
  children?: React.ReactNode;
}
```

**useDashboardStream Hook**:
```typescript
const {
  data,           // Latest dashboard data
  events,         // Real-time events array
  isConnected,    // WebSocket connection status
  error           // Error message if any
} = useDashboardStream();
```

---

## Testing

### Automated Tests

**Run All Tests:**
```bash
python tests/run_dashboard_v3_tests.py
```

**Run Individual Test Suites:**
```bash
# Frontend validation
python tests/validate_dashboard_v3_frontend.py

# Backend API tests (requires pytest)
pytest tests/test_dashboard_v3_bff.py -v

# Integration tests
python tests/test_dashboard_v3_integration.py
```

### Manual Testing Checklist

**Frontend:**
- [ ] Dashboard page loads without errors
- [ ] All 4 tabs render correctly
- [ ] Tab navigation works smoothly
- [ ] Real-time updates appear (check timestamps)
- [ ] Cards display data properly
- [ ] Responsive layout works on mobile
- [ ] WebSocket connection indicator shows status
- [ ] No console errors

**Backend:**
- [ ] `/health` returns 200 OK
- [ ] `/api/dashboard/snapshot` returns valid JSON
- [ ] WebSocket accepts connections
- [ ] Events stream in real-time
- [ ] No Python exceptions in logs
- [ ] Docker container stays running

**Integration:**
- [ ] Frontend receives backend data
- [ ] WebSocket events update UI
- [ ] Timestamp consistency (< 60s drift)
- [ ] Error handling works (disconnect/reconnect)
- [ ] Performance acceptable (< 5s page load)

---

## Troubleshooting

### Frontend Issues

**Problem**: "Cannot connect to backend"  
**Solution**:
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check NEXT_PUBLIC_API_URL in `.env.local`
3. Check CORS settings in backend
4. Verify network/firewall rules

**Problem**: "WebSocket connection failed"  
**Solution**:
1. Test WebSocket manually: `wscat -c ws://localhost:8000/ws/dashboard`
2. Check NEXT_PUBLIC_WS_URL uses `ws://` or `wss://`
3. Verify backend WebSocket route registered
4. Check browser console for specific error

**Problem**: "No real-time updates"  
**Solution**:
1. Check `isConnected` in dashboard UI
2. Verify events are being generated (check backend logs)
3. Test WebSocket with `wscat` to confirm events sent
4. Check browser network tab for WebSocket frames

**Problem**: "Dashboard shows 'Loading...'"  
**Solution**:
1. Open browser DevTools → Network tab
2. Check if `/api/dashboard/snapshot` returns data
3. Verify response structure matches TypeScript interface
4. Check console for JSON parsing errors

### Backend Issues

**Problem**: "404 Not Found on /api/dashboard/*"  
**Solution**:
1. Check backend logs for route registration: `"Dashboard API routes registered"`
2. Verify router imports in `main.py`
3. Check FastAPI route listing: `docker exec quantum_backend python -c "from backend.main import app; print([r.path for r in app.routes if hasattr(r, 'path')])"`
4. Restart backend: `systemctl restart backend`

**Problem**: "WebSocket immediately disconnects"  
**Solution**:
1. Check backend logs for WebSocket errors
2. Verify `/ws/dashboard` route exists
3. Test connection with: `wscat -c ws://localhost:8000/ws/dashboard`
4. Check for authentication/authorization requirements

**Problem**: "Backend logs show 'TimeoutError'"  
**Solution**:
1. Check if microservices are running (if using BFF)
2. Reduce timeout in `fetch_json()` function
3. Use `/api/dashboard/snapshot` instead of BFF endpoints
4. Verify network connectivity to external services

**Problem**: "High CPU/Memory usage"  
**Solution**:
1. Check number of WebSocket connections
2. Reduce WebSocket event rate (increase batch size)
3. Enable event filtering in websocket.py
4. Check for infinite loops in async tasks

### Docker Issues

**Problem**: "Container keeps restarting"  
**Solution**:
1. Check logs: `journalctl -u quantum_backend.service --tail 100`
2. Look for Python import errors
3. Verify all dependencies in requirements.txt
4. Check environment variables in systemctl.yml

**Problem**: "Code changes not reflected"  
**Solution**:
1. Rebuild image: `systemctl build backend`
2. Restart container: `systemctl up -d backend`
3. Verify volume mounts in systemctl.yml
4. Check file permissions on host

**Problem**: "Port 8000 already in use"  
**Solution**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

---

## Performance Optimization

### Frontend

**Code Splitting:**
```typescript
// Lazy load heavy components
const SystemTab = dynamic(() => import('./SystemTab'), {
  loading: () => <div>Loading...</div>
});
```

**Memo & Callback:**
```typescript
// Prevent unnecessary re-renders
const MemoizedCard = React.memo(DashboardCard);

// Stable callback references
const handleTabChange = useCallback((tab) => {
  setActiveTab(tab);
}, []);
```

**WebSocket Optimization:**
```typescript
// Batch updates
const [events, setEvents] = useState([]);
useEffect(() => {
  const newEvents = wsEvents.slice(-10); // Keep last 10
  setEvents(newEvents);
}, [wsEvents]);
```

### Backend

**Async Optimization:**
```python
# Use asyncio.gather for parallel requests
async def fetch_all_data():
    results = await asyncio.gather(
        fetch_portfolio(),
        fetch_positions(),
        fetch_risk(),
        return_exceptions=True
    )
    return results
```

**Caching:**
```python
# Add Redis caching for expensive operations
from redis import asyncio as aioredis

async def get_cached_data(key: str):
    redis = await aioredis.from_url("redis://localhost")
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)
    # Fetch and cache
```

**WebSocket Batching:**
```python
# Already implemented in websocket.py
BATCH_SIZE = 10
MAX_RATE_PER_SEC = 50
```

---

## Security Considerations

### API Security

**Authentication** (future):
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@router.get("/snapshot")
async def get_snapshot(token: str = Depends(security)):
    # Verify JWT token
    if not verify_token(token):
        raise HTTPException(401, "Invalid token")
```

**CORS Configuration:**
```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

**Rate Limiting:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.get("/snapshot")
@limiter.limit("10/minute")
async def get_snapshot(request: Request):
    ...
```

### Frontend Security

**Environment Variables:**
- Never commit API keys to git
- Use `.env.local` for secrets
- Validate `process.env` at runtime

**XSS Prevention:**
- React escapes by default
- Avoid `dangerouslySetInnerHTML`
- Sanitize user inputs

**WebSocket Security:**
```typescript
// Validate messages
const isValidEvent = (event: any): boolean => {
  return event && 
         typeof event.event_type === 'string' &&
         event.timestamp &&
         event.data;
};
```

---

## Monitoring & Logging

### Backend Logging

**Log Levels:**
- `DEBUG`: Detailed diagnostic info
- `INFO`: General system events
- `WARNING`: Potential issues
- `ERROR`: Errors requiring attention
- `CRITICAL`: System failures

**Key Log Messages:**
```python
logger.info(f"[Dashboard] Snapshot generated in {elapsed:.2f}s")
logger.warning(f"[WebSocket] Client disconnected: {client_id}")
logger.error(f"[API] Failed to fetch portfolio data: {error}")
```

**Docker Logs:**
```bash
# Tail logs
docker logs -f quantum_backend

# Search logs
journalctl -u quantum_backend.service 2>&1 | grep "ERROR"

# Export logs
journalctl -u quantum_backend.service > backend.log
```

### Frontend Monitoring

**Console Logging:**
```typescript
console.log('[Dashboard] Loaded with data:', data);
console.warn('[WebSocket] Connection unstable');
console.error('[API] Failed to fetch:', error);
```

**Error Boundaries:**
```typescript
class ErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    console.error('Dashboard error:', error, errorInfo);
    // Send to monitoring service
  }
}
```

### Production Monitoring

**Recommended Tools:**
- **Sentry**: Error tracking
- **Datadog/New Relic**: APM
- **Grafana**: Metrics visualization
- **Prometheus**: Metrics collection

---

## Maintenance

### Regular Tasks

**Daily:**
- Check backend logs for errors
- Verify WebSocket connections stable
- Monitor API response times

**Weekly:**
- Review error rates
- Check disk space usage
- Update dependencies (security patches)

**Monthly:**
- Full system backup
- Performance audit
- Security review

### Updates

**Backend Updates:**
```bash
# Update Python packages
pip list --outdated
pip install --upgrade <package>

# Update requirements.txt
pip freeze > requirements.txt

# Test thoroughly before deploying
```

**Frontend Updates:**
```bash
# Check for updates
npm outdated

# Update packages
npm update

# Major version updates
npm install <package>@latest

# Test build
npm run build
```

### Backup & Recovery

**Backup Checklist:**
- [ ] Source code (git repository)
- [ ] Environment variables
- [ ] Database (if used)
- [ ] Configuration files
- [ ] Docker images

**Recovery Steps:**
1. Clone repository: `git clone <repo>`
2. Restore environment variables
3. Rebuild Docker images: `systemctl build`
4. Start services: `systemctl up -d`
5. Verify health endpoints
6. Run test suite

---

## Production Readiness Checklist

### Before Production Deployment

**Code Quality:**
- [ ] All tests passing
- [ ] No console.log in frontend
- [ ] No debug logging in backend
- [ ] TypeScript strict mode enabled
- [ ] Linting passes (ESLint, Pylint)

**Performance:**
- [ ] Page load < 3 seconds
- [ ] API response < 2 seconds
- [ ] WebSocket latency < 100ms
- [ ] Lighthouse score > 90

**Security:**
- [ ] HTTPS enabled
- [ ] WSS (WebSocket Secure) enabled
- [ ] CORS configured properly
- [ ] Rate limiting enabled
- [ ] Environment variables secured

**Monitoring:**
- [ ] Error tracking configured
- [ ] Logging enabled
- [ ] Alerts configured
- [ ] Health checks automated

**Documentation:**
- [ ] API documentation complete
- [ ] Deployment guide reviewed
- [ ] Runbook created
- [ ] Team trained

---

## Support & Resources

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Next.js: https://nextjs.org/docs
- React: https://react.dev/
- WebSocket API: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket

### Internal Files
- `AI_INTEGRATION_COMPLETE.md` - System integration overview
- `ARCHITECTURE_V2_INTEGRATION_COMPLETE.md` - Architecture details
- `tests/` - Comprehensive test suite

### Contact
- Developer: Quantum Trader Team
- Repository: quantum_trader
- Issues: Use GitHub Issues

---

## Appendix

### Environment Variables Reference

**Backend:**
```bash
# Required
BINANCE_TESTNET=true
BINANCE_API_KEY=<key>
BINANCE_API_SECRET=<secret>

# Optional
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
DASHBOARD_TIMEOUT=5
WS_BATCH_SIZE=10
WS_MAX_RATE=50
```

**Frontend:**
```bash
# Required
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Optional
NEXT_PUBLIC_POLLING_INTERVAL=10000
NEXT_PUBLIC_WS_RECONNECT_DELAY=3000
```

### Port Reference
- **3000**: Frontend (Next.js dev server)
- **8000**: Backend (FastAPI)
- **6379**: Redis (optional)
- **5432**: PostgreSQL (future)

### File Sizes
- **Frontend**: ~1.5MB (minified)
- **Backend**: ~500MB (Docker image)
- **Total**: ~1GB (all services)

---

**Dashboard V3.0 - Production Ready** ✅  
*Last Updated: December 5, 2025*

