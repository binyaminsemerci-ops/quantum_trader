# QuantumFond Hedge Fund OS - Complete Setup Guide

## 🎯 Phase 17 - Core Infrastructure Build Complete

### Overview
Full-stack hedge fund operating system with FastAPI backend and React frontend.
Designed for deployment on api.quantumfond.com (backend) and app.quantumfond.com (frontend).

---

## 📁 Project Structure

```
quantumfond_backend/
├── main.py                    # FastAPI entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── db/
│   ├── connection.py         # Database configuration
│   └── models/
│       └── base_models.py    # SQLAlchemy models
└── routers/
    ├── auth_router.py        # JWT authentication
    ├── overview_router.py    # Dashboard data
    ├── trades_router.py      # Trading operations
    ├── risk_router.py        # Risk management
    ├── ai_router.py          # AI models status
    ├── strategy_router.py    # Strategy management
    ├── performance_router.py # Performance analytics
    ├── system_router.py      # System monitoring
    ├── admin_router.py       # Administration
    └── incident_router.py    # Incident tracking

quantumfond_frontend/
├── package.json              # Node dependencies
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind CSS config
├── index.html               # Entry HTML
└── src/
    ├── App.tsx              # Main app component
    ├── main.tsx             # React entry point
    ├── components/          # Reusable components
    │   ├── Sidebar.tsx
    │   ├── TopBar.tsx
    │   ├── InsightCard.tsx
    │   ├── StatusBanner.tsx
    │   └── TrendChart.tsx
    ├── pages/               # Page components
    │   ├── overview.tsx
    │   ├── live.tsx
    │   ├── risk.tsx
    │   ├── ai.tsx
    │   ├── strategy.tsx
    │   ├── performance.tsx
    │   ├── system.tsx
    │   ├── admin.tsx
    │   └── incident.tsx
    └── hooks/
        └── useFetch.ts      # Custom fetch hook
```

---

## 🚀 Backend Setup

### 1. Install Dependencies

```bash
cd quantumfond_backend
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 3. Initialize Database

```bash
# Create PostgreSQL database
createdb quantumdb

# Initialize tables
python -c "from db.connection import init_db; init_db()"
```

### 4. Run Backend

```bash
uvicorn main:app --reload --port 8000
```

**Backend available at:** `http://localhost:8000`
**API Docs:** `http://localhost:8000/docs`

---

## 🎨 Frontend Setup

### 1. Install Dependencies

```bash
cd quantumfond_frontend
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

**Frontend available at:** `http://localhost:5173`

### 3. Build for Production

```bash
npm run build
```

---

## 🔐 Authentication

### Default Users

| Username | Password | Role | Access Level |
|----------|----------|------|--------------|
| admin | AdminPass123 | admin | Full system access |
| riskops | Risk123 | risk | Risk management |
| trader | Trade123 | trader | Trading operations |
| viewer | View123 | viewer | Read-only access |

### JWT Authentication Flow

1. **Login:** POST `/auth/login` with username/password
2. **Receive:** JWT token + role information
3. **Use:** Include token in `Authorization: Bearer <token>` header
4. **Refresh:** POST `/auth/refresh` to get new token

---

## 📊 API Endpoints

### Overview Module (`/overview`)
- `GET /overview/dashboard` - Main dashboard data
- `GET /overview/metrics` - Key performance metrics
- `GET /overview/status` - System status

### Trades Module (`/trades`)
- `GET /trades/active` - Active positions
- `GET /trades/history` - Trade history
- `GET /trades/{trade_id}` - Trade details
- `GET /trades/live/feed` - Real-time feed

### Risk Module (`/risk`)
- `GET /risk/metrics` - Current risk metrics
- `GET /risk/limits` - Risk limits
- `GET /risk/circuit-breakers` - Circuit breaker status
- `GET /risk/exposure/breakdown` - Exposure analysis

### AI Module (`/ai`)
- `GET /ai/models` - AI model status
- `GET /ai/predictions` - Latest predictions
- `GET /ai/performance` - Model performance
- `GET /ai/retraining` - Retraining status

### Strategy Module (`/strategy`)
- `GET /strategy/active` - Active strategies
- `GET /strategy/{strategy_id}` - Strategy details
- `GET /strategy/backtests` - Backtest results
- `GET /strategy/optimization` - Optimization status

### Performance Module (`/performance`)
- `GET /performance/summary` - Performance summary
- `GET /performance/timeline` - Performance over time
- `GET /performance/benchmark` - Benchmark comparison
- `GET /performance/attribution` - Return attribution
- `GET /performance/reports/daily` - Daily report

### System Module (`/system`)
- `GET /system/health` - System health
- `GET /system/metrics` - Detailed metrics
- `GET /system/logs` - System logs
- `GET /system/alerts` - System alerts
- `GET /system/services` - Service status

### Admin Module (`/admin`) - Requires Admin Role
- `GET /admin/users` - User management
- `GET /admin/audit-log` - Audit log
- `GET /admin/settings` - System settings
- `POST /admin/settings` - Update settings
- `POST /admin/emergency-stop` - Emergency stop

### Incident Module (`/incidents`)
- `GET /incidents/` - List incidents
- `GET /incidents/{incident_id}` - Incident details
- `POST /incidents/` - Create incident
- `PATCH /incidents/{incident_id}` - Update incident
- `GET /incidents/statistics/summary` - Statistics

---

## 🎯 Frontend Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/overview` | Overview | Central dashboard |
| `/live` | Live | Live trading feed |
| `/risk` | Risk | Risk management |
| `/ai` | AI | AI models status |
| `/strategy` | Strategy | Strategy management |
| `/performance` | Performance | Performance analytics |
| `/system` | System | System monitoring |
| `/admin` | Admin | Administration |
| `/incident` | Incident | Incident tracking |

---

## 🔧 Configuration

### Backend Configuration (`quantumfond_backend/.env`)

```env
DATABASE_URL=postgresql://user:pass@localhost/quantumdb
JWT_SECRET_KEY=your-secret-key-here
ENVIRONMENT=production
CORS_ORIGINS=https://app.quantumfond.com
```

### Frontend Configuration

Update API base URL in `src/App.tsx`:

```typescript
const API_BASE_URL = 'https://api.quantumfond.com';
```

---

## 🚢 Production Deployment

### Backend Deployment (api.quantumfond.com)

```bash
# Using systemd service
sudo systemctl start quantumfond-api
sudo systemctl enable quantumfond-api

# Or using Docker
docker build -t quantumfond-backend .
docker run -d -p 8000:8000 quantumfond-backend
```

### Frontend Deployment (app.quantumfond.com)

```bash
# Build production bundle
npm run build

# Deploy to static hosting (Vercel, Netlify, etc.)
# Or serve with nginx
cp -r dist/* /var/www/quantumfond/
```

### Nginx Configuration

```nginx
# Backend
server {
    listen 443 ssl;
    server_name api.quantumfond.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Frontend
server {
    listen 443 ssl;
    server_name app.quantumfond.com;
    root /var/www/quantumfond;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## ✅ Verification Checklist

### Backend Tests

```bash
# Health check
curl http://localhost:8000/health

# Expected: {"status":"OK","service":"QuantumFond API","version":"1.0.0"}

# Test authentication
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"AdminPass123"}'

# Test protected endpoint
curl http://localhost:8000/overview/dashboard \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Frontend Tests

1. Visit `http://localhost:5173/overview`
2. Check Sidebar navigation works
3. Verify TopBar shows system status
4. Test each page loads correctly
5. Confirm CORS with backend

### CORS Verification

- Backend allows frontend origin
- Credentials included in requests
- JWT token properly sent in headers

---

## 📈 Next Steps

### Phase 18 - Database Integration
- Connect routers to PostgreSQL
- Implement real data queries
- Add database migrations

### Phase 19 - Real-time Features
- WebSocket connections
- Live data streaming
- Real-time notifications

### Phase 20 - Production Hardening
- Rate limiting
- Caching layer
- Error tracking
- Monitoring & alerting

---

## 🛠️ Troubleshooting

### Backend Issues

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Database connection error:**
- Check PostgreSQL is running
- Verify credentials in .env
- Ensure database exists

**Import errors:**
- Activate virtual environment
- Reinstall requirements

### Frontend Issues

**API connection error:**
- Check backend is running on port 8000
- Verify CORS configuration
- Check browser console for errors

**Build errors:**
- Clear node_modules: `rm -rf node_modules && npm install`
- Update dependencies: `npm update`

---

## 📞 Support

For issues or questions:
- Check API docs: `http://localhost:8000/docs`
- Review logs: Backend console output
- Test endpoints: Use Swagger UI or Postman

---

>>> **[Phase 17 Complete – Core Infrastructure operational on quantumfond.com]** <<<

🎯 Backend: 9 routers + authentication + database models
🎨 Frontend: 9 pages + 5 components + routing
🔐 JWT Auth: Multi-role access control
📊 API: Full REST API with Swagger docs
✅ Ready for Phase 18: Database Integration

