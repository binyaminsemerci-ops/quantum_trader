# QuantumFond Hedge Fund OS - Architecture Document

## System Architecture

### Technology Stack

**Backend:**
- FastAPI (Python 3.11+)
- SQLAlchemy ORM
- PostgreSQL Database
- JWT Authentication
- Redis (planned for caching)

**Frontend:**
- React 18 with TypeScript
- Vite (Build tool)
- Tailwind CSS
- React Router v6
- Chart.js for visualizations

---

## Database Schema

### Core Tables

```sql
-- Users & Authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Audit Logging
CREATE TABLE control_log (
    id SERIAL PRIMARY KEY,
    user VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL,
    action VARCHAR(200) NOT NULL,
    status VARCHAR(50) NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading Records
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    pnl FLOAT,
    strategy VARCHAR(100),
    status VARCHAR(20) DEFAULT 'open',
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    metadata JSONB
);

-- Risk Metrics
CREATE TABLE risk_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    threshold FLOAT,
    status VARCHAR(20) DEFAULT 'normal',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI Models
CREATE TABLE ai_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    status VARCHAR(20) DEFAULT 'active',
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- System Events
CREATE TABLE system_events (
    id SERIAL PRIMARY KEY,
    event VARCHAR(200) NOT NULL,
    cpu FLOAT,
    ram FLOAT,
    disk FLOAT,
    severity VARCHAR(20) DEFAULT 'info',
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Incidents
CREATE TABLE incidents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    assigned_to VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    metadata JSONB
);
```

---

## API Architecture

### Module Organization

```
/auth          - Authentication & JWT
/overview      - Dashboard & metrics
/trades        - Trading operations
/risk          - Risk management
/ai            - AI models & predictions
/strategy      - Strategy management
/performance   - Performance analytics
/system        - System monitoring
/admin         - Administration
/incidents     - Incident tracking
```

### Authentication Flow

```
1. User submits credentials → /auth/login
2. Backend validates → checks static users (will be DB)
3. Generate JWT with role claims
4. Return token + role to frontend
5. Frontend stores token
6. All subsequent requests include: Authorization: Bearer <token>
7. Backend validates token on protected routes
8. Role-based access control enforced
```

---

## Security Model

### Role-Based Access Control (RBAC)

**Admin:**
- Full system access
- User management
- System configuration
- Emergency controls

**Risk:**
- Risk management features
- Circuit breakers
- Emergency stop
- Risk analytics

**Trader:**
- Trading operations
- View positions
- Execute trades
- View strategies

**Viewer:**
- Read-only access
- View dashboards
- View reports
- No modifications

### Security Features

- JWT token-based authentication
- Role-based endpoint protection
- CORS configuration
- Password hashing (planned)
- Audit logging
- Rate limiting (planned)

---

## Frontend Architecture

### Component Hierarchy

```
App
├── Router
│   ├── Sidebar (navigation)
│   ├── TopBar (status)
│   └── Routes
│       ├── Overview
│       ├── Live
│       ├── Risk
│       ├── AI
│       ├── Strategy
│       ├── Performance
│       ├── System
│       ├── Admin
│       └── Incident
```

### State Management

Currently using React hooks:
- `useState` for component state
- `useEffect` for data fetching
- Custom `useFetch` hook

Future: Consider Redux or Zustand for global state

### Styling Architecture

- Tailwind CSS utility-first
- Dark theme default
- Responsive grid system
- Consistent color palette:
  - Green: Success, positive metrics
  - Red: Errors, negative metrics
  - Yellow: Warnings
  - Blue: Information
  - Gray: Base UI

---

## Deployment Architecture

### Development

```
localhost:8000  → Backend API
localhost:5173  → Frontend Dev Server
localhost:5432  → PostgreSQL
localhost:6379  → Redis (planned)
```

### Production

```
api.quantumfond.com   → Backend API (FastAPI + Uvicorn)
app.quantumfond.com   → Frontend (Static hosting)
db.internal           → PostgreSQL (private network)
cache.internal        → Redis (private network)
```

### Infrastructure Components

- **Load Balancer:** Nginx/Caddy
- **Backend:** Uvicorn + FastAPI
- **Database:** PostgreSQL 15+
- **Cache:** Redis 7+
- **Monitoring:** Prometheus + Grafana (planned)
- **Logging:** ELK Stack (planned)

---

## Data Flow

### Real-time Trading Flow

```
1. Market Data → Backend ingestion
2. AI Models → Generate predictions
3. Strategy Engine → Decision making
4. Execution Engine → Place orders
5. Position Tracking → Update trades table
6. Risk Monitoring → Check limits
7. Frontend Updates → WebSocket (planned)
```

### Dashboard Flow

```
1. Frontend requests → /overview/dashboard
2. Backend aggregates → Multiple data sources
3. Response assembled → JSON payload
4. Frontend receives → Update state
5. Components render → Display metrics
```

---

## Scalability Considerations

### Horizontal Scaling

- Stateless API design
- JWT tokens (no server sessions)
- Database connection pooling
- Redis for distributed caching

### Performance Optimization

- Database indexing on key fields
- API response caching
- Frontend code splitting
- Lazy loading for routes
- Pagination for large datasets

### Monitoring & Observability

- Health check endpoints
- Metrics collection
- Error tracking
- Performance monitoring
- Audit logging

---

## Future Enhancements

### Phase 18 - Database Integration
- Real database connections
- Data persistence
- Migration system

### Phase 19 - Real-time Features
- WebSocket connections
- Live data streaming
- Push notifications

### Phase 20 - Advanced Features
- Advanced charting
- Custom dashboards
- Report generation
- Export functionality

### Phase 21 - ML Integration
- Connect to actual AI models
- Real-time predictions
- Model retraining pipeline

---

## Development Workflow

### Backend Development

```bash
1. Create new router in /routers
2. Define Pydantic models
3. Implement endpoints
4. Add to main.py
5. Test with /docs (Swagger)
6. Write tests
```

### Frontend Development

```bash
1. Create page component in /pages
2. Add route in App.tsx
3. Implement API calls with useFetch
4. Style with Tailwind
5. Test navigation
6. Verify responsiveness
```

---

>>> **System designed for enterprise-grade hedge fund operations** <<<
