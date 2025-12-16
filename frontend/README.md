# Quantum Trader Dashboard - Frontend

Modern Next.js + React + Tailwind CSS dashboard for real-time trading monitoring.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend API running on http://localhost:8000

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

### Production Build

```bash
npm run build
npm start
```

## 📁 Project Structure

```
frontend/
├── components/           # React components
│   ├── Sidebar.tsx      # Navigation sidebar
│   ├── TopBar.tsx       # Top status bar
│   ├── PortfolioPanel.tsx
│   ├── PositionsPanel.tsx
│   ├── SignalsPanel.tsx
│   ├── RiskPanel.tsx
│   └── SystemHealthPanel.tsx
├── lib/                 # Utilities
│   ├── types.ts        # TypeScript types
│   ├── api.ts          # REST API client
│   ├── websocket.ts    # WebSocket client
│   ├── store.ts        # Zustand state management
│   └── utils.ts        # Helper functions
├── pages/              # Next.js pages
│   ├── _app.tsx        # App wrapper
│   └── index.tsx       # Main dashboard
├── styles/
│   └── globals.css     # Global styles + Tailwind
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.js
```

## 🎨 Features

### Real-time Updates
- WebSocket connection to backend (`ws://localhost:8000/ws/dashboard`)
- Auto-reconnect with exponential backoff
- Ping/pong heartbeat (every 30s)
- Event handlers for 7 event types:
  - position_updated
  - pnl_updated
  - signal_generated
  - ess_state_changed
  - health_alert
  - trade_executed
  - order_placed

### Dashboard Panels
- **Portfolio**: Equity, PnL, margin, position count
- **Positions**: Open positions table with live PnL
- **Signals**: Recent AI signals (ensemble/meta/RL)
- **Risk**: ESS state, drawdown, exposure, risk limits
- **System Health**: Microservices status, alerts

### State Management
- Zustand store for global state
- Optimistic updates from WebSocket events
- Automatic snapshot refresh on mount

## 🔧 Configuration

### Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### API Proxy

`next.config.js` proxies `/api/*` to backend:

```javascript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:8000/api/:path*',
    },
  ]
}
```

## 🎯 Usage

### Load Initial Snapshot

```typescript
import { fetchDashboardSnapshot } from '@/lib/api';

const snapshot = await fetchDashboardSnapshot();
```

### Connect to WebSocket

```typescript
import { dashboardWebSocket } from '@/lib/websocket';

// Subscribe to events
const unsubscribe = dashboardWebSocket.subscribe((event) => {
  console.log('Event:', event.type, event.payload);
});

// Connect
dashboardWebSocket.connect();

// Cleanup
unsubscribe();
dashboardWebSocket.disconnect();
```

### Access Global State

```typescript
import { useDashboardStore } from '@/lib/store';

function MyComponent() {
  const { snapshot, loading, error } = useDashboardStore();
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return <div>Equity: ${snapshot.portfolio.equity}</div>;
}
```

## 🧪 Testing

```bash
npm run type-check  # TypeScript validation
npm run lint        # ESLint
```

## 📊 Performance

- Initial load: Single REST call (~500ms-2s)
- Real-time updates: WebSocket push (~100ms latency)
- No polling overhead
- Optimized re-renders with Zustand

## 🎨 Styling

Uses Tailwind CSS with custom theme:

```javascript
// tailwind.config.js
theme: {
  extend: {
    colors: {
      success: '#10b981',  // Green
      danger: '#ef4444',   // Red
      warning: '#f59e0b',  // Orange
      primary: '#3b82f6',  // Blue
    },
  },
}
```

## 🐛 Troubleshooting

### WebSocket won't connect
- Check backend is running: `curl http://localhost:8000/api/dashboard/health`
- Check WebSocket endpoint: `websocat ws://localhost:8000/ws/dashboard`
- Check browser console for errors

### Snapshot fails to load
- Verify all microservices are running
- Check backend logs: `docker-compose logs -f`
- Test endpoint: `curl http://localhost:8000/api/dashboard/snapshot`

### Styles not loading
- Clear Next.js cache: `rm -rf .next`
- Rebuild: `npm run build`

## 📝 TODO

- [ ] Add PnL chart (Recharts)
- [ ] Add order history panel
- [ ] Add trade execution panel
- [ ] Add settings page
- [ ] Add authentication
- [ ] Add dark mode toggle
- [ ] Add responsive mobile layout
- [ ] Add error boundary
- [ ] Add loading skeletons

## 🚀 Deployment

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Docker Compose

```yaml
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
      - NEXT_PUBLIC_WS_URL=ws://backend:8000
```

---

**Sprint 4 - Part 4 & 5 Complete** ✅
