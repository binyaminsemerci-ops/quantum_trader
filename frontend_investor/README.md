# QuantumFond Investor Portal

**Secure Read-Only Access to Fund Performance**

> Phase 22: Investor Portal & Reporting Layer Operational on investor.quantumfond.com

---

## 🎯 Overview

The QuantumFond Investor Portal provides verified investors with secure, read-only access to:

- **Real-time Performance Metrics** - Sharpe, Sortino, Win Rate, Profit Factor, Max Drawdown
- **Portfolio Positions** - Current active trades with P&L tracking
- **Equity Curve Visualization** - Interactive charts of fund performance
- **Risk Analytics** - VaR, Expected Shortfall, exposure metrics
- **AI Model Insights** - Ensemble model status and weights
- **Downloadable Reports** - JSON, CSV, and PDF export formats

---

## 🏗️ Architecture

### Technology Stack

```
Frontend:  Next.js 14 + React 18 + TypeScript + Tailwind CSS
Charts:    Recharts (equity curve visualization)
Auth:      JWT tokens from auth.quantumfond.com
API:       RESTful integration with api.quantumfond.com
Hosting:   Node.js on Hetzner VPS (PM2 process manager)
Domain:    investor.quantumfond.com
```

### Directory Structure

```
frontend_investor/
├── pages/
│   ├── _app.tsx              # App wrapper with auth guard
│   ├── _document.tsx         # HTML document structure
│   ├── index.tsx             # Dashboard (main page)
│   ├── login.tsx             # Authentication page
│   ├── portfolio.tsx         # Active positions table
│   ├── performance.tsx       # Equity curve chart
│   ├── risk.tsx              # Risk metrics dashboard
│   ├── models.tsx            # AI model insights
│   └── reports.tsx           # Download center
├── components/
│   ├── InvestorNavbar.tsx    # Top navigation bar
│   ├── MetricCard.tsx        # KPI display cards
│   ├── EquityChart.tsx       # Recharts line chart
│   ├── ReportCard.tsx        # Download buttons
│   └── LoadingSpinner.tsx    # Loading state
├── hooks/
│   └── useAuth.ts            # Authentication hook
├── styles/
│   └── globals.css           # Global styles + Tailwind
├── public/                   # Static assets
├── package.json              # Dependencies
├── next.config.js            # Next.js configuration
├── tailwind.config.js        # Tailwind customization
└── tsconfig.json             # TypeScript config
```

---

## 🔐 Authentication Flow

### Login Process

1. User enters credentials on `/login`
2. POST request to `https://auth.quantumfond.com/login`
3. JWT token stored in localStorage
4. Token included in all API requests via Authorization header
5. Auto-redirect to `/` on successful login

### Protected Routes

All pages except `/login` require authentication:

```typescript
// useAuth hook provides:
- user: Current user object (username, role)
- login(username, password): Login function
- logout(): Clear session and redirect
- getToken(): Retrieve JWT for API calls
- isAuthenticated: Boolean auth status
```

---

## 📊 Pages & Features

### 1️⃣ Dashboard (`/`)

**Key Metrics Grid:**
- Total Return (currency format)
- Win Rate (percentage)
- Profit Factor (ratio)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk)
- Max Drawdown (peak-to-trough decline)

**Quick Info Cards:**
- AI Engine Status (models, mode, health)
- Risk Overview (level, governor, exposure)

**API Endpoint:** `GET /performance/metrics`

---

### 2️⃣ Portfolio (`/portfolio`)

**Active Positions Table:**
- Symbol, Direction (BUY/SELL)
- Entry Price, Current Price
- P&L (profit/loss)
- Take Profit / Stop Loss levels
- Confidence score

**API Endpoint:** `GET /trades/open`

---

### 3️⃣ Performance (`/performance`)

**Equity Curve Visualization:**
- Interactive Recharts line chart
- Hover tooltips with timestamps
- Responsive design (mobile-friendly)
- Real-time data updates

**API Endpoint:** `GET /performance/metrics` (curve field)

---

### 4️⃣ Risk (`/risk`)

**Risk Metrics Dashboard:**
- Portfolio Exposure (% of capital)
- VaR (95% Value at Risk)
- Expected Shortfall
- Current Drawdown
- Governor State (ESS status)
- Risk Level (LOW/MODERATE/HIGH)

**Educational Section:**
- Metric explanations for investors

**API Endpoint:** `GET /risk/summary`

---

### 5️⃣ AI Models (`/models`)

**Ensemble Overview:**
- Active model count
- Total ensemble weight
- Average latency

**Model Table:**
- Model name and status
- Weight distribution (visual bar)
- Error rate
- Inference latency

**Architecture Info:**
- Ensemble strategy explanation
- Continuous learning details

**API Endpoint:** `GET /ai/models`

---

### 6️⃣ Reports (`/reports`)

**Download Center:**
- **JSON Export** - Full data with precision
- **CSV Export** - Excel-compatible format
- **PDF Report** - Formatted professional report

**Features:**
- One-click downloads
- Automatic filename generation
- Token-authenticated requests

**API Endpoints:**
- `GET /reports/export/json`
- `GET /reports/export/csv`
- `GET /reports/export/pdf`

---

## 🎨 Design System

### Color Palette

```css
--quantum-bg:      #0a0a0f  /* Background */
--quantum-dark:    #111118  /* Dark sections */
--quantum-card:    #1a1a24  /* Card backgrounds */
--quantum-border:  #2a2a38  /* Borders */
--quantum-text:    #e5e7eb  /* Primary text */
--quantum-muted:   #9ca3af  /* Secondary text */
--quantum-accent:  #22c55e  /* Brand green */
--quantum-danger:  #ef4444  /* Red alerts */
```

### Responsive Breakpoints

```
Mobile:  < 768px
Tablet:  768px - 1024px
Desktop: > 1024px
```

---

## 🚀 Deployment

### Prerequisites

- Node.js 18+ installed
- Access to VPS (46.224.116.254)
- SSH key configured (~/.ssh/hetzner_fresh)
- Domain DNS configured (investor.quantumfond.com → VPS IP)

### Local Development

```bash
cd frontend_investor
npm install
npm run dev
# Opens on http://localhost:3001
```

### Production Deployment

**Option 1: Bash Script (Linux/WSL)**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Option 2: PowerShell Script (Windows)**
```powershell
.\deploy.ps1
```

**Manual Deployment:**
```bash
# 1. Build
npm run build

# 2. Upload to VPS
tar -czf investor_build.tar.gz .next package.json
scp -i ~/.ssh/hetzner_fresh investor_build.tar.gz root@46.224.116.254:/home/qt/quantum_trader/frontend_investor/

# 3. Start on VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
cd /home/qt/quantum_trader/frontend_investor
tar -xzf investor_build.tar.gz
npm install --production
pm2 start npm --name "quantumfond-investor" -- start
pm2 save

# 4. Configure Nginx (see deploy.sh for config)
```

---

## 🔧 Configuration

### Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=https://api.quantumfond.com
NEXT_PUBLIC_AUTH_URL=https://auth.quantumfond.com
NEXT_PUBLIC_WS_URL=wss://api.quantumfond.com/ws
```

### Backend CORS Configuration

**In `backend/main.py`:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://investor.quantumfond.com",
        "http://localhost:3001"  # Development
    ],
    allow_credentials=True,
    allow_methods=["GET"],  # Read-only
    allow_headers=["*"],
)
```

---

## 🧪 Testing

### Manual Testing Checklist

```
✅ Login page loads and accepts credentials
✅ Dashboard displays metrics from API
✅ Portfolio table shows open positions
✅ Performance chart renders equity curve
✅ Risk page displays VaR and ESS status
✅ AI Models page shows ensemble data
✅ Reports download in all formats (JSON/CSV/PDF)
✅ Navigation between pages works
✅ Logout clears session and redirects
✅ Mobile responsive design works
✅ HTTPS certificate valid
✅ API authentication tokens working
```

### Automated Tests (Future Enhancement)

```bash
# Install testing dependencies
npm install --save-dev @testing-library/react jest

# Run tests
npm test
```

---

## 📈 Performance Optimization

### Next.js Optimizations

- **Static Site Generation (SSG)** for login page
- **Incremental Static Regeneration (ISR)** for dashboard
- **Code Splitting** - Automatic by Next.js
- **Image Optimization** - Next.js Image component

### Best Practices

```typescript
// Use React hooks efficiently
const [data, setData] = useState<Type>({});

// Memoize expensive calculations
const metric = useMemo(() => calculateMetric(data), [data]);

// Debounce API calls
const debouncedFetch = useCallback(debounce(fetchData, 500), []);
```

---

## 🔒 Security

### Authentication

- **JWT tokens** stored in localStorage (client-side)
- **Bearer token** in Authorization header
- **Token expiration** handled by backend
- **Auto-logout** on 401 responses

### Best Practices

```typescript
// Never expose sensitive data in client code
// Always validate responses from API
// Use HTTPS for all production traffic
// Implement rate limiting on backend
// Log all authentication attempts
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CORS Errors**
```
Solution: Verify backend CORS allows investor.quantumfond.com
Check: backend/main.py CORSMiddleware configuration
```

**2. Authentication Failures**
```
Solution: Check auth.quantumfond.com endpoint is accessible
Verify: JWT token format and expiration
```

**3. API Connection Refused**
```
Solution: Ensure api.quantumfond.com backend is running
Check: Docker container status on VPS
```

**4. Build Errors**
```
Solution: Clear .next directory and rebuild
Commands: rm -rf .next && npm run build
```

---

## 📚 API Integration

### Endpoint Summary

| Page | Endpoint | Method | Auth Required |
|------|----------|--------|---------------|
| Dashboard | `/performance/metrics` | GET | ✅ |
| Portfolio | `/trades/open` | GET | ✅ |
| Performance | `/performance/metrics` | GET | ✅ |
| Risk | `/risk/summary` | GET | ✅ |
| Models | `/ai/models` | GET | ✅ |
| Reports | `/reports/export/{format}` | GET | ✅ |
| Login | `/login` (auth domain) | POST | ❌ |

### Response Formats

**Performance Metrics:**
```json
{
  "metrics": {
    "total_return": 12345.67,
    "winrate": 0.68,
    "profit_factor": 2.3,
    "sharpe": 1.8,
    "sortino": 2.1,
    "max_drawdown": -0.12
  },
  "curve": [
    {"timestamp": "2025-01-01T00:00:00Z", "equity": 10000},
    {"timestamp": "2025-01-02T00:00:00Z", "equity": 10150}
  ]
}
```

---

## 🛠️ Development

### Adding New Pages

```typescript
// 1. Create page file
// pages/newpage.tsx
import InvestorNavbar from '@/components/InvestorNavbar';

export default function NewPage() {
  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      <div className="max-w-7xl mx-auto px-4 py-8">
        <h1>New Page</h1>
      </div>
    </div>
  );
}

// 2. Add to navigation
// components/InvestorNavbar.tsx
const navItems = [
  // ... existing items
  { icon: '🆕', label: 'New Page', path: '/newpage' },
];
```

### Creating Components

```typescript
// components/CustomComponent.tsx
interface CustomComponentProps {
  title: string;
  value: number;
}

export default function CustomComponent({ title, value }: CustomComponentProps) {
  return (
    <div className="bg-quantum-card border border-quantum-border rounded-lg p-4">
      <h3>{title}</h3>
      <p>{value}</p>
    </div>
  );
}
```

---

## 📞 Support

**Technical Issues:**
- Email: tech@quantumfond.com
- GitHub Issues: binyaminsemerci-ops/quantum_trader

**Investor Relations:**
- Email: investors@quantumfond.com
- Portal: investor.quantumfond.com/support

---

## 📝 License

© 2025 QuantumFond. All rights reserved.

**Proprietary Software** - Internal Use Only

---

## ✅ Verification Checklist

```
✅ Next.js application structure created
✅ Authentication system implemented
✅ All 6 pages built and functional
✅ Shared components created
✅ Tailwind CSS styling applied
✅ API integration with auth tokens
✅ Recharts visualization configured
✅ Deployment scripts ready
✅ Documentation complete
✅ Responsive design implemented
✅ Security best practices applied
✅ Error handling implemented
```

---

>>> **[Phase 22 Complete – Investor Portal & Reporting Layer Operational on investor.quantumfond.com]**

