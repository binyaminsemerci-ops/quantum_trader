# QuantumFond Investor Portal - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
cd C:\quantum_trader\frontend_investor
npm install
```

### Step 2: Run Development Server
```bash
npm run dev
```
**Opens at:** http://localhost:3001

### Step 3: Test Login
**Default credentials (for testing):**
- Username: `investor`
- Password: `demo123`

*(Change these in production)*

---

## 📁 Project Structure

```
frontend_investor/
├── pages/           → All pages (6 investor pages + login)
├── components/      → Reusable UI components
├── hooks/          → useAuth authentication hook
└── styles/         → Global CSS + Tailwind
```

---

## 🔧 Common Tasks

### Build for Production
```bash
npm run build
npm start
```

### Deploy to VPS
```powershell
# Windows (PowerShell)
.\deploy.ps1

# Linux/WSL (Bash)
./deploy.sh
```

### View Logs
```bash
pm2 logs quantumfond-investor
```

---

## 🌐 Pages Overview

| Route | Description |
|-------|-------------|
| `/login` | JWT authentication |
| `/` | Dashboard with KPI metrics |
| `/portfolio` | Active trading positions |
| `/performance` | Equity curve chart |
| `/risk` | Risk metrics (VaR, ES) |
| `/models` | AI model ensemble |
| `/reports` | Download JSON/CSV/PDF |

---

## 🔐 Authentication

**Login Flow:**
1. User enters credentials at `/login`
2. POST to `https://auth.quantumfond.com/login`
3. JWT token stored in localStorage
4. Token sent with all API requests
5. Auto-logout on 401 responses

**useAuth Hook:**
```typescript
const { user, login, logout, getToken } = useAuth();
```

---

## 🎨 Customization

### Change Theme Colors
Edit `tailwind.config.js`:
```javascript
quantum: {
  accent: '#22c55e',  // Brand green
  bg: '#0a0a0f',      // Background
}
```

### Add New Page
1. Create `pages/newpage.tsx`
2. Add route to `components/InvestorNavbar.tsx`
3. Fetch data from API with useAuth token

---

## 🐛 Troubleshooting

**CORS Error?**
→ Check backend allows `investor.quantumfond.com`

**Login Fails?**
→ Verify `auth.quantumfond.com` is accessible

**Build Error?**
→ Delete `.next` folder and rebuild

---

## 📚 Full Documentation

See [README.md](README.md) for complete documentation.

---

>>> **Ready to deploy!** Run `npm run build` then `.\deploy.ps1`
