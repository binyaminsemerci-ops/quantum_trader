# Quantum Trader - Frontend Setup

## 🎨 Frontend Architecture

**Framework:** React 18 + Vite 5 + TypeScript  
**UI Library:** Tailwind CSS + Lucide Icons  
**Charts:** Recharts  
**Port:** http://localhost:3000 (mapped from container port 5173)

---

## 🚀 Quick Start

Frontend starts automatically when you run:
```bash
.\start_quantum_trader.ps1
```
or
```bash
start_quantum_trader.bat
```

---

## 🐳 Docker Configuration

The frontend runs in a Node.js 20 Alpine container:

```yaml
frontend:
  image: node:20-alpine
  container_name: quantum_frontend
  profiles: ["dev"]
  command: sh -c "npm install && npm run dev -- --host 0.0.0.0"
  ports:
    - "3000:5173"
  volumes:
    - ./qt-agent-ui:/app
  environment:
    - VITE_API_URL=http://localhost:8000
```

---

## 🔧 Manual Development

If you want to run frontend outside Docker:

```bash
cd qt-agent-ui
npm install
npm run dev
```

Then access: http://localhost:5173

---

## 📊 API Proxy Configuration

The frontend uses Vite's proxy for backend API calls:

```typescript
// vite.config.ts
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://quantum_backend:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

**Usage in React:**
```typescript
// Call backend API
fetch('/api/health')
// Or use full URL
fetch('http://localhost:8000/health')
```

---

## 🛠️ Troubleshooting

### Frontend not loading
```bash
# Check container status
docker ps | grep quantum_frontend

# Check logs
docker logs quantum_frontend

# Restart frontend only
docker restart quantum_frontend
```

### Port 3000 already in use
Stop the conflicting service or change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "3001:5173"  # Change 3000 to 3001
```

### Build errors
```bash
# Clear node_modules and reinstall
cd qt-agent-ui
rm -rf node_modules
docker-compose --profile dev up -d --build frontend
```

---

## 📦 Project Structure

```
qt-agent-ui/
├── src/
│   ├── components/      # React components
│   ├── pages/          # Page components
│   ├── hooks/          # Custom React hooks
│   ├── utils/          # Utility functions
│   └── App.tsx         # Main app component
├── index.html          # HTML entry point
├── vite.config.ts      # Vite configuration
├── tailwind.config.ts  # Tailwind configuration
└── package.json        # Dependencies
```

---

## 🎯 Features

- ✅ Real-time trading dashboard
- ✅ Live position monitoring
- ✅ P&L visualization with charts
- ✅ AI confidence indicators
- ✅ System health monitoring
- ✅ Trading profile configuration
- ✅ Responsive design (mobile-friendly)

---

## 🔗 Useful URLs

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 📝 Next Steps

1. **Access UI:** Open http://localhost:3000 in your browser
2. **View Dashboard:** Monitor live trading activity
3. **Check API:** Backend health at http://localhost:8000/health
4. **Customize:** Edit components in `qt-agent-ui/src/`
5. **Build for Prod:** Run `npm run build` for production bundle

---

## 🚨 Important Notes

- Frontend auto-installs dependencies on first run (may take 1-2 minutes)
- Hot reload is enabled - changes auto-refresh
- Container uses Node.js 20 Alpine (lightweight)
- Port mapping: Container 5173 → Host 3000
- Backend must be running for API calls to work
