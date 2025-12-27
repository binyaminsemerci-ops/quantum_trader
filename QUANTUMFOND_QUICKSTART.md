# Quick Start Guide - QuantumFond Hedge Fund OS

## 🚀 5-Minute Setup

### Backend

```bash
# 1. Navigate to backend
cd quantumfond_backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run backend
uvicorn main:app --reload --port 8000
```

✅ **Backend running at:** http://localhost:8000  
📚 **API Docs at:** http://localhost:8000/docs

---

### Frontend

```bash
# 1. Navigate to frontend
cd quantumfond_frontend

# 2. Install dependencies
npm install

# 3. Run frontend
npm run dev
```

✅ **Frontend running at:** http://localhost:5173

---

## 🔐 Test Login

**Visit:** http://localhost:5173/overview

**Test Credentials:**
- Username: `admin`
- Password: `AdminPass123`

---

## 📊 Test API

```bash
# Health check
curl http://localhost:8000/health

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"AdminPass123"}'
```

---

## ✅ Verification

- [ ] Backend health endpoint returns OK
- [ ] Frontend loads at localhost:5173
- [ ] Sidebar navigation works
- [ ] TopBar shows system status
- [ ] API endpoints return data
- [ ] CORS allows frontend→backend

---

>>> **[Phase 17 Complete – Ready for deployment]** <<<
