# Dashboard V3.0 Test Suite - Quick Start Guide

## 🚀 Quick Test Execution

### 1. Quick Smoke Test (2 minutes)
**Purpose**: Verify basic functionality
```powershell
cd c:\quantum_trader
python tests\run_dashboard_v3_tests.py --quick
```

**What it tests**:
- API overview endpoint responds
- Portfolio integration works
- Basic schema validation

---

### 2. Backend API Tests (5 minutes)
**Purpose**: Validate all API contracts
```powershell
python tests\run_dashboard_v3_tests.py --api
```

**What it tests** (168 tests):
- `/api/dashboard/overview` - 32 tests
- `/api/dashboard/trading` - 25 tests
- `/api/dashboard/risk` - 28 tests
- `/api/dashboard/system` - 30 tests
- `/ws/dashboard` - 18 tests
- Numeric safety - 20 tests
- Logging validation - 15 tests

---

### 3. Integration Tests (3 minutes)
**Purpose**: Validate service-to-service communication
```powershell
python tests\run_dashboard_v3_tests.py --integration
```

**What it tests** (38 tests):
- Portfolio → Dashboard data flow
- Risk → Dashboard integration
- ESS status propagation
- Binance testnet mock

**Prerequisites**:
- Backend must be running: `docker-compose up backend -d`
- Portfolio service: `http://localhost:8004`

---

### 4. Frontend Tests (4 minutes)
**Purpose**: Validate React components
```powershell
python tests\run_dashboard_v3_tests.py --frontend
```

**What it tests** (69 tests):
- OverviewTab rendering - 15 tests
- TradingTab position table - 18 tests
- RiskTab metrics - 16 tests
- SystemTab health - 20 tests

**Prerequisites**:
```powershell
cd frontend
npm install --save-dev @testing-library/react @testing-library/jest-dom jest jest-environment-jsdom
# See frontend/__tests__/README_TEST_SETUP.md for full setup
```

---

### 5. E2E Tests (5 minutes)
**Purpose**: Validate complete user flows
```powershell
python tests\run_dashboard_v3_tests.py --e2e
```

**What it tests** (6 tests):
- Dashboard loads successfully
- Tab navigation works
- Position data displays
- No NaN values in UI
- Real-time updates

**Prerequisites**:
```powershell
pip install pytest-playwright
playwright install
# Frontend must be running: npm run dev
```

---

### 6. Full Test Suite (20 minutes)
**Purpose**: Complete validation before deployment
```powershell
python tests\run_dashboard_v3_tests.py
```

**Total tests**: 281
- Backend API: 168
- Integration: 38
- Frontend: 69
- E2E: 6

---

## 📋 Manual Testing Checklist

After automated tests pass, complete manual validation:

```powershell
# Open checklist
code tests\MANUAL_TEST_CHECKLIST.md
```

**Manual checks** (100+ items):
1. Pre-test setup verification
2. Overview tab visual validation
3. Trading tab position accuracy
4. Risk tab ESS status
5. System tab service health
6. Real-time update behavior
7. Data consistency cross-checks
8. Error handling
9. Responsive design
10. Performance metrics

---

## ✅ Pre-Deployment Checklist

### Required Services
```powershell
# 1. Start backend
docker-compose up backend -d

# 2. Check backend health
curl http://localhost:8000/health

# 3. Check Portfolio service
curl http://localhost:8004/health

# 4. Start frontend (for E2E)
cd frontend
npm run dev
```

### Test Execution Order
1. ✅ Quick smoke test: `--quick`
2. ✅ API tests: `--api`
3. ✅ Integration tests: `--integration` (requires backend)
4. ✅ Frontend tests: `--frontend` (after npm install)
5. ✅ E2E tests: `--e2e` (requires frontend running)
6. ✅ Manual checklist validation

### Success Criteria
- [ ] All automated tests pass (281/281)
- [ ] No NaN values anywhere in UI
- [ ] Manual checklist 100% complete
- [ ] Performance < 3s dashboard load
- [ ] WebSocket connects successfully
- [ ] QA sign-off obtained
- [ ] Tech Lead approval

---

## 🐛 Troubleshooting

### Test Failures

**API tests fail**:
```powershell
# Check backend is running
curl http://localhost:8000/health

# Check logs
docker-compose logs backend | Select-Object -Last 50
```

**Integration tests fail**:
```powershell
# Verify Portfolio service
curl http://localhost:8004/health

# Check service connectivity
docker-compose ps
```

**Frontend tests fail**:
```powershell
# Verify Jest setup
cd frontend
npm test -- --listTests

# Check configuration
cat jest.config.js
cat jest.setup.js
```

**E2E tests fail**:
```powershell
# Verify frontend running
curl http://localhost:3000

# Check Playwright installation
playwright --version

# Run with headed browser (for debugging)
pytest tests/e2e/test_dashboard_v3_e2e.py --headed
```

---

## 📊 Test Results Interpretation

### Expected Output (Success)
```
✓ ALL TESTS PASSED (281/281)
✓ Dashboard V3.0 is production ready!

Next Steps:
  1. Review manual checklist
  2. Deploy to staging
  3. Run E2E against staging
  4. Get QA sign-off
```

### Expected Output (Failure)
```
✗ SOME TESTS FAILED
✗ 5 test(s) failed across 2 suite(s)

Troubleshooting:
  • Review test output above
  • Check services running
  • Verify dependencies installed
```

---

## 📖 Additional Documentation

- **Full Test Suite Overview**: `tests/DASHBOARD_V3_TEST_SUITE_README.md`
- **Manual Testing Guide**: `tests/MANUAL_TEST_CHECKLIST.md`
- **Frontend Test Setup**: `frontend/__tests__/README_TEST_SETUP.md`
- **Test Files**:
  - API: `tests/api/test_dashboard_*.py`
  - Integration: `tests/integrations/dashboard/test_*.py`
  - Frontend: `frontend/__tests__/*.test.tsx`
  - E2E: `tests/e2e/test_dashboard_v3_e2e.py`

---

## 🎯 Quick Reference Commands

```powershell
# Smoke test (fastest)
python tests\run_dashboard_v3_tests.py --quick

# API only
python tests\run_dashboard_v3_tests.py --api

# Integration only  
python tests\run_dashboard_v3_tests.py --integration

# Frontend only
python tests\run_dashboard_v3_tests.py --frontend

# E2E only
python tests\run_dashboard_v3_tests.py --e2e

# Everything
python tests\run_dashboard_v3_tests.py

# Individual test file
pytest tests/api/test_dashboard_overview.py -v

# Specific test
pytest tests/api/test_dashboard_overview.py::test_overview_schema_valid -v

# Frontend specific component
cd frontend
npm test -- OverviewTab.test.tsx

# E2E with browser visible (debugging)
pytest tests/e2e/test_dashboard_v3_e2e.py --headed --slowmo 1000
```

---

**Last Updated**: December 5, 2025  
**Test Suite Version**: 1.0.0  
**Dashboard Version**: 3.0.0
