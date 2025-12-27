# 🧪 PHASE 23.0 – TESTING QUICK REFERENCE

**Status:** ✅ OPERATIONAL  
**Purpose:** Catch numeric rendering errors before production

---

## ⚡ QUICK COMMANDS

```bash
# Unit Tests
npm run test              # Watch mode
npm run test:unit         # Run once
npm run test:coverage     # With coverage

# E2E Tests
npm run cypress           # Interactive UI
npm run test:e2e          # Headless

# All Tests
npm run test:all          # Unit + E2E

# Type Check
npm run type-check        # Verify TypeScript
```

---

## 🛡️ SAFE FORMATTERS

```typescript
import { safeNum, safePercent, safeCurrency } from '@/lib/formatters';

// Replace unsafe code:
value.toFixed(2)              → safeNum(value, 2)
`${value.toFixed(2)}%`        → safePercent(value)
`$${value.toFixed(2)}`        → safeCurrency(value)
Math.round(value).toString()  → safeInt(value)
```

---

## ✅ WHAT'S TESTED

**Unit Tests (30):**
- ✅ Undefined/null handling
- ✅ NaN/Infinity handling
- ✅ Valid number formatting
- ✅ Currency & percentage
- ✅ Integer formatting
- ✅ Safe parsing

**E2E Tests (8):**
- ✅ AI page loads without errors
- ✅ No console .toFixed errors
- ✅ No NaN rendering
- ✅ API error handling
- ✅ Missing data handling
- ✅ Extreme values
- ✅ Navigation stability
- ✅ Dashboard metrics

---

## 🚨 ERROR PREVENTION

**Before (Crashes):**
```javascript
{metrics.confidence.toFixed(2)}%  // ❌ TypeError
```

**After (Safe):**
```javascript
{safePercent(metrics.confidence)}  // ✅ "0.00%"
```

---

## 🤖 CI/CD PIPELINE

**Triggers:**
- Push to `main`
- Pull requests

**Steps:**
1. Type check (TypeScript)
2. Unit tests (Vitest)
3. Build (Next.js)
4. E2E tests (Cypress)
5. Lint (ESLint)
6. **Block if any fail** ❌

**Location:** `.github/workflows/test.yml`

---

## 📊 TEST STATUS

```bash
# Check test results
npm run test:unit -- --run

# Expected output:
✓ __tests__/formatters.test.ts (30 tests)
  Test Files  1 passed (1)
       Tests  30 passed (30)
```

---

## 🔍 FIND & REPLACE

```bash
# Find all unsafe .toFixed() usage:
grep -r "\.toFixed(" frontend/

# Replace pattern:
OLD: value?.toFixed(2)
NEW: safeNum(value, 2)

OLD: `${value.toFixed(2)}%`
NEW: safePercent(value)

OLD: `$${value.toFixed(2)}`
NEW: safeCurrency(value)
```

---

## 📁 FILES CREATED

```
✅ frontend/lib/formatters.ts              # Safe functions
✅ frontend/__tests__/formatters.test.ts   # 30 unit tests
✅ frontend/vitest.config.ts               # Vitest config
✅ frontend/vitest.setup.ts                # Test setup
✅ frontend/cypress.config.ts              # Cypress config
✅ frontend/cypress/e2e/ai_page.cy.ts      # 8 E2E tests
✅ .github/workflows/test.yml              # CI pipeline
```

---

## 🎯 INTEGRATION CHECKLIST

- [ ] Replace all `.toFixed()` with safe formatters
- [ ] Import formatters in components
- [ ] Run `npm run test:all`
- [ ] Commit and push
- [ ] Check GitHub Actions
- [ ] Monitor CI results

---

## 🚀 NEXT PHASE

**Phase 23.1:** Component integration
- Update all existing components
- Add component-specific tests
- Expand E2E coverage

---

>>> [Phase 23.0 Complete – Automated Numeric Guardrail & CI Stability Layer Operational ✅]
