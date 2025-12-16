# Quantum Trader Test Suite

Comprehensive test suite for Quantum Trader covering unit, integration, and scenario tests.

## Structure

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_p0_patches.py # Core foundation tests
│   └── test_p1_patches.py # Production readiness tests
│
├── integration/           # Multi-module tests
│   └── test_system_integration.py
│
└── scenario/              # IB compliance scenarios
    └── test_ib_scenarios.py
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests (slower)
pytest tests/integration/ -v

# Scenario tests (IB compliance)
pytest -m scenario -v

# Specific test file
pytest tests/unit/test_p1_patches.py -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest --cov=backend --cov-report=html

# View report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

### Run Async Tests
```bash
pytest -m asyncio -v
```

## Test Markers

Tests are organized with pytest markers:

| Marker | Description | Usage |
|--------|-------------|-------|
| `unit` | Fast, isolated tests | `pytest -m unit` |
| `integration` | Multi-module tests | `pytest -m integration` |
| `scenario` | IB compliance scenarios | `pytest -m scenario` |
| `asyncio` | Async tests | `pytest -m asyncio` |
| `slow` | Slow tests (skip by default) | `pytest -m "not slow"` |

## Test Coverage

### Current Status

| Category | Tests | Coverage | Status |
|----------|-------|----------|--------|
| **Unit** | 17 | ~60% | 🟡 In Progress |
| **Integration** | 4 | ~30% | 🔴 TODO |
| **Scenario** | 7 | ~70% | 🟡 In Progress |
| **Total** | 28 | ~50% | 🟡 In Progress |

**Target:** 80% coverage before mainnet deployment

### Coverage by Module

| Module | Coverage | Priority |
|--------|----------|----------|
| `backend/services/risk/` | 75% | ✅ High |
| `backend/core/` | 65% | ✅ High |
| `backend/services/execution/` | 45% | 🟡 Medium |
| `backend/services/ai/` | 40% | 🟡 Medium |
| `backend/services/monitoring/` | 30% | 🔴 Low |

## Test Scenarios (IB Compliance)

The `tests/scenario/` directory contains 7 critical scenarios directly linked to IB requirements:

### Scenario 1: Normal Market Conditions
- **Status:** ✅ Implemented
- **Tests:** Signal quality, position sizing, TP/SL execution
- **Expected:** Full trading with $500 positions

### Scenario 2: High Volatility Regime
- **Status:** ✅ Implemented
- **Tests:** Signal filtering (65% confidence), reduced position sizes
- **Expected:** Only high-quality signals pass, 50% position reduction

### Scenario 3: Flash Crash Event
- **Status:** ✅ Implemented
- **Tests:** ESS activation, position closure, trading halt
- **Expected:** All positions closed within 30 seconds

### Scenario 4: PolicyStore Unavailable
- **Status:** ⏳ TODO
- **Tests:** Fallback to defaults, warning logs
- **Expected:** Trading continues with conservative settings

### Scenario 5: Model Prediction Disagreement
- **Status:** ✅ Implemented
- **Tests:** Consensus requirement (≥3/4 models)
- **Expected:** Conflicting signals rejected

### Scenario 6: Drawdown Recovery
- **Status:** ✅ Implemented
- **Tests:** Auto-recovery transitions (EMERGENCY → PROTECTIVE → CAUTIOUS → NORMAL)
- **Expected:** Gradual recovery as DD improves

### Scenario 7: Multi-Symbol Correlation
- **Status:** ⏳ TODO
- **Tests:** Correlation detection, position blocking
- **Expected:** Over-concentration prevented

## Writing New Tests

### Unit Test Template

```python
import pytest
from backend.services.risk.risk_guard import RiskGuard

class TestRiskGuard:
    """Unit tests for RiskGuard."""
    
    @pytest.fixture
    def risk_guard(self):
        return RiskGuard(max_position_size=500)
    
    def test_position_limit_exceeded(self, risk_guard):
        """Test risk guard blocks oversized positions."""
        result = risk_guard.check_position_size(
            symbol="BTCUSDT",
            size_usd=1000
        )
        assert result.blocked is True
        assert "limit exceeded" in result.reason.lower()
```

### Async Test Template

```python
import pytest

@pytest.mark.asyncio
async def test_ess_activation():
    """Test ESS activates on drawdown."""
    from backend.services.risk.emergency_stop_system import ESS
    
    ess = ESS()
    await ess.activate("Test emergency")
    
    assert ess.is_active is True
```

### Scenario Test Template

```python
@pytest.mark.scenario
@pytest.mark.asyncio
async def test_scenario_custom():
    """
    Scenario: Custom Test
    
    Description:
        Describe the scenario and expected behavior.
    
    Expected Behavior:
        - Step 1
        - Step 2
        - Step 3
    """
    # Setup
    # Execute
    # Assert
    pass
```

## Continuous Integration

### GitHub Actions (Future)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'backend'`  
**Solution:** Run tests from project root: `cd c:\quantum_trader && pytest`

**Issue:** Async tests fail with `RuntimeError: Event loop is closed`  
**Solution:** Add `pytest-asyncio` to requirements and set `asyncio_mode = "auto"` in `pyproject.toml`

**Issue:** Tests take too long  
**Solution:** Run unit tests only: `pytest tests/unit/ -v`

## Best Practices

1. **Test Isolation:** Each test should be independent
2. **Fast Unit Tests:** Keep unit tests under 100ms each
3. **Clear Names:** Use descriptive test names (test_what_when_then)
4. **Fixtures:** Use pytest fixtures for common setup
5. **Mocking:** Mock external dependencies (Redis, exchange API)
6. **Coverage:** Aim for 80%+ coverage on critical modules
7. **Documentation:** Document complex test scenarios

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [P2_PATCHES_COMPLETE.md](../P2_PATCHES_COMPLETE.md)
