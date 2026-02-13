# Test: Signal Advisory Has No Execution Power

> **Test ID**: SA-FT-01  
> **Service**: Signal Advisory  
> **Type**: Architectural Constraint Test  
> **Priority**: CRITICAL – Verifies core safety constraint

---

## 1. Purpose

Verify that Signal Advisory has **NO execution capability** and cannot:
- Place orders on exchange
- Modify existing orders
- Cancel orders
- Access exchange API directly

This is a fundamental architectural constraint, not a feature flag.

---

## 2. Test Cases

### SA-FT-01-A: No Exchange Client

**Scenario**: Verify Signal Advisory has no exchange client instantiated

```python
def test_no_exchange_client():
    """
    Signal Advisory MUST NOT have exchange client.
    """
    
    signal_advisory = SignalAdvisory()
    
    # Verify no exchange-related attributes
    assert not hasattr(signal_advisory, 'exchange_client')
    assert not hasattr(signal_advisory, 'exchange')
    assert not hasattr(signal_advisory, 'binance_client')
    assert not hasattr(signal_advisory, 'api_key')
    assert not hasattr(signal_advisory, 'api_secret')
    
    # Verify no execution methods
    assert not hasattr(signal_advisory, 'place_order')
    assert not hasattr(signal_advisory, 'cancel_order')
    assert not hasattr(signal_advisory, 'modify_order')
    assert not hasattr(signal_advisory, 'execute_trade')
```

**Expected**: All assertions pass  
**Failure Handling**: BLOCK DEPLOYMENT – architectural violation

---

### SA-FT-01-B: Output Contains No Execution Instructions

**Scenario**: Verify signal output format has no execution instructions

```python
def test_output_is_advisory_only():
    """
    Signal output MUST be advisory only.
    No execution instructions in payload.
    """
    
    signal = SignalAdvisory().generate_signal(mock_inputs)
    
    # Must have execution_power = "NONE"
    assert signal.execution_power == "NONE"
    
    # Must NOT have execution fields
    assert not hasattr(signal, 'order_id')
    assert not hasattr(signal, 'order_type')
    assert not hasattr(signal, 'limit_price')
    assert not hasattr(signal, 'quantity')
    assert not hasattr(signal, 'leverage')
    
    # Action is RECOMMENDATION, not ORDER
    assert signal.recommended_action in ["LONG", "SHORT", "NO_SIGNAL"]
    assert "ORDER" not in signal.recommended_action
    assert "EXECUTE" not in signal.reasoning
```

**Expected**: Signal is purely advisory  
**Failure Handling**: Reject signal, fix output format

---

### SA-FT-01-C: Cannot Import Execution Modules

**Scenario**: Signal Advisory module cannot import execution-related modules

```python
def test_no_execution_imports():
    """
    Signal Advisory code MUST NOT import execution modules.
    """
    import ast
    
    with open('services/signal_advisory/main.py', 'r') as f:
        tree = ast.parse(f.read())
    
    imports = [node for node in ast.walk(tree) 
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    
    forbidden_imports = [
        'execution_engine',
        'order_manager',
        'binance',
        'ccxt',
        'exchange_client',
    ]
    
    for imp in imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                for forbidden in forbidden_imports:
                    assert forbidden not in alias.name.lower(), \
                        f"Forbidden import: {alias.name}"
        elif isinstance(imp, ast.ImportFrom):
            for forbidden in forbidden_imports:
                assert forbidden not in (imp.module or '').lower(), \
                    f"Forbidden import from: {imp.module}"
```

**Expected**: No execution-related imports found  
**Failure Handling**: BLOCK DEPLOYMENT – code review required

---

### SA-FT-01-D: Network Isolation

**Scenario**: Signal Advisory cannot reach exchange endpoints

```python
def test_network_isolation():
    """
    Signal Advisory should not have network path to exchange.
    """
    import socket
    
    # Run within Signal Advisory container context
    exchange_endpoints = [
        ('api.binance.com', 443),
        ('fapi.binance.com', 443),
        ('api.bybit.com', 443),
    ]
    
    for host, port in exchange_endpoints:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            
            # Should NOT be able to connect
            assert result != 0, f"Signal Advisory should not reach {host}:{port}"
        except (socket.timeout, socket.error):
            pass  # Expected - cannot connect
```

**Expected**: All connection attempts fail/timeout  
**Failure Handling**: Review network policies, update Docker network config

---

### SA-FT-01-E: Redis Event Has No Execution Instructions

**Scenario**: Events published to Redis contain no execution instructions

```python
def test_redis_event_is_advisory():
    """
    Redis events from Signal Advisory must be advisory only.
    """
    
    # Mock Redis to capture published event
    captured_events = []
    
    def capture_publish(stream, event):
        captured_events.append((stream, event))
    
    signal_advisory = SignalAdvisory(redis_publisher=capture_publish)
    signal_advisory.generate_and_publish(mock_inputs)
    
    for stream, event in captured_events:
        # Stream should be advisory stream
        assert 'advisory' in stream
        assert 'execution' not in stream
        assert 'order' not in stream
        
        # Event should not contain execution fields
        assert 'order_id' not in event
        assert 'execute' not in event.get('action', '').lower()
        assert event.get('execution_power') == 'NONE'
```

**Expected**: All events are advisory format only  
**Failure Handling**: Fix event schema, review publisher

---

### SA-FT-01-F: API Endpoint Returns Advisory Only

**Scenario**: HTTP API returns advisory signals, not execution orders

```python
def test_api_returns_advisory():
    """
    API endpoints return advisory responses only.
    """
    import httpx
    
    response = httpx.get('http://signal_advisory:8006/signal/latest')
    data = response.json()
    
    # Must have advisory fields
    assert 'execution_power' in data
    assert data['execution_power'] == 'NONE'
    
    # Must have advisory action
    assert data.get('recommended_action') in ['LONG', 'SHORT', 'NO_SIGNAL', None]
    
    # Must NOT have execution fields
    assert 'order' not in data
    assert 'execute' not in str(data).lower()
    assert 'api_key' not in str(data).lower()
```

**Expected**: API is advisory-only  
**Failure Handling**: Fix API response format

---

## 3. Continuous Verification

### Runtime Assertion

```python
# In SignalAdvisory.__init__
def __init__(self):
    # Runtime verification of architectural constraint
    self._verify_no_execution_capability()

def _verify_no_execution_capability(self):
    """
    Runtime check that execution capability is absent.
    Runs on every startup.
    """
    
    forbidden_attrs = [
        'exchange_client', 'execution_engine', 'order_manager',
        'api_key', 'api_secret', 'place_order', 'execute_trade'
    ]
    
    for attr in forbidden_attrs:
        if hasattr(self, attr):
            raise ArchitecturalViolation(
                f"Signal Advisory has forbidden attribute: {attr}. "
                f"This violates the advisory-only constraint."
            )
    
    # Log verification
    logger.info("Verified: Signal Advisory has no execution capability")
```

### Health Check

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "execution_power": "NONE",
        "advisory_only": True,
        "verified_no_exchange_access": True
    }
```

---

## 4. Test Matrix

| Test ID | Description | Priority | Automation |
|---------|-------------|----------|------------|
| SA-FT-01-A | No exchange client | CRITICAL | CI/CD gate |
| SA-FT-01-B | Output advisory only | CRITICAL | Every signal |
| SA-FT-01-C | No execution imports | CRITICAL | CI/CD gate |
| SA-FT-01-D | Network isolation | HIGH | Weekly |
| SA-FT-01-E | Redis event advisory | HIGH | Every event |
| SA-FT-01-F | API advisory only | HIGH | Hourly |

---

## 5. Failure Consequences

If ANY of these tests fail:

```
┌─────────────────────────────────────────────────────────────────┐
│ ARCHITECTURAL VIOLATION DETECTED                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Signal Advisory has execution capability – THIS IS FORBIDDEN    │
│                                                                 │
│ Immediate Actions:                                              │
│ 1. BLOCK DEPLOYMENT                                             │
│ 2. Alert security team                                          │
│ 3. Code review required                                         │
│ 4. Identify how violation occurred                              │
│ 5. Remove execution capability                                  │
│ 6. Re-verify all constraints                                    │
│                                                                 │
│ This is a CRITICAL SAFETY CONSTRAINT.                           │
│ Signal Advisory must NEVER have execution power.                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Signal Advisory advises. Risk Kernel decides. Execution Engine executes. No exceptions.*
