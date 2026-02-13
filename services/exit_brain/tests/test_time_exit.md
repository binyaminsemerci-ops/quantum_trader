# EB-FT-01: Time Exit Test

**Test ID**: EB-FT-01  
**Category**: Mandatory Exit  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify that positions are closed when exceeding maximum hold time.

---

## Test Cases

### TC-01.1: Trending + Normal Vol = 72h Max

**Given**: Position in TRENDING regime, normal volatility  
**And**: Position age = 73 hours  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent`

```python
def test_time_exit_trending_normal():
    position = Position(
        opened_at=now() - timedelta(hours=73),
        entry_regime=Regime.TRENDING_UP,
        entry_volatility=0.02
    )
    
    result = exit_brain.check_time_exit(position, Regime.TRENDING_UP, vol=0.02)
    
    assert result.action == "CLOSE_FULL"
    assert "time" in result.reason.lower()
```

### TC-01.2: Ranging + High Vol = 24h Max

**Given**: Position in RANGING regime, high volatility  
**And**: Position age = 25 hours  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent`

```python
def test_time_exit_ranging_highvol():
    position = Position(
        opened_at=now() - timedelta(hours=25),
        entry_regime=Regime.RANGING,
        entry_volatility=0.05
    )
    
    result = exit_brain.check_time_exit(position, Regime.RANGING, vol=0.05)
    
    assert result.action == "CLOSE_FULL"
```

### TC-01.3: Chaotic = 12h Max

**Given**: Position in any regime  
**And**: Current regime = CHAOTIC  
**And**: Position age = 13 hours  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent`

```python
def test_time_exit_chaotic():
    position = Position(opened_at=now() - timedelta(hours=13))
    
    result = exit_brain.check_time_exit(position, Regime.CHAOTIC, vol=0.08)
    
    assert result.action == "CLOSE_FULL"
```

### TC-01.4: Position Within Time Limit

**Given**: Position age = 24 hours  
**And**: Max hold = 72 hours (trending + normal)  
**When**: Exit Brain evaluates  
**Then**: Result = None (no time exit)

```python
def test_no_time_exit_within_limit():
    position = Position(opened_at=now() - timedelta(hours=24))
    
    result = exit_brain.check_time_exit(position, Regime.TRENDING_UP, vol=0.02)
    
    assert result is None
```

### TC-01.5: Warning at 80% of Max Time

**Given**: Position age = 58 hours (80% of 72)  
**When**: Exit Brain evaluates  
**Then**: Result = Warning (not exit)

```python
def test_time_warning():
    position = Position(opened_at=now() - timedelta(hours=58))
    
    result = exit_brain.check_time_exit(position, Regime.TRENDING_UP, vol=0.02)
    
    assert result.type == "TIME_WARNING"
    assert result.action == "MONITOR"
```

---

## Acceptance Criteria

- [ ] All time limits enforced correctly
- [ ] Regime affects time limit
- [ ] Volatility affects time limit
- [ ] Warning issued before exit
