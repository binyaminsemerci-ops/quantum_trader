from microservices.risk_policy_enforcer import create_enforcer

# LAYER 1: capital guards
enforcer = create_enforcer('redis://localhost:6379')

# Over-leverage clamp
ok, clamped = enforcer.check_leverage(25.0)
print(f"Over-leverage check: ok={ok}, clamped={clamped}")

# Daily loss limit breach
for _ in range(5):
    enforcer.record_trade_outcome(pnl=-300.0, symbol='BTCUSDT')

metrics = enforcer.compute_system_state(symbol='BTCUSDT')
print(f"Daily loss state: {metrics.system_state.value}, reason={metrics.failure_reason}")

# Rolling drawdown breach
for eq in [10000, 9500, 9000, 8000, 7000]:
    enforcer.update_equity(eq)

metrics = enforcer.compute_system_state(symbol='BTCUSDT')
print(f"Drawdown state: {metrics.system_state.value}, reason={metrics.failure_reason}")
