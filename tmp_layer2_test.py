from microservices.risk_policy_enforcer import create_enforcer

enforcer = create_enforcer('redis://localhost:6379')

# Volatility gate breach (too high)
allowed, state, reason = enforcer.allow_trade(
    symbol='BTCUSDT',
    requested_leverage=2.0,
    volatility=0.5,
    spread_bps=5.0
)
print(f"Volatility breach: allowed={allowed}, state={state.value}, reason={reason}")

# Liquidity gate breach (spread too wide)
allowed, state, reason = enforcer.allow_trade(
    symbol='BTCUSDT',
    requested_leverage=2.0,
    volatility=0.02,
    spread_bps=100.0
)
print(f"Liquidity breach: allowed={allowed}, state={state.value}, reason={reason}")

# Whitelist enforcement
allowed, state, reason = enforcer.allow_trade(
    symbol='DOGEUSDT',
    requested_leverage=2.0,
    volatility=0.02,
    spread_bps=5.0
)
print(f"Whitelist breach: allowed={allowed}, state={state.value}, reason={reason}")
