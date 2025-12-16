"""Verify live trading configuration"""
from backend.config.execution import load_execution_config
from backend.config.risk import load_risk_config
from backend.services.execution.execution import build_execution_adapter

exec_cfg = load_execution_config()
risk_cfg = load_risk_config()

print('[TARGET] LIVE TRADING CONFIGURATION')
print('=' * 80)
print(f'Exchange: {exec_cfg.exchange}')
print(f'Testnet: {exec_cfg.binance_testnet}')
print(f'Min Notional: ${exec_cfg.min_notional}')
print(f'Max Orders: {exec_cfg.max_orders}')
print()
print('[MONEY] RISK LIMITS:')
print(f'Max per trade: ${risk_cfg.max_notional_per_trade}')
print(f'Max per symbol: ${risk_cfg.max_position_per_symbol}')
print(f'Max total exposure: ${risk_cfg.max_gross_exposure}')
print(f'Max daily loss: ${risk_cfg.max_daily_loss}')
print()
print(f'[CHART] Allowed symbols: {", ".join(risk_cfg.allowed_symbols)}')
print()

adapter = build_execution_adapter(exec_cfg)
print(f'[OK] Adapter: {type(adapter).__name__}')
if hasattr(adapter, 'ready'):
    print(f'Ready: {adapter.ready}')
