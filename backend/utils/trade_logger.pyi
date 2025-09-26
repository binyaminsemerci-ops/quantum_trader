from typing import Any, Optional

def log_trade(
    trade: dict[str, Any], status: str, reason: Optional[str] = None
) -> Any: ...

# The codebase references these helpers; provide liberal Any signatures so
# mypy in CI can import them even if implementations are more dynamic.
def get_trades(limit: int | None = None) -> Any: ...
def get_balance_and_pnl() -> Any: ...
