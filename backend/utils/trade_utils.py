"""Small helpers for choosing execution markets (spot vs futures) and building pairs.

Most trades should execute on USDT-margined futures / cross-margin by default.
Use the helpers here to centralize that decision so UI, backtests and bots are consistent.
"""

from typing import Literal
from config.config import DEFAULT_QUOTE, FUTURES_QUOTE, make_pair

MarketType = Literal["spot", "futures"]


def select_execution_pair(base: str, market: MarketType = "futures") -> str:
    """Return the exchange symbol to use for execution.

    By default we prefer futures (USDT-margined) for most automated trading
    flows. Pass market='spot' to explicitly use the spot quote (DEFAULT_QUOTE).
    """
    if market == "futures":
        return make_pair(base, quote=FUTURES_QUOTE)
    return make_pair(base, quote=DEFAULT_QUOTE)
