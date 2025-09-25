"""Pluggable exchange client abstraction.

This module defines a small runtime-friendly interface and a factory for
exchange adapters. The initial adapter wraps Binance-like functionality.
Other adapters (Coinbase, KuCoin) can be added by implementing the same
methods and registering them in get_exchange_client.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol
import datetime
from config.config import DEFAULT_EXCHANGE


class ExchangeClient(Protocol):
    def spot_balance(self) -> Dict[str, Any]:
        ...

    def futures_balance(self) -> Dict[str, Any]:
        ...

    def fetch_recent_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        ...

    def create_order(self, symbol: str, side: str, qty: float, order_type: str = 'MARKET') -> Dict[str, Any]:
        ...


class _BinanceAdapter:
    """Lightweight wrapper around python-binance client methods used by the repo.

    This adapter delays importing the heavy `binance` package so tests/CI that
    don't install it can still import the module.
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None
        try:
            from binance.client import Client  # type: ignore

            if api_key and api_secret:
                self._client = Client(api_key, api_secret)  # type: ignore
        except Exception:
            self._client = None

    def spot_balance(self) -> Dict[str, Any]:
        # return a small shape the rest of the app expects
        if not self._client:
            return {"asset": "USDC", "free": 1000.0}
        try:
            # map to simple representation
            acct = self._client.get_account()
            # find USDC entry if present
            for b in acct.get('balances', []):
                if b.get('asset') == 'USDC':
                    return {"asset": 'USDC', "free": float(b.get('free', 0))}
        except Exception:
            pass
        return {"asset": "USDC", "free": 0.0}

    def futures_balance(self) -> Dict[str, Any]:
        if not self._client:
            return {"asset": "USDT", "balance": 0.0}
        try:
            # python-binance exposes futures account endpoints differently; attempt a pragmatic fetch
            fut = None
            try:
                fut = self._client.futures_account_balance()
            except Exception:
                fut = None
            if fut:
                # find USDT entry
                for b in fut:
                    if b.get('asset') == 'USDT':
                        return {"asset": 'USDT', "balance": float(b.get('balance', 0))}
        except Exception:
            pass
        return {"asset": "USDT", "balance": 0.0}

    def fetch_recent_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self._client:
            # deterministic mock trades
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return [{"symbol": symbol, "qty": "0.01", "price": "100.0", "isBuyerMaker": True, "time": now} for _ in range(limit)]
        try:
            return self._client.get_recent_trades(symbol=symbol, limit=limit)  # type: ignore
        except Exception:
            return []

    def create_order(self, symbol: str, side: str, qty: float, order_type: str = 'MARKET') -> Dict[str, Any]:
        if not self._client:
            return {"symbol": symbol, "status": "mock", "side": side, "qty": qty}
        try:
            return self._client.create_order(symbol=symbol, side=side, type=order_type, quantity=qty)  # type: ignore
        except Exception as exc:
            return {"error": str(exc)}


from typing import Type


_ADAPTER_REGISTRY: Dict[str, Type] = {
    'binance': _BinanceAdapter,
    # future adapters: 'coinbase': CoinbaseAdapter, 'kucoin': KuCoinAdapter
}


class _CoinbaseAdapter:
    """Stub adapter for Coinbase-like APIs.

    This is intentionally lightweight: it provides the same methods as the
    Binance adapter but delays any heavy imports and returns mock-safe
    responses when credentials are not supplied. Implementors can extend
    this to call coinbase-pro or the newer Coinbase Exchange APIs.
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None
        # Attempt to create a real ccxt client lazily. If ccxt isn't installed
        # or credentials aren't provided, keep _client as None so tests/CI
        # continue to use the mock-safe responses.
        try:
            import ccxt  # type: ignore

            # ccxt uses the id 'coinbasepro' for the Coinbase Pro (Exchange) API.
            if api_key and api_secret:
                ex = ccxt.coinbasepro({
                    'apiKey': api_key,
                    'secret': api_secret,
                })
                self._client = ex
        except Exception:
            self._client = None

    def spot_balance(self) -> Dict[str, Any]:
        # Prefer real client if available
        if not self._client:
            return {"asset": "USDC", "free": 500.0}
        try:
            bal = self._client.fetch_balance()
            # ccxt returns balances in a dict under 'total'/'free' keyed by currency
            total = bal.get('total') or bal
            free = bal.get('free') or bal.get('free') or {}
            if 'USDC' in total:
                return {"asset": "USDC", "free": float(free.get('USDC', 0))}
            # fallback: return first non-zero asset
            for k, v in (free or {}).items():
                try:
                    if float(v) > 0:
                        return {"asset": k, "free": float(v)}
                except Exception:
                    continue
        except Exception:
            pass
        return {"asset": "USDC", "free": 0.0}

    def futures_balance(self) -> Dict[str, Any]:
        # Best-effort: try futures-type balance via ccxt params; otherwise mock
        if not self._client:
            return {"asset": "USDT", "balance": 0.0}
        try:
            # many ccxt exchanges accept a 'type' param for futures (exchange-specific)
            bal = None
            try:
                bal = self._client.fetch_balance({'type': 'future'})
            except Exception:
                try:
                    bal = self._client.fetch_balance({'type': 'futures'})
                except Exception:
                    bal = None
            if bal:
                total = bal.get('total') or bal
                if 'USDT' in total:
                    return {"asset": 'USDT', "balance": float(total.get('USDT', 0))}
        except Exception:
            pass
        return {"asset": "USDT", "balance": 0.0}

    def fetch_recent_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not self._client:
            return [{"symbol": symbol, "qty": "0.01", "price": "100.0", "isBuyerMaker": True, "time": now} for _ in range(limit)]
        try:
            # ccxt expects a symbol like 'ETH/USDC'
            if symbol.upper().endswith('USDC'):
                ccxt_sym = f"{symbol[:-4]}/USDC"
            elif symbol.upper().endswith('USDT'):
                ccxt_sym = f"{symbol[:-4]}/USDT"
            else:
                # fallback: try to insert a slash before last 3 chars
                ccxt_sym = f"{symbol[:-3]}/{symbol[-3:]}"
            trades = self._client.fetch_trades(ccxt_sym, limit=limit)
            # normalize minimal fields expected by callers
            out = []
            for t in trades[:limit]:
                out.append({
                        'symbol': symbol,
                        'qty': str(t.get('amount') or t.get('size') or 0),
                        'price': str(t.get('price') or 0),
                        'isBuyerMaker': t.get('side') == 'buy',
                        'time': datetime.datetime.fromtimestamp(int(t.get('timestamp', 0) / 1000), tz=datetime.timezone.utc).isoformat() if t.get('timestamp') else now,
                    })
            return out
        except Exception:
            return [{"symbol": symbol, "qty": "0.01", "price": "100.0", "isBuyerMaker": True, "time": now} for _ in range(limit)]

    def create_order(self, symbol: str, side: str, qty: float, order_type: str = 'MARKET') -> Dict[str, Any]:
        if not self._client:
            return {"symbol": symbol, "status": "mock", "side": side, "qty": qty}
        try:
            # normalize symbol for ccxt
            if symbol.upper().endswith('USDC'):
                ccxt_sym = f"{symbol[:-4]}/USDC"
            elif symbol.upper().endswith('USDT'):
                ccxt_sym = f"{symbol[:-4]}/USDT"
            else:
                ccxt_sym = symbol
            # ccxt create_order(symbol, type, side, amount, params)
            res = self._client.create_order(ccxt_sym, order_type.lower(), side.lower(), float(qty))
            return res
        except Exception as exc:
            return {"error": str(exc)}


class _KuCoinAdapter:
    """Stub adapter for KuCoin-like APIs.

    Implements the same minimal surface as other adapters so the rest of the
    application can call through the factory without special-casing.
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None
        # lazy import ccxt and instantiate kucoin client when possible
        try:
            import ccxt  # type: ignore
            if api_key and api_secret:
                ex = ccxt.kucoin({
                    'apiKey': api_key,
                    'secret': api_secret,
                })
                self._client = ex
        except Exception:
            self._client = None

    def spot_balance(self) -> Dict[str, Any]:
        if not self._client:
            return {"asset": "USDC", "free": 200.0}
        try:
            bal = self._client.fetch_balance()
            total = bal.get('total') or bal
            free = bal.get('free') or {}
            if 'USDC' in total:
                return {"asset": "USDC", "free": float(free.get('USDC', 0))}
            for k, v in (free or {}).items():
                try:
                    if float(v) > 0:
                        return {"asset": k, "free": float(v)}
                except Exception:
                    continue
        except Exception:
            pass
        return {"asset": "USDC", "free": 0.0}

    def futures_balance(self) -> Dict[str, Any]:
        if not self._client:
            return {"asset": "USDT", "balance": 0.0}
        try:
            bal = None
            try:
                bal = self._client.fetch_balance({'type': 'future'})
            except Exception:
                try:
                    bal = self._client.fetch_balance({'type': 'futures'})
                except Exception:
                    bal = None
            if bal:
                total = bal.get('total') or bal
                if 'USDT' in total:
                    return {"asset": 'USDT', "balance": float(total.get('USDT', 0))}
        except Exception:
            pass
        return {"asset": "USDT", "balance": 0.0}

    def fetch_recent_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not self._client:
            return [{"symbol": symbol, "qty": "0.01", "price": "100.0", "isBuyerMaker": True, "time": now} for _ in range(limit)]
        try:
            if symbol.upper().endswith('USDC'):
                ccxt_sym = f"{symbol[:-4]}/USDC"
            elif symbol.upper().endswith('USDT'):
                ccxt_sym = f"{symbol[:-4]}/USDT"
            else:
                ccxt_sym = f"{symbol[:-3]}/{symbol[-3:]}"
            trades = self._client.fetch_trades(ccxt_sym, limit=limit)
            out = []
            for t in trades[:limit]:
                out.append({
                    'symbol': symbol,
                    'qty': str(t.get('amount') or t.get('size') or 0),
                    'price': str(t.get('price') or 0),
                    'isBuyerMaker': t.get('side') == 'buy',
                    'time': datetime.datetime.fromtimestamp(int(t.get('timestamp', 0) / 1000), tz=datetime.timezone.utc).isoformat() if t.get('timestamp') else now,
                })
            return out
        except Exception:
            return [{"symbol": symbol, "qty": "0.01", "price": "100.0", "isBuyerMaker": True, "time": now} for _ in range(limit)]

    def create_order(self, symbol: str, side: str, qty: float, order_type: str = 'MARKET') -> Dict[str, Any]:
        if not self._client:
            return {"symbol": symbol, "status": "mock", "side": side, "qty": qty}
        try:
            if symbol.upper().endswith('USDC'):
                ccxt_sym = f"{symbol[:-4]}/USDC"
            elif symbol.upper().endswith('USDT'):
                ccxt_sym = f"{symbol[:-4]}/USDT"
            else:
                ccxt_sym = symbol
            res = self._client.create_order(ccxt_sym, order_type.lower(), side.lower(), float(qty))
            return res
        except Exception as exc:
            return {"error": str(exc)}


_ADAPTER_REGISTRY.update({
    'coinbase': _CoinbaseAdapter,
    'kucoin': _KuCoinAdapter,
})


def get_exchange_client(name: Optional[str] = None, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> ExchangeClient:
    # If no name provided, use repository-wide default
    if not name:
        name = DEFAULT_EXCHANGE
    cls = _ADAPTER_REGISTRY.get(name.lower())
    if not cls:
        raise ValueError(f'Unknown exchange adapter: {name}')
    # mypy: cls is a Type and calling it returns an ExchangeClient at runtime
    return cls(api_key=api_key, api_secret=api_secret)


__all__ = ['ExchangeClient', 'get_exchange_client']
