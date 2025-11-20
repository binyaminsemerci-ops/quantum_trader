"""Automated execution loop utilities."""

from __future__ import annotations

import asyncio
import atexit
import logging
import json
from pathlib import Path
from dataclasses import dataclass, replace
import os
import weakref
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.config import (
    ExecutionConfig,
    LiquidityConfig,
    load_config,
    load_execution_config,
    load_liquidity_config,
)
from backend.config.risk import load_risk_config
from backend.models.liquidity import ExecutionJournal, LiquidityRun, LiquiditySnapshot, PortfolioAllocation
from backend.services.positions import PortfolioPositionService
from backend.services.risk_guard import RiskGuardService, SqliteRiskStateStore
from backend.services.ai_trading_engine import create_ai_trading_engine


logger = logging.getLogger(__name__)


_STABLE_QUOTE_SUFFIXES: Tuple[str, ...] = ("USDT", "USDC", "BUSD", "USD")


@dataclass(slots=True)
class OrderIntent:
    symbol: str
    side: str
    target_weight: float
    quantity: float
    price: float
    notional: float
    reason: str


class ExchangeAdapter(Protocol):
    async def get_positions(self) -> Mapping[str, float]: ...

    async def get_cash_balance(self) -> float: ...

    async def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> str: ...


class PaperExchangeAdapter:
    """Simple in-memory adapter useful for tests and dry-runs."""

    def __init__(self, *, positions: Optional[Mapping[str, float]] = None, cash: float = 500.0) -> None:
        self._positions: Dict[str, float] = {symbol.upper(): float(qty) for symbol, qty in (positions or {}).items()}
        self._cash = float(cash)
        self._lock = asyncio.Lock()
        self._order_counter = 0

    async def get_positions(self) -> Mapping[str, float]:
        async with self._lock:
            return dict(self._positions)

    async def get_cash_balance(self) -> float:
        async with self._lock:
            return float(self._cash)

    async def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> str:
        symbol = symbol.upper()
        qty = float(quantity)
        notional = qty * float(price)
        async with self._lock:
            if side.upper() == "BUY":
                self._cash -= notional
                self._positions[symbol] = self._positions.get(symbol, 0.0) + qty
            else:
                self._cash += notional
                self._positions[symbol] = self._positions.get(symbol, 0.0) - qty
            self._order_counter += 1
            return f"paper-{symbol}-{self._order_counter}"


class BinanceExecutionAdapter:
    """Minimal live adapter wrapping the python-binance client."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        api_secret: Optional[str],
        quote_asset: str = "USDT",
        testnet: bool = False,
    ) -> None:
        self._quote = (quote_asset or "USDT").upper()
        self._client = None
        self._testnet = bool(testnet)
        if api_key and api_secret:
            try:
                from binance.client import Client  # type: ignore

                self._client = Client(api_key, api_secret, testnet=self._testnet)  # type: ignore[assignment]
                if self._testnet:
                    # Ensure the client routes to the spot testnet endpoints.
                    try:
                        self._client.API_URL = "https://testnet.binance.vision/api"  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - property missing in older libs
                        pass
            except Exception as exc:  # pragma: no cover - best-effort init
                logger.warning("Failed to initialise Binance client: %s", exc)
                self._client = None
        else:
            self._client = None

    @property
    def ready(self) -> bool:
        return self._client is not None

    async def _fetch_account_balances(self) -> Dict[str, Dict[str, float]]:
        if self._client is None:
            return {}
        try:
            account = await asyncio.to_thread(self._client.get_account)
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Binance account snapshot failed: %s", exc)
            return {}

        balances: Dict[str, Dict[str, float]] = {}
        for entry in account.get("balances", []):
            asset = str(entry.get("asset", "")).upper()
            if not asset:
                continue
            try:
                free = float(entry.get("free", 0.0))
                locked = float(entry.get("locked", 0.0))
            except (TypeError, ValueError):
                continue
            balances[asset] = {"free": free, "locked": locked}
        return balances

    async def get_positions(self) -> Mapping[str, float]:
        balances = await self._fetch_account_balances()
        positions: Dict[str, float] = {}
        for asset, entry in balances.items():
            if asset == self._quote:
                continue
            total = float(entry.get("free", 0.0)) + float(entry.get("locked", 0.0))
            if total <= 0:
                continue
            positions[f"{asset}{self._quote}"] = total
        return positions

    async def get_cash_balance(self) -> float:
        balances = await self._fetch_account_balances()
        entry = balances.get(self._quote)
        if not entry:
            return 0.0
        return float(entry.get("free", 0.0))

    async def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> str:
        if self._client is None:
            raise RuntimeError("Binance client is not configured")
        try:
            order = await asyncio.to_thread(
                self._client.create_order,
                symbol=symbol,
                side=side.upper(),
                type="MARKET",
                quantity=quantity,
            )
            return str(order.get("orderId") or order.get("clientOrderId") or "")
        except Exception as exc:  # pragma: no cover - live API failure path
            logger.exception("Failed to submit Binance order", exc_info=exc)
            raise


class BinanceFuturesExecutionAdapter:
    """Minimal Binance USDM/COINM Futures adapter with HMAC-signed REST calls.

    Notes:
    - Uses USDM by default when market_type='usdm_perp' (fapi.binance.com)
    - Uses COINM when market_type='coinm_perp' (dapi.binance.com)
    - Honors STAGING_MODE=true by simulating order submission (no network call)
    - Sets leverage and margin mode lazily per-symbol before first order
    """

    def __init__(
        self,
        *,
        api_key: Optional[str],
        api_secret: Optional[str],
        quote_asset: str = "USDT",
        market_type: str = "usdm_perp",
        default_leverage: int = 30,
        margin_mode: str = "cross",
        recv_window: int = 5000,
    ) -> None:
        import hmac
        import hashlib
        import time
        from urllib.parse import urlencode

        self._hmac = hmac
        self._hashlib = hashlib
        self._time = time
        self._urlencode = urlencode

        self._api_key = api_key or ""
        self._api_secret = (api_secret or "").encode("utf-8")
        requested_quote = (quote_asset or "USDT").upper()
        # Slightly more tolerant default recv window (timestamp skew issues in containers)
        self._recv_window = max(5000, int(recv_window))
        self._market_type = market_type
        self._quote = self._select_settlement_quote(requested_quote)
        if self._quote != requested_quote:
            logger.info(
                "Binance %s futures do not support %s quote; using %s",
                market_type,
                requested_quote,
                self._quote,
            )
        self._base_url = "https://fapi.binance.com" if market_type == "usdm_perp" else "https://dapi.binance.com"
        # Always use environment variable for leverage
        env_leverage = os.getenv("QT_DEFAULT_LEVERAGE")
        self._default_leverage = max(1, min(int(env_leverage or default_leverage or 30), 125))
        self._margin_mode = (margin_mode or "cross").lower()
        # Lazy-imported aiohttp session holder; type as Any to avoid import timing issues
        self._aiohttp_session = None  # will hold aiohttp.ClientSession
        self._configured_symbols: set[str] = set()
        # Track server time offset in ms to avoid -1021 timestamp errors
        self._time_offset_ms = 0
        # Cache exchange info filters for precision rounding
        self._symbol_filters: dict[str, dict[str, Any]] = {}
        self._symbol_aliases: Dict[str, str] = {}
        self._atexit_registered = False
        self._register_session_cleanup()

    @property
    def ready(self) -> bool:
        return bool(self._api_key and self._api_secret)

    def _register_session_cleanup(self) -> None:
        if self._atexit_registered:
            return
        weak_self = weakref.ref(self)

        def _cleanup() -> None:
            inst = weak_self()
            if inst is not None:
                inst._sync_close_session()

        atexit.register(_cleanup)
        self._atexit_registered = True

    def _sync_close_session(self) -> None:
        session = self._aiohttp_session
        if session is None or getattr(session, "closed", True):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close())
            return
        if loop.is_closed():
            asyncio.run(self.close())
            return
        if loop.is_running():
            loop.create_task(self.close())
        else:
            loop.run_until_complete(self.close())

    def _select_settlement_quote(self, requested: str) -> str:
        if self._market_type != "usdm_perp":
            return requested
        if requested in {"USDT", "BUSD"}:
            return requested
        return "USDT"

    def normalize_symbol(self, symbol: str) -> str:
        """Map requested symbols to the canonical futures quote."""

        sym = (symbol or "").upper()
        if not sym:
            return sym
        cached = self._symbol_aliases.get(sym)
        if cached:
            return cached
        if self._market_type != "usdm_perp":
            self._symbol_aliases[sym] = sym
            return sym
        for suffix in _STABLE_QUOTE_SUFFIXES:
            if sym.endswith(suffix) and suffix != self._quote:
                base = sym[: -len(suffix)]
                if base:
                    canonical = f"{base}{self._quote}"
                    self._symbol_aliases[sym] = canonical
                    return canonical
        self._symbol_aliases[sym] = sym
        return sym

    async def _session(self):  # returns aiohttp.ClientSession
        import aiohttp
        ses = self._aiohttp_session
        if ses is None or getattr(ses, "closed", True):
            ses = aiohttp.ClientSession()
            self._aiohttp_session = ses
        return ses

    def _ts(self) -> int:
        # Use server time offset if known
        return int(self._time.time() * 1000) + int(self._time_offset_ms)

    async def _server_time(self) -> Optional[int]:
        """Fetch server time for selected futures API (no auth required)."""
        ses = await self._session()
        path = "/fapi/v1/time" if self._market_type == "usdm_perp" else "/dapi/v1/time"
        url = f"{self._base_url}{path}"
        try:
            async with ses.get(url, timeout=10) as resp:
                data = await resp.json()
                st = data.get("serverTime") if isinstance(data, dict) else None
                return int(st) if st is not None else None
        except Exception:  # pragma: no cover - best effort
            return None

    async def _sync_time(self) -> None:
        st = await self._server_time()
        if st is None:
            return
        local_ms = int(self._time.time() * 1000)
        self._time_offset_ms = int(st) - local_ms

    def _sign(self, params: Mapping[str, Any]) -> str:
        query = self._urlencode(params, doseq=True)
        return self._hmac.new(self._api_secret, query.encode("utf-8"), self._hashlib.sha256).hexdigest()

    async def _signed_request(self, method: str, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        ses = await self._session()
        base_params: Dict[str, Any] = {"timestamp": self._ts(), "recvWindow": self._recv_window}
        if params:
            base_params.update(params)
        sig = self._sign(base_params)
        headers = {"X-MBX-APIKEY": self._api_key}
        url = f"{self._base_url}{path}?{self._urlencode(base_params)}&signature={sig}"

        async def _do_request() -> Any:
            async with ses.request(method.upper(), url, headers=headers, timeout=20) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    # Try parse error body for Binance error codes
                    try:
                        import json as _json
                        payload = _json.loads(text)
                    except Exception:
                        payload = {"msg": text}
                    # Raise with more context
                    from aiohttp import ClientResponseError
                    raise ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=str(payload.get("msg") or payload),
                        headers=resp.headers,
                    )
                try:
                    return await resp.json()
                except Exception:
                    return text

        try:
            return await _do_request()
        except Exception as exc:
            # If timestamp issue, resync time and retry once
            msg = str(exc)
            if "-1021" in msg or "Timestamp" in msg or "recvWindow" in msg:
                await self._sync_time()
                # Rebuild URL with new timestamp
                base_params.update({"timestamp": self._ts()})
                sig2 = self._sign(base_params)
                url = f"{self._base_url}{path}?{self._urlencode(base_params)}&signature={sig2}"
                try:
                    return await _do_request()
                except Exception:
                    raise
            raise

    async def _configure_symbol(self, symbol: str) -> None:
        # Always set leverage (not just on first config)
        # This ensures new positions get correct leverage even if symbol was used before
        
        # Fetch and cache symbol filters for precision rounding
        if symbol not in self._symbol_filters:
            try:
                ses = await self._session()
                path = "/fapi/v1/exchangeInfo" if self._market_type == "usdm_perp" else "/dapi/v1/exchangeInfo"
                url = f"{self._base_url}{path}"
                async with ses.get(url, timeout=10) as resp:
                    info = await resp.json()
                    for sym_info in info.get("symbols", []):
                        if sym_info["symbol"] == symbol:
                            filters = {f["filterType"]: f for f in sym_info.get("filters", [])}
                            self._symbol_filters[symbol] = filters
                            break
            except Exception as exc:
                logger.warning("Failed to fetch exchange info for %s: %s", symbol, exc)
        
        # Always set leverage for autonomous trading (not cached)
        margin_type = "CROSSED" if self._margin_mode == "cross" else "ISOLATED"
        try:
            env_leverage = os.getenv("QT_DEFAULT_LEVERAGE")
            leverage = max(1, min(int(env_leverage or self._default_leverage or 30), 125))
            await self._signed_request("POST", "/fapi/v1/leverage" if self._market_type == "usdm_perp" else "/dapi/v1/leverage", {  # type: ignore[dict-item]
                "symbol": symbol,
                "leverage": leverage,
            })
            logger.info(f"✅ Set {leverage}x leverage for {symbol}")
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("Failed to set leverage for %s: %s", symbol, exc)
        
        # Set margin type only once (can fail if already set)
        if symbol not in self._configured_symbols:
            try:
                await self._signed_request(
                    "POST",
                    "/fapi/v1/marginType" if self._market_type == "usdm_perp" else "/dapi/v1/marginType",
                    {"symbol": symbol, "marginType": margin_type},
                )
            except Exception:  # pragma: no cover - best-effort
                logger.warning("Failed to set margin mode for %s", symbol)
            self._configured_symbols.add(symbol)

    async def get_positions(self) -> Mapping[str, float]:
        # Use account endpoint to map positions to symbol -> quantity
        try:
            data = await self._signed_request("GET", "/fapi/v2/account" if self._market_type == "usdm_perp" else "/dapi/v2/account")
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Futures account snapshot failed: %s", exc)
            return {}
        result: Dict[str, float] = {}
        for p in data.get("positions", []) if isinstance(data, dict) else []:
            sym = str(p.get("symbol", "")).upper()
            if not sym:
                continue
            try:
                amt = float(p.get("positionAmt", 0.0))
            except (TypeError, ValueError):
                continue
            if abs(amt) <= 0.0:
                continue
            result[sym] = amt
        return result

    async def get_cash_balance(self) -> float:
        try:
            data = await self._signed_request("GET", "/fapi/v2/account" if self._market_type == "usdm_perp" else "/dapi/v2/account")
        except Exception as exc:  # pragma: no cover
            logger.warning("Futures account balance failed: %s", exc)
            return 0.0
        
        # OPTIMIZATION: Sum ALL available balances (USDT, BNFCR, BFUSD, etc.)
        # Many users have balance split across multiple collateral assets
        total_available = 0.0
        for a in data.get("assets", []) if isinstance(data, dict) else []:
            try:
                available = float(a.get("availableBalance", 0.0))
                if available > 0:
                    total_available += available
            except (TypeError, ValueError):
                continue
        
        return total_available

    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker price for a symbol."""
        try:
            path = "/fapi/v1/ticker/price" if self._market_type == "usdm_perp" else "/dapi/v1/ticker/price"
            data = await self._signed_request("GET", path, {"symbol": symbol})
            if isinstance(data, dict) and "price" in data:
                return {"last": float(data["price"])}
            return {}
        except Exception as exc:
            logger.warning("Failed to get ticker for %s: %s", symbol, exc)
            return {}

    def _round_quantity(self, symbol: str, quantity: float) -> str:
        """Round quantity to the LOT_SIZE filter's step size and format to correct precision."""
        filters = self._symbol_filters.get(symbol, {})
        lot_size = filters.get("LOT_SIZE", {})
        step_str = str(lot_size.get("stepSize", "1.0"))
        step = float(step_str)
        if step <= 0:
            step = 1.0
            step_str = "1.0"
        # Round to nearest step
        import math
        rounded = math.floor(quantity / step) * step
        rounded = max(rounded, step)
        # Determine precision from step size (e.g., '0.1' -> 1 decimal place)
        if '.' in step_str:
            decimals = len(step_str.split('.')[1].rstrip('0'))
        else:
            decimals = 0
        # Format to the correct decimal places to avoid precision errors
        return f"{rounded:.{decimals}f}"

    async def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> str:
        # Dry-run in staging mode
        if (os.getenv("STAGING_MODE") or "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("[DRY-RUN] Futures order %s %s qty=%s price=%s", side.upper(), symbol, quantity, price)
            return f"dryrun-{symbol}-{int(self._time.time())}"
        await self._configure_symbol(symbol)
        # Round quantity to symbol's LOT_SIZE
        rounded_qty = self._round_quantity(symbol, quantity)
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": rounded_qty,
        }
        data = await self._signed_request("POST", "/fapi/v1/order" if self._market_type == "usdm_perp" else "/dapi/v1/order", params)
        return str(data.get("orderId") or data.get("clientOrderId") or "")

    async def close(self) -> None:
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()


# --- Simple trade state store for exits (JSON-backed) ---

class TradeStateStore:
    """Persist minimal per-symbol trade state to support TP/SL/trailing exits.

    Schema per symbol:
        {
            "side": "LONG" | "SHORT",
            "qty": float,                 # open quantity (abs)
            "avg_entry": float,           # average entry price
            "peak": float | None,         # highest price since entry (LONG)
            "trough": float | None,       # lowest price since entry (SHORT)
            "opened_at": str              # ISO timestamp
        }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8") or "{}")
            except Exception:
                self._data = {}
        self._loaded = True

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get(self, symbol: str) -> Optional[dict]:
        self._load()
        return self._data.get(symbol.upper())

    def set(self, symbol: str, state: dict) -> None:
        self._load()
        self._data[symbol.upper()] = state
        self._save()

    def delete(self, symbol: str) -> None:
        self._load()
        self._data.pop(symbol.upper(), None)
        self._save()

    def move(self, old_symbol: str, new_symbol: str) -> None:
        """Rename a stored state without losing history."""

        self._load()
        old_key = old_symbol.upper()
        new_key = new_symbol.upper()
        if old_key == new_key:
            return
        if old_key not in self._data:
            return
        if new_key in self._data:
            return
        self._data[new_key] = self._data.pop(old_key)
        self._save()

    def keys(self) -> List[str]:
        self._load()
        return list(self._data.keys())

    def update_on_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Update state given a new fill. Handles averaging and closures.

        For LONG: BUY increases qty and averages price; SELL decreases qty; zero qty deletes state.
        For SHORT: SELL increases qty and averages entry; BUY decreases qty.
        """
        from datetime import datetime, timezone
        self._load()
        sym = symbol.upper()
        side_u = side.upper()
        qty = float(qty)
        price = float(price)
        state = self._data.get(sym)

        def _new(side_name: str, q: float, p: float) -> dict:
            now = datetime.now(timezone.utc).isoformat()
            return {"side": side_name, "qty": float(q), "avg_entry": float(p), "peak": p if side_name == "LONG" else None, "trough": p if side_name == "SHORT" else None, "opened_at": now}

        if state is None:
            if side_u == "BUY":
                state = _new("LONG", qty, price)
            else:
                state = _new("SHORT", qty, price)
            self._data[sym] = state
            self._save()
            return

        # Existing state present
        cur_side = str(state.get("side", "LONG")).upper()
        cur_qty = float(state.get("qty", 0.0))
        cur_entry = float(state.get("avg_entry", price))

        if side_u == "BUY":
            if cur_side == "LONG":
                # increase long position (average up/down)
                new_qty = cur_qty + qty
                new_entry = (cur_entry * cur_qty + price * qty) / max(new_qty, 1e-9)
                state.update({"qty": new_qty, "avg_entry": new_entry, "peak": max(float(state.get("peak", price)), price)})
            else:  # SHORT reduced
                new_qty = max(cur_qty - qty, 0.0)
                state.update({"qty": new_qty})
                if new_qty <= 1e-12:
                    self._data.pop(sym, None)
                    self._save()
                    return
        else:  # SELL
            if cur_side == "LONG":
                new_qty = max(cur_qty - qty, 0.0)
                state.update({"qty": new_qty})
                if new_qty <= 1e-12:
                    self._data.pop(sym, None)
                    self._save()
                    return
            else:  # SHORT add
                new_qty = cur_qty + qty
                new_entry = (cur_entry * cur_qty + price * qty) / max(new_qty, 1e-9)
                state.update({"qty": new_qty, "avg_entry": new_entry, "trough": min(float(state.get("trough", price)), price)})

        # Update peak/trough based on latest price
        if state.get("side") == "LONG":
            state["peak"] = max(float(state.get("peak", price)), price)
        else:
            state["trough"] = min(float(state.get("trough", price)), price)
        self._data[sym] = state
        self._save()

def build_execution_adapter(config: ExecutionConfig) -> ExchangeAdapter:
    """Select an exchange adapter based on execution configuration."""

    exchange = (config.exchange or "paper").lower()
    if exchange in {"binance-futures", "binance_futures", "binancefutures"}:
        # Build futures adapter using liquidity config for market type/leverage
        cfg = load_config()
        liq = load_liquidity_config()
        api_key = getattr(cfg, "binance_api_key", None)
        api_secret = getattr(cfg, "binance_api_secret", None)
        adapter = BinanceFuturesExecutionAdapter(
            api_key=api_key,
            api_secret=api_secret,
            quote_asset=config.quote_asset,
            market_type=getattr(liq, "market_type", "usdm_perp"),
            default_leverage=getattr(liq, "default_leverage", 30),
            margin_mode=getattr(liq, "margin_mode", "cross"),
        )
        if adapter.ready:
            return adapter
        logger.warning("Binance Futures adapter unavailable; falling back to paper mode")
    if exchange == "binance":
        cfg = load_config()
        api_key = getattr(cfg, "binance_api_key", None)
        api_secret = getattr(cfg, "binance_api_secret", None)
        adapter = BinanceExecutionAdapter(
            api_key=api_key,
            api_secret=api_secret,
            quote_asset=config.quote_asset,
            testnet=config.binance_testnet,
        )
        if adapter.ready:
            return adapter
        logger.warning("Binance execution adapter unavailable; falling back to paper mode")
    return PaperExchangeAdapter()


# --- Exit rules config helpers ---

_DEFAULT_FORCE_EXIT_SL = 0.03   # 3% hard stop
_DEFAULT_FORCE_EXIT_TP = 0.05   # 5% take profit
_DEFAULT_FORCE_EXIT_TRAIL = 0.02  # 2% give-back from peak/trough
_DEFAULT_FORCE_EXIT_PARTIAL = 0.0  # take full size on TP unless overridden


def _exit_config_from_env() -> Dict[str, Optional[float]]:
    def _get_float(name: str, fallback: float) -> float:
        v = (os.getenv(name) or "").strip()
        if not v:
            return fallback
        try:
            return float(v)
        except Exception:
            return fallback

    def _env_flag(name: str, *, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on"}:
            return True
        if normalised in {"0", "false", "no", "off"}:
            return False
        return default

    enabled = _env_flag("QT_FORCE_EXITS_ENABLED", default=True)
    if not enabled:
        return {"sl_pct": None, "tp_pct": None, "trail_pct": None, "partial_tp": None}

    sl_default = _get_float("QT_SL_PCT", _DEFAULT_FORCE_EXIT_SL)
    tp_default = _get_float("QT_TP_PCT", _DEFAULT_FORCE_EXIT_TP)
    trail_default = _get_float("QT_TRAIL_PCT", _DEFAULT_FORCE_EXIT_TRAIL)
    partial_default = _get_float("QT_PARTIAL_TP", _DEFAULT_FORCE_EXIT_PARTIAL)

    return {
        "sl_pct": _get_float("QT_SL_PCT", sl_default),
        "tp_pct": _get_float("QT_TP_PCT", tp_default),
        "trail_pct": _get_float("QT_TRAIL_PCT", trail_default),
        "partial_tp": _get_float("QT_PARTIAL_TP", partial_default),
    }


def _evaluate_forced_exits(
    *,
    prices: Mapping[str, float],
    positions: Mapping[str, float],
    min_notional: float,
    store: TradeStateStore,
    ai_signals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[OrderIntent]:
    """Create forced exit intents based on AI-driven dynamic TP/SL levels.
    
    🎯 HYBRID AI + TP/SL SYSTEM:
    - Uses AI-generated TP/SL levels if available (stored in trade state)
    - Falls back to static config if no AI levels
    - AI dynamically adjusts based on confidence, volatility, momentum

    Returns a list of OrderIntent that should take precedence over allocation-based intents.
    """
    cfg = _exit_config_from_env()
    default_sl = cfg["sl_pct"]
    default_tp = cfg["tp_pct"]
    default_trail = cfg["trail_pct"]
    default_partial = cfg["partial_tp"] or 0.0

    logger.info(f"🔍 FORCED EXIT EVAL: positions={len(positions)}, prices={len(prices)}")
    logger.info(f"🔍 Config: TP={default_tp}% SL={default_sl}% Trail={default_trail}% min_notional={min_notional}")

    if default_sl is None and default_tp is None and default_trail is None:
        logger.warning("⚠️ All TP/SL/Trail are None - forced exits disabled!")
        return []

    intents: List[OrderIntent] = []
    ai_signal_map = ai_signals or {}

    for sym, qty in positions.items():
        price = float(prices.get(sym.upper(), 0.0))
        logger.info(f"🔍 Position {sym}: qty={qty}, price={price}")
        
        if price <= 0:
            logger.warning(f"⚠️ Skipping {sym}: price={price} <= 0")
            continue
        pos_qty = float(qty)
        if abs(pos_qty) <= 0:
            logger.warning(f"⚠️ Skipping {sym}: qty={pos_qty} == 0")
            continue
        
        # Get position state - MUST exist for TP/SL calculation
        state = store.get(sym)
        if state is None:
            logger.warning(f"⚠️ {sym} has no entry state - cannot calculate P&L for forced exits. Position may have been opened before state tracking started.")
            continue
        
        # Use state's side (set correctly by update_on_fill)
        side = str(state.get("side", "LONG")).upper()
        logger.info(f"🔍 {sym} state: side={side}, qty={state.get('qty')}, entry={state.get('avg_entry')}, current={price}")

        # Get entry tracking values from state
        avg_entry = float(state.get("avg_entry", price))
        open_qty = float(state.get("qty", abs(pos_qty)))
        peak = float(state.get("peak", price)) if side == "LONG" else None
        trough = float(state.get("trough", price)) if side == "SHORT" else None
        
        # Calculate actual P&L percentage
        if side == "LONG":
            pnl_pct = ((price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
        else:  # SHORT
            pnl_pct = ((avg_entry - price) / avg_entry) * 100 if avg_entry > 0 else 0
        
        logger.info(f"💰 {sym} {side} P&L: {pnl_pct:+.2f}% (entry={avg_entry:.4f}, current={price:.4f})")

        # 🎯 AI-DRIVEN HYBRID: Use AI-generated TP/SL if available
        # Check trade state for AI-set levels first
        sl = float(state.get("ai_sl_pct", default_sl)) if state.get("ai_sl_pct") else default_sl
        tp = float(state.get("ai_tp_pct", default_tp)) if state.get("ai_tp_pct") else default_tp
        trail = float(state.get("ai_trail_pct", default_trail)) if state.get("ai_trail_pct") else default_trail
        partial = float(state.get("ai_partial_tp", default_partial)) if state.get("ai_partial_tp") else default_partial
        
        ai_source = "AI" if state.get("ai_sl_pct") or state.get("ai_tp_pct") else "static"
        logger.info(f"🎯 {sym} using {ai_source} TP/SL: TP={tp*100 if tp else 0:.1f}%, SL={sl*100 if sl else 0:.1f}%, Trail={trail*100 if trail else 0:.1f}%")

        # Update peak/trough with latest price in-memory (persist on next fill)
        if side == "LONG":
            peak = max(peak or price, price)
        else:
            trough = min(trough or price, price)

        exit_size_qty: float = 0.0
        exit_reason: str = ""
        exit_side: str = "SELL" if side == "LONG" else "BUY"

        # 1) Hard stop loss
        if sl is not None and sl > 0:
            if side == "LONG" and price <= avg_entry * (1.0 - sl):
                exit_size_qty = open_qty
                exit_reason = f"{ai_source}-SL {sl*100:.2f}%"
                logger.info(f"💰 {sym} LONG SL triggered: {price} <= {avg_entry * (1.0 - sl):.4f}")
            elif side == "SHORT" and price >= avg_entry * (1.0 + sl):
                exit_size_qty = open_qty
                exit_reason = f"{ai_source}-SL {sl*100:.2f}%"
                logger.info(f"💰 {sym} SHORT SL triggered: {price} >= {avg_entry * (1.0 + sl):.4f}")

        # 2) Take profit (partial if configured)
        if exit_size_qty <= 0 and tp is not None and tp > 0:
            if side == "LONG" and price >= avg_entry * (1.0 + tp):
                exit_size_qty = open_qty * (partial if 0.0 < partial < 1.0 else 1.0)
                exit_reason = f"{ai_source}-TP {tp*100:.2f}%{' partial' if 0.0 < partial < 1.0 else ''}"
                logger.info(f"💰 {sym} LONG TP triggered: {price:.4f} >= {avg_entry * (1.0 + tp):.4f} (P&L: {pnl_pct:+.2f}%)")
            elif side == "SHORT" and price <= avg_entry * (1.0 - tp):
                exit_size_qty = open_qty * (partial if 0.0 < partial < 1.0 else 1.0)
                exit_reason = f"{ai_source}-TP {tp*100:.2f}%{' partial' if 0.0 < partial < 1.0 else ''}"
                logger.info(f"💰 {sym} SHORT TP triggered: {price:.4f} <= {avg_entry * (1.0 - tp):.4f} (P&L: {pnl_pct:+.2f}%)")
            else:
                logger.info(f"🔍 {sym} {side} TP not met: P&L {pnl_pct:+.2f}% < TP {tp*100:.1f}%")

        # 3) Trailing stop
        if exit_size_qty <= 0 and trail is not None and trail > 0:
            if side == "LONG" and peak is not None and price <= peak * (1.0 - trail):
                exit_size_qty = open_qty
                exit_reason = f"{ai_source}-TRAIL {trail*100:.2f}% from peak"
                logger.info(f"💰 {sym} LONG TRAIL triggered: {price} <= {peak * (1.0 - trail):.4f}")
            elif side == "SHORT" and trough is not None and price >= trough * (1.0 + trail):
                exit_size_qty = open_qty
                exit_reason = f"{ai_source}-TRAIL {trail*100:.2f}% from trough"
                logger.info(f"💰 {sym} SHORT TRAIL triggered: {price} >= {trough * (1.0 + trail):.4f}")

        if exit_size_qty <= 0:
            logger.info(f"🔍 {sym} no exit triggered")
            continue

        notional = abs(exit_size_qty) * price
        if notional < float(min_notional):
            logger.warning(f"⚠️ {sym} exit skipped: notional {notional:.2f} < min {min_notional}")
            continue

        logger.info(f"✅ {sym} exit intent created: {exit_reason}")
        intents.append(
            OrderIntent(
                symbol=sym.upper(),
                side=exit_side,
                target_weight=0.0,
                quantity=float(abs(exit_size_qty)),
                price=price,
                notional=notional,
                reason=f"FORCED_EXIT {exit_reason} entry={avg_entry:.4f} price={price:.4f}",
            )
        )
        try:
            logger.info(
                "🔔 Forced exit planned: %s %s qty=%.6f @ %.6f (%s)",
                exit_side,
                sym.upper(),
                float(abs(exit_size_qty)),
                price,
                exit_reason,
            )
        except Exception:
            pass

    # Sort largest first to exit biggest exposures first
    intents.sort(key=lambda i: i.notional, reverse=True)
    logger.info(f"🔍 FORCED EXIT EVAL COMPLETE: generated {len(intents)} exit intents")
    return intents


def get_latest_portfolio(db: Session) -> Optional[Tuple[LiquidityRun, List[PortfolioAllocation], Dict[str, float]]]:
    run: LiquidityRun | None = (
        db.query(LiquidityRun).order_by(LiquidityRun.fetched_at.desc()).first()
    )
    if run is None or run.selection_size == 0:
        return None

    allocations = (
        db.query(PortfolioAllocation)
        .filter(PortfolioAllocation.run_id == run.id)
        .order_by(PortfolioAllocation.weight.desc())
        .all()
    )
    if not allocations:
        return None

    snapshots = db.query(LiquiditySnapshot).filter(LiquiditySnapshot.run_id == run.id).all()
    prices = {snapshot.symbol.upper(): float(snapshot.price or 0.0) for snapshot in snapshots}
    return run, allocations, prices


def _effective_equity(total_equity: float, config: ExecutionConfig) -> float:
    effective = max(total_equity - config.cash_buffer, 0.0)
    return effective


def compute_ai_signal_orders(
    ai_signals: List[Dict[str, Any]],
    prices: Mapping[str, float],
    *,
    positions: Mapping[str, float],
    total_equity: float,
    config: ExecutionConfig,
) -> List[OrderIntent]:
    """
    Convert AI signals directly to order intents for event-driven trading.
    This bypasses portfolio allocations and trades purely on AI recommendations.
    """
    effective_equity = _effective_equity(total_equity, config)
    intents: List[OrderIntent] = []
    position_map = {symbol.upper(): float(qty) for symbol, qty in positions.items()}
    
    # Filter for actionable signals (BUY/SELL with high confidence)
    for signal in ai_signals:
        symbol = signal.get("symbol", "").upper()
        action = signal.get("action", "HOLD")
        confidence = abs(float(signal.get("confidence", 0.0)))
        
        logger.info(f"🔍 Processing AI signal: {symbol} {action} conf={confidence:.2f}")
        
        if action == "HOLD":
            logger.info(f"⏭️ Skipping {symbol}: action is HOLD")
            continue
            
        price = float(prices.get(symbol, 0.0))
        if price <= 0:
            logger.info(f"⏭️ Skipping {symbol}: no valid price (price={price})")
            continue
        
        current_qty = position_map.get(symbol, 0.0)
        logger.info(f"📊 {symbol}: current_qty={current_qty}, price={price}")
        
        # For event-driven mode, use fixed position size per signal
        # Each trade should be optimized for 30x leverage
        size_multiplier = signal.get("size_multiplier", 1.0)
        
        # AGGRESSIVE: FAST $600 per position for better profit potential
        # With 30x leverage: $600 margin = $18,000 position size
        # NO longer multiplied by confidence - we use fixed size!
        base_notional = 600.0  # FAST $600 per trade
        target_notional = base_notional * size_multiplier  # Only apply size_multiplier, NOT confidence
        
        # Cap at max configured (QT_MAX_NOTIONAL_PER_TRADE)
        max_per_trade = float(os.getenv("QT_MAX_NOTIONAL_PER_TRADE", "5000.0"))
        target_notional = min(target_notional, max_per_trade)
        
        logger.info(f"💰 {symbol} sizing: base=${base_notional:.0f}, conf={confidence:.2f}, mult={size_multiplier:.2f}, target=${target_notional:.2f}")
        
        # Determine side based on AI action and current position
        if action == "BUY":
            # Open LONG or add to existing LONG
            side = "BUY"
            current_notional = max(current_qty, 0.0) * price
            delta = target_notional - current_notional
            logger.info(f"💰 {symbol} BUY: target={target_notional:.2f}, current={current_notional:.2f}, delta={delta:.2f}, min={config.min_notional}")
            if delta < config.min_notional:
                logger.info(f"⏭️ Skipping {symbol} BUY: delta {delta:.2f} < min {config.min_notional}")
                continue
            quantity_units = delta / price
            
        elif action == "SELL":
            # Open SHORT or add to existing SHORT
            side = "SELL"
            current_notional = abs(min(current_qty, 0.0)) * price
            delta = target_notional - current_notional
            logger.info(f"💰 {symbol} SELL: target={target_notional:.2f}, current={current_notional:.2f}, delta={delta:.2f}, min={config.min_notional}")
            if delta < config.min_notional:
                logger.info(f"⏭️ Skipping {symbol} SELL: delta {delta:.2f} < min {config.min_notional}")
                continue
            quantity_units = delta / price
        
        else:
            continue
        
        notional = abs(quantity_units) * price
        target_weight = target_notional / effective_equity if effective_equity > 0 else 0.0
        reason = f"AI: {action} conf={confidence:.2f} size_mult={size_multiplier:.2f}"
        
        logger.info(
            f"🎯 Creating order intent: {symbol} {side} | "
            f"confidence={confidence:.2f}, target_notional={target_notional:.2f}, "
            f"notional={notional:.2f}, qty={abs(quantity_units):.4f}, price={price:.2f}"
        )
        
        intents.append(
            OrderIntent(
                symbol=symbol,
                side=side,
                target_weight=target_weight,
                quantity=abs(quantity_units),
                price=price,
                notional=notional,
                reason=reason,
            )
        )
    
    # Sort by confidence * notional (prioritize high-confidence large positions)
    intents.sort(key=lambda item: item.notional, reverse=True)
    if config.max_orders and len(intents) > config.max_orders:
        intents = intents[: config.max_orders]
    
    return intents


def compute_target_orders(
    allocations: Sequence[PortfolioAllocation],
    prices: Mapping[str, float],
    *,
    positions: Mapping[str, float],
    total_equity: float,
    config: ExecutionConfig,
) -> List[OrderIntent]:
    effective_equity = _effective_equity(total_equity, config)
    intents: List[OrderIntent] = []
    position_map = {symbol.upper(): float(qty) for symbol, qty in positions.items()}

    for allocation in allocations:
        symbol = allocation.symbol.upper()
        price = float(prices.get(symbol, 0.0))
        if price <= 0:
            continue

        target_weight = max(float(allocation.weight), 0.0)
        target_notional = target_weight * effective_equity
        current_qty = position_map.get(symbol, 0.0)
        current_notional = current_qty * price
        delta = target_notional - current_notional

        side = "BUY" if delta > 0 else "SELL"
        quantity_units = delta / price
        if side == "SELL":
            holding_qty = abs(current_qty)
            if holding_qty <= 0:
                continue
            if config.allow_partial:
                quantity_units = -min(abs(quantity_units), holding_qty)
            else:
                quantity_units = -holding_qty

        notional = abs(quantity_units) * price
        if notional < config.min_notional:
            continue

        reason = f"target={target_weight:.4f}; delta={delta:.2f}; price={price:.4f}"
        intents.append(
            OrderIntent(
                symbol=symbol,
                side=side,
                target_weight=target_weight,
                quantity=abs(quantity_units),
                price=price,
                notional=notional,
                reason=reason,
            )
        )

    intents.sort(key=lambda item: item.notional, reverse=True)
    if config.max_orders and len(intents) > config.max_orders:
        intents = intents[: config.max_orders]
    return intents


def _build_risk_guard() -> RiskGuardService:
    risk_config = load_risk_config()
    store = None
    if risk_config.risk_state_db_path:
        store = SqliteRiskStateStore(risk_config.risk_state_db_path)
    return RiskGuardService(risk_config, store=store)


async def run_portfolio_rebalance(
    db: Session,
    *,
    adapter: Optional[ExchangeAdapter] = None,
    execution_config: Optional[ExecutionConfig] = None,
    liquidity_config: Optional[LiquidityConfig] = None,
    risk_guard: Optional[RiskGuardService] = None,
) -> Dict[str, Any]:
    execution_config = execution_config or load_execution_config()
    liquidity_config = liquidity_config or load_liquidity_config()
    adapter = adapter or build_execution_adapter(execution_config)
    # Build local trade state store for exits
    trade_state_path = os.getenv("QT_TRADE_STATE_DB") or str(Path("backend/data/trade_state.json"))
    trade_store = TradeStateStore(trade_state_path)
    portfolio = get_latest_portfolio(db)
    if portfolio is None:
        return {"status": "no_portfolio"}

    run, allocations, prices = portfolio
    position_service = PortfolioPositionService(db)

    price_map = {symbol.upper(): float(price) for symbol, price in prices.items()}
    symbol_normalizer = getattr(adapter, "normalize_symbol", None)
    if callable(symbol_normalizer):
        for key, price in list(price_map.items()):
            canonical = symbol_normalizer(key)
            if canonical != key and canonical not in price_map:
                price_map[canonical] = price
        for key in trade_store.keys():
            canonical = symbol_normalizer(key)
            if canonical != key:
                trade_store.move(key, canonical)

    raw_positions = await adapter.get_positions()
    positions = {symbol.upper(): float(qty) for symbol, qty in raw_positions.items()}
    cash = await adapter.get_cash_balance()
    total_equity = cash + sum(abs(qty) * price_map.get(symbol, 0.0) for symbol, qty in positions.items())
    total_equity = max(total_equity, 0.0)
    symbol_exposure = {symbol: abs(qty) * price_map.get(symbol, 0.0) for symbol, qty in positions.items()}
    gross_exposure = sum(symbol_exposure.values())

    # Fetch AI trading signals early so they can be used for both forced exits AND position sizing
    ai_signal_map = {}
    try:
        from ai_engine.agents.xgb_agent import make_default_agent
        agent = make_default_agent()
        ai_engine = create_ai_trading_engine(agent=agent, db_session=db)
        # FUTURES OPTIMIZATION: Only analyze allowed symbols to avoid timeout
        # Use allowed_symbols from risk config instead of all allocations
        from backend.config.risk import load_risk_config
        risk_cfg = load_risk_config()
        allowed = risk_cfg.allowed_symbols
        position_symbols = list(positions.keys())
        # Only use allowed symbols + existing positions
        all_symbols = list(set(position_symbols + allowed)) if allowed else list(set(position_symbols + [alloc.symbol.upper() for alloc in allocations]))
        logger.info(f"🎯 Analyzing {len(all_symbols)} symbols for AI signals (allowed: {len(allowed) if allowed else 'all'})")
        ai_signals = await ai_engine.get_trading_signals(all_symbols, positions)
        ai_signal_map = {sig["symbol"]: sig for sig in ai_signals}
        logger.info(f"🧠 AI engine generated {len(ai_signals)} signals: BUY={sum(1 for s in ai_signals if s.get('action')=='BUY')} SELL={sum(1 for s in ai_signals if s.get('action')=='SELL')} HOLD={sum(1 for s in ai_signals if s.get('action')=='HOLD')}")
    except Exception as exc:
        logger.warning(f"AI engine failed during signal fetch: {exc}", exc_info=True)

    # Before computing target intents, evaluate any forced exits (TP/SL/TRAIL)
    # Now includes AI signals so exits can use AI-generated TP/SL levels
    forced_exits = _evaluate_forced_exits(
        prices=price_map,
        positions=positions,
        min_notional=execution_config.min_notional,
        store=trade_store,
        ai_signals=ai_signal_map,
    )

    # EVENT-DRIVEN MODE: If we have strong AI signals, trade directly on them
    # instead of relying on portfolio allocations
    # Read confidence threshold from environment (default 0.50)
    confidence_threshold = float(os.getenv("QT_CONFIDENCE_THRESHOLD", "0.72"))
    logger.info(f"🎯 Filtering AI signals: threshold={confidence_threshold:.2f}, total_signals={len(ai_signal_map)}")
    strong_ai_signals = [
        sig for sig in (ai_signal_map.values() if ai_signal_map else [])
        if sig.get("action") in ("BUY", "SELL") and abs(float(sig.get("confidence", 0.0))) >= confidence_threshold
    ]
    logger.info(f"🎯 Strong AI signals after filtering: {len(strong_ai_signals)} (BUY/SELL with conf>={confidence_threshold:.2f})")
    
    if strong_ai_signals:
        logger.info(f"🤖 Event-driven mode: Trading on {len(strong_ai_signals)} strong AI signals (threshold={confidence_threshold:.2f}, bypassing portfolio allocations)")
        intents = compute_ai_signal_orders(
            strong_ai_signals,
            prices,
            positions=positions,
            total_equity=total_equity,
            config=execution_config,
        )
        logger.info(f"📋 AI signals generated {len(intents)} order intents")
    else:
        # Fallback to traditional portfolio-based trading
        logger.info(f"📊 Standard mode: Using portfolio allocations ({len(allocations)} symbols)")
        intents = compute_target_orders(
            allocations,
            prices,
            positions=positions,
            total_equity=total_equity,
            config=execution_config,
        )

    # Apply AI trading signals to adjust position sizes (already fetched earlier)
    if ai_signal_map and not strong_ai_signals:
        # Only adjust if we're in standard mode (not event-driven)
        try:
            # Adjust intents based on AI recommendations
            adjusted_intents = []
            for intent in intents:
                ai_sig = ai_signal_map.get(intent.symbol)
                if ai_sig:
                    action = ai_sig["action"]
                    size_mult = ai_sig.get("size_multiplier", 1.0)
                    confidence = ai_sig.get("confidence", 0.5)
                    
                    # Skip if AI strongly recommends opposite action
                    if action == "SELL" and intent.side.upper() == "BUY" and confidence > 0.7:
                        logger.info(f"AI skipping BUY intent for {intent.symbol}: AI={action} confidence={confidence:.2f}")
                        continue
                    if action == "BUY" and intent.side.upper() == "SELL" and confidence > 0.7:
                        logger.info(f"AI skipping SELL intent for {intent.symbol}: AI={action} confidence={confidence:.2f}")
                        continue
                    
                    # Adjust quantity based on AI size multiplier
                    if action in ("BUY", "SELL"):
                        adjusted_qty = intent.quantity * size_mult
                        adjusted_notional = intent.notional * size_mult
                        adjusted_intent = OrderIntent(
                            symbol=intent.symbol,
                            side=intent.side,
                            target_weight=intent.target_weight * size_mult,
                            quantity=adjusted_qty,
                            price=intent.price,
                            notional=adjusted_notional,
                            reason=f"AI={action}({confidence:.2f})x{size_mult:.2f}; {intent.reason}"
                        )
                        adjusted_intents.append(adjusted_intent)
                        logger.info(f"AI adjusted {intent.symbol}: qty={intent.quantity:.4f}->{adjusted_qty:.4f}, mult={size_mult:.2f}")
                    else:
                        # HOLD - keep original intent but note AI signal
                        adjusted_intent = OrderIntent(
                            symbol=intent.symbol,
                            side=intent.side,
                            target_weight=intent.target_weight,
                            quantity=intent.quantity,
                            price=intent.price,
                            notional=intent.notional,
                            reason=f"AI=HOLD({confidence:.2f}); {intent.reason}"
                        )
                        adjusted_intents.append(adjusted_intent)
                else:
                    # No AI signal, keep original
                    adjusted_intents.append(intent)
            
            intents = adjusted_intents
            logger.info(f"AI engine adjusted {len(intents)} order intents")
        except Exception as exc:
            logger.warning(f"AI intent adjustment failed: {exc}", exc_info=True)

    def _normalize_intent_symbols(items: List[OrderIntent]) -> List[OrderIntent]:
        if not callable(symbol_normalizer):
            return items
        normalized: List[OrderIntent] = []
        for entry in items:
            canonical = symbol_normalizer(entry.symbol)
            if canonical != entry.symbol:
                price_map.setdefault(canonical, price_map.get(entry.symbol, entry.price))
                trade_store.move(entry.symbol, canonical)
                entry = replace(entry, symbol=canonical)
            normalized.append(entry)
        return normalized

    forced_exits = _normalize_intent_symbols(forced_exits)
    intents = _normalize_intent_symbols(intents)

    # Prepend forced exit intents and remove conflicting intents for the same symbols
    if forced_exits:
        forced_symbols = {i.symbol for i in forced_exits}
        keep: List[OrderIntent] = []
        for intent in intents:
            if intent.symbol in forced_symbols:
                continue
            keep.append(intent)
        intents = forced_exits + keep
    
    # ⚠️ ENFORCE MAX POSITIONS LIMIT
    max_positions = int(os.getenv("QT_MAX_POSITIONS", "0"))
    if max_positions > 0:
        # Count current open positions (excluding those being closed by forced exits)
        forced_exit_symbols = {i.symbol for i in forced_exits}
        current_open = {sym for sym, qty in positions.items() if abs(qty) > 0 and sym not in forced_exit_symbols}
        current_count = len(current_open)
        
        # Count how many NEW positions we're trying to open (BUY when no position, SELL SHORT when no position)
        new_opens = []
        for intent in intents:
            if intent.symbol in forced_exit_symbols:
                continue  # Skip exits
            existing_qty = positions.get(intent.symbol, 0.0)
            # New position: going from 0 to long/short, or flipping direction
            is_new_open = (abs(existing_qty) < 1e-9) or (
                (existing_qty > 0 and intent.side == "SELL") or
                (existing_qty < 0 and intent.side == "BUY")
            )
            if is_new_open and intent.symbol not in current_open:
                new_opens.append(intent)
        
        available_slots = max(0, max_positions - current_count)
        logger.info(f"⚠️ Position limit: {current_count}/{max_positions} open, {len(new_opens)} new orders planned, {available_slots} slots available")
        
        if len(new_opens) > available_slots:
            # Keep exits and position adjustments, limit new opens by largest notional
            new_opens_sorted = sorted(new_opens, key=lambda x: x.notional, reverse=True)
            allowed_new = set(i.symbol for i in new_opens_sorted[:available_slots])
            
            filtered_intents = []
            for intent in intents:
                if intent in forced_exits:
                    filtered_intents.append(intent)  # Always allow exits
                elif intent in new_opens:
                    if intent.symbol in allowed_new:
                        filtered_intents.append(intent)
                    else:
                        logger.warning(f"⚠️ Skipping {intent.symbol} - exceeds max {max_positions} positions limit")
                else:
                    filtered_intents.append(intent)  # Keep adjustments to existing positions
            
            intents = filtered_intents
            logger.info(f"⚠️ Limited to {available_slots} new positions (total after: {current_count + available_slots}/{max_positions})")

    risk_guard = risk_guard or _build_risk_guard()

    summary = {
        "status": "ok",
        "run_id": run.id,
        "orders_planned": len(intents),
        "orders_submitted": 0,
        "orders_skipped": 0,
        "orders_failed": 0,
        "gross_exposure": gross_exposure,
        "positions_synced": False,
    }

    if not intents:
        snapshot = position_service.sync_from_holdings(positions, price_map)
        summary["gross_exposure"] = float(snapshot.get("total_notional", 0.0))
        summary["positions_synced"] = True
        return summary

    for intent in intents:
        price_map.setdefault(intent.symbol, intent.price)

    entries: List[ExecutionJournal] = []
    working_positions = dict(positions)
    exposure_by_symbol = dict(symbol_exposure)
    current_gross = gross_exposure

    for intent in intents:
        price = price_map.get(intent.symbol, intent.price)
        current_qty = working_positions.get(intent.symbol, 0.0)
        direction = 1.0 if intent.side.upper() == "BUY" else -1.0
        projected_qty = current_qty + (direction * intent.quantity)
        projected_notional = abs(projected_qty) * price
        existing_exposure = exposure_by_symbol.get(intent.symbol, 0.0)
        other_exposure = current_gross - existing_exposure
        projected_total = other_exposure + projected_notional

        allowed, reason = await risk_guard.can_execute(
            symbol=intent.symbol,
            notional=intent.notional,
            projected_notional=projected_notional,
            total_exposure=projected_total,
            price=price,
            price_as_of=run.fetched_at,
        )
        if not allowed:
            logger.warning(
                f"🚫 Order SKIPPED by risk_guard: {intent.symbol} {intent.side} {intent.quantity} - Reason: {reason} "
                f"(notional={intent.notional:.2f}, projected={projected_notional:.2f}, total_exposure={projected_total:.2f})"
            )
            summary["orders_skipped"] += 1
            entries.append(
                ExecutionJournal(
                    run_id=run.id,
                    symbol=intent.symbol,
                    side=intent.side,
                    target_weight=intent.target_weight,
                    quantity=intent.quantity,
                    status="skipped",
                    reason=f"risk_guard={reason}; {intent.reason}",
                    created_at=datetime.now(timezone.utc),
                )
            )
            continue

        try:
            order_id = await adapter.submit_order(intent.symbol, intent.side, intent.quantity, intent.price)
            await risk_guard.record_execution(
                symbol=intent.symbol,
                notional=intent.notional,
                pnl=0.0,
            )
            
            # Get TP/SL percentages from AI signal or use defaults from config
            tp_pct = float(os.getenv("QT_TP_PCT", "0.5")) / 100  # 0.5% default
            sl_pct = float(os.getenv("QT_SL_PCT", "0.75")) / 100  # 0.75% default
            
            # Update local trade state on fill
            try:
                trade_store.update_on_fill(intent.symbol, intent.side, intent.quantity, price)
                
                # 🎯 HYBRID AI + TP/SL: Store AI-generated TP/SL levels in trade state
                ai_sig = ai_signal_map.get(intent.symbol)
                
                if ai_sig and any(k in ai_sig for k in ("tp_percent", "sl_percent", "trail_percent", "partial_tp")):
                    state = trade_store.get(intent.symbol)
                    if state:
                        # Store AI-generated dynamic TP/SL values for this position
                        if "tp_percent" in ai_sig:
                            state["ai_tp_pct"] = float(ai_sig["tp_percent"])
                            tp_pct = float(ai_sig["tp_percent"])
                        if "sl_percent" in ai_sig:
                            state["ai_sl_pct"] = float(ai_sig["sl_percent"])
                            sl_pct = float(ai_sig["sl_percent"])
                        if "trail_percent" in ai_sig:
                            state["ai_trail_pct"] = float(ai_sig["trail_percent"])
                        if "partial_tp" in ai_sig:
                            state["ai_partial_tp"] = float(ai_sig["partial_tp"])
                        trade_store.set(intent.symbol, state)
                        logger.info(
                            f"🎯 AI TP/SL stored for {intent.symbol}: "
                            f"TP={state.get('ai_tp_pct', 0)*100:.2f}% "
                            f"SL={state.get('ai_sl_pct', 0)*100:.2f}% "
                            f"Trail={state.get('ai_trail_pct', 0)*100:.2f}%"
                        )
            except Exception as store_exc:
                logger.warning(f"Failed to update trade state for {intent.symbol}: {store_exc}")
            
            # 🛡️ CRITICAL: Place TP/SL orders on Binance Futures (OUTSIDE trade_store try block!)
            # ALWAYS run this after order submission - no conditions!
            if execution_config.exchange == "binance-futures":
                logger.info(f"🎯 Attempting to place TP/SL orders for {intent.symbol} (side={intent.side}, price={price})")
                try:
                    # Import Binance client and asyncio for async execution
                    from binance.client import Client as BinanceClient
                    import asyncio
                    binance_api_key = os.getenv("BINANCE_API_KEY")
                    binance_api_secret = os.getenv("BINANCE_API_SECRET")
                    
                    if binance_api_key and binance_api_secret:
                        logger.info(f"🔑 Binance credentials found for {intent.symbol}")
                        
                        # Run sync Binance API calls in a thread pool to avoid blocking
                        def place_tpsl_orders():
                            client = BinanceClient(binance_api_key, binance_api_secret)
                            
                            # 🚨 CRITICAL: Get ACTUAL entry price from Binance position (not signal price!)
                            actual_entry_price = price  # Default to signal price
                            try:
                                positions = client.futures_position_information(symbol=intent.symbol)
                                if positions:
                                    pos = positions[0]
                                    pos_amt = float(pos.get('positionAmt', 0))
                                    if abs(pos_amt) > 0:
                                        actual_entry_price = float(pos.get('entryPrice', price))
                                        logger.info(f"📊 {intent.symbol} actual entry: ${actual_entry_price} (signal was ${price})")
                            except Exception as pos_exc:
                                logger.warning(f"Could not fetch position for {intent.symbol}, using signal price: {pos_exc}")
                            
                            # Get price precision for this symbol from exchange info
                            try:
                                exchange_info = client.futures_exchange_info()
                                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == intent.symbol), None)
                                if symbol_info:
                                    # Get price precision from filters
                                    price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                                    if price_filter:
                                        tick_size = float(price_filter['tickSize'])
                                        # Calculate decimal places from tick size (e.g., 0.0001 = 4 decimals)
                                        if tick_size >= 1:
                                            price_precision = 0
                                        elif '.' in str(tick_size):
                                            price_precision = len(str(tick_size).rstrip('0').split('.')[-1])
                                        else:
                                            price_precision = 0
                                    else:
                                        price_precision = 2  # Default fallback
                                else:
                                    price_precision = 2  # Default fallback
                            except Exception as prec_exc:
                                logger.warning(f"Could not get price precision for {intent.symbol}, using default 2: {prec_exc}")
                                price_precision = 2
                            
                            logger.info(f"📏 {intent.symbol} price precision: {price_precision} decimals")
                            
                            # Calculate TP and SL prices using ACTUAL ENTRY
                            # 🎯 LEVERAGE-ADJUSTED TP/SL
                            # Goal: 2% SL = 2% margin loss (not 40% loss!)
                            # With 30x leverage: 2% margin loss = 0.067% price move (2% / 30)
                            leverage = float(intent.leverage) if intent.leverage else 30.0
                            price_tp_pct = tp_pct / leverage  # 3% / 30 = 0.1% price
                            price_sl_pct = sl_pct / leverage  # 2% / 30 = 0.067% price
                            
                            logger.info(f"🎯 Leverage {leverage}x: TP {tp_pct*100:.1f}% margin = {price_tp_pct*100:.2f}% price, SL {sl_pct*100:.1f}% margin = {price_sl_pct*100:.2f}% price")
                            
                            if intent.side.upper() == "BUY":  # LONG position
                                tp_price = round(actual_entry_price * (1 + price_tp_pct), price_precision)
                                sl_price = round(actual_entry_price * (1 - price_sl_pct), price_precision)
                                tp_side = 'SELL'
                                sl_side = 'SELL'
                            else:  # SHORT position
                                tp_price = round(actual_entry_price * (1 - price_tp_pct), price_precision)
                                sl_price = round(actual_entry_price * (1 + price_sl_pct), price_precision)
                                tp_side = 'BUY'
                                sl_side = 'BUY'
                            
                            logger.info(f"💰 {intent.symbol} TP=${tp_price}, SL=${sl_price} (from entry ${actual_entry_price})")
                            
                            # Cancel any existing TP/SL orders for this symbol to avoid conflicts
                            try:
                                open_orders = client.futures_get_open_orders(symbol=intent.symbol)
                                for order in open_orders:
                                    if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'TAKE_PROFIT', 'STOP_LOSS']:
                                        client.futures_cancel_order(symbol=intent.symbol, orderId=order['orderId'])
                                        logger.info(f"🗑️  Cancelled existing {order['type']} order {order['orderId']} for {intent.symbol}")
                            except Exception as cancel_exc:
                                logger.warning(f"Could not cancel existing orders for {intent.symbol}: {cancel_exc}")
                            
                            # Place TAKE_PROFIT_MARKET order using closePosition=True
                            tp_order = client.futures_create_order(
                                symbol=intent.symbol,
                                side=tp_side,
                                type='TAKE_PROFIT_MARKET',
                                stopPrice=tp_price,
                                closePosition=True,  # Use closePosition instead of quantity
                                workingType='MARK_PRICE'
                            )
                            logger.info(f"✅ TP order placed for {intent.symbol}: {tp_order['orderId']} @ ${tp_price} (+{tp_pct*100:.2f}%)")
                            
                            # Place STOP_MARKET order using closePosition=True
                            sl_order = client.futures_create_order(
                                symbol=intent.symbol,
                                side=sl_side,
                                type='STOP_MARKET',
                                stopPrice=sl_price,
                                closePosition=True,  # Use closePosition instead of quantity
                                workingType='MARK_PRICE'
                            )
                            logger.info(f"✅ SL order placed for {intent.symbol}: {sl_order['orderId']} @ ${sl_price} (-{sl_pct*100:.2f}%)")
                            return True
                        
                        # Execute in thread pool
                        success = await asyncio.to_thread(place_tpsl_orders)
                        if success:
                            logger.info(f"🛡️  TP/SL orders successfully placed for {intent.symbol}")
                    else:
                        logger.warning(f"⚠️  Cannot set TP/SL for {intent.symbol}: Missing Binance credentials")
                except Exception as tpsl_exc:
                    logger.error(f"❌ Failed to set TP/SL orders for {intent.symbol}: {tpsl_exc}", exc_info=True)
            
            working_positions[intent.symbol] = projected_qty
            exposure_by_symbol[intent.symbol] = projected_notional
            current_gross = projected_total
            summary["orders_submitted"] += 1
            entries.append(
                ExecutionJournal(
                    run_id=run.id,
                    symbol=intent.symbol,
                    side=intent.side,
                    target_weight=intent.target_weight,
                    quantity=intent.quantity,
                    status="filled",
                    reason=f"order_id={order_id}; {intent.reason}",
                    created_at=datetime.now(timezone.utc),
                    executed_at=datetime.now(timezone.utc),
                )
            )
            
            # Record AI prediction outcome for learning
            if "AI=" in intent.reason:
                try:
                    ai_action = intent.reason.split("AI=")[1].split("(")[0]
                    ai_confidence = float(intent.reason.split("(")[1].split(")")[0])
                    if hasattr(ai_engine, 'record_execution_outcome'):
                        # Get features from AI signal cache
                        ai_sig = ai_signal_map.get(intent.symbol, {})
                        features = ai_sig.get("features", {})
                        
                        await ai_engine.record_execution_outcome(
                            symbol=intent.symbol,
                            predicted_action=ai_action,
                            confidence=ai_confidence,
                            actual_outcome=intent.side.upper(),
                            entry_price=price,
                            features=features
                        )
                        logger.debug(f"Recorded AI outcome: {intent.symbol} {ai_action}({ai_confidence:.2f}) -> {intent.side} with {len(features)} features")
                except Exception as record_exc:
                    logger.debug(f"Could not record AI outcome for {intent.symbol}: {record_exc}")
        except Exception as exc:  # pragma: no cover - defensive guard
            # If the requested quote (e.g., USDC) is unavailable, optionally retry with USDT
            retried = False
            alt_symbol = ""
            try:
                if intent.symbol.upper().endswith("USDC"):
                    alt_symbol = intent.symbol[:-4] + "USDT"
                    order_id = await adapter.submit_order(alt_symbol, intent.side, intent.quantity, intent.price)
                    # Ensure we can value the alt symbol later
                    try:
                        price_map[alt_symbol] = price
                    except Exception:
                        pass
                    # Record success under alt symbol
                    await risk_guard.record_execution(
                        symbol=alt_symbol,
                        notional=intent.notional,
                        pnl=0.0,
                    )
                    working_positions[alt_symbol] = projected_qty
                    exposure_by_symbol[alt_symbol] = projected_notional
                    current_gross = projected_total
                    summary["orders_submitted"] += 1
                    entries.append(
                        ExecutionJournal(
                            run_id=run.id,
                            symbol=alt_symbol,
                            side=intent.side,
                            target_weight=intent.target_weight,
                            quantity=intent.quantity,
                            status="filled",
                            reason=f"order_id={order_id}; fallback_from={intent.symbol}; {intent.reason}",
                            created_at=datetime.now(timezone.utc),
                            executed_at=datetime.now(timezone.utc),
                        )
                    )
                    retried = True
            except Exception:
                retried = False

            if not retried:
                summary["orders_failed"] += 1
                entries.append(
                    ExecutionJournal(
                        run_id=run.id,
                        symbol=intent.symbol,
                        side=intent.side,
                        target_weight=intent.target_weight,
                        quantity=intent.quantity,
                        status="failed",
                        reason=intent.reason,
                        error=str(exc),
                        created_at=datetime.now(timezone.utc),
                    )
                )

    if entries:
        db.bulk_save_objects(entries)
        db.commit()

    final_positions_raw = await adapter.get_positions()
    final_positions = {symbol.upper(): float(qty) for symbol, qty in final_positions_raw.items()}
    snapshot = position_service.sync_from_holdings(final_positions, price_map)
    summary["gross_exposure"] = float(snapshot.get("total_notional", 0.0))
    summary["positions_synced"] = True

    return summary


__all__ = [
    "OrderIntent",
    "ExchangeAdapter",
    "PaperExchangeAdapter",
    "BinanceExecutionAdapter",
    "build_execution_adapter",
    "compute_target_orders",
    "compute_ai_signal_orders",
    "get_latest_portfolio",
    "run_portfolio_rebalance",
]
