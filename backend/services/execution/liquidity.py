"""Liquidity universe ingestion and selection utilities."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import aiohttp
from sqlalchemy.orm import Session

try:  # pragma: no cover - support package relative imports
    from backend.config import LiquidityConfig, load_liquidity_config
    from backend.models.liquidity import LiquidityRun, LiquiditySnapshot, PortfolioAllocation
    from backend.services.selection_engine import blend_liquidity_and_model, score_symbols_with_agent, summarize_selection
except ImportError:  # pragma: no cover
    from config import LiquidityConfig, load_liquidity_config  # type: ignore
    from models.liquidity import LiquidityRun, LiquiditySnapshot, PortfolioAllocation  # type: ignore
    from services.selection_engine import blend_liquidity_and_model, score_symbols_with_agent, summarize_selection  # type: ignore


logger = logging.getLogger(__name__)

SPOT_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
USDM_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
COINM_TICKER_URL = "https://dapi.binance.com/dapi/v1/ticker/24hr"
COINGECKO_MARKET_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=volume_desc&per_page=250&page=1&sparkline=false"
)


@dataclass(slots=True)
class ProviderRecord:
    symbol: str
    price: float
    base_volume: float
    quote_volume: float
    change_percent: float
    market_cap: Optional[float]
    provider: str
    raw: Mapping[str, Any]


@dataclass(slots=True)
class LiquidityRecord:
    symbol: str
    price: float
    change_percent: float
    base_volume: float
    quote_volume: float
    market_cap: Optional[float]
    liquidity_score: float
    momentum_score: float
    aggregate_score: float
    providers: Mapping[str, ProviderRecord]
    model_action: str = "HOLD"
    model_score: float = 0.0
    allocation_score: float = 0.0


def _stable_quote_match(symbol: str, quotes: Sequence[str]) -> Optional[str]:
    for quote in quotes:
        if symbol.endswith(quote):
            return quote
    return None


def _blacklisted(symbol: str, suffixes: Sequence[str]) -> bool:
    upper = symbol.upper()
    return any(upper.endswith(suffix) for suffix in suffixes)


def _filter_symbol(symbol: str, config: LiquidityConfig) -> bool:
    quote = _stable_quote_match(symbol, config.stable_quote_assets)
    if not quote:
        return False
    base = symbol[: -len(quote)]
    if not base or base.upper() in config.stable_quote_assets:
        return False
    if _blacklisted(symbol, config.blacklist_suffixes):
        return False
    return True


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    try:
        async with session.get(url, timeout=20) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {text[:200]}")
            return await response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch %s: %s", url, exc)
        raise


def _select_binance_endpoint(config: LiquidityConfig) -> str:
    if getattr(config, "market_type", "spot") == "usdm_perp":
        return USDM_TICKER_URL
    if getattr(config, "market_type", "spot") == "coinm_perp":
        return COINM_TICKER_URL
    return SPOT_TICKER_URL


async def fetch_binance_liquidity(
    session: aiohttp.ClientSession,
    config: LiquidityConfig,
) -> Dict[str, ProviderRecord]:
    payload = {}
    try:
        endpoint = _select_binance_endpoint(config)
        data = await _fetch_json(session, endpoint)
    except Exception:
        return payload

    for item in data or []:
        symbol = str(item.get("symbol", "")).upper()
        # Futures endpoints include all contracts including delivery. Filter if needed.
        if getattr(config, "market_type", "spot") != "spot":
            # Perpetual contracts on Binance futures often end with USDT (USDM) or USD (COINM)
            contract_type = item.get("contractType") or item.get("contract_type")
            if contract_type and contract_type != "PERPETUAL":
                continue
        if not _filter_symbol(symbol, config):
            continue
        quote_volume = _parse_float(item.get("quoteVolume"))
        if quote_volume < config.min_quote_volume:
            continue
        last_price_key = "lastPrice"
        if getattr(config, "market_type", "spot") != "spot":
            # Futures 24hr ticker uses 'lastPrice' as well, but volume keys might differ (e.g., 'volume' base asset volume)
            pass
        record = ProviderRecord(
            symbol=symbol,
            price=_parse_float(item.get(last_price_key)),
            base_volume=_parse_float(item.get("volume")),
            quote_volume=quote_volume,
            change_percent=_parse_float(item.get("priceChangePercent")),
            market_cap=None,
            provider="binance-futures" if getattr(config, "market_type", "spot") != "spot" else "binance",
            raw=item,
        )
        payload[symbol] = record
    return payload


async def fetch_coingecko_liquidity(
    session: aiohttp.ClientSession,
    config: LiquidityConfig,
) -> Dict[str, ProviderRecord]:
    payload = {}
    try:
        data = await _fetch_json(session, COINGECKO_MARKET_URL)
    except Exception:
        return payload

    for item in data or []:
        symbol = str(item.get("symbol", "")).upper()
        coin_symbol = f"{symbol}USDT"
        if not _filter_symbol(coin_symbol, config):
            continue
        total_volume = _parse_float(item.get("total_volume"))
        if total_volume <= 0:
            continue
        record = ProviderRecord(
            symbol=coin_symbol,
            price=_parse_float(item.get("current_price")),
            base_volume=0.0,
            quote_volume=total_volume,
            change_percent=_parse_float(item.get("price_change_percentage_24h")),
            market_cap=_parse_float(item.get("market_cap")),
            provider="coingecko",
            raw=item,
        )
        payload[coin_symbol] = record
    return payload


def _combine_records(
    binance: Mapping[str, ProviderRecord],
    coingecko: Mapping[str, ProviderRecord],
    config: LiquidityConfig,
) -> List[LiquidityRecord]:
    combined: Dict[str, LiquidityRecord] = {}
    symbols = set(binance.keys()) | set(coingecko.keys())
    for symbol in symbols:
        providers: Dict[str, ProviderRecord] = {}
        binance_record = binance.get(symbol)
        cg_record = coingecko.get(symbol)
        if binance_record:
            providers["binance"] = binance_record
        if cg_record:
            providers["coingecko"] = cg_record
        if not providers:
            continue

        price = binance_record.price if binance_record else (cg_record.price if cg_record else 0.0)
        change_percent = (
            binance_record.change_percent
            if binance_record
            else (cg_record.change_percent if cg_record else 0.0)
        )
        base_volume = binance_record.base_volume if binance_record else 0.0
        quote_volume = max(
            binance_record.quote_volume if binance_record else 0.0,
            cg_record.quote_volume if cg_record else 0.0,
        )
        market_cap = None
        if cg_record and cg_record.market_cap:
            market_cap = cg_record.market_cap

        liquidity_score = (
            config.binance_weight * (binance_record.quote_volume if binance_record else 0.0)
            + config.coingecko_weight * (cg_record.quote_volume if cg_record else 0.0)
        )
        momentum = change_percent / 100.0
        momentum_score = momentum
        aggregate_multiplier = 1 + config.score_momentum_weight * momentum
        if aggregate_multiplier < 0.1:
            aggregate_multiplier = 0.1
        aggregate_score = liquidity_score * aggregate_multiplier

        record = LiquidityRecord(
            symbol=symbol,
            price=price,
            change_percent=change_percent,
            base_volume=base_volume,
            quote_volume=quote_volume,
            market_cap=market_cap,
            liquidity_score=liquidity_score,
            momentum_score=momentum_score,
            aggregate_score=aggregate_score,
            providers=providers,
        )
        combined[symbol] = record

    ranked = sorted(
        combined.values(),
        key=lambda item: (item.aggregate_score, item.quote_volume),
        reverse=True,
    )
    return ranked[: config.universe_max]


def _select_portfolio(records: Sequence[LiquidityRecord], config: LiquidityConfig) -> List[LiquidityRecord]:
    if not records:
        return []
    max_size = min(config.selection_max, len(records))
    size = max(config.selection_min, max_size)
    return list(records[:size])


def _serialize_providers(providers: Mapping[str, ProviderRecord]) -> str:
    payload = {
        name: {
            "price": record.price,
            "quote_volume": record.quote_volume,
            "change_percent": record.change_percent,
            "market_cap": record.market_cap,
            "provider": record.provider,
            "raw": record.raw,
        }
        for name, record in providers.items()
    }
    return json.dumps(payload, ensure_ascii=False)


def _primary_provider(binance_records: Mapping[str, ProviderRecord]) -> str:
    if binance_records:
        return "binance"
    return "coingecko"


def persist_liquidity_run(
    db: Session,
    *,
    records: Sequence[LiquidityRecord],
    selection: Sequence[LiquidityRecord],
    provider_primary: str,
    status: str = "completed",
    message: Optional[str] = None,
    signals: Optional[Mapping[str, Mapping[str, Any]]] = None,
    analytics: Optional[Mapping[str, Any]] = None,
) -> LiquidityRun:
    stored_message = message
    if analytics:
        payload: Dict[str, Any] = {"analytics": analytics}
        if message:
            payload["note"] = message
        try:
            stored_message = json.dumps(payload, ensure_ascii=False)
        except TypeError as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to serialize selection analytics: %s", exc)

    run = LiquidityRun(
        fetched_at=datetime.now(timezone.utc),
        universe_size=len(records),
        selection_size=len(selection),
        provider_primary=provider_primary,
        status=status,
        message=stored_message,
    )
    db.add(run)
    db.flush()

    snapshots = []
    for idx, record in enumerate(records, start=1):
        snapshot = LiquiditySnapshot(
            run_id=run.id,
            rank=idx,
            symbol=record.symbol,
            price=record.price,
            change_percent=record.change_percent,
            base_volume=record.base_volume,
            quote_volume=record.quote_volume,
            market_cap=record.market_cap,
            liquidity_score=record.liquidity_score,
            momentum_score=record.momentum_score,
            aggregate_score=record.aggregate_score,
            payload=_serialize_providers(record.providers),
        )
        snapshots.append(snapshot)
    db.bulk_save_objects(snapshots)

    if selection:
        weight_basis = {
            item.symbol: max(
                getattr(item, "allocation_score", item.aggregate_score),
                0.0,
            )
            for item in selection
        }
        total_score = sum(weight_basis.values())
        if total_score <= 0 and selection:
            weight = 1 / len(selection)
            weights = {item.symbol: weight for item in selection}
        else:
            weights = {
                symbol: basis / total_score if total_score else 0.0
                for symbol, basis in weight_basis.items()
            }
        signals = signals or {}
        allocations = []
        for record in selection:
            signal = signals.get(record.symbol.upper()) or signals.get(record.symbol)
            parts = [
                f"liquidity={record.liquidity_score:.2f}",
                f"momentum={record.change_percent:.2f}%",
            ]
            if hasattr(record, "model_action"):
                parts.append(f"model={record.model_action}({record.model_score:.2f})")
            elif signal:
                action = str(signal.get("action", "HOLD"))
                score = float(signal.get("score", 0.0))
                parts.append(f"model={action}({score:.2f})")
            allocation_component = getattr(record, "allocation_score", None)
            if allocation_component is not None and allocation_component > 0:
                parts.append(f"blend={allocation_component:.3f}")
            reason = "; ".join(parts)
            allocation = PortfolioAllocation(
                run_id=run.id,
                symbol=record.symbol,
                weight=weights.get(record.symbol, 0.0),
                score=record.aggregate_score,
                reason=reason,
            )
            allocations.append(allocation)
        db.bulk_save_objects(allocations)

    db.commit()
    db.refresh(run)
    return run


async def refresh_liquidity(
    db: Session,
    *,
    config: Optional[LiquidityConfig] = None,
) -> Dict[str, Any]:
    config = config or load_liquidity_config()
    # Create a client with conservative total/connect/read timeouts to avoid scheduler hangs
    timeout = aiohttp.ClientTimeout(total=25, connect=5, sock_read=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        binance_task = asyncio.create_task(fetch_binance_liquidity(session, config))
        coingecko_task = None
        if getattr(config, "coingecko_weight", 0.0) > 0.0:
            coingecko_task = asyncio.create_task(fetch_coingecko_liquidity(session, config))
        try:
            if coingecko_task is not None:
                binance_records, coingecko_records = await asyncio.wait_for(
                    asyncio.gather(binance_task, coingecko_task, return_exceptions=False),
                    timeout=30,
                )
            else:
                binance_records = await asyncio.wait_for(binance_task, timeout=25)
                coingecko_records = {}
        except asyncio.TimeoutError:
            logger.warning("Liquidity provider fetch timed out; proceeding with partial/empty data")
            # Best-effort cancellation of in-flight tasks
            if coingecko_task is not None:
                for t in (binance_task, coingecko_task):
                    t.cancel()
            else:
                binance_task.cancel()
            binance_records, coingecko_records = {}, {}

    combined = _combine_records(binance_records, coingecko_records, config)
    selection = _select_portfolio(combined, config)
    provider_primary = _primary_provider(binance_records)

    model_signals: Dict[str, Any] = {}
    if not selection:
        # Diagnostic logging to help understand empty selections in live runs
        try:
            logger.warning(
                "Liquidity selection empty: binance=%s coingecko=%s combined=%s min_quote_volume=%s market_type=%s selection_min=%s selection_max=%s weights(binance=%s, coingecko=%s, momentum=%s)",
                len(binance_records),
                len(coingecko_records),
                len(combined),
                getattr(config, "min_quote_volume", None),
                getattr(config, "market_type", None),
                getattr(config, "selection_min", None),
                getattr(config, "selection_max", None),
                getattr(config, "binance_weight", None),
                getattr(config, "coingecko_weight", None),
                getattr(config, "score_momentum_weight", None),
            )
        except Exception:
            pass
    if selection:
        try:
            symbols = [record.symbol for record in selection]
            # Guard model scoring with a timeout so we never block the scheduler end-to-end
            model_signals = await asyncio.wait_for(
                score_symbols_with_agent(symbols, limit=240),
                timeout=15,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Model scoring failed or timed out; continuing with liquidity-only selection: %s", exc)
            model_signals = {}
        final_selection = blend_liquidity_and_model(selection, model_signals, config)
        if not final_selection:
            final_selection = selection
    else:
        final_selection = selection

    selection_analytics: Dict[str, Any] = {}
    if final_selection:
        selection_analytics = summarize_selection(final_selection, config=config)

    run = persist_liquidity_run(
        db,
        records=combined,
        selection=final_selection,
        provider_primary=provider_primary,
        signals=model_signals,
        analytics=selection_analytics or None,
    )

    return {
        "run_id": run.id,
        "fetched_at": run.fetched_at.isoformat() if run.fetched_at else None,
        "universe_size": run.universe_size,
        "provider_primary": provider_primary,
        "selection_size": run.selection_size,
        "model_summary": {
            "scored": len(model_signals),
            "buy": sum(1 for payload in model_signals.values() if str(payload.get("action")).upper() == "BUY"),
            "sell": sum(1 for payload in model_signals.values() if str(payload.get("action")).upper() == "SELL"),
        },
        "selection_analytics": selection_analytics,
    }


__all__ = [
    "LiquidityRecord",
    "ProviderRecord",
    "fetch_binance_liquidity",
    "fetch_coingecko_liquidity",
    "refresh_liquidity",
    "persist_liquidity_run",
]


