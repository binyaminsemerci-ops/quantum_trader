"""Runtime risk guard utilities for trading execution.

MIGRATION STEP 1: Risk parameters now driven by PolicyStore v2.
All hardcoded risk limits replaced with dynamic policy reads.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, Optional, Protocol, Tuple
import math

from backend.config.risk import RiskConfig
from backend.utils.telemetry import record_risk_denial, set_risk_daily_loss

# [NEW] ARCHITECTURE V2: Import PolicyStore and logger
from backend.core.policy_store import PolicyStore, get_policy_store
from backend.core.logger import get_logger

# P2-02: Import logging APIs
from backend.core import metrics_logger
from backend.core import audit_logger

logger = get_logger(__name__)


@dataclass
class _TradeRecord:
    timestamp: datetime
    notional: float
    pnl: float


@dataclass(frozen=True)
class KillSwitchState:
    enabled: bool
    reason: str
    updated_at: datetime


class RiskStateStore(Protocol):
    async def get_records(self) -> Iterable[_TradeRecord]: ...

    async def add_record(self, record: _TradeRecord) -> None: ...

    async def clear(self) -> None: ...

    async def get_kill_switch_state(self) -> Optional[KillSwitchState]: ...

    async def set_kill_switch_state(self, state: Optional[KillSwitchState]) -> None: ...

    async def get_kill_switch_override(self) -> Optional[bool]: ...

    async def set_kill_switch_override(self, enabled: Optional[bool]) -> None: ...


class InMemoryRiskStateStore:
    """Simple in-memory risk state store with rolling 24 hour window."""

    def __init__(self, window: timedelta | None = None) -> None:
        self._window = window or timedelta(hours=24)
        self._records: Deque[_TradeRecord] = deque()
        self._lock = asyncio.Lock()
        self._kill_switch_state: Optional[KillSwitchState] = None

    async def get_records(self) -> Iterable[_TradeRecord]:
        async with self._lock:
            self._prune_locked()
            return list(self._records)

    async def add_record(self, record: _TradeRecord) -> None:
        async with self._lock:
            self._records.append(record)
            self._prune_locked()

    async def clear(self) -> None:
        async with self._lock:
            self._records.clear()
            self._kill_switch_state = None

    def _prune_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._window
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    async def get_kill_switch_state(self) -> Optional[KillSwitchState]:
        async with self._lock:
            return self._kill_switch_state

    async def set_kill_switch_state(self, state: Optional[KillSwitchState]) -> None:
        async with self._lock:
            self._kill_switch_state = state

    async def get_kill_switch_override(self) -> Optional[bool]:
        state = await self.get_kill_switch_state()
        if state is None:
            return None
        return state.enabled

    async def set_kill_switch_override(self, enabled: Optional[bool]) -> None:
        if enabled is None:
            await self.set_kill_switch_state(None)
            return
        state = KillSwitchState(
            enabled=bool(enabled),
            reason="legacy_override",
            updated_at=datetime.now(timezone.utc),
        )
        await self.set_kill_switch_state(state)


class SqliteRiskStateStore:
    """SQLite-backed risk state store with rolling trade history."""

    def __init__(self, db_path: str | Path, window: timedelta | None = None) -> None:
        self._window = window or timedelta(hours=24)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialise()

    def _initialise(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    notional REAL NOT NULL,
                    pnl REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kill_switch_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    override INTEGER,
                    reason TEXT,
                    updated_at TEXT
                )
                """
            )
            try:
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(kill_switch_state)")
                }
            except sqlite3.OperationalError:
                columns = set()
            if "reason" not in columns:
                try:
                    conn.execute("ALTER TABLE kill_switch_state ADD COLUMN reason TEXT")
                except sqlite3.OperationalError:
                    pass
            if "updated_at" not in columns:
                try:
                    conn.execute("ALTER TABLE kill_switch_state ADD COLUMN updated_at TEXT")
                except sqlite3.OperationalError:
                    pass

    async def get_records(self) -> Iterable[_TradeRecord]:
        async with self._lock:
            return await asyncio.to_thread(self._get_records_sync)

    def _get_records_sync(self) -> Iterable[_TradeRecord]:
        cutoff = (datetime.now(timezone.utc) - self._window).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM risk_records WHERE timestamp < ?", (cutoff,))
            rows = conn.execute(
                "SELECT timestamp, notional, pnl FROM risk_records ORDER BY timestamp ASC"
            ).fetchall()
        records = []
        for timestamp, notional, pnl in rows:
            ts = datetime.fromisoformat(timestamp)
            records.append(_TradeRecord(timestamp=ts, notional=float(notional), pnl=float(pnl)))
        return records

    async def add_record(self, record: _TradeRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._add_record_sync, record)

    def _add_record_sync(self, record: _TradeRecord) -> None:
        cutoff = (datetime.now(timezone.utc) - self._window).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO risk_records (timestamp, notional, pnl) VALUES (?, ?, ?)",
                (record.timestamp.isoformat(), record.notional, record.pnl),
            )
            conn.execute("DELETE FROM risk_records WHERE timestamp < ?", (cutoff,))

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM risk_records")
            conn.execute("DELETE FROM kill_switch_state")

    async def get_kill_switch_state(self) -> Optional[KillSwitchState]:
        async with self._lock:
            return await asyncio.to_thread(self._get_kill_switch_state_sync)

    def _get_kill_switch_state_sync(self) -> Optional[KillSwitchState]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT override, reason, updated_at FROM kill_switch_state WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        override, reason, updated_at = row
        if override is None:
            return None
        try:
            timestamp = (
                datetime.fromisoformat(updated_at)
                if isinstance(updated_at, str)
                else datetime.now(timezone.utc)
            )
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except Exception:
            timestamp = datetime.now(timezone.utc)
        return KillSwitchState(
            enabled=bool(override),
            reason=str(reason or ""),
            updated_at=timestamp,
        )

    async def set_kill_switch_state(self, state: Optional[KillSwitchState]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._set_kill_switch_state_sync, state)

    def _set_kill_switch_state_sync(self, state: Optional[KillSwitchState]) -> None:
        if state is None:
            override_value = None
            reason_value = None
            timestamp = datetime.now(timezone.utc).isoformat()
        else:
            override_value = 1 if state.enabled else 0
            reason_value = state.reason
            timestamp = state.updated_at.isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO kill_switch_state (id, override, reason, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET override = excluded.override, reason = excluded.reason, updated_at = excluded.updated_at
                """,
                (override_value, reason_value, timestamp),
            )

    async def get_kill_switch_override(self) -> Optional[bool]:
        state = await self.get_kill_switch_state()
        if state is None:
            return None
        return state.enabled

    async def set_kill_switch_override(self, enabled: Optional[bool]) -> None:
        if enabled is None:
            await self.set_kill_switch_state(None)
            return
        reason = "legacy_override_on" if enabled else "legacy_override_off"
        await self.set_kill_switch_state(
            KillSwitchState(
                enabled=bool(enabled),
                reason=reason,
                updated_at=datetime.now(timezone.utc),
            )
        )


class RiskGuardService:
    """Evaluate whether trade executions are permitted under current constraints."""

    def __init__(
        self,
        config: RiskConfig,
        store: Optional[RiskStateStore] = None,
        policy_store = None,  # [NEW] PolicyStore for dynamic limits
    ) -> None:
        self._config = config
        self._store = store or InMemoryRiskStateStore()
        self._kill_switch_state: Optional[KillSwitchState] = None
        self._kill_switch_loaded = False
        self._lock = asyncio.Lock()
        self._position_loader: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None
        self.policy_store = policy_store  # [NEW] Store reference for dynamic reads
        set_risk_daily_loss(0.0)
        
        if self.policy_store:
            logger.info("✅ PolicyStore integration enabled in RiskGuard (dynamic limits)")
        
        # [REMOVED] MSC AI Policy Reader - Module does not exist
        self._msc_policy_store = None

    @property
    def config(self) -> RiskConfig:
        return self._config

    async def can_execute(
        self,
        *,
        symbol: str,
        notional: float,
        projected_notional: Optional[float] = None,
        total_exposure: Optional[float] = None,
        price: Optional[float] = None,
        price_as_of: Optional[datetime] = None,
        leverage: Optional[float] = None,  # [NEW] For leverage check
        trade_risk_pct: Optional[float] = None,  # [NEW] For risk % check
        position_size_usd: Optional[float] = None,  # [NEW] For position cap check
        trace_id: Optional[str] = None,  # [NEW] For structured logging
    ) -> Tuple[bool, str]:
        """Evaluate trade execution against dynamic risk profile.
        
        MIGRATION STEP 1: Risk limits now read from PolicyStore v2 RiskProfile.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            notional: Trade notional value
            projected_notional: Projected position size after trade
            total_exposure: Total portfolio exposure
            price: Current price
            price_as_of: Price timestamp
            leverage: Requested leverage (if applicable)
            trade_risk_pct: Risk % of this trade (0-100)
            position_size_usd: Position size in USD
            trace_id: Trace ID for logging correlation
            
        Returns:
            (can_execute, denial_reason)
        """
        symbol = symbol.upper()
        
        # [ARCHITECTURE V2] READ DYNAMIC RISK PROFILE FROM POLICYSTORE
        risk_profile = None
        profile_name = "FALLBACK_NORMAL"
        
        if self.policy_store:
            try:
                risk_profile = await self.policy_store.get_active_risk_profile()
                profile_name = risk_profile.name
                
                logger.info(
                    "risk_guard_risk_profile_loaded",
                    trace_id=trace_id or "unknown",
                    profile_name=profile_name,
                    max_leverage=risk_profile.max_leverage,
                    max_risk_pct_per_trade=risk_profile.max_risk_pct_per_trade,
                    max_daily_drawdown_pct=risk_profile.max_daily_drawdown_pct,
                    max_open_positions=risk_profile.max_open_positions,
                    position_size_cap_usd=risk_profile.position_size_cap_usd,
                    allow_new_positions=risk_profile.allow_new_positions,
                )
                
                # [NEW CHECK 1] Leverage limit
                if leverage is not None and leverage > risk_profile.max_leverage:
                    record_risk_denial("max_leverage_exceeded")
                    
                    # P2-02: Log risk block to audit
                    audit_logger.log_risk_block(
                        symbol=symbol,
                        action=f"trade_leverage_{leverage}x",
                        reason=f"Leverage {leverage}x exceeds limit {risk_profile.max_leverage}x",
                        blocker="RiskGuard"
                    )
                    
                    # P2-02: Record risk denial metrics
                    metrics_logger.record_counter(
                        "risk_denials",
                        value=1.0,
                        labels={"reason": "max_leverage", "symbol": symbol}
                    )
                    
                    logger.warning(
                        "risk_guard_denied_leverage",
                        trace_id=trace_id or "unknown",
                        profile_name=profile_name,
                        requested_leverage=leverage,
                        max_leverage=risk_profile.max_leverage,
                    )
                    return False, f"max_leverage ({leverage} > {risk_profile.max_leverage})"
                
                # [NEW CHECK 2] Risk per trade % limit
                if trade_risk_pct is not None and trade_risk_pct > risk_profile.max_risk_pct_per_trade:
                    record_risk_denial("max_risk_pct_per_trade_exceeded")
                    logger.warning(
                        "risk_guard_denied_risk_pct",
                        trace_id=trace_id or "unknown",
                        profile_name=profile_name,
                        trade_risk_pct=trade_risk_pct,
                        max_risk_pct_per_trade=risk_profile.max_risk_pct_per_trade,
                    )
                    return False, f"max_risk_pct ({trade_risk_pct:.2f}% > {risk_profile.max_risk_pct_per_trade}%)"
                
                # [NEW CHECK 3] Position size cap
                if position_size_usd is not None and position_size_usd > risk_profile.position_size_cap_usd:
                    record_risk_denial("position_size_cap_exceeded")
                    
                    # P2-02: Log risk block to audit
                    audit_logger.log_risk_block(
                        symbol=symbol,
                        action=f"trade_size_${position_size_usd:.0f}",
                        reason=f"Position size ${position_size_usd:.2f} exceeds cap ${risk_profile.position_size_cap_usd}",
                        blocker="RiskGuard"
                    )
                    
                    # P2-02: Record risk denial metrics
                    metrics_logger.record_counter(
                        "risk_denials",
                        value=1.0,
                        labels={"reason": "position_cap", "symbol": symbol}
                    )
                    
                    logger.warning(
                        "risk_guard_denied_position_cap",
                        trace_id=trace_id or "unknown",
                        profile_name=profile_name,
                        position_size_usd=position_size_usd,
                        position_size_cap_usd=risk_profile.position_size_cap_usd,
                    )
                    return False, f"position_cap (${position_size_usd:.2f} > ${risk_profile.position_size_cap_usd})"
                
                # [NEW CHECK 4] Allow new positions flag
                if not risk_profile.allow_new_positions:
                    record_risk_denial("new_positions_disabled")
                    logger.warning(
                        "risk_guard_denied_new_positions_disabled",
                        trace_id=trace_id or "unknown",
                        profile_name=profile_name,
                    )
                    return False, "new_positions_disabled"
                
                # [EXISTING CHECK] Max open positions (enhanced with policy)
                if self._position_loader:
                    positions = await self._position_loader()
                    current_count = len([p for p in positions.values() if p.get('size', 0) != 0])
                    if current_count >= risk_profile.max_open_positions:
                        record_risk_denial("max_positions_exceeded")
                        logger.warning(
                            "risk_guard_denied_max_positions",
                            trace_id=trace_id or "unknown",
                            profile_name=profile_name,
                            current_positions=current_count,
                            max_positions=risk_profile.max_open_positions,
                        )
                        return False, f"max_positions ({current_count} >= {risk_profile.max_open_positions})"
                
            except Exception as e:
                logger.error(
                    "risk_guard_policystore_error",
                    trace_id=trace_id or "unknown",
                    error=str(e),
                    fallback="using legacy hardcoded limits",
                )
                # Fallback to legacy checks below
                # Fallback to legacy checks below
        
        # [LEGACY CHECKS] Kill switch, symbol whitelist, basic limits (preserved for backward compat)
        if await self._is_kill_switch_enabled():
            record_risk_denial("kill_switch")
            logger.warning(
                "risk_guard_denied_kill_switch",
                trace_id=trace_id or "unknown",
                profile_name=profile_name,
            )
            return False, "kill_switch"

        if symbol not in {sym.upper() for sym in self._config.allowed_symbols}:
            record_risk_denial("symbol_not_allowed")
            logger.warning(
                "risk_guard_denied_symbol_not_allowed",
                trace_id=trace_id or "unknown",
                symbol=symbol,
                profile_name=profile_name,
            )
            return False, "symbol_not_allowed"

        if notional > self._config.max_notional_per_trade:
            record_risk_denial("notional_limit")
            logger.warning(
                "risk_guard_denied_notional",
                trace_id=trace_id or "unknown",
                notional=notional,
                max_notional=self._config.max_notional_per_trade,
                profile_name=profile_name,
            )
            return False, "notional_limit"

        if price is not None:
            if not math.isfinite(price) or price <= 0:
                record_risk_denial("price_invalid")
                return False, "price_invalid"
            floor = self._config.min_unit_price
            if floor is not None and price < floor:
                record_risk_denial("price_floor")
                return False, "price_floor"
            ceiling = self._config.max_unit_price
            if ceiling is not None and price > ceiling:
                record_risk_denial("price_ceiling")
                return False, "price_ceiling"

        if price_as_of is not None and self._config.max_price_staleness_seconds is not None:
            as_of = price_as_of
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            staleness = (datetime.now(timezone.utc) - as_of).total_seconds()
            if staleness > self._config.max_price_staleness_seconds:
                record_risk_denial("price_stale")
                return False, "price_stale"

        # [NEW CHECK 5] Daily drawdown limit (enhanced with risk profile)
        daily_loss = await self._daily_loss()
        max_daily_drawdown_pct = risk_profile.max_daily_drawdown_pct if risk_profile else 5.0  # Fallback
        
        # Calculate current daily drawdown % (assuming we track account balance)
        # TODO: This needs account balance tracking - for now using absolute loss limit
        if daily_loss >= self._config.max_daily_loss:
            record_risk_denial("daily_loss_limit")
            logger.warning(
                "risk_guard_denied_daily_loss",
                trace_id=trace_id or "unknown",
                daily_loss=daily_loss,
                max_daily_loss=self._config.max_daily_loss,
                profile_name=profile_name,
                max_daily_drawdown_pct=max_daily_drawdown_pct,
            )
            return False, "daily_loss_limit"

        per_symbol_limit = self._config.max_position_per_symbol
        if per_symbol_limit is not None and projected_notional is not None:
            if projected_notional > per_symbol_limit:
                record_risk_denial("position_limit")
                return False, "position_limit"

        gross_limit = self._config.max_gross_exposure
        if gross_limit is not None and total_exposure is not None:
            if total_exposure > gross_limit:
                record_risk_denial("gross_exposure_limit")
                return False, "gross_exposure_limit"

        # ✅ ALL CHECKS PASSED
        logger.info(
            "risk_guard_trade_approved",
            trace_id=trace_id or "unknown",
            profile_name=profile_name,
            symbol=symbol,
            notional=notional,
            leverage=leverage,
            trade_risk_pct=trade_risk_pct,
        )
        return True, ""

    async def record_execution(
        self,
        *,
        symbol: str,
        notional: float,
        pnl: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        _ = symbol  # currently unused, reserved for per-symbol stats
        ts = timestamp or datetime.now(timezone.utc)
        await self._store.add_record(_TradeRecord(timestamp=ts, notional=notional, pnl=pnl))
        await self._refresh_metrics()

    async def reset(self) -> None:
        await self._store.clear()
        async with self._lock:
            self._kill_switch_state = None
            self._kill_switch_loaded = True
        set_risk_daily_loss(0.0)

    async def set_kill_switch(self, enabled: Optional[bool], *, reason: Optional[str] = None) -> None:
        if enabled is None:
            state: Optional[KillSwitchState] = None
        else:
            state = KillSwitchState(
                enabled=bool(enabled),
                reason=reason
                or ("manual_override_on" if enabled else "manual_override_off"),
                updated_at=datetime.now(timezone.utc),
            )
        async with self._lock:
            self._kill_switch_state = state
            self._kill_switch_loaded = True
        await self._store.set_kill_switch_state(state)

    def set_position_loader(self, loader: Callable[[], Awaitable[Dict[str, Any]]]) -> None:
        self._position_loader = loader

    async def kill_switch_state(self) -> Optional[KillSwitchState]:
        return await self._get_kill_switch_state()

    async def snapshot(self) -> Dict[str, Any]:
        records = list(await self._store.get_records())
        daily_loss = sum(-min(record.pnl, 0.0) for record in records)
        set_risk_daily_loss(daily_loss)
        kill_switch_state = await self._get_kill_switch_state()
        override = None if kill_switch_state is None else kill_switch_state.enabled
        positions_snapshot = await self._load_positions_snapshot()
        return {
            "config": {
                "staging_mode": self._config.staging_mode,
            "kill_switch": await self._is_kill_switch_enabled(),
                "max_notional_per_trade": self._config.max_notional_per_trade,
                "max_daily_loss": self._config.max_daily_loss,
                "allowed_symbols": list(self._config.allowed_symbols),
                "failsafe_reset_minutes": self._config.failsafe_reset_minutes,
                "risk_state_db_path": self._config.risk_state_db_path,
                "max_position_per_symbol": self._config.max_position_per_symbol,
                "max_gross_exposure": self._config.max_gross_exposure,
            },
            "state": {
                "kill_switch_override": override,
                "records": [
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "notional": record.notional,
                        "pnl": record.pnl,
                    }
                    for record in records
                ],
                "daily_loss": daily_loss,
                "trade_count": len(records),
                "kill_switch_state": None
                if kill_switch_state is None
                else {
                    "enabled": kill_switch_state.enabled,
                    "reason": kill_switch_state.reason,
                    "updated_at": kill_switch_state.updated_at.isoformat(),
                },
            },
            "positions": positions_snapshot,
        }

    async def _load_positions_snapshot(self) -> Dict[str, Any]:
        if self._position_loader is None:
            return {
                "positions": [],
                "total_notional": 0.0,
                "as_of": None,
            }
        try:
            snapshot = await self._position_loader()
        except Exception:  # pragma: no cover - resilience guard
            logger.exception("Position loader failed")
            return {
                "positions": [],
                "total_notional": 0.0,
                "as_of": None,
            }
        if not isinstance(snapshot, dict):
            return {
                "positions": [],
                "total_notional": 0.0,
                "as_of": None,
            }
        return snapshot

    async def _daily_loss(self) -> float:
        records = await self._store.get_records()
        return sum(-min(record.pnl, 0.0) for record in records)

    async def _refresh_metrics(self) -> None:
        set_risk_daily_loss(await self._daily_loss())

    async def _is_kill_switch_enabled(self) -> bool:
        await self._evaluate_kill_switch_failsafe()
        kill_switch_state = await self._get_kill_switch_state()
        if kill_switch_state is not None:
            return kill_switch_state.enabled
        return self._config.kill_switch

    async def _get_kill_switch_state(self) -> Optional[KillSwitchState]:
        async with self._lock:
            if self._kill_switch_loaded:
                return self._kill_switch_state
        value = await self._store.get_kill_switch_state()
        async with self._lock:
            self._kill_switch_state = value
            self._kill_switch_loaded = True
            return self._kill_switch_state

    async def _evaluate_kill_switch_failsafe(self) -> None:
        minutes = self._config.failsafe_reset_minutes
        if minutes <= 0:
            return
        kill_switch_state = await self._get_kill_switch_state()
        if kill_switch_state is None or not kill_switch_state.enabled:
            return
        elapsed = datetime.now(timezone.utc) - kill_switch_state.updated_at
        if elapsed < timedelta(minutes=minutes):
            return
        positions_snapshot = await self._load_positions_snapshot()
        try:
            total_exposure = float(positions_snapshot.get("total_notional", 0.0))
        except (TypeError, ValueError, AttributeError):
            total_exposure = 0.0
        threshold = self._config.max_gross_exposure or self._config.max_notional_per_trade
        if threshold is None:
            threshold = 0.0
        if total_exposure <= threshold:
            logger.warning(
                "Kill switch auto-reset after %.2f minutes; exposure %.2f <= %.2f",
                elapsed.total_seconds() / 60.0,
                total_exposure,
                threshold,
            )
            await self.set_kill_switch(None)
    
    async def check_exchange_limits(
        self,
        exchange_name: str,
        symbol: str,
        notional: float,
        trace_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check per-exchange exposure limits.
        
        EPIC-EXCH-ROUTING-001: Placeholder for exchange-specific risk controls.
        
        Future implementation will check:
        - Per-exchange exposure caps
        - Per-exchange position limits
        - Exchange-specific VaR limits
        - Cross-exchange correlation constraints
        
        Args:
            exchange_name: Target exchange (e.g., "bybit", "okx")
            symbol: Trading symbol
            notional: Trade notional value
            trace_id: Trace ID for logging
        
        Returns:
            (allowed, denial_reason)
        
        TODO:
        - [ ] Implement per-exchange exposure tracking
        - [ ] Add exchange-specific position limits to PolicyStore
        - [ ] Integrate with Global Risk v3 (EPIC-RISK3-001)
        - [ ] Add cross-exchange correlation matrix
        """
        logger.debug(
            "check_exchange_limits called (NO-OP for now)",
            extra={
                "exchange": exchange_name,
                "symbol": symbol,
                "notional": notional,
                "trace_id": trace_id or "unknown"
            }
        )
        
        # TODO: Implement actual checks here
        # For now, always allow (no-op)
        return (True, "")

    async def evaluate_trade_intent(
        self,
        trade_intent: Dict[str, Any],
        trace_id: str = "",
    ) -> Tuple[bool, str]:
        """
        [RL v3 PRODUCTION] Evaluate trade intent from RL v3 Live Orchestrator.
        
        CRITICAL: This is the ONLY entry point for RL v3 risk checking.
        NO futures math should be done in RL v3 - all calculations here.
        
        Args:
            trade_intent: {
                "symbol": str,
                "side": str (LONG/SHORT/FLAT),
                "size_pct": float (0-1, portfolio allocation),
                "leverage": int,
                "confidence": float,
                "source": str,
                "trace_id": str
            }
            trace_id: Correlation ID
            
        Returns:
            (approved, reason)
        """
        symbol = trade_intent.get("symbol", "").upper()
        side = trade_intent.get("side", "FLAT")
        size_pct = trade_intent.get("size_pct", 0.0)
        leverage = trade_intent.get("leverage", 1)
        confidence = trade_intent.get("confidence", 0.0)
        
        # GUARD 1: Validate inputs
        if not symbol or side == "FLAT":
            return (False, "Invalid trade intent: no symbol or FLAT side")
        
        if size_pct <= 0 or size_pct > 1.0:
            return (False, f"Invalid size_pct: {size_pct} (must be 0-1)")
        
        # GUARD 2: Get RL v3 policy limits
        rl_v3_config = {}
        if self.policy_store:
            try:
                policy = await self.policy_store.get_policy()
                rl_v3_config = getattr(policy, "rl_v3_live", {})
            except Exception as e:
                logger.error(f"[RiskGuard] Failed to get rl_v3_live config: {e}")
        
        max_size_pct = rl_v3_config.get("max_size_pct", 0.15)
        max_loss_pct = rl_v3_config.get("max_loss_per_trade_pct", 0.02)
        
        # GUARD 3: Size limit
        if size_pct > max_size_pct:
            return (False, f"size_pct {size_pct:.2%} exceeds limit {max_size_pct:.2%}")
        
        # GUARD 4: Leverage check (use active risk profile max_leverage)
        risk_profile = None
        if self.policy_store:
            try:
                risk_profile = await self.policy_store.get_active_risk_profile()
            except:
                pass
        
        max_leverage = risk_profile.get("max_leverage", 5.0) if risk_profile else 5.0
        if leverage > max_leverage:
            return (False, f"leverage {leverage}x exceeds limit {max_leverage}x")
        
        # GUARD 5: Check open position (prevent double intent)
        # NOTE: This requires state store integration - placeholder for now
        # TODO: Integrate with execution_adapter.get_open_positions()
        
        # GUARD 6: Estimate liquidation buffer (futures cross margin safety)
        liq_buffer_pct = rl_v3_config.get("liq_buffer_pct", 0.20)
        # In cross margin, if size_pct * leverage approaches 1.0, liquidation risk is high
        margin_usage = size_pct * leverage
        if margin_usage > (1.0 - liq_buffer_pct):
            return (False, f"margin usage {margin_usage:.2%} too high (liq_buffer={liq_buffer_pct:.2%})")
        
        logger.info(
            f"[RiskGuard] ✅ Trade intent approved: {symbol} {side} {size_pct:.2%} {leverage}x",
            extra={
                "symbol": symbol,
                "side": side,
                "size_pct": size_pct,
                "leverage": leverage,
                "confidence": confidence,
                "trace_id": trace_id,
            }
        )
        
        return (True, "")


__all__ = [
    "RiskGuardService",
    "RiskStateStore",
    "InMemoryRiskStateStore",
    "SqliteRiskStateStore",
    "KillSwitchState",
]
