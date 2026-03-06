"""Tests for position_source.py — read-only parsing from quantum:position:* hashes.

Coverage goals
--------------
* _parse_hash: field parsing, side normalisation, zero/negative/missing quantity,
  missing entry_price, optional field defaults, mark_price derivation.
* PositionSource.get_open_positions(): SCAN → HGETALL → PositionSnapshot round-trip,
  allowlist filtering, max_positions cap, SCAN exception returns [], HGETALL
  exception skips the symbol but continues.
* Read-only guarantee: _FakeRedis exposes ONLY scan_position_keys() and
  hgetall_position(). Any write call (xadd, set_with_ttl, hset, …) would
  raise AttributeError and immediately fail the test.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.models import PositionSnapshot
from microservices.exit_management_agent.position_source import (
    PositionSource,
    _parse_hash,
)


# ── Fake Redis ─────────────────────────────────────────────────────────────────


class _FakeRedis:
    """
    Implements only the two READ methods used by PositionSource.
    No write methods are defined — any write call raises AttributeError,
    which immediately fails the test. This makes the read-only contract
    testable by construction without explicit mock assertions.
    """

    def __init__(
        self,
        keys: list | None = None,
        data_map: dict | None = None,
        scan_raises: bool = False,
        hgetall_raises_for: set | None = None,
    ) -> None:
        self._keys = keys or []
        self._data_map = data_map or {}
        self._scan_raises = scan_raises
        self._hgetall_raises_for = hgetall_raises_for or set()
        self.hgetall_calls: list = []

    async def scan_position_keys(
        self, match: str = "quantum:position:*", batch: int = 200
    ) -> list:
        if self._scan_raises:
            raise RuntimeError("SCAN failed — simulated Redis error")
        return list(self._keys)

    async def hgetall_position(self, key: str) -> dict:
        self.hgetall_calls.append(key)
        if key in self._hgetall_raises_for:
            raise RuntimeError(f"HGETALL failed for {key} — simulated")
        return dict(self._data_map.get(key, {}))


# ── Canonical test hash data ───────────────────────────────────────────────────

_VALID_LONG: dict = {
    "side": "LONG",
    "quantity": "0.01",
    "entry_price": "30000.0",
    "unrealized_pnl": "15.0",   # mark = 30000 + 15/0.01 = 31500
    "leverage": "10.0",
    "stop_loss": "29000.0",
    "take_profit": "35000.0",
    "entry_risk_usdt": "100.0",
    "sync_timestamp": "1700000000",
}

_VALID_SHORT: dict = {
    "side": "SHORT",
    "quantity": "0.02",
    "entry_price": "50000.0",
    "unrealized_pnl": "10.0",   # SHORT in profit: price dropped
    "leverage": "5.0",
    "stop_loss": "51000.0",
    "take_profit": "45000.0",
    "entry_risk_usdt": "50.0",
    "sync_timestamp": "1700000000",
}


# ── _parse_hash unit tests ─────────────────────────────────────────────────────


class TestParseHash:
    def test_valid_long_returns_snapshot(self):
        snap = _parse_hash("BTCUSDT", _VALID_LONG)
        assert isinstance(snap, PositionSnapshot)
        assert snap.symbol == "BTCUSDT"
        assert snap.side == "LONG"
        assert snap.quantity == pytest.approx(0.01)
        assert snap.entry_price == pytest.approx(30_000.0)
        assert snap.leverage == pytest.approx(10.0)

    def test_valid_long_derives_mark_price(self):
        # LONG: mark = entry + unrealized_pnl / quantity = 30000 + 15/0.01 = 31500
        snap = _parse_hash("BTCUSDT", _VALID_LONG)
        assert snap.mark_price == pytest.approx(31_500.0)

    def test_valid_short_returns_snapshot(self):
        snap = _parse_hash("ETHUSDT", _VALID_SHORT)
        assert isinstance(snap, PositionSnapshot)
        assert snap.side == "SHORT"
        assert snap.quantity == pytest.approx(0.02)
        assert snap.entry_price == pytest.approx(50_000.0)

    def test_empty_dict_returns_none(self):
        assert _parse_hash("BTCUSDT", {}) is None

    def test_zero_quantity_returns_none(self):
        data = {**_VALID_LONG, "quantity": "0"}
        assert _parse_hash("BTCUSDT", data) is None

    def test_negative_quantity_returns_none(self):
        data = {**_VALID_LONG, "quantity": "-0.01"}
        assert _parse_hash("BTCUSDT", data) is None

    def test_non_numeric_quantity_returns_none(self):
        data = {**_VALID_LONG, "quantity": "not_a_number"}
        assert _parse_hash("BTCUSDT", data) is None

    def test_missing_quantity_key_returns_none(self):
        data = {k: v for k, v in _VALID_LONG.items() if k != "quantity"}
        assert _parse_hash("BTCUSDT", data) is None

    def test_zero_entry_price_returns_none(self):
        data = {**_VALID_LONG, "entry_price": "0"}
        assert _parse_hash("BTCUSDT", data) is None

    def test_missing_entry_price_returns_none(self):
        data = {k: v for k, v in _VALID_LONG.items() if k != "entry_price"}
        assert _parse_hash("BTCUSDT", data) is None

    def test_buy_side_normalised_to_long(self):
        data = {**_VALID_LONG, "side": "BUY"}
        snap = _parse_hash("BTCUSDT", data)
        assert snap is not None
        assert snap.side == "LONG"

    def test_sell_side_normalised_to_short(self):
        data = {**_VALID_SHORT, "side": "SELL"}
        snap = _parse_hash("ETHUSDT", data)
        assert snap is not None
        assert snap.side == "SHORT"

    def test_unknown_side_defaults_to_long(self):
        data = {**_VALID_LONG, "side": "UNDEFINED_DIRECTION"}
        snap = _parse_hash("BTCUSDT", data)
        assert snap is not None
        assert snap.side == "LONG"

    def test_missing_optional_fields_use_safe_defaults(self):
        # Only the two required fields; everything else should default safely.
        minimal = {
            "quantity": "0.01",
            "entry_price": "30000.0",
        }
        snap = _parse_hash("BTCUSDT", minimal)
        assert isinstance(snap, PositionSnapshot)
        assert snap.stop_loss == pytest.approx(0.0)
        assert snap.take_profit == pytest.approx(0.0)
        assert snap.entry_risk_usdt == pytest.approx(0.0)
        assert snap.leverage == pytest.approx(1.0)  # min-clamped to 1.0

    def test_optional_stop_loss_parsed_correctly(self):
        snap = _parse_hash("BTCUSDT", _VALID_LONG)
        assert snap.stop_loss == pytest.approx(29_000.0)

    def test_optional_entry_risk_usdt_parsed_correctly(self):
        snap = _parse_hash("BTCUSDT", _VALID_LONG)
        assert snap.entry_risk_usdt == pytest.approx(100.0)


# ── PositionSource integration tests ──────────────────────────────────────────


class TestPositionSource:
    @pytest.mark.asyncio
    async def test_no_keys_returns_empty_list(self):
        src = PositionSource(_FakeRedis(keys=[], data_map={}))
        assert await src.get_open_positions() == []

    @pytest.mark.asyncio
    async def test_one_valid_long_position_returned(self):
        fake = _FakeRedis(
            keys=["quantum:position:BTCUSDT"],
            data_map={"quantum:position:BTCUSDT": _VALID_LONG},
        )
        result = await PositionSource(fake).get_open_positions()
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"
        assert result[0].side == "LONG"

    @pytest.mark.asyncio
    async def test_invalid_hash_is_skipped(self):
        fake = _FakeRedis(
            keys=["quantum:position:BTCUSDT"],
            data_map={"quantum:position:BTCUSDT": {}},  # empty → _parse_hash returns None
        )
        result = await PositionSource(fake).get_open_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_exception_returns_empty_list(self):
        fake = _FakeRedis(scan_raises=True)
        result = await PositionSource(fake).get_open_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_hgetall_exception_skips_symbol_continues_processing(self):
        """HGETALL failure for one key must not abort the loop."""
        fake = _FakeRedis(
            keys=["quantum:position:BTCUSDT", "quantum:position:ETHUSDT"],
            data_map={"quantum:position:ETHUSDT": _VALID_LONG},
            hgetall_raises_for={"quantum:position:BTCUSDT"},
        )
        result = await PositionSource(fake).get_open_positions()
        # BTCUSDT errored → skipped; ETHUSDT should still come through
        assert len(result) == 1
        assert result[0].symbol == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_allowlist_filters_out_unlisted_symbols(self):
        fake = _FakeRedis(
            keys=["quantum:position:BTCUSDT", "quantum:position:ETHUSDT"],
            data_map={
                "quantum:position:BTCUSDT": _VALID_LONG,
                "quantum:position:ETHUSDT": {**_VALID_LONG, "quantity": "0.5"},
            },
        )
        result = await PositionSource(fake).get_open_positions(
            allowlist=frozenset({"BTCUSDT"})
        )
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_empty_allowlist_returns_all_symbols(self):
        fake = _FakeRedis(
            keys=["quantum:position:BTCUSDT", "quantum:position:ETHUSDT"],
            data_map={
                "quantum:position:BTCUSDT": _VALID_LONG,
                "quantum:position:ETHUSDT": {**_VALID_LONG, "quantity": "0.5"},
            },
        )
        result = await PositionSource(fake).get_open_positions(allowlist=None)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_max_positions_caps_number_of_results(self):
        keys = [f"quantum:position:TOKEN{i}USDT" for i in range(10)]
        data_map = {k: dict(_VALID_LONG) for k in keys}
        fake = _FakeRedis(keys=keys, data_map=data_map)
        result = await PositionSource(fake, max_positions=3).get_open_positions()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_position_source_is_read_only(self):
        """
        _FakeRedis defines NO write methods.  Any call to xadd(), set_with_ttl(),
        hset(), etc. would raise AttributeError and fail this test immediately.
        Passing this test proves PositionSource only called read methods.
        """
        key = "quantum:position:BTCUSDT"
        fake = _FakeRedis(keys=[key], data_map={key: _VALID_LONG})
        src = PositionSource(fake)
        result = await src.get_open_positions()
        assert len(result) == 1
        # Confirm only hgetall_position was called (not any write method)
        assert fake.hgetall_calls == [key]
