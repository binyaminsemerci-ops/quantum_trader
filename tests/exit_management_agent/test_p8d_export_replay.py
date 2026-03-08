"""test_p8d_export_replay: tests for PATCH-8D replay schema and serialization.

Scope: ops.offline.replay_schema (ReplayRecord, helpers) and the pure
utility functions in ops.offline.export_replay (_parse_stream_id,
_date_to_epoch_ms, _ms_to_stream_id, _write_jsonl).

No Redis connections are made in any test — all I/O is either in-memory or
uses tmp_path (pytest fixture).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ops.offline.replay_schema import (
    ReplayRecord,
    _bool_field,
    _decode,
    _opt_float,
    _opt_int,
)
from ops.offline.export_replay import (
    _date_to_epoch_ms,
    _ms_to_stream_id,
    _parse_stream_id,
    _write_jsonl,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────────

def _full_fields_str() -> dict[str, str]:
    """A complete replay record dict with string keys and values."""
    return {
        "decision_id": "dec-abc123",
        "symbol": "BTCUSDT",
        "record_time_epoch": "1741200000",
        "patch": "PATCH-8C",
        "source": "exit_management_agent",
        "live_action": "FULL_CLOSE",
        "formula_action": "FULL_CLOSE",
        "qwen3_action": "FULL_CLOSE",
        "diverged": "false",
        "exit_score": "0.85",
        "entry_price": "95000.50",
        "side": "LONG",
        "quantity": "0.01",
        "hold_duration_sec": "3720",
        "close_price": "97500.00",
        "closed_by": "exit_management_agent",
        "outcome_action": "FULL_CLOSE",
        "reward": "0.850000",
        "regret_label": "none",
        "regret_score": "0.0000",
        "preferred_action": "FULL_CLOSE",
    }


def _full_fields_bytes() -> dict[bytes, bytes]:
    """Same record with bytes keys and bytes values (as returned by redis-py)."""
    return {k.encode(): v.encode() for k, v in _full_fields_str().items()}


# ────────────────────────────────────────────────────────────────────────────────
# TestDecodeHelper
# ────────────────────────────────────────────────────────────────────────────────

class TestDecodeHelper:

    def test_bytes_decoded_to_str(self):
        assert _decode(b"hello") == "hello"

    def test_str_passthrough(self):
        assert _decode("world") == "world"

    def test_none_returns_empty_str(self):
        assert _decode(None) == ""

    def test_int_stringified(self):
        assert _decode(42) == "42"

    def test_bytes_with_replacement(self):
        # Invalid UTF-8 bytes should not raise
        result = _decode(b"\xff\xfe")
        assert isinstance(result, str)


# ────────────────────────────────────────────────────────────────────────────────
# TestOptFloat
# ────────────────────────────────────────────────────────────────────────────────

class TestOptFloat:

    def test_valid_float(self):
        assert _opt_float("0.85") == pytest.approx(0.85)

    def test_negative_float(self):
        assert _opt_float("-0.35") == pytest.approx(-0.35)

    def test_null_string_returns_none(self):
        assert _opt_float("null") is None

    def test_none_string_returns_none(self):
        assert _opt_float("None") is None

    def test_empty_string_returns_none(self):
        assert _opt_float("") is None

    def test_nan_returns_none(self):
        assert _opt_float("nan") is None

    def test_garbage_returns_none(self):
        assert _opt_float("UNKNOWN") is None

    def test_integer_string(self):
        assert _opt_float("1") == pytest.approx(1.0)


# ────────────────────────────────────────────────────────────────────────────────
# TestOptInt
# ────────────────────────────────────────────────────────────────────────────────

class TestOptInt:

    def test_valid_int(self):
        assert _opt_int("3720") == 3720

    def test_zero(self):
        assert _opt_int("0") == 0

    def test_null_string_returns_none(self):
        assert _opt_int("null") is None

    def test_empty_string_returns_none(self):
        assert _opt_int("") is None

    def test_float_string_returns_none(self):
        # int() does not accept "3720.5"
        assert _opt_int("3720.5") is None

    def test_garbage_returns_none(self):
        assert _opt_int("abc") is None


# ────────────────────────────────────────────────────────────────────────────────
# TestBoolField
# ────────────────────────────────────────────────────────────────────────────────

class TestBoolField:

    def test_true_lowercase(self):
        assert _bool_field("true") is True

    def test_false_lowercase(self):
        assert _bool_field("false") is False

    def test_true_mixed_case(self):
        assert _bool_field("True") is True

    def test_false_with_space(self):
        assert _bool_field(" false ") is False

    def test_empty_is_false(self):
        assert _bool_field("") is False

    def test_null_is_false(self):
        assert _bool_field("null") is False


# ────────────────────────────────────────────────────────────────────────────────
# TestReplayRecordFromRedisEntry — str keys
# ────────────────────────────────────────────────────────────────────────────────

class TestReplayRecordFromRedisEntryStr:

    def _make(self, **overrides) -> ReplayRecord:
        fields = _full_fields_str()
        fields.update(overrides)
        return ReplayRecord.from_redis_entry("1741200000123-0", fields)

    def test_stream_id_stored(self):
        rec = self._make()
        assert rec.stream_id == "1741200000123-0"

    def test_basic_str_fields(self):
        rec = self._make()
        assert rec.decision_id == "dec-abc123"
        assert rec.symbol == "BTCUSDT"
        assert rec.patch == "PATCH-8C"
        assert rec.source == "exit_management_agent"
        assert rec.live_action == "FULL_CLOSE"
        assert rec.closed_by == "exit_management_agent"
        assert rec.regret_label == "none"
        assert rec.preferred_action == "FULL_CLOSE"

    def test_numeric_fields_parsed(self):
        rec = self._make()
        assert rec.record_time_epoch == 1741200000
        assert rec.exit_score == pytest.approx(0.85)
        assert rec.entry_price == pytest.approx(95000.50)
        assert rec.quantity == pytest.approx(0.01)
        assert rec.hold_duration_sec == 3720
        assert rec.close_price == pytest.approx(97500.0)
        assert rec.reward == pytest.approx(0.85)
        assert rec.regret_score == pytest.approx(0.0)

    def test_diverged_false(self):
        rec = self._make()
        assert rec.diverged is False

    def test_diverged_true(self):
        rec = self._make(diverged="true")
        assert rec.diverged is True

    def test_side_and_outcome_action(self):
        rec = self._make()
        assert rec.side == "LONG"
        assert rec.outcome_action == "FULL_CLOSE"


# ────────────────────────────────────────────────────────────────────────────────
# TestReplayRecordFromRedisEntry — bytes keys/values
# ────────────────────────────────────────────────────────────────────────────────

class TestReplayRecordFromRedisEntryBytes:

    def test_bytes_keys_and_values_parsed(self):
        rec = ReplayRecord.from_redis_entry(b"1741200000999-0", _full_fields_bytes())
        assert rec.stream_id == "1741200000999-0"
        assert rec.decision_id == "dec-abc123"
        assert rec.symbol == "BTCUSDT"
        assert rec.diverged is False
        assert rec.reward == pytest.approx(0.85)

    def test_bytes_stream_id_decoded(self):
        rec = ReplayRecord.from_redis_entry(b"9999000000001-5", _full_fields_bytes())
        assert rec.stream_id == "9999000000001-5"


# ────────────────────────────────────────────────────────────────────────────────
# TestNullAndMissingFields
# ────────────────────────────────────────────────────────────────────────────────

class TestNullAndMissingFields:

    def test_exit_score_null_is_none(self):
        rec = ReplayRecord.from_redis_entry("0-0", {"exit_score": "null"})
        assert rec.exit_score is None

    def test_record_time_epoch_missing_is_none(self):
        rec = ReplayRecord.from_redis_entry("0-0", {})
        assert rec.record_time_epoch is None

    def test_hold_duration_null_is_none(self):
        rec = ReplayRecord.from_redis_entry("0-0", {"hold_duration_sec": "null"})
        assert rec.hold_duration_sec is None

    def test_reward_null_is_none(self):
        rec = ReplayRecord.from_redis_entry("0-0", {"reward": "null"})
        assert rec.reward is None

    def test_regret_score_null_is_none(self):
        rec = ReplayRecord.from_redis_entry("0-0", {"regret_score": "null"})
        assert rec.regret_score is None

    def test_empty_dict_uses_defaults(self):
        rec = ReplayRecord.from_redis_entry("0-1", {})
        assert rec.decision_id == ""
        assert rec.symbol == ""
        assert rec.live_action == "UNKNOWN"
        assert rec.formula_action == "null"
        assert rec.qwen3_action == "null"
        assert rec.diverged is False
        assert rec.closed_by == "unknown"
        assert rec.regret_label == "none"
        assert rec.preferred_action == "HOLD"
        assert rec.exit_score is None
        assert rec.reward is None

    def test_partial_dict_fills_remaining_defaults(self):
        rec = ReplayRecord.from_redis_entry("1-0", {
            "symbol": "ETHUSDT",
            "live_action": "HOLD",
        })
        assert rec.symbol == "ETHUSDT"
        assert rec.live_action == "HOLD"
        assert rec.decision_id == ""
        assert rec.reward is None


# ────────────────────────────────────────────────────────────────────────────────
# TestToDict
# ────────────────────────────────────────────────────────────────────────────────

class TestToDict:

    def _full_record(self) -> ReplayRecord:
        return ReplayRecord.from_redis_entry("1741200000123-0", _full_fields_str())

    def test_returns_dict(self):
        assert isinstance(self._full_record().to_dict(), dict)

    def test_all_fields_present(self):
        d = self._full_record().to_dict()
        expected_keys = {
            "stream_id", "decision_id", "symbol", "record_time_epoch",
            "patch", "source", "live_action", "formula_action", "qwen3_action",
            "diverged", "exit_score", "entry_price", "side", "quantity",
            "hold_duration_sec", "close_price", "closed_by", "outcome_action",
            "reward", "regret_label", "regret_score", "preferred_action",
        }
        assert expected_keys == set(d.keys())

    def test_numeric_field_types(self):
        d = self._full_record().to_dict()
        assert isinstance(d["exit_score"], float)
        assert isinstance(d["reward"], float)
        assert isinstance(d["hold_duration_sec"], int)
        assert isinstance(d["record_time_epoch"], int)
        assert isinstance(d["diverged"], bool)

    def test_none_fields_remain_none(self):
        rec = ReplayRecord.from_redis_entry("0-0", {})
        d = rec.to_dict()
        assert d["exit_score"] is None
        assert d["reward"] is None
        assert d["record_time_epoch"] is None

    def test_json_serializable(self):
        d = self._full_record().to_dict()
        # Must not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ────────────────────────────────────────────────────────────────────────────────
# TestToJsonLine
# ────────────────────────────────────────────────────────────────────────────────

class TestToJsonLine:

    def _full_record(self) -> ReplayRecord:
        return ReplayRecord.from_redis_entry("1741200000123-0", _full_fields_str())

    def test_returns_str(self):
        assert isinstance(self._full_record().to_json_line(), str)

    def test_no_trailing_newline(self):
        line = self._full_record().to_json_line()
        assert not line.endswith("\n")

    def test_valid_json(self):
        line = self._full_record().to_json_line()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_round_trip_values(self):
        rec = self._full_record()
        parsed = json.loads(rec.to_json_line())
        assert parsed["symbol"] == rec.symbol
        assert parsed["decision_id"] == rec.decision_id
        assert parsed["reward"] == pytest.approx(rec.reward)
        assert parsed["hold_duration_sec"] == rec.hold_duration_sec
        assert parsed["diverged"] is rec.diverged
        assert parsed["regret_label"] == rec.regret_label

    def test_null_numeric_fields_as_json_null(self):
        rec = ReplayRecord.from_redis_entry("0-0", {})
        parsed = json.loads(rec.to_json_line())
        assert parsed["exit_score"] is None
        assert parsed["reward"] is None
        assert parsed["record_time_epoch"] is None

    def test_single_no_extra_whitespace(self):
        line = self._full_record().to_json_line()
        # Compact JSON: re-parsing and re-serializing should be identical
        assert json.dumps(json.loads(line), ensure_ascii=False) == line


# ────────────────────────────────────────────────────────────────────────────────
# TestExportUtilFunctions
# ────────────────────────────────────────────────────────────────────────────────

class TestParseStreamId:

    def test_str_standard(self):
        ts, seq = _parse_stream_id("1741200000123-7")
        assert ts == 1741200000123
        assert seq == 7

    def test_bytes_standard(self):
        ts, seq = _parse_stream_id(b"1741200000000-0")
        assert ts == 1741200000000
        assert seq == 0

    def test_seq_defaults_to_zero_if_absent(self):
        # Redis IDs always have a "-seq" part, but handle edge case gracefully
        ts, seq = _parse_stream_id("1741200000000")
        assert ts == 1741200000000
        assert seq == 0

    def test_large_seq(self):
        ts, seq = _parse_stream_id("1000-999")
        assert ts == 1000
        assert seq == 999


class TestDateHelpers:

    def test_date_to_epoch_ms_known_date(self):
        # 2026-03-01T00:00:00 UTC
        ms = _date_to_epoch_ms("2026-03-01")
        from datetime import datetime, timezone as _tz
        dt = datetime(2026, 3, 1, tzinfo=_tz.utc)
        assert ms == int(dt.timestamp() * 1000)

    def test_ms_to_stream_id(self):
        assert _ms_to_stream_id(1741200000000) == "1741200000000-0"

    def test_round_trip_date_to_stream_id(self):
        ms = _date_to_epoch_ms("2026-03-09")
        sid = _ms_to_stream_id(ms)
        assert sid.endswith("-0")
        ts, _ = _parse_stream_id(sid)
        assert ts == ms


# ────────────────────────────────────────────────────────────────────────────────
# TestWriteJsonl
# ────────────────────────────────────────────────────────────────────────────────

class TestWriteJsonl:

    def _make_records(self, n: int) -> list[ReplayRecord]:
        records = []
        for i in range(n):
            fields = _full_fields_str()
            fields["decision_id"] = f"dec-{i:04d}"
            fields["symbol"] = "BTCUSDT"
            fields["reward"] = f"{0.1 * i:.6f}"
            records.append(ReplayRecord.from_redis_entry(f"100000{i}-0", fields))
        return records

    def test_creates_file(self, tmp_path):
        out = tmp_path / "replay_2026-03-09.jsonl"
        _write_jsonl(out, self._make_records(3))
        assert out.exists()

    def test_correct_line_count(self, tmp_path):
        out = tmp_path / "replay_test.jsonl"
        _write_jsonl(out, self._make_records(5))
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5

    def test_each_line_valid_json(self, tmp_path):
        out = tmp_path / "replay_test.jsonl"
        _write_jsonl(out, self._make_records(4))
        for line in out.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            assert "decision_id" in obj

    def test_appends_on_second_call(self, tmp_path):
        out = tmp_path / "replay_append.jsonl"
        _write_jsonl(out, self._make_records(2))
        _write_jsonl(out, self._make_records(3))
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5

    def test_creates_parent_dir_indirectly(self, tmp_path):
        # _write_jsonl itself does not mkdir; caller (export()) does.
        # But the file write itself should work when dir already exists.
        subdir = tmp_path / "logs" / "replay"
        subdir.mkdir(parents=True)
        out = subdir / "replay_2026-03-09.jsonl"
        _write_jsonl(out, self._make_records(1))
        assert out.exists()

    def test_empty_record_list_creates_empty_file(self, tmp_path):
        out = tmp_path / "replay_empty.jsonl"
        _write_jsonl(out, [])
        assert out.exists()
        assert out.read_text() == ""

    def test_unicode_symbol_roundtrip(self, tmp_path):
        fields = _full_fields_str()
        fields["symbol"] = "BNBUSDT"
        rec = ReplayRecord.from_redis_entry("1-0", fields)
        out = tmp_path / "replay_unicode.jsonl"
        _write_jsonl(out, [rec])
        parsed = json.loads(out.read_text(encoding="utf-8").strip())
        assert parsed["symbol"] == "BNBUSDT"
