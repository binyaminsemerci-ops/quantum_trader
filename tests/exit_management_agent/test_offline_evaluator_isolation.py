"""Tests for PATCH-11 offline evaluator isolation.

The critical safety property: OfflineEvaluator must NEVER be importable
or callable from the live tick path (main.py).
"""
from __future__ import annotations

import ast
import importlib
import inspect

import pytest

from microservices.exit_management_agent.llm.offline_evaluator import (
    OfflineEvaluator,
    OfflineEvaluation,
)


class TestOfflineEvaluatorIsolation:
    """Structural isolation — offline evaluator must not be reachable from live path."""

    def test_main_does_not_import_offline_evaluator(self):
        """main.py must NEVER import offline_evaluator or OfflineEvaluator."""
        import microservices.exit_management_agent.main as main_mod
        source = inspect.getsource(main_mod)
        assert "offline_evaluator" not in source
        assert "OfflineEvaluator" not in source

    def test_main_ast_no_offline_import(self):
        """Parse main.py AST to detect any form of offline_evaluator import."""
        import microservices.exit_management_agent.main as main_mod
        source = inspect.getsource(main_mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "offline_evaluator" not in node.module
                if hasattr(node, "names"):
                    for alias in node.names:
                        assert "offline_evaluator" not in (alias.name or "")
                        assert "OfflineEvaluator" not in (alias.name or "")

    def test_judge_orchestrator_does_not_import_offline_evaluator(self):
        """JudgeOrchestrator must NEVER use offline evaluator."""
        import microservices.exit_management_agent.llm.judge_orchestrator as orch_mod
        source = inspect.getsource(orch_mod)
        assert "offline_evaluator" not in source
        assert "OfflineEvaluator" not in source

    def test_ensemble_bridge_does_not_import_offline_evaluator(self):
        """EnsembleBridge must not reference offline evaluator."""
        import microservices.exit_management_agent.ensemble_bridge as bridge_mod
        source = inspect.getsource(bridge_mod)
        assert "offline_evaluator" not in source
        assert "OfflineEvaluator" not in source


class TestOfflineEvaluatorInterface:
    """OfflineEvaluator only accepts closed trade records, not live state."""

    def test_evaluate_requires_trade_record(self):
        """The method signature takes a dict (trade record), not live objects."""
        sig = inspect.signature(OfflineEvaluator.evaluate_closed_trade)
        params = list(sig.parameters.keys())
        assert "trade_record" in params
        # Must NOT accept position state, pipeline ctx, or snapshot
        assert "state" not in params
        assert "ctx" not in params
        assert "snapshot" not in params
        assert "pipeline_ctx" not in params

    def test_no_redis_in_constructor(self):
        """Constructor must not accept Redis client."""
        sig = inspect.signature(OfflineEvaluator.__init__)
        params = list(sig.parameters.keys())
        assert "redis" not in params
        assert "redis_client" not in params
        assert "redis_host" not in params

    def test_evaluation_result_is_immutable(self):
        """OfflineEvaluation should be frozen dataclass."""
        r = OfflineEvaluation(
            symbol="BTCUSDT",
            trade_id="t-001",
            overall_grade="B+",
            exit_timing="LATE",
            missed_opportunity_pct=5.0,
            should_have_action="REDUCE_50",
            key_learnings=["lesson"],
            threshold_suggestions={},
            confidence=0.7,
            raw="{}",
            evaluator_model="llama-3.3-70b-versatile",
            success=True,
        )
        with pytest.raises(AttributeError):
            r.symbol = "ETHUSDT"
