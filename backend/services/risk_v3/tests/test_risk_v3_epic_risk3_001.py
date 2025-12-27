"""
Tests for Risk v3 - EPIC-RISK3-001

Tests all components:
- Exposure Matrix Engine
- VaR/ES Engine
- Systemic Risk Detector
- Risk Rules Engine
- Risk Orchestrator (end-to-end)
"""

import pytest
import asyncio
from datetime import datetime
from typing import List

# Import Risk v3 components
import sys
sys.path.insert(0, "c:/quantum_trader")

from backend.services.risk_v3.models import (
    RiskSnapshot,
    PositionExposure,
    ExposureMatrix,
    VaRResult,
    ESResult,
    SystemicRiskSignal,
    RiskLevel,
    SystemicRiskType,
    RiskLimits,
)
from backend.services.risk_v3.exposure_matrix import (
    ExposureMatrixEngine,
    compute_symbol_exposure,
    compute_exchange_exposure,
    compute_strategy_exposure,
)
from backend.services.risk_v3.var_es import VaRESEngine
from backend.services.risk_v3.systemic import SystemicRiskDetector
from backend.services.risk_v3.rules import RiskRulesEngine, ESSTier
from backend.services.risk_v3.orchestrator import RiskOrchestrator


# === Fixtures ===

@pytest.fixture
def sample_positions() -> List[PositionExposure]:
    """Sample positions for testing"""
    return [
        PositionExposure(
            symbol="BTCUSDT",
            exchange="binance",
            strategy="trend_following",
            quantity=0.5,
            notional_usd=15000.0,
            leverage=3.0,
            unrealized_pnl=500.0,
            entry_price=30000.0,
            current_price=31000.0,
            risk_pct=2.0,
        ),
        PositionExposure(
            symbol="ETHUSDT",
            exchange="binance",
            strategy="mean_reversion",
            quantity=5.0,
            notional_usd=10000.0,
            leverage=2.0,
            unrealized_pnl=-200.0,
            entry_price=2000.0,
            current_price=1960.0,
            risk_pct=1.5,
        ),
        PositionExposure(
            symbol="SOLUSDT",
            exchange="okx",
            strategy="trend_following",
            quantity=50.0,
            notional_usd=5000.0,
            leverage=2.0,
            unrealized_pnl=100.0,
            entry_price=100.0,
            current_price=102.0,
            risk_pct=1.0,
        ),
    ]


@pytest.fixture
def sample_snapshot(sample_positions) -> RiskSnapshot:
    """Sample risk snapshot for testing"""
    return RiskSnapshot(
        timestamp=datetime.utcnow(),
        positions=sample_positions,
        symbol_exposure={"BTCUSDT": 15000.0, "ETHUSDT": 10000.0, "SOLUSDT": 5000.0},
        symbol_leverage={"BTCUSDT": 3.0, "ETHUSDT": 2.0, "SOLUSDT": 2.0},
        exchange_exposure={"binance": 25000.0, "okx": 5000.0},
        exchange_position_count={"binance": 2, "okx": 1},
        strategy_exposure={"trend_following": 20000.0, "mean_reversion": 10000.0},
        strategy_position_count={"trend_following": 2, "mean_reversion": 1},
        account_balance=10000.0,
        total_equity=10400.0,  # balance + unrealized PnL
        total_notional=30000.0,
        total_unrealized_pnl=400.0,
        total_leverage=2.88,  # 30000 / 10400
        drawdown_pct=0.02,
        daily_pnl=400.0,
        weekly_pnl=1500.0,
        volatility_cluster="medium",
        regime="trending",
    )


@pytest.fixture
def sample_returns():
    """Sample returns data for VaR/ES calculation"""
    return {
        "BTCUSDT": [0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.01, 0.02, -0.005],
        "ETHUSDT": [0.015, -0.02, 0.025, -0.015, 0.01, -0.01, 0.02, -0.015, 0.015, -0.01],
        "SOLUSDT": [0.03, -0.025, 0.02, -0.01, 0.015, -0.02, 0.025, -0.01, 0.01, -0.015],
    }


@pytest.fixture
def risk_limits() -> RiskLimits:
    """Sample risk limits"""
    return RiskLimits(
        max_leverage=5.0,
        max_position_size_usd=15000.0,
        max_daily_drawdown_pct=5.0,
        max_correlation=0.80,
        var_95_limit=1000.0,
        var_99_limit=2000.0,
        es_975_limit=2500.0,
        max_symbol_concentration=0.60,
        max_exchange_concentration=0.85,
        min_liquidity_score=0.50,
        correlation_spike_threshold=0.20,
        volatility_spike_threshold=2.0,
    )


# === Exposure Matrix Tests ===

def test_compute_symbol_exposure(sample_positions):
    """Test symbol exposure computation"""
    exposure = compute_symbol_exposure(sample_positions)
    
    assert len(exposure) == 3
    assert exposure["BTCUSDT"] == 15000.0
    assert exposure["ETHUSDT"] == 10000.0
    assert exposure["SOLUSDT"] == 5000.0


def test_compute_exchange_exposure(sample_positions):
    """Test exchange exposure computation"""
    exposure = compute_exchange_exposure(sample_positions)
    
    assert len(exposure) == 2
    assert exposure["binance"] == 25000.0
    assert exposure["okx"] == 5000.0


def test_compute_strategy_exposure(sample_positions):
    """Test strategy exposure computation"""
    exposure = compute_strategy_exposure(sample_positions)
    
    assert len(exposure) == 2
    assert exposure["trend_following"] == 20000.0
    assert exposure["mean_reversion"] == 10000.0


def test_exposure_matrix_basic(sample_snapshot):
    """Test basic exposure matrix computation"""
    engine = ExposureMatrixEngine()
    
    exposure_matrix = engine.compute_exposure_matrix(sample_snapshot)
    
    assert isinstance(exposure_matrix, ExposureMatrix)
    assert len(exposure_matrix.normalized_symbol_exposure) == 3
    assert 0.0 <= exposure_matrix.symbol_concentration_hhi <= 1.0
    assert 0.0 <= exposure_matrix.exchange_concentration_hhi <= 1.0
    
    # Check normalized exposures sum to ~1.0
    total_normalized = sum(exposure_matrix.normalized_symbol_exposure.values())
    assert abs(total_normalized - 1.0) < 0.01


def test_exposure_matrix_hotspots(sample_snapshot):
    """Test risk hotspot identification"""
    engine = ExposureMatrixEngine()
    
    exposure_matrix = engine.compute_exposure_matrix(sample_snapshot)
    
    # BTCUSDT should be identified as hotspot (50% exposure)
    assert len(exposure_matrix.risk_hotspots) > 0
    
    btc_hotspot = next(
        (h for h in exposure_matrix.risk_hotspots if h["name"] == "BTCUSDT"),
        None
    )
    assert btc_hotspot is not None
    assert btc_hotspot["exposure_pct"] == 0.50  # 15000 / 30000


# === VaR/ES Tests ===

def test_var_delta_normal():
    """Test delta-normal VaR calculation"""
    engine = VaRESEngine()
    
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.01, 0.02, -0.005]
    portfolio_value = 10000.0
    
    var = engine.compute_var(
        returns=returns,
        portfolio_value=portfolio_value,
        method="delta_normal",
        level=0.95,
    )
    
    assert var > 0  # VaR should be positive
    assert var < portfolio_value  # VaR should be less than portfolio


def test_var_historical():
    """Test historical VaR calculation"""
    engine = VaRESEngine()
    
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.01, 0.02, -0.005]
    portfolio_value = 10000.0
    
    var = engine.compute_var(
        returns=returns,
        portfolio_value=portfolio_value,
        method="historical",
        level=0.95,
    )
    
    assert var > 0
    assert var < portfolio_value


def test_es_calculation():
    """Test Expected Shortfall calculation"""
    engine = VaRESEngine()
    
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.01, 0.02, -0.005]
    portfolio_value = 10000.0
    
    es = engine.compute_es(
        returns=returns,
        portfolio_value=portfolio_value,
        method="historical",
        level=0.975,
    )
    
    assert es > 0
    # ES should be >= VaR (more conservative)
    var = engine.compute_var(returns, portfolio_value, "historical", 0.95)
    assert es >= var


def test_var_result_complete(sample_snapshot):
    """Test complete VaR result generation"""
    engine = VaRESEngine()
    
    var_result = engine.compute_var_result(
        snapshot=sample_snapshot,
        returns_data={},  # Will trigger placeholder
        method="delta_normal",
        threshold_95=1000.0,
        threshold_99=2000.0,
    )
    
    assert isinstance(var_result, VaRResult)
    assert var_result.var_95 > 0
    assert var_result.var_99 > var_result.var_95  # 99% VaR should be higher
    assert isinstance(var_result.pass_95, bool)
    assert isinstance(var_result.pass_99, bool)


# === Systemic Risk Tests ===

def test_systemic_detector_initialization():
    """Test systemic risk detector initialization"""
    detector = SystemicRiskDetector(
        correlation_spike_threshold=0.20,
        volatility_spike_threshold=2.0,
    )
    
    assert detector.correlation_spike_threshold == 0.20
    assert detector.volatility_spike_threshold == 2.0


def test_systemic_concentration_detection(sample_snapshot):
    """Test concentration risk detection"""
    detector = SystemicRiskDetector(concentration_threshold=0.40)
    
    # Create exposure matrix with high concentration
    engine = ExposureMatrixEngine()
    exposure_matrix = engine.compute_exposure_matrix(sample_snapshot)
    
    signals = detector.detect(
        snapshot=sample_snapshot,
        exposure_matrix=exposure_matrix,
        var_result=None,
        market_state=None,
    )
    
    # Should detect concentration (BTCUSDT = 50%)
    conc_signals = [s for s in signals if s.risk_type == SystemicRiskType.CONCENTRATION_RISK]
    assert len(conc_signals) > 0


def test_systemic_cascading_risk(sample_snapshot):
    """Test cascading risk detection"""
    detector = SystemicRiskDetector()
    
    # Modify snapshot for cascading risk scenario
    sample_snapshot.total_leverage = 4.0  # High leverage
    
    # Create exposure matrix with high correlation
    engine = ExposureMatrixEngine()
    exposure_matrix = engine.compute_exposure_matrix(sample_snapshot)
    exposure_matrix.avg_correlation = 0.75  # High correlation
    
    signals = detector.detect(
        snapshot=sample_snapshot,
        exposure_matrix=exposure_matrix,
        var_result=None,
        market_state=None,
    )
    
    # Should detect cascading risk
    cascade_signals = [s for s in signals if s.risk_type == SystemicRiskType.CASCADING_RISK]
    assert len(cascade_signals) > 0


# === Risk Rules Tests ===

def test_rules_engine_initialization(risk_limits):
    """Test rules engine initialization"""
    engine = RiskRulesEngine(risk_limits)
    
    assert engine.risk_limits == risk_limits


def test_rules_leverage_check(sample_snapshot, risk_limits):
    """Test leverage threshold check"""
    engine = RiskRulesEngine(risk_limits)
    
    # Modify snapshot to exceed leverage
    sample_snapshot.total_leverage = 6.0  # Exceeds limit of 5.0
    
    threshold = engine._check_leverage(sample_snapshot)
    
    assert threshold is not None
    assert threshold.breached is True
    assert threshold.current_value == 6.0
    assert threshold.severity in [RiskLevel.WARNING, RiskLevel.CRITICAL]


def test_rules_evaluate_all(sample_snapshot, risk_limits):
    """Test complete rule evaluation"""
    engine = RiskRulesEngine(risk_limits)
    
    # Create dummy results
    from backend.services.risk_v3.exposure_matrix import ExposureMatrixEngine
    from backend.services.risk_v3.var_es import VaRESEngine
    
    exp_engine = ExposureMatrixEngine()
    exposure_matrix = exp_engine.compute_exposure_matrix(sample_snapshot)
    
    var_engine = VaRESEngine()
    var_result = var_engine.compute_var_result(sample_snapshot, {}, "delta_normal", 1000.0, 2000.0)
    es_result = var_engine.compute_es_result(sample_snapshot, {}, "historical", 2500.0)
    
    risk_level, breached, critical, warnings = engine.evaluate_all_rules(
        snapshot=sample_snapshot,
        exposure_matrix=exposure_matrix,
        var_result=var_result,
        es_result=es_result,
        systemic_signals=[],
    )
    
    assert risk_level in [RiskLevel.INFO, RiskLevel.WARNING, RiskLevel.CRITICAL]
    assert isinstance(breached, list)
    assert isinstance(critical, list)
    assert isinstance(warnings, list)


def test_ess_tier_recommendation(risk_limits):
    """Test ESS tier recommendation"""
    engine = RiskRulesEngine(risk_limits)
    
    # NORMAL scenario
    tier = engine.recommend_ess_tier(RiskLevel.INFO, [], [])
    assert tier == ESSTier.NORMAL
    
    # REDUCED scenario (1 critical issue)
    tier = engine.recommend_ess_tier(RiskLevel.CRITICAL, ["Issue 1"], [])
    assert tier == ESSTier.REDUCED
    
    # EMERGENCY scenario (2 critical issues)
    tier = engine.recommend_ess_tier(RiskLevel.CRITICAL, ["Issue 1", "Issue 2"], [])
    assert tier == ESSTier.EMERGENCY


# === Orchestrator Tests ===

@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    orchestrator = RiskOrchestrator()
    
    assert orchestrator.exposure_engine is not None
    assert orchestrator.var_es_engine is not None
    assert orchestrator.systemic_detector is not None
    assert orchestrator.evaluation_count == 0


@pytest.mark.asyncio
async def test_orchestrator_evaluate_risk():
    """Test end-to-end risk evaluation"""
    orchestrator = RiskOrchestrator()
    
    # Run evaluation (will use placeholder data from adapters)
    signal = await orchestrator.evaluate_risk()
    
    assert signal is not None
    assert signal.risk_level in [RiskLevel.INFO, RiskLevel.WARNING, RiskLevel.CRITICAL]
    assert signal.snapshot is not None
    assert signal.exposure_matrix is not None
    assert 0.0 <= signal.overall_risk_score <= 1.0
    assert orchestrator.evaluation_count == 1


@pytest.mark.asyncio
async def test_orchestrator_status():
    """Test orchestrator status retrieval"""
    orchestrator = RiskOrchestrator()
    
    status = await orchestrator.get_status()
    
    assert "evaluation_count" in status
    assert "engines" in status
    assert status["engines"]["exposure_matrix"] == "active"


# === Integration Tests ===

@pytest.mark.asyncio
async def test_full_risk_pipeline(sample_snapshot, sample_returns, risk_limits):
    """Test complete risk evaluation pipeline"""
    # Initialize all components
    exp_engine = ExposureMatrixEngine()
    var_engine = VaRESEngine()
    systemic_detector = SystemicRiskDetector()
    rules_engine = RiskRulesEngine(risk_limits)
    
    # 1. Compute exposure matrix
    exposure_matrix = exp_engine.compute_exposure_matrix(
        sample_snapshot,
        returns_data=sample_returns
    )
    assert exposure_matrix is not None
    
    # 2. Calculate VaR/ES
    var_result = var_engine.compute_var_result(
        sample_snapshot,
        sample_returns,
        "delta_normal",
        risk_limits.var_95_limit,
        risk_limits.var_99_limit,
    )
    es_result = var_engine.compute_es_result(
        sample_snapshot,
        sample_returns,
        "historical",
        risk_limits.es_975_limit,
    )
    assert var_result is not None
    assert es_result is not None
    
    # 3. Detect systemic risks
    systemic_signals = systemic_detector.detect(
        sample_snapshot,
        exposure_matrix,
        var_result,
        {"liquidity_score": 0.80},
    )
    assert isinstance(systemic_signals, list)
    
    # 4. Evaluate rules
    risk_level, breached, critical, warnings = rules_engine.evaluate_all_rules(
        sample_snapshot,
        exposure_matrix,
        var_result,
        es_result,
        systemic_signals,
    )
    assert risk_level in [RiskLevel.INFO, RiskLevel.WARNING, RiskLevel.CRITICAL]
    
    # 5. ESS tier recommendation
    ess_tier = rules_engine.recommend_ess_tier(risk_level, critical, systemic_signals)
    assert ess_tier in [ESSTier.NORMAL, ESSTier.REDUCED, ESSTier.EMERGENCY]
    
    print(f"\n✅ Full pipeline test passed:")
    print(f"  Risk Level: {risk_level.value}")
    print(f"  ESS Tier: {ess_tier.value}")
    print(f"  Critical Issues: {len(critical)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Systemic Signals: {len(systemic_signals)}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
