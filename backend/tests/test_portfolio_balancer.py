"""
Tests for Portfolio Balancer AI (PBA)
"""

import pytest
from backend.services.portfolio_balancer import (
    PortfolioBalancerAI,
    Position,
    CandidateTrade,
    PortfolioConstraints,
    RiskMode,
)


class TestPortfolioBalancerAI:
    """Test Portfolio Balancer AI functionality"""
    
    @pytest.fixture
    def balancer(self, tmp_path):
        """Create balancer with test data directory"""
        return PortfolioBalancerAI(data_dir=str(tmp_path))
    
    @pytest.fixture
    def sample_positions(self):
        """Sample open positions"""
        return [
            Position(
                symbol="BTCUSDT",
                side="LONG",
                size=0.1,
                entry_price=40000,
                current_price=41000,
                margin=1000,
                leverage=10,
                category="CORE",
                sector="L1",
                risk_amount=100
            ),
            Position(
                symbol="ETHUSDT",
                side="LONG",
                size=1.0,
                entry_price=2500,
                current_price=2550,
                margin=500,
                leverage=10,
                category="CORE",
                sector="L1",
                risk_amount=50
            )
        ]
    
    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate trades"""
        return [
            CandidateTrade(
                symbol="SOLUSDT",
                action="BUY",
                confidence=0.75,
                size=10,
                margin_required=300,
                risk_amount=30,
                category="CORE",
                stability_score=0.8
            ),
            CandidateTrade(
                symbol="DOGEUSDT",
                action="BUY",
                confidence=0.60,
                size=1000,
                margin_required=200,
                risk_amount=20,
                category="EXPANSION",
                stability_score=0.3
            )
        ]
    
    def test_portfolio_state_computation(self, balancer, sample_positions):
        """Test portfolio state computation"""
        state = balancer._compute_portfolio_state(
            positions=sample_positions,
            total_equity=10000,
            used_margin=1500,
            free_margin=8500
        )
        
        assert state.total_positions == 2
        assert state.long_positions == 2
        assert state.short_positions == 0
        assert state.total_equity == 10000
        assert state.total_risk_pct == 1.5  # (100 + 50) / 10000 * 100
    
    def test_no_violations_healthy_portfolio(self, balancer, sample_positions):
        """Test that healthy portfolio has no violations"""
        state = balancer._compute_portfolio_state(
            sample_positions, 10000, 1500, 8500
        )
        violations = balancer._detect_violations(state, sample_positions)
        
        assert len(violations) == 0
    
    def test_max_positions_violation(self, balancer):
        """Test max positions violation detection"""
        # Create 9 positions (exceeds default max of 8)
        positions = [
            Position(
                symbol=f"SYM{i}USDT",
                side="LONG",
                size=1.0,
                entry_price=100,
                current_price=100,
                margin=100,
                leverage=10,
                risk_amount=10
            )
            for i in range(9)
        ]
        
        state = balancer._compute_portfolio_state(positions, 10000, 900, 9100)
        violations = balancer._detect_violations(state, positions)
        
        # Should have critical violation for max positions
        assert len(violations) > 0
        assert any(v.constraint == "max_positions" for v in violations)
        assert any(v.severity == "CRITICAL" for v in violations)
    
    def test_risk_exceeded_violation(self, balancer):
        """Test total risk exceeded violation"""
        # Create positions with high total risk (20% > 15% limit)
        positions = [
            Position(
                symbol="BTCUSDT",
                side="LONG",
                size=1.0,
                entry_price=40000,
                current_price=40000,
                margin=5000,
                leverage=10,
                risk_amount=2000  # 20% of 10k equity
            )
        ]
        
        state = balancer._compute_portfolio_state(positions, 10000, 5000, 5000)
        violations = balancer._detect_violations(state, positions)
        
        assert any(v.constraint == "max_total_risk_pct" for v in violations)
        assert any(v.severity == "CRITICAL" for v in violations)
    
    def test_trade_prioritization(self, balancer, sample_candidates):
        """Test trade prioritization logic"""
        state = balancer._compute_portfolio_state([], 10000, 0, 10000)
        prioritized = balancer._prioritize_trades(sample_candidates, state)
        
        # CORE with higher confidence should rank first
        assert prioritized[0].symbol == "SOLUSDT"
        assert prioritized[0].priority_score > prioritized[1].priority_score
    
    def test_filter_trades_max_positions(self, balancer, sample_positions, sample_candidates):
        """Test trade filtering when at max positions"""
        # Set max positions to 2 (already have 2 positions)
        balancer.constraints.max_positions = 2
        
        state = balancer._compute_portfolio_state(sample_positions, 10000, 1500, 8500)
        prioritized = balancer._prioritize_trades(sample_candidates, state)
        allowed, dropped = balancer._filter_trades(prioritized, state, sample_positions, [])
        
        # Should block all trades (no room for new positions)
        assert len(allowed) == 0
        assert len(dropped) == 2
    
    def test_filter_trades_duplicate_symbol(self, balancer, sample_positions, sample_candidates):
        """Test that trades for symbols with open positions are blocked"""
        # Add candidate for BTCUSDT (already have position)
        duplicate_candidate = CandidateTrade(
            symbol="BTCUSDT",
            action="BUY",
            confidence=0.80,
            size=0.05,
            margin_required=200,
            risk_amount=20,
            category="CORE"
        )
        candidates = [duplicate_candidate] + sample_candidates
        
        state = balancer._compute_portfolio_state(sample_positions, 10000, 1500, 8500)
        prioritized = balancer._prioritize_trades(candidates, state)
        allowed, dropped = balancer._filter_trades(prioritized, state, sample_positions, [])
        
        # BTCUSDT trade should be dropped
        dropped_symbols = [t.symbol for t in dropped]
        assert "BTCUSDT" in dropped_symbols
    
    def test_risk_mode_safe_on_critical_violation(self, balancer):
        """Test that critical violations trigger SAFE mode"""
        from backend.services.portfolio_balancer import PortfolioViolation, PortfolioState
        
        state = PortfolioState(
            timestamp="2025-11-23T00:00:00Z",
            total_equity=10000,
            used_margin=1500,
            free_margin=8500,
            total_risk_pct=20.0  # High risk
        )
        
        violations = [
            PortfolioViolation(
                constraint="max_total_risk_pct",
                current_value=20.0,
                limit_value=15.0,
                severity="CRITICAL",
                message="Risk exceeded"
            )
        ]
        
        mode = balancer._determine_risk_mode(state, violations, None, None)
        assert mode == RiskMode.SAFE
    
    def test_risk_mode_safe_on_high_drawdown(self, balancer):
        """Test that high drawdown triggers SAFE mode"""
        from backend.services.portfolio_balancer import PortfolioState
        
        state = PortfolioState(
            timestamp="2025-11-23T00:00:00Z",
            total_equity=10000,
            used_margin=1500,
            free_margin=8500
        )
        
        risk_manager_state = {
            "daily_drawdown_pct": -4.0  # > -3.0 threshold
        }
        
        mode = balancer._determine_risk_mode(state, [], None, risk_manager_state)
        assert mode == RiskMode.SAFE
    
    def test_full_analysis_workflow(self, balancer, sample_positions, sample_candidates):
        """Test complete analysis workflow"""
        output = balancer.analyze_portfolio(
            positions=sample_positions,
            candidates=sample_candidates,
            total_equity=10000,
            used_margin=1500,
            free_margin=8500
        )
        
        # Verify output structure
        assert output.timestamp is not None
        assert output.risk_mode in ["SAFE", "NEUTRAL", "AGGRESSIVE"]
        assert output.portfolio_state is not None
        assert isinstance(output.violations, list)
        assert isinstance(output.allowed_trades, list)
        assert isinstance(output.dropped_trades, list)
        assert isinstance(output.recommendations, list)
        
        # Should have some allowed trades (we have room)
        assert len(output.allowed_trades) > 0
    
    def test_recommendations_generation(self, balancer):
        """Test recommendation generation logic"""
        from backend.services.portfolio_balancer import PortfolioState
        
        # Create state near limits
        state = PortfolioState(
            timestamp="2025-11-23T00:00:00Z",
            total_equity=10000,
            used_margin=1500,
            free_margin=8500,
            total_positions=7,  # Near max of 8
            total_risk_pct=12.0,  # 80% of 15% limit
            max_symbol_concentration_pct=25.0
        )
        
        recommendations = balancer._generate_recommendations(
            state, [], RiskMode.NEUTRAL, []
        )
        
        assert len(recommendations) > 0
        # Should mention near max positions
        assert any("position" in rec.lower() for rec in recommendations)
    
    def test_critical_violations_block_all_trades(self, balancer, sample_candidates):
        """Test that critical violations block all new trades"""
        from backend.services.portfolio_balancer import PortfolioState, PortfolioViolation
        
        state = PortfolioState(
            timestamp="2025-11-23T00:00:00Z",
            total_equity=10000,
            used_margin=9000,
            free_margin=1000,
            total_positions=10  # Way over limit
        )
        
        violations = [
            PortfolioViolation(
                constraint="max_positions",
                current_value=10,
                limit_value=8,
                severity="CRITICAL",
                message="Too many positions"
            )
        ]
        
        prioritized = balancer._prioritize_trades(sample_candidates, state)
        allowed, dropped = balancer._filter_trades(prioritized, state, [], violations)
        
        # All trades should be dropped
        assert len(allowed) == 0
        assert len(dropped) == len(sample_candidates)
    
    def test_insufficient_margin_blocks_trade(self, balancer, sample_candidates):
        """Test that insufficient margin blocks trades"""
        from backend.services.portfolio_balancer import PortfolioState
        
        state = PortfolioState(
            timestamp="2025-11-23T00:00:00Z",
            total_equity=10000,
            used_margin=9800,
            free_margin=200,  # Not enough for 300 margin trade
            total_positions=2
        )
        
        prioritized = balancer._prioritize_trades(sample_candidates, state)
        allowed, dropped = balancer._filter_trades(prioritized, state, [], [])
        
        # SOLUSDT requires 300 margin, should be dropped
        dropped_symbols = [t.symbol for t in dropped]
        assert "SOLUSDT" in dropped_symbols


class TestPositionDataClass:
    """Test Position data class"""
    
    def test_position_creation(self):
        """Test position creation and calculations"""
        pos = Position(
            symbol="BTCUSDT",
            side="LONG",
            size=0.1,
            entry_price=40000,
            current_price=42000,
            margin=1000,
            leverage=10,
            risk_amount=100
        )
        
        assert pos.symbol == "BTCUSDT"
        assert pos.exposure == 0.1 * 42000  # size * current_price
        assert pos.unrealized_pnl_pct == 0.05  # (42000 - 40000) / 40000
    
    def test_short_position_pnl(self):
        """Test short position PnL calculation"""
        pos = Position(
            symbol="BTCUSDT",
            side="SHORT",
            size=0.1,
            entry_price=40000,
            current_price=38000,  # Price dropped = profit for short
            margin=1000,
            leverage=10,
            risk_amount=100
        )
        
        # Short profit when price drops
        assert pos.unrealized_pnl_pct == 0.05  # (40000 - 38000) / 40000


class TestCandidateTradeDataClass:
    """Test CandidateTrade data class"""
    
    def test_candidate_creation(self):
        """Test candidate trade creation"""
        trade = CandidateTrade(
            symbol="ETHUSDT",
            action="BUY",
            confidence=0.75,
            size=1.0,
            margin_required=250,
            risk_amount=25,
            category="CORE",
            stability_score=0.8
        )
        
        assert trade.symbol == "ETHUSDT"
        assert trade.confidence == 0.75
        assert trade.category == "CORE"
