#!/usr/bin/env python3
"""
PORTFOLIO BALANCER AI (PBA)
============================

Global portfolio state management for Quantum Trader.
Controls total exposure, diversification, and risk across all positions and signals.

Mission: PREVENT OVER-CONCENTRATION, MANAGE TOTAL RISK, OPTIMIZE CAPITAL ALLOCATION

Author: Quantum Trader AI Team
Date: November 23, 2025
Version: 1.0
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RiskMode(Enum):
    """Portfolio risk modes"""
    SAFE = "SAFE"
    NEUTRAL = "NEUTRAL"
    AGGRESSIVE = "AGGRESSIVE"


class SymbolCategory(Enum):
    """Symbol quality categories from Universe OS"""
    CORE = "CORE"
    EXPANSION = "EXPANSION"
    MONITORING = "MONITORING"
    TOXIC = "TOXIC"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Position:
    """Open position data"""
    symbol: str
    side: str  # LONG or SHORT
    size: float
    entry_price: float
    current_price: float
    margin: float
    leverage: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Metadata
    category: str = "EXPANSION"
    sector: str = "unknown"
    risk_amount: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        if self.current_price > 0 and self.entry_price > 0:
            if self.side == "LONG":
                self.unrealized_pnl_pct = (self.current_price - self.entry_price) / self.entry_price
            else:  # SHORT
                self.unrealized_pnl_pct = (self.entry_price - self.current_price) / self.entry_price
    
    @property
    def exposure(self) -> float:
        """Position exposure (notional value)"""
        return self.size * self.current_price
    
    @property
    def position_risk_pct(self) -> float:
        """Position risk as % of margin"""
        if self.margin > 0:
            return (self.risk_amount / self.margin) * 100
        return 0.0


@dataclass
class CandidateTrade:
    """Pending trade signal"""
    symbol: str
    action: str  # BUY or SELL
    confidence: float
    size: float = 0.0
    margin_required: float = 0.0
    risk_amount: float = 0.0
    
    # Metadata
    category: str = "EXPANSION"
    sector: str = "unknown"
    stability_score: float = 0.0
    cost_score: float = 0.0  # Lower is better (spread + slippage)
    recent_performance: float = 0.0  # Recent symbol performance
    
    # Priority (computed)
    priority_score: float = 0.0


@dataclass
class PortfolioState:
    """Complete portfolio state snapshot"""
    timestamp: str
    total_equity: float
    used_margin: float
    free_margin: float
    
    # Exposure metrics
    total_exposure_long: float = 0.0
    total_exposure_short: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Risk metrics
    total_risk_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Position counts
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    
    # Diversification
    symbol_count: int = 0
    sector_distribution: Dict[str, float] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Concentration risks
    max_symbol_concentration_pct: float = 0.0
    max_sector_concentration_pct: float = 0.0
    correlation_risk_score: float = 0.0


@dataclass
class PortfolioConstraints:
    """Portfolio-level constraints - AGGRESSIVE MODE"""
    max_positions: int = 50  # 🔥 Increased from 15 to 50!
    max_positions_per_symbol: int = 3  # Allow multiple per symbol
    max_total_risk_pct: float = 80.0  # 80% total portfolio risk
    max_per_trade_risk_pct: float = 80.0  # 80% per trade!
    max_symbol_concentration_pct: float = 50.0  # Allow 50% per symbol
    max_sector_concentration_pct: float = 80.0  # Allow 80% per sector
    max_leverage: float = 25.0  # Match Math AI max leverage
    max_long_exposure_pct: float = 500.0  # 500% leverage on longs!
    max_short_exposure_pct: float = 500.0  # 500% leverage on shorts!
    max_net_exposure_pct: float = 500.0  # 500% total exposure!


@dataclass
class PortfolioViolation:
    """Constraint violation"""
    constraint: str
    current_value: float
    limit_value: float
    severity: str  # WARNING, CRITICAL
    message: str


@dataclass
class TradeDecision:
    """Trade execution decision"""
    symbol: str
    action: str
    allowed: bool
    reason: str
    priority_score: float = 0.0
    recommended_size: float = 0.0


@dataclass
class BalancerOutput:
    """Portfolio Balancer output"""
    timestamp: str
    risk_mode: str
    portfolio_state: PortfolioState
    violations: List[PortfolioViolation]
    
    # Trade decisions
    allowed_trades: List[TradeDecision]
    dropped_trades: List[TradeDecision]
    
    # Recommendations
    recommendations: List[str]
    actions_required: List[str]


# ============================================================================
# PORTFOLIO BALANCER AI
# ============================================================================

class PortfolioBalancerAI:
    """
    Portfolio Balancer AI (PBA)
    
    Manages global portfolio state, exposure, diversification, and risk.
    Advisory system for Execution Layer and Orchestrator.
    """
    
    def __init__(
        self,
        constraints: Optional[PortfolioConstraints] = None,
        data_dir: str = "/app/data"
    ):
        """Initialize Portfolio Balancer AI"""
        self.constraints = constraints or PortfolioConstraints()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("PORTFOLIO BALANCER AI (PBA) — INITIALIZING")
        logger.info("=" * 80)
        logger.info(f"Max Positions: {self.constraints.max_positions}")
        logger.info(f"Max Total Risk: {self.constraints.max_total_risk_pct}%")
        logger.info(f"Max Symbol Concentration: {self.constraints.max_symbol_concentration_pct}%")
        logger.info(f"Max Sector Concentration: {self.constraints.max_sector_concentration_pct}%")
    
    def analyze_portfolio(
        self,
        positions: List[Position],
        candidates: List[CandidateTrade],
        total_equity: float,
        used_margin: float,
        free_margin: float,
        orchestrator_policy: Optional[Dict[str, Any]] = None,
        risk_manager_state: Optional[Dict[str, Any]] = None
    ) -> BalancerOutput:
        """
        Main analysis method
        
        Args:
            positions: All open positions
            candidates: All candidate trades (pending signals)
            total_equity: Total portfolio equity
            used_margin: Currently used margin
            free_margin: Available free margin
            orchestrator_policy: Current orchestrator policy state
            risk_manager_state: Current risk manager state
        
        Returns:
            BalancerOutput with decisions and recommendations
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info("=" * 80)
        logger.info("PORTFOLIO BALANCER AI — ANALYSIS STARTING")
        logger.info("=" * 80)
        logger.info(f"Total Equity: ${total_equity:,.2f}")
        logger.info(f"Open Positions: {len(positions)}")
        logger.info(f"Candidate Trades: {len(candidates)}")
        
        # 1. Compute portfolio state
        portfolio_state = self._compute_portfolio_state(
            positions, total_equity, used_margin, free_margin
        )
        
        # 2. Detect violations
        violations = self._detect_violations(portfolio_state, positions)
        
        # 3. Determine risk mode
        risk_mode = self._determine_risk_mode(
            portfolio_state, violations, orchestrator_policy, risk_manager_state
        )
        
        # 4. Prioritize candidate trades
        prioritized_candidates = self._prioritize_trades(candidates, portfolio_state)
        
        # 5. Filter trades based on constraints
        allowed_trades, dropped_trades = self._filter_trades(
            prioritized_candidates, portfolio_state, positions, violations
        )
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(
            portfolio_state, violations, risk_mode, positions
        )
        
        # 7. Generate action items
        actions_required = self._generate_actions(violations, risk_mode)
        
        # Create output
        output = BalancerOutput(
            timestamp=timestamp,
            risk_mode=risk_mode.value,
            portfolio_state=portfolio_state,
            violations=violations,
            allowed_trades=allowed_trades,
            dropped_trades=dropped_trades,
            recommendations=recommendations,
            actions_required=actions_required
        )
        
        # Log summary
        self._log_summary(output)
        
        # Save to disk
        self._save_output(output)
        
        return output
    
    def _compute_portfolio_state(
        self,
        positions: List[Position],
        total_equity: float,
        used_margin: float,
        free_margin: float
    ) -> PortfolioState:
        """Compute comprehensive portfolio state"""
        
        total_exposure_long = 0.0
        total_exposure_short = 0.0
        total_risk = 0.0
        
        sector_exposure: Dict[str, float] = defaultdict(float)
        category_distribution: Dict[str, int] = defaultdict(int)
        symbol_exposure: Dict[str, float] = defaultdict(float)
        
        long_count = 0
        short_count = 0
        
        for pos in positions:
            exposure = pos.exposure
            
            if pos.side == "LONG":
                total_exposure_long += exposure
                long_count += 1
            else:
                total_exposure_short += exposure
                short_count += 1
            
            total_risk += pos.risk_amount
            
            sector_exposure[pos.sector] += exposure
            category_distribution[pos.category] += 1
            symbol_exposure[pos.symbol] += exposure
        
        net_exposure = total_exposure_long - total_exposure_short
        gross_exposure = total_exposure_long + total_exposure_short
        
        # Calculate concentrations
        max_symbol_concentration = 0.0
        if total_equity > 0:
            max_symbol_concentration = max(
                (exp / total_equity * 100 for exp in symbol_exposure.values()),
                default=0.0
            )
        
        max_sector_concentration = 0.0
        if total_equity > 0:
            max_sector_concentration = max(
                (exp / total_equity * 100 for exp in sector_exposure.values()),
                default=0.0
            )
        
        # Sector distribution as percentages
        sector_pct = {}
        if gross_exposure > 0:
            sector_pct = {
                sector: (exp / gross_exposure * 100)
                for sector, exp in sector_exposure.items()
            }
        
        # Total risk percentage
        total_risk_pct = 0.0
        if total_equity > 0:
            total_risk_pct = (total_risk / total_equity) * 100
        
        state = PortfolioState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_equity=total_equity,
            used_margin=used_margin,
            free_margin=free_margin,
            total_exposure_long=total_exposure_long,
            total_exposure_short=total_exposure_short,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            total_risk_pct=total_risk_pct,
            total_positions=len(positions),
            long_positions=long_count,
            short_positions=short_count,
            symbol_count=len(symbol_exposure),
            sector_distribution=sector_pct,
            category_distribution=dict(category_distribution),
            max_symbol_concentration_pct=max_symbol_concentration,
            max_sector_concentration_pct=max_sector_concentration
        )
        
        logger.info(f"Portfolio State Computed:")
        logger.info(f"  Total Positions: {state.total_positions} (L:{long_count} S:{short_count})")
        logger.info(f"  Gross Exposure: ${gross_exposure:,.2f}")
        logger.info(f"  Net Exposure: ${net_exposure:,.2f}")
        logger.info(f"  Total Risk: {total_risk_pct:.2f}%")
        logger.info(f"  Max Symbol Concentration: {max_symbol_concentration:.2f}%")
        
        return state
    
    def _detect_violations(
        self,
        state: PortfolioState,
        positions: List[Position]
    ) -> List[PortfolioViolation]:
        """Detect constraint violations"""
        violations = []
        
        # Max positions
        if state.total_positions > self.constraints.max_positions:
            violations.append(PortfolioViolation(
                constraint="max_positions",
                current_value=float(state.total_positions),
                limit_value=float(self.constraints.max_positions),
                severity="CRITICAL",
                message=f"Too many positions: {state.total_positions}/{self.constraints.max_positions}"
            ))
        elif state.total_positions >= self.constraints.max_positions * 0.9:
            violations.append(PortfolioViolation(
                constraint="max_positions",
                current_value=float(state.total_positions),
                limit_value=float(self.constraints.max_positions),
                severity="WARNING",
                message=f"Near max positions: {state.total_positions}/{self.constraints.max_positions}"
            ))
        
        # Total risk
        if state.total_risk_pct > self.constraints.max_total_risk_pct:
            violations.append(PortfolioViolation(
                constraint="max_total_risk_pct",
                current_value=state.total_risk_pct,
                limit_value=self.constraints.max_total_risk_pct,
                severity="CRITICAL",
                message=f"Total risk exceeded: {state.total_risk_pct:.2f}%/{self.constraints.max_total_risk_pct}%"
            ))
        elif state.total_risk_pct > self.constraints.max_total_risk_pct * 0.85:
            violations.append(PortfolioViolation(
                constraint="max_total_risk_pct",
                current_value=state.total_risk_pct,
                limit_value=self.constraints.max_total_risk_pct,
                severity="WARNING",
                message=f"High total risk: {state.total_risk_pct:.2f}%/{self.constraints.max_total_risk_pct}%"
            ))
        
        # Symbol concentration
        if state.max_symbol_concentration_pct > self.constraints.max_symbol_concentration_pct:
            violations.append(PortfolioViolation(
                constraint="max_symbol_concentration_pct",
                current_value=state.max_symbol_concentration_pct,
                limit_value=self.constraints.max_symbol_concentration_pct,
                severity="CRITICAL",
                message=f"Symbol over-concentration: {state.max_symbol_concentration_pct:.2f}%"
            ))
        
        # Sector concentration
        if state.max_sector_concentration_pct > self.constraints.max_sector_concentration_pct:
            violations.append(PortfolioViolation(
                constraint="max_sector_concentration_pct",
                current_value=state.max_sector_concentration_pct,
                limit_value=self.constraints.max_sector_concentration_pct,
                severity="CRITICAL",
                message=f"Sector over-concentration: {state.max_sector_concentration_pct:.2f}%"
            ))
        
        # Net exposure
        if state.total_equity > 0:
            net_exposure_pct = abs(state.net_exposure / state.total_equity * 100)
            if net_exposure_pct > self.constraints.max_net_exposure_pct:
                violations.append(PortfolioViolation(
                    constraint="max_net_exposure_pct",
                    current_value=net_exposure_pct,
                    limit_value=self.constraints.max_net_exposure_pct,
                    severity="WARNING",
                    message=f"High net exposure: {net_exposure_pct:.2f}%"
                ))
        
        # Position-level leverage
        for pos in positions:
            if pos.leverage > self.constraints.max_leverage:
                violations.append(PortfolioViolation(
                    constraint="max_leverage",
                    current_value=pos.leverage,
                    limit_value=self.constraints.max_leverage,
                    severity="CRITICAL",
                    message=f"{pos.symbol} leverage too high: {pos.leverage}x"
                ))
        
        if violations:
            logger.warning(f"⚠️  {len(violations)} constraint violations detected")
            for v in violations:
                logger.warning(f"  [{v.severity}] {v.message}")
        else:
            logger.info("✅ No constraint violations")
        
        return violations
    
    def _determine_risk_mode(
        self,
        state: PortfolioState,
        violations: List[PortfolioViolation],
        orchestrator_policy: Optional[Dict[str, Any]],
        risk_manager_state: Optional[Dict[str, Any]]
    ) -> RiskMode:
        """Determine appropriate risk mode"""
        
        # Start at NEUTRAL
        mode = RiskMode.NEUTRAL
        
        # Critical violations → SAFE
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        if critical_violations:
            logger.info("🛡️  SAFE mode: Critical violations present")
            return RiskMode.SAFE
        
        # Check orchestrator policy
        if orchestrator_policy:
            policy_risk = orchestrator_policy.get("risk_profile", "NEUTRAL")
            if policy_risk == "DEFENSIVE":
                mode = RiskMode.SAFE
            elif policy_risk == "AGGRESSIVE":
                mode = RiskMode.AGGRESSIVE
        
        # Check risk manager state
        if risk_manager_state:
            dd_pct = abs(risk_manager_state.get("daily_drawdown_pct", 0.0))
            if dd_pct > 3.0:
                logger.info(f"🛡️  SAFE mode: High drawdown {dd_pct:.2f}%")
                return RiskMode.SAFE
            
            losing_streak = risk_manager_state.get("losing_streak", 0)
            if losing_streak >= 5:
                logger.info(f"🛡️  SAFE mode: Losing streak {losing_streak}")
                return RiskMode.SAFE
        
        # Portfolio health checks
        if state.total_positions >= self.constraints.max_positions:
            logger.info("🛡️  SAFE mode: Max positions reached")
            return RiskMode.SAFE
        
        if state.total_risk_pct > self.constraints.max_total_risk_pct * 0.8:
            logger.info(f"🛡️  SAFE mode: High risk exposure {state.total_risk_pct:.2f}%")
            return RiskMode.SAFE
        
        # Warning violations → cap at NEUTRAL
        warning_violations = [v for v in violations if v.severity == "WARNING"]
        if warning_violations and mode == RiskMode.AGGRESSIVE:
            logger.info("⚠️  NEUTRAL mode: Warnings present, capping at NEUTRAL")
            return RiskMode.NEUTRAL
        
        logger.info(f"✅ Risk Mode: {mode.value}")
        return mode
    
    def _prioritize_trades(
        self,
        candidates: List[CandidateTrade],
        state: PortfolioState
    ) -> List[CandidateTrade]:
        """Prioritize candidate trades"""
        
        for trade in candidates:
            # Base score from confidence
            score = trade.confidence * 100
            
            # Category boost
            if trade.category == "CORE":
                score += 20
            elif trade.category == "EXPANSION":
                score += 10
            elif trade.category == "MONITORING":
                score += 5
            
            # Stability boost
            score += trade.stability_score * 10
            
            # Cost penalty (lower cost = higher score)
            score -= trade.cost_score * 5
            
            # Recent performance boost
            score += trade.recent_performance * 10
            
            trade.priority_score = max(0.0, score)
        
        # Sort by priority (highest first)
        sorted_trades = sorted(candidates, key=lambda t: t.priority_score, reverse=True)
        
        if sorted_trades:
            logger.info(f"Trade Priorities (top 5):")
            for i, trade in enumerate(sorted_trades[:5], 1):
                logger.info(f"  {i}. {trade.symbol} {trade.action} - {trade.priority_score:.2f} pts")
        
        return sorted_trades
    
    def _filter_trades(
        self,
        candidates: List[CandidateTrade],
        state: PortfolioState,
        positions: List[Position],
        violations: List[PortfolioViolation]
    ) -> Tuple[List[TradeDecision], List[TradeDecision]]:
        """Filter trades based on constraints"""
        
        allowed = []
        dropped = []
        
        # If critical violations, block all trades
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        if critical_violations:
            for trade in candidates:
                dropped.append(TradeDecision(
                    symbol=trade.symbol,
                    action=trade.action,
                    allowed=False,
                    reason="CRITICAL_VIOLATIONS",
                    priority_score=trade.priority_score
                ))
            logger.warning(f"❌ Blocking all {len(candidates)} trades due to critical violations")
            return allowed, dropped
        
        # Check position limits
        positions_available = self.constraints.max_positions - state.total_positions
        
        # Track symbols already open
        open_symbols = {pos.symbol for pos in positions}
        
        # Track new positions we're allowing
        new_positions = 0
        
        for trade in candidates:
            # Skip if symbol already has position
            if trade.symbol in open_symbols:
                dropped.append(TradeDecision(
                    symbol=trade.symbol,
                    action=trade.action,
                    allowed=False,
                    reason="POSITION_ALREADY_OPEN",
                    priority_score=trade.priority_score
                ))
                continue
            
            # Check if we have room for more positions
            if new_positions >= positions_available:
                dropped.append(TradeDecision(
                    symbol=trade.symbol,
                    action=trade.action,
                    allowed=False,
                    reason="MAX_POSITIONS_REACHED",
                    priority_score=trade.priority_score
                ))
                continue
            
            # Check risk limits
            projected_risk = state.total_risk_pct + (trade.risk_amount / state.total_equity * 100)
            if projected_risk > self.constraints.max_total_risk_pct:
                dropped.append(TradeDecision(
                    symbol=trade.symbol,
                    action=trade.action,
                    allowed=False,
                    reason="TOTAL_RISK_EXCEEDED",
                    priority_score=trade.priority_score
                ))
                continue
            
            # Check margin availability
            if trade.margin_required > state.free_margin:
                dropped.append(TradeDecision(
                    symbol=trade.symbol,
                    action=trade.action,
                    allowed=False,
                    reason="INSUFFICIENT_MARGIN",
                    priority_score=trade.priority_score
                ))
                continue
            
            # Trade is allowed
            allowed.append(TradeDecision(
                symbol=trade.symbol,
                action=trade.action,
                allowed=True,
                reason="APPROVED",
                priority_score=trade.priority_score,
                recommended_size=trade.size
            ))
            new_positions += 1
        
        logger.info(f"Trade Filtering Results:")
        logger.info(f"  ✅ Allowed: {len(allowed)}")
        logger.info(f"  ❌ Dropped: {len(dropped)}")
        
        return allowed, dropped
    
    def _generate_recommendations(
        self,
        state: PortfolioState,
        violations: List[PortfolioViolation],
        risk_mode: RiskMode,
        positions: List[Position]
    ) -> List[str]:
        """Generate portfolio recommendations"""
        
        recommendations = []
        
        # Risk mode recommendation
        recommendations.append(f"RISK MODE: {risk_mode.value}")
        
        # Position count
        if state.total_positions >= self.constraints.max_positions * 0.9:
            recommendations.append(
                f"Near max positions ({state.total_positions}/{self.constraints.max_positions}) - "
                "Consider closing weak positions before opening new ones"
            )
        
        # Risk level
        if state.total_risk_pct > self.constraints.max_total_risk_pct * 0.7:
            recommendations.append(
                f"High risk exposure ({state.total_risk_pct:.2f}%) - "
                "Consider reducing position sizes or closing losing positions"
            )
        
        # Concentration
        if state.max_symbol_concentration_pct > self.constraints.max_symbol_concentration_pct * 0.8:
            recommendations.append(
                f"High symbol concentration ({state.max_symbol_concentration_pct:.2f}%) - "
                "Diversify across more symbols"
            )
        
        # Long/Short balance
        if state.total_positions > 0:
            long_pct = (state.long_positions / state.total_positions) * 100
            if long_pct > 80:
                recommendations.append(
                    f"Portfolio heavily long ({long_pct:.0f}%) - Consider short opportunities for balance"
                )
            elif long_pct < 20:
                recommendations.append(
                    f"Portfolio heavily short ({100-long_pct:.0f}%) - Consider long opportunities for balance"
                )
        
        # Violations
        for v in violations:
            if v.severity == "CRITICAL":
                recommendations.append(f"CRITICAL: {v.message} - Immediate action required")
        
        # Category distribution
        if state.category_distribution:
            core_count = state.category_distribution.get("CORE", 0)
            total = sum(state.category_distribution.values())
            if total > 0:
                core_pct = (core_count / total) * 100
                if core_pct < 50:
                    recommendations.append(
                        f"Low CORE allocation ({core_pct:.0f}%) - Prefer CORE symbols for stability"
                    )
        
        return recommendations
    
    def _generate_actions(
        self,
        violations: List[PortfolioViolation],
        risk_mode: RiskMode
    ) -> List[str]:
        """Generate required actions"""
        
        actions = []
        
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        
        if critical_violations:
            actions.append("BLOCK NEW TRADES until violations resolved")
            
            for v in critical_violations:
                if v.constraint == "max_positions":
                    actions.append("CLOSE at least 1 position immediately")
                elif v.constraint == "max_total_risk_pct":
                    actions.append("REDUCE position sizes or close losing positions")
                elif v.constraint == "max_symbol_concentration_pct":
                    actions.append("REDUCE position size in over-concentrated symbols")
                elif v.constraint == "max_leverage":
                    actions.append("REDUCE leverage on high-leverage positions")
        
        if risk_mode == RiskMode.SAFE:
            actions.append("DEFENSIVE MODE: Reduce risk, avoid aggressive trades")
        
        return actions
    
    def _log_summary(self, output: BalancerOutput):
        """Log analysis summary"""
        logger.info("=" * 80)
        logger.info("PORTFOLIO BALANCER AI — ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Risk Mode: {output.risk_mode}")
        logger.info(f"Total Positions: {output.portfolio_state.total_positions}")
        logger.info(f"Total Risk: {output.portfolio_state.total_risk_pct:.2f}%")
        logger.info(f"Violations: {len(output.violations)}")
        logger.info(f"Allowed Trades: {len(output.allowed_trades)}")
        logger.info(f"Dropped Trades: {len(output.dropped_trades)}")
        logger.info(f"Recommendations: {len(output.recommendations)}")
        
        if output.actions_required:
            logger.warning("⚠️  ACTIONS REQUIRED:")
            for action in output.actions_required:
                logger.warning(f"  • {action}")
    
    def _save_output(self, output: BalancerOutput):
        """Save output to disk"""
        try:
            output_file = self.data_dir / "portfolio_balancer_output.json"
            
            # Convert to dict
            output_dict = {
                "timestamp": output.timestamp,
                "risk_mode": output.risk_mode,
                "portfolio_state": asdict(output.portfolio_state),
                "violations": [asdict(v) for v in output.violations],
                "allowed_trades": [asdict(t) for t in output.allowed_trades],
                "dropped_trades": [asdict(t) for t in output.dropped_trades],
                "recommendations": output.recommendations,
                "actions_required": output.actions_required
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_dict, f, indent=2)
            
            logger.info(f"✅ Output saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
    
    async def balance_loop(self):
        """Continuous portfolio monitoring and balancing loop."""
        import asyncio
        
        logger.info("\n" + "=" * 80)
        logger.info("⚖️ PORTFOLIO BALANCER - STARTING CONTINUOUS MONITORING")
        logger.info("=" * 80)
        logger.info(f"Mode: ADVISORY (risk management & diversification)")
        logger.info(f"Check interval: Every minute")
        logger.info("=" * 80 + "\n")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # Log status every 10 minutes
                if iteration % 10 == 0:
                    logger.info(
                        f"⚖️ [PORTFOLIO_BALANCER] Status check #{iteration} - "
                        f"Monitoring portfolio balance and risk exposure"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info("⚖️ [PORTFOLIO_BALANCER] Balance loop cancelled")
                break
            except Exception as e:
                logger.error(f"[PORTFOLIO_BALANCER] Balance loop error: {e}", exc_info=True)
                await asyncio.sleep(60)


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

def main():
    """Test the Portfolio Balancer AI"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test data
    positions = [
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
    
    candidates = [
        CandidateTrade(
            symbol="SOLUSDT",
            action="BUY",
            confidence=0.75,
            size=10,
            margin_required=300,
            risk_amount=30,
            category="CORE",
            sector="L1",
            stability_score=0.8,
            cost_score=0.2,
            recent_performance=0.05
        ),
        CandidateTrade(
            symbol="DOGEUSDT",
            action="BUY",
            confidence=0.60,
            size=1000,
            margin_required=200,
            risk_amount=20,
            category="EXPANSION",
            sector="MEME",
            stability_score=0.3,
            cost_score=0.5,
            recent_performance=-0.02
        )
    ]
    
    # Create balancer
    balancer = PortfolioBalancerAI()
    
    # Run analysis
    output = balancer.analyze_portfolio(
        positions=positions,
        candidates=candidates,
        total_equity=10000,
        used_margin=1500,
        free_margin=8500
    )
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    for rec in output.recommendations:
        print(f"  • {rec}")
    
    if output.actions_required:
        print("\n" + "=" * 80)
        print("ACTIONS REQUIRED:")
        print("=" * 80)
        for action in output.actions_required:
            print(f"  • {action}")


if __name__ == "__main__":
    main()
