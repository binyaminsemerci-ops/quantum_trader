"""
Meta Strategy Controller (MSC AI) - Live System Integration

Integrates the MSC AI with Quantum Trader's existing infrastructure:
- Connects to SQLAlchemy database for metrics and strategies
- Reads from Redis/DB PolicyStore
- Provides REST API endpoints for monitoring
- Scheduled periodic evaluation (every 30 minutes)
- Prometheus metrics

Author: Quantum Trader Team
Date: 2025-11-30
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json

from sqlalchemy.orm import Session
from sqlalchemy import text
import redis
from prometheus_client import Counter, Histogram, Gauge

# Import from the comprehensive MSC file (not the directory package)
import importlib.util
from pathlib import Path

# Load meta_strategy_controller.py directly to avoid package conflict
spec = importlib.util.spec_from_file_location(
    "meta_strategy_controller_direct",
    Path(__file__).parent.parent / "meta_strategy_controller.py"
)
msc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msc_module)

MetaStrategyController = msc_module.MetaStrategyController
MetricsRepository = msc_module.MetricsRepository
StrategyRepository = msc_module.StrategyRepository
StrategyStats = msc_module.StrategyStats
StrategyConfig = msc_module.StrategyConfig
SystemHealth = msc_module.SystemHealth
RegimeType = msc_module.RegimeType

from backend.services.policy_store import PolicyStore
from backend.database import SessionLocal

logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

msc_evaluations_total = Counter(
    'msc_ai_evaluations_total',
    'Total MSC AI policy evaluations',
    ['risk_mode']
)

msc_policy_changes = Counter(
    'msc_ai_policy_changes_total',
    'Total MSC AI policy changes',
    ['from_mode', 'to_mode']
)

msc_active_strategies = Gauge(
    'msc_ai_active_strategies',
    'Number of strategies selected by MSC AI'
)

msc_evaluation_duration = Histogram(
    'msc_ai_evaluation_duration_seconds',
    'Time taken for MSC AI evaluation',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

msc_system_health_drawdown = Gauge(
    'msc_ai_system_health_drawdown_pct',
    'Current system drawdown percentage'
)

msc_system_health_winrate = Gauge(
    'msc_ai_system_health_winrate_pct',
    'Current system winrate percentage'
)


# ============================================================================
# Quantum Trader Metrics Repository Adapter
# ============================================================================

class QuantumMetricsRepository:
    """
    Adapter that reads system-wide metrics from Quantum Trader's database
    to provide data for MSC AI decision-making.
    """
    
    def __init__(self, session_factory=None):
        """
        Initialize metrics repository.
        
        Args:
            session_factory: SQLAlchemy session factory (defaults to SessionLocal)
        """
        self.session_factory = session_factory or SessionLocal
        logger.info("[MSC AI] QuantumMetricsRepository initialized")
    
    def get_current_drawdown_pct(self) -> float:
        """
        Get current drawdown percentage (last 30 days by default).
        
        Returns drawdown as positive percentage (e.g., 5.0 for 5% DD).
        Alias for get_drawdown() for compatibility with MSC AI controller.
        """
        return self.get_drawdown(period_days=30)
    
    def get_drawdown(self, period_days: int = 30) -> float:
        """
        Get current drawdown percentage over the specified period.
        
        Returns drawdown as positive percentage (e.g., 5.0 for 5% DD).
        """
        db: Session = self.session_factory()
        try:
            # Query execution_journal for filled orders
            query = text("""
                SELECT 
                    MAX(equity_before) as peak_equity,
                    MIN(equity_after) as trough_equity
                FROM execution_journal
                WHERE status = 'filled'
                AND created_at >= datetime('now', :period)
            """)
            
            result = db.execute(query, {"period": f"-{period_days} days"}).fetchone()
            
            if result and result[0] and result[1]:
                peak = float(result[0])
                trough = float(result[1])
                
                if peak > 0:
                    drawdown = ((peak - trough) / peak) * 100
                    return max(0.0, drawdown)
            
            # Fallback: check trade_logs table
            query_fallback = text("""
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as winning_trades
                FROM trade_logs
                WHERE status = 'filled'
                AND timestamp >= datetime('now', :period)
            """)
            
            result = db.execute(query_fallback, {"period": f"-{period_days} days"}).fetchone()
            
            if result and result[0]:
                # Rough estimate: if winrate < 45%, assume ~5% DD, else ~2%
                total = result[0]
                wins = result[1] or 0
                winrate = (wins / total) * 100 if total > 0 else 50
                
                if winrate < 45:
                    return 5.0
                elif winrate < 50:
                    return 3.0
                else:
                    return 1.5
            
            return 2.0  # Conservative default
            
        except Exception as e:
            logger.error(f"[MSC AI] Failed to get drawdown: {e}")
            return 2.0
        finally:
            db.close()
    
    def get_winrate(self, period_days: int = 30) -> float:
        """
        Get overall winrate percentage over the specified period.
        
        Returns winrate as decimal (e.g., 0.55 for 55%).
        """
        db: Session = self.session_factory()
        try:
            # Query execution_journal
            query = text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM execution_journal
                WHERE status = 'filled'
                AND created_at >= datetime('now', :period)
            """)
            
            result = db.execute(query, {"period": f"-{period_days} days"}).fetchone()
            
            if result and result[0] and result[0] > 0:
                total = result[0]
                wins = result[1] or 0
                return wins / total
            
            # Fallback: use trade_logs
            query_fallback = text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as wins
                FROM trade_logs
                WHERE status = 'filled'
                AND timestamp >= datetime('now', :period)
            """)
            
            result = db.execute(query_fallback, {"period": f"-{period_days} days"}).fetchone()
            
            if result and result[0] and result[0] > 0:
                total = result[0]
                wins = result[1] or 0
                return wins / total
            
            return 0.52  # Neutral default
            
        except Exception as e:
            logger.error(f"[MSC AI] Failed to get winrate: {e}")
            return 0.52
        finally:
            db.close()
    
    def get_equity_slope(self, period_days: int = 7) -> float:
        """
        Get equity curve slope (% change per day) over recent period.
        
        Positive = upward trend, negative = downward trend.
        """
        db: Session = self.session_factory()
        try:
            query = text("""
                SELECT 
                    julianday('now') - julianday(MIN(created_at)) as days,
                    (SELECT equity_after FROM execution_journal 
                     WHERE status = 'filled' 
                     ORDER BY created_at DESC LIMIT 1) as latest_equity,
                    (SELECT equity_before FROM execution_journal 
                     WHERE status = 'filled' 
                     ORDER BY created_at ASC LIMIT 1) as first_equity
                FROM execution_journal
                WHERE status = 'filled'
                AND created_at >= datetime('now', :period)
            """)
            
            result = db.execute(query, {"period": f"-{period_days} days"}).fetchone()
            
            if result and result[0] and result[1] and result[2]:
                days = result[0]
                latest = float(result[1])
                first = float(result[2])
                
                if days > 0 and first > 0:
                    total_change_pct = ((latest - first) / first) * 100
                    slope = total_change_pct / days
                    return slope
            
            return 0.0  # Flat if no data
            
        except Exception as e:
            logger.error(f"[MSC AI] Failed to get equity slope: {e}")
            return 0.0
        finally:
            db.close()
    
    def get_system_health(self) -> SystemHealth:
        """
        Gather comprehensive system health metrics.
        
        Returns:
            SystemHealth dataclass with all metrics populated
        """
        try:
            drawdown = self.get_drawdown(period_days=30)
            winrate = self.get_winrate(period_days=30)
            equity_slope = self.get_equity_slope(period_days=7)
            
            # Determine regime (simplified - you can enhance this with RegimeDetector)
            regime = RegimeType.RANGING
            if equity_slope > 0.5:
                regime = RegimeType.BULL_TRENDING
            elif equity_slope < -0.5:
                regime = RegimeType.BEAR_TRENDING
            elif abs(equity_slope) < 0.1:
                regime = RegimeType.CHOPPY
            
            # Determine volatility (simplified)
            volatility = "NORMAL"
            if drawdown > 5.0:
                volatility = "HIGH"
            elif drawdown > 7.0:
                volatility = "EXTREME"
            elif drawdown < 2.0:
                volatility = "LOW"
            
            # Track consecutive losses (simplified)
            consecutive_losses = 0
            days_since_profit = 0
            
            db: Session = self.session_factory()
            try:
                query = text("""
                    SELECT pnl 
                    FROM execution_journal 
                    WHERE status = 'filled' 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                results = db.execute(query).fetchall()
                
                for row in results:
                    pnl = row[0]
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        break
                
                # Days since last profit
                query_profit = text("""
                    SELECT julianday('now') - julianday(created_at) as days
                    FROM execution_journal
                    WHERE status = 'filled' AND pnl > 0
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                result = db.execute(query_profit).fetchone()
                if result and result[0]:
                    days_since_profit = int(result[0])
                    
            finally:
                db.close()
            
            health = SystemHealth(
                current_drawdown=drawdown,
                global_winrate=winrate,
                equity_slope_pct_per_day=equity_slope,
                regime=regime,
                volatility=volatility,
                consecutive_losses=consecutive_losses,
                days_since_profit=days_since_profit
            )
            
            # Update Prometheus metrics
            msc_system_health_drawdown.set(drawdown)
            msc_system_health_winrate.set(winrate * 100)
            
            logger.info(f"[MSC AI] System Health: DD={drawdown:.2f}%, WR={winrate*100:.1f}%, Slope={equity_slope:+.2f}%/day, Regime={regime.value}")
            
            return health
            
        except Exception as e:
            logger.error(f"[MSC AI] Failed to get system health: {e}", exc_info=True)
            # Return safe defaults
            return SystemHealth(
                current_drawdown=3.0,
                global_winrate=0.50,
                equity_slope_pct_per_day=0.0,
                regime=RegimeType.RANGING,
                volatility="NORMAL",
                consecutive_losses=0,
                days_since_profit=0
            )


# ============================================================================
# Quantum Trader Strategy Repository Adapter (MSC AI)
# ============================================================================

class QuantumStrategyRepositoryMSC:
    """
    Adapter that reads LIVE strategy configurations from the database
    for MSC AI to evaluate and select from.
    """
    
    def __init__(self, session_factory=None):
        """
        Initialize strategy repository for MSC AI.
        
        Args:
            session_factory: SQLAlchemy session factory (defaults to SessionLocal)
        """
        self.session_factory = session_factory or SessionLocal
        logger.info("[MSC AI] QuantumStrategyRepositoryMSC initialized")
    
    def list_live_strategies(self) -> List[StrategyConfig]:
        """
        Get all LIVE strategies from the database.
        
        Returns:
            List of StrategyConfig objects representing active strategies
        """
        db: Session = self.session_factory()
        try:
            # Query the runtime_strategies table for LIVE strategies
            query = text("""
                SELECT 
                    id,
                    name,
                    description,
                    status,
                    regime_compatibility,
                    created_at,
                    updated_at
                FROM runtime_strategies
                WHERE status = 'LIVE'
                ORDER BY created_at DESC
            """)
            
            results = db.execute(query).fetchall()
            
            strategies = []
            for row in results:
                strategy = StrategyConfig(
                    strategy_id=str(row[0]),
                    strategy_name=row[1] or f"STRATEGY_{row[0]}",
                    description=row[2] or "No description",
                    regime_compatibility=row[4] or "TRENDING",
                    created_at=row[5] or datetime.now(timezone.utc),
                    updated_at=row[6] or datetime.now(timezone.utc)
                )
                strategies.append(strategy)
            
            logger.info(f"[MSC AI] Found {len(strategies)} LIVE strategies")
            return strategies
            
        except Exception as e:
            logger.error(f"[MSC AI] Failed to list LIVE strategies: {e}")
            # Return mock strategies for fallback
            return self._get_fallback_strategies()
        finally:
            db.close()
    
    def get_strategy_stats(self, strategy_id: str, period_days: int = 30) -> Optional[StrategyStats]:
        """
        Get performance statistics for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            period_days: Lookback period
            
        Returns:
            StrategyStats with performance metrics, or None if not found
        """
        db: Session = self.session_factory()
        try:
            # Query execution_journal for strategy-specific metrics
            query = text("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM execution_journal
                WHERE status = 'filled'
                AND strategy_id = :strategy_id
                AND created_at >= datetime('now', :period)
            """)
            
            result = db.execute(query, {
                "strategy_id": strategy_id,
                "period": f"-{period_days} days"
            }).fetchone()
            
            if result and result[0] and result[0] > 0:
                total = result[0]
                wins = result[1] or 0
                avg_pnl = result[2] or 0.0
                total_pnl = result[3] or 0.0
                best = result[4] or 0.0
                worst = result[5] or 0.0
                
                winrate = wins / total
                
                # Calculate profit factor
                gross_profit = sum([p for p in [best] if p > 0])
                gross_loss = abs(sum([p for p in [worst] if p < 0]))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0
                
                # Calculate max drawdown (simplified)
                max_drawdown = abs(worst / 100) if worst < 0 else 0.0
                
                stats = StrategyStats(
                    total_trades=total,
                    winrate=winrate,
                    profit_factor=profit_factor,
                    avg_R_multiple=avg_pnl / 100 if avg_pnl else 0.5,
                    max_drawdown=max_drawdown,
                    total_pnl=total_pnl
                )
                
                return stats
            
            # No data found - return conservative estimates
            return StrategyStats(
                total_trades=50,
                winrate=0.50,
                profit_factor=1.5,
                avg_R_multiple=0.8,
                max_drawdown=0.03,
                total_pnl=0.0
            )
            
        except Exception as e:
            logger.warning(f"[MSC AI] Failed to get strategy stats for {strategy_id}: {e}")
            return None
        finally:
            db.close()
    
    def _get_fallback_strategies(self) -> List[StrategyConfig]:
        """Return fallback strategies when database is unavailable."""
        return [
            StrategyConfig(
                strategy_id="FALLBACK_001",
                strategy_name="Safe Momentum Strategy",
                description="Conservative momentum trading",
                regime_compatibility="TRENDING",
                created_at=datetime.now(timezone.utc) - timedelta(days=30),
                updated_at=datetime.now(timezone.utc)
            ),
            StrategyConfig(
                strategy_id="FALLBACK_002",
                strategy_name="Mean Reversion Strategy",
                description="Range-bound mean reversion",
                regime_compatibility="RANGING",
                created_at=datetime.now(timezone.utc) - timedelta(days=20),
                updated_at=datetime.now(timezone.utc)
            )
        ]


# ============================================================================
# Quantum Trader Policy Store (MSC AI)
# ============================================================================

class QuantumPolicyStoreMSC:
    """
    Policy store that writes MSC AI decisions to both Redis and database
    for consumption by other trading components.
    """
    
    def __init__(self, redis_url: Optional[str] = None, session_factory=None):
        """
        Initialize policy store with Redis and database backends.
        
        Args:
            redis_url: Redis connection URL (defaults to localhost)
            session_factory: SQLAlchemy session factory
        """
        self.redis_client = None
        self.session_factory = session_factory or SessionLocal
        
        # Try to connect to Redis
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"[MSC AI] PolicyStore connected to Redis: {redis_url}")
        except Exception as e:
            logger.warning(f"[MSC AI] Redis not available: {e}. Using DB-only mode.")
            self.redis_client = None
    
    def write_policy(self, policy: Dict) -> None:
        """
        Write policy to both Redis and database.
        
        Args:
            policy: Policy dictionary from MSC AI
        """
        try:
            # Write to Redis (fast access for trading components)
            if self.redis_client:
                try:
                    self.redis_client.set(
                        "msc_ai:current_policy",
                        json.dumps(policy),
                        ex=3600  # Expire after 1 hour
                    )
                    
                    # Also write individual keys for easy access
                    self.redis_client.set("msc_ai:risk_mode", policy["risk_mode"], ex=3600)
                    self.redis_client.set("msc_ai:max_risk_per_trade", policy["max_risk_per_trade"], ex=3600)
                    self.redis_client.set("msc_ai:max_positions", policy["max_positions"], ex=3600)
                    self.redis_client.set("msc_ai:min_confidence", policy["global_min_confidence"], ex=3600)
                    
                    # Store allowed strategies as a set
                    self.redis_client.delete("msc_ai:allowed_strategies")
                    if policy["allowed_strategies"]:
                        self.redis_client.sadd("msc_ai:allowed_strategies", *policy["allowed_strategies"])
                        self.redis_client.expire("msc_ai:allowed_strategies", 3600)
                    
                    logger.info(f"[MSC AI] Policy written to Redis: {policy['risk_mode']}")
                except Exception as e:
                    logger.error(f"[MSC AI] Failed to write to Redis: {e}")
            
            # Write to database (persistent storage and audit trail)
            db: Session = self.session_factory()
            try:
                # Create msc_policies table if it doesn't exist
                create_table_query = text("""
                    CREATE TABLE IF NOT EXISTS msc_policies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        risk_mode TEXT NOT NULL,
                        max_risk_per_trade REAL NOT NULL,
                        max_positions INTEGER NOT NULL,
                        min_confidence REAL NOT NULL,
                        max_daily_trades INTEGER NOT NULL,
                        allowed_strategies TEXT,
                        system_drawdown REAL,
                        system_winrate REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                db.execute(create_table_query)
                db.commit()
                
                # Insert policy record
                insert_query = text("""
                    INSERT INTO msc_policies (
                        risk_mode,
                        max_risk_per_trade,
                        max_positions,
                        min_confidence,
                        max_daily_trades,
                        allowed_strategies,
                        system_drawdown,
                        system_winrate
                    ) VALUES (
                        :risk_mode,
                        :max_risk_per_trade,
                        :max_positions,
                        :min_confidence,
                        :max_daily_trades,
                        :allowed_strategies,
                        :system_drawdown,
                        :system_winrate
                    )
                """)
                
                db.execute(insert_query, {
                    "risk_mode": policy["risk_mode"],
                    "max_risk_per_trade": policy["max_risk_per_trade"],
                    "max_positions": policy["max_positions"],
                    "min_confidence": policy["global_min_confidence"],
                    "max_daily_trades": policy.get("max_daily_trades", 30),
                    "allowed_strategies": json.dumps(policy["allowed_strategies"]),
                    "system_drawdown": policy.get("system_health", {}).get("current_drawdown", 0.0),
                    "system_winrate": policy.get("system_health", {}).get("global_winrate", 0.0)
                })
                db.commit()
                
                logger.info(f"[MSC AI] Policy written to database")
                
            except Exception as e:
                logger.error(f"[MSC AI] Failed to write policy to database: {e}")
                db.rollback()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"[MSC AI] Failed to write policy: {e}", exc_info=True)
    
    def read_policy(self) -> Optional[Dict]:
        """
        Read current policy from Redis or database.
        
        Returns:
            Current policy dictionary, or None if not found
        """
        # Try Redis first (fastest)
        if self.redis_client:
            try:
                policy_json = self.redis_client.get("msc_ai:current_policy")
                if policy_json:
                    return json.loads(policy_json)
            except Exception as e:
                logger.warning(f"[MSC AI] Failed to read from Redis: {e}")
        
        # Fallback to database
        db: Session = self.session_factory()
        try:
            query = text("""
                SELECT 
                    risk_mode,
                    max_risk_per_trade,
                    max_positions,
                    min_confidence,
                    max_daily_trades,
                    allowed_strategies,
                    created_at
                FROM msc_policies
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            result = db.execute(query).fetchone()
            
            if result:
                return {
                    "risk_mode": result[0],
                    "max_risk_per_trade": result[1],
                    "max_positions": result[2],
                    "global_min_confidence": result[3],
                    "max_daily_trades": result[4],
                    "allowed_strategies": json.loads(result[5]) if result[5] else [],
                    "updated_at": result[6]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"[MSC AI] Failed to read policy from database: {e}")
            return None
        finally:
            db.close()


# ============================================================================
# MSC AI Controller Instance
# ============================================================================

# Global reference to MSC controller (set once at startup)
_msc_controller_instance: Optional[MetaStrategyController] = None


def initialize_msc_controller(opportunity_ranker=None) -> MetaStrategyController:
    """
    Initialize MSC AI controller with OpportunityRanker integration.
    
    Call this ONCE at startup with opportunity_ranker from app state.
    
    Args:
        opportunity_ranker: OpportunityRanker instance from main app
        
    Returns:
        MetaStrategyController configured with Quantum Trader adapters
    """
    global _msc_controller_instance
    
    logger.info("[MSC AI] Initializing Meta Strategy Controller...")
    
    metrics_repo = QuantumMetricsRepository()
    strategy_repo = QuantumStrategyRepositoryMSC()
    policy_store = QuantumPolicyStoreMSC()
    
    _msc_controller_instance = MetaStrategyController(
        metrics_repo=metrics_repo,
        strategy_repo=strategy_repo,
        policy_store=policy_store,
        evaluation_period_days=30,
        min_strategies=2,
        max_strategies=8,
        opportunity_ranker=opportunity_ranker  # NEW: OpportunityRanker integration
    )
    
    logger.info("[MSC AI] Meta Strategy Controller ready ✓")
    if opportunity_ranker:
        logger.info("[MSC AI] OpportunityRanker integration: ENABLED")
    
    return _msc_controller_instance


@lru_cache(maxsize=1)
def get_msc_controller() -> MetaStrategyController:
    """
    Get existing MSC AI controller instance.
    
    Falls back to creating one without OpportunityRanker if not initialized.
    
    Returns:
        MetaStrategyController configured with Quantum Trader adapters
    """
    global _msc_controller_instance
    
    if _msc_controller_instance is not None:
        return _msc_controller_instance
    
    # Fallback: create without OpportunityRanker
    logger.warning("[MSC AI] Controller not initialized, creating fallback instance")
    return initialize_msc_controller(opportunity_ranker=None)


def run_msc_evaluation() -> Dict:
    """
    Execute MSC AI evaluation and update global policy.
    
    Returns:
        Dictionary with evaluation results and updated policy
    """
    import time
    
    start_time = time.time()
    
    try:
        logger.info("=" * 80)
        logger.info("[MSC AI] Starting policy evaluation cycle")
        logger.info("=" * 80)
        
        controller = get_msc_controller()
        
        # Run evaluation
        policy = controller.evaluate_and_update_policy()
        
        # Track metrics
        duration = time.time() - start_time
        msc_evaluation_duration.observe(duration)
        msc_evaluations_total.labels(risk_mode=policy["risk_mode"]).inc()
        msc_active_strategies.set(len(policy["allowed_strategies"]))
        
        logger.info("=" * 80)
        logger.info(f"[MSC AI] Evaluation completed in {duration:.2f}s")
        logger.info(f"[MSC AI] Risk Mode: {policy['risk_mode']}")
        logger.info(f"[MSC AI] Active Strategies: {len(policy['allowed_strategies'])}")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "duration_seconds": duration,
            "policy": policy,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[MSC AI] Evaluation failed: {e}", exc_info=True)
        duration = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
