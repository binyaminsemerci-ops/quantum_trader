"""
Health check endpoints for Strategy Generator AI services.

Provides status monitoring for continuous_runner and shadow_runner.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from backend.database import SessionLocal
from backend.research.postgres_repository import PostgresStrategyRepository
from backend.research.models import StrategyStatus


def get_strategy_generator_health() -> Dict[str, Any]:
    """
    Get health status of strategy generator service.
    
    Returns:
        dict: Health status including generation count, last run time, etc.
    """
    repo = PostgresStrategyRepository(SessionLocal)
    
    try:
        # Count strategies by status
        candidate_count = len(repo.get_strategies_by_status(StrategyStatus.CANDIDATE))
        shadow_count = len(repo.get_strategies_by_status(StrategyStatus.SHADOW))
        live_count = len(repo.get_strategies_by_status(StrategyStatus.LIVE))
        disabled_count = len(repo.get_strategies_by_status(StrategyStatus.DISABLED))
        
        # Get all strategies and aggregate stats
        all_strategies = (
            repo.get_strategies_by_status(StrategyStatus.CANDIDATE) +
            repo.get_strategies_by_status(StrategyStatus.SHADOW) +
            repo.get_strategies_by_status(StrategyStatus.LIVE)
        )
        
        # Collect recent backtest stats from all strategies
        recent_stats = []
        for strategy in all_strategies:
            stats = repo.get_stats(strategy.strategy_id, source="BACKTEST", days=7)
            recent_stats.extend(stats)
        
        # Calculate average performance
        avg_fitness = sum(s.fitness_score for s in recent_stats) / len(recent_stats) if recent_stats else 0
        avg_pf = sum(s.profit_factor for s in recent_stats) / len(recent_stats) if recent_stats else 0
        avg_wr = sum(s.win_rate for s in recent_stats) / len(recent_stats) if recent_stats else 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "strategies": {
                "candidate": candidate_count,
                "shadow": shadow_count,
                "live": live_count,
                "disabled": disabled_count,
                "total": candidate_count + shadow_count + live_count + disabled_count
            },
            "recent_performance": {
                "avg_fitness": round(avg_fitness, 2),
                "avg_profit_factor": round(avg_pf, 2),
                "avg_win_rate": round(avg_wr, 3),
                "backtest_count": len(recent_stats)
            }
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


def get_shadow_tester_health() -> Dict[str, Any]:
    """
    Get health status of shadow tester service.
    
    Returns:
        dict: Health status including shadow test results, deployment status, etc.
    """
    repo = PostgresStrategyRepository(SessionLocal)
    
    try:
        # Get shadow strategies
        shadow_strategies = repo.get_strategies_by_status(StrategyStatus.SHADOW)
        
        # Get recent shadow test results from all shadow strategies
        shadow_stats = []
        for strategy in shadow_strategies:
            stats = repo.get_stats(strategy.strategy_id, source="SHADOW", days=7)
            shadow_stats.extend(stats)
        
        # Calculate shadow performance
        avg_fitness = sum(s.fitness_score for s in shadow_stats) / len(shadow_stats) if shadow_stats else 0
        avg_pf = sum(s.profit_factor for s in shadow_stats) / len(shadow_stats) if shadow_stats else 0
        avg_wr = sum(s.win_rate for s in shadow_stats) / len(shadow_stats) if shadow_stats else 0
        
        # Count deployment-ready strategies (fitness >= 70)
        deployment_ready = sum(1 for s in shadow_stats if s.fitness_score >= 70.0)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "shadow_testing": {
                "active_strategies": len(shadow_strategies),
                "recent_tests": len(shadow_stats),
                "deployment_ready": deployment_ready
            },
            "shadow_performance": {
                "avg_fitness": round(avg_fitness, 2),
                "avg_profit_factor": round(avg_pf, 2),
                "avg_win_rate": round(avg_wr, 3)
            }
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    print("=== Strategy Generator Health ===")
    print(get_strategy_generator_health())
    
    print("\n=== Shadow Tester Health ===")
    print(get_shadow_tester_health())
