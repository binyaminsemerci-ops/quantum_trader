"""
RL v2 Monitoring Dashboard
===========================

Real-time monitoring for RL v2 Q-learning agents.

Features:
- Q-table growth tracking
- Learning progress visualization
- Hyperparameter monitoring
- Performance comparison (RL v1 vs v2)

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class RLv2Monitor:
    """Real-time monitoring for RL v2 agents."""
    
    def __init__(
        self,
        meta_q_table_path: Path = Path("data/rl_v2/meta_strategy_q_table.json"),
        sizing_q_table_path: Path = Path("data/rl_v2/position_sizing_q_table.json"),
        metrics_path: Path = Path("data/rl_v2/metrics.json")
    ):
        """
        Initialize RL v2 monitor.
        
        Args:
            meta_q_table_path: Path to meta strategy Q-table
            sizing_q_table_path: Path to position sizing Q-table
            metrics_path: Path to metrics file
        """
        self.meta_q_table_path = meta_q_table_path
        self.sizing_q_table_path = sizing_q_table_path
        self.metrics_path = metrics_path
        
        # Historical metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(
            "[RL v2 Monitor] Initialized",
            meta_q_table=str(meta_q_table_path),
            sizing_q_table=str(sizing_q_table_path)
        )
    
    def get_q_table_stats(self, q_table_path: Path) -> Dict[str, Any]:
        """
        Get Q-table statistics.
        
        Args:
            q_table_path: Path to Q-table file
            
        Returns:
            Q-table stats
        """
        try:
            if not q_table_path.exists():
                return {
                    "exists": False,
                    "total_states": 0,
                    "total_state_actions": 0,
                    "epsilon": 0.0,
                    "update_count": 0,
                    "file_size_kb": 0
                }
            
            # Load Q-table
            with open(q_table_path, 'r') as f:
                data = json.load(f)
            
            q_table = data.get("q_table", {})
            
            # Calculate stats
            total_states = len(q_table)
            total_state_actions = sum(len(actions) for actions in q_table.values())
            avg_actions_per_state = total_state_actions / total_states if total_states > 0 else 0
            
            # Get file size
            file_size_kb = q_table_path.stat().st_size / 1024
            
            # Get best Q-values
            all_q_values = [q for actions in q_table.values() for q in actions.values()]
            max_q = max(all_q_values) if all_q_values else 0.0
            min_q = min(all_q_values) if all_q_values else 0.0
            avg_q = sum(all_q_values) / len(all_q_values) if all_q_values else 0.0
            
            return {
                "exists": True,
                "total_states": total_states,
                "total_state_actions": total_state_actions,
                "avg_actions_per_state": round(avg_actions_per_state, 2),
                "epsilon": data.get("epsilon", 0.0),
                "update_count": data.get("update_count", 0),
                "file_size_kb": round(file_size_kb, 2),
                "max_q_value": round(max_q, 4),
                "min_q_value": round(min_q, 4),
                "avg_q_value": round(avg_q, 4)
            }
            
        except Exception as e:
            logger.error(
                "[RL v2 Monitor] Failed to get Q-table stats",
                error=str(e),
                q_table_path=str(q_table_path)
            )
            return {
                "exists": False,
                "error": str(e)
            }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.
        
        Returns:
            Current metrics
        """
        meta_stats = self.get_q_table_stats(self.meta_q_table_path)
        sizing_stats = self.get_q_table_stats(self.sizing_q_table_path)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "meta_agent": meta_stats,
            "sizing_agent": sizing_stats,
            "total_updates": meta_stats.get("update_count", 0) + sizing_stats.get("update_count", 0),
            "total_states": meta_stats.get("total_states", 0) + sizing_stats.get("total_states", 0),
            "total_size_kb": meta_stats.get("file_size_kb", 0) + sizing_stats.get("file_size_kb", 0)
        }
        
        return metrics
    
    def record_metrics(self):
        """Record current metrics to history and file."""
        metrics = self.get_current_metrics()
        self.metrics_history.append(metrics)
        
        # Save to file
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(
                "[RL v2 Monitor] Failed to save metrics",
                error=str(e)
            )
    
    def print_dashboard(self):
        """Print monitoring dashboard to console."""
        metrics = self.get_current_metrics()
        
        print("\n" + "=" * 80)
        print("RL v2 MONITORING DASHBOARD")
        print("=" * 80)
        print(f"Timestamp: {metrics['timestamp']}")
        print()
        
        # Meta Strategy Agent
        print("📊 META STRATEGY AGENT")
        print("-" * 80)
        meta = metrics["meta_agent"]
        if meta.get("exists"):
            print(f"  States Learned:        {meta['total_states']:,}")
            print(f"  State-Action Pairs:    {meta['total_state_actions']:,}")
            print(f"  Avg Actions/State:     {meta['avg_actions_per_state']}")
            print(f"  Total Updates:         {meta['update_count']:,}")
            print(f"  Exploration Rate (ε):  {meta['epsilon']:.4f}")
            print(f"  Q-Value Range:         [{meta['min_q_value']:.4f}, {meta['max_q_value']:.4f}]")
            print(f"  Avg Q-Value:           {meta['avg_q_value']:.4f}")
            print(f"  Q-Table Size:          {meta['file_size_kb']:.2f} KB")
        else:
            print("  ⚠️  Q-table not found - agent not yet trained")
        print()
        
        # Position Sizing Agent
        print("📈 POSITION SIZING AGENT")
        print("-" * 80)
        sizing = metrics["sizing_agent"]
        if sizing.get("exists"):
            print(f"  States Learned:        {sizing['total_states']:,}")
            print(f"  State-Action Pairs:    {sizing['total_state_actions']:,}")
            print(f"  Avg Actions/State:     {sizing['avg_actions_per_state']}")
            print(f"  Total Updates:         {sizing['update_count']:,}")
            print(f"  Exploration Rate (ε):  {sizing['epsilon']:.4f}")
            print(f"  Q-Value Range:         [{sizing['min_q_value']:.4f}, {sizing['max_q_value']:.4f}]")
            print(f"  Avg Q-Value:           {sizing['avg_q_value']:.4f}")
            print(f"  Q-Table Size:          {sizing['file_size_kb']:.2f} KB")
        else:
            print("  ⚠️  Q-table not found - agent not yet trained")
        print()
        
        # Overall Stats
        print("🎯 OVERALL STATISTICS")
        print("-" * 80)
        print(f"  Total Updates:         {metrics['total_updates']:,}")
        print(f"  Total States:          {metrics['total_states']:,}")
        print(f"  Total Q-Table Size:    {metrics['total_size_kb']:.2f} KB")
        print()
        
        # Learning Progress
        if len(self.metrics_history) > 1:
            print("📈 LEARNING PROGRESS (Last vs Current)")
            print("-" * 80)
            last = self.metrics_history[-2] if len(self.metrics_history) > 1 else metrics
            
            # Meta agent growth
            meta_states_growth = metrics["meta_agent"].get("total_states", 0) - last["meta_agent"].get("total_states", 0)
            meta_updates_growth = metrics["meta_agent"].get("update_count", 0) - last["meta_agent"].get("update_count", 0)
            
            # Sizing agent growth
            sizing_states_growth = metrics["sizing_agent"].get("total_states", 0) - last["sizing_agent"].get("total_states", 0)
            sizing_updates_growth = metrics["sizing_agent"].get("update_count", 0) - last["sizing_agent"].get("update_count", 0)
            
            print(f"  Meta Agent:")
            print(f"    New States:          +{meta_states_growth}")
            print(f"    New Updates:         +{meta_updates_growth}")
            print(f"  Sizing Agent:")
            print(f"    New States:          +{sizing_states_growth}")
            print(f"    New Updates:         +{sizing_updates_growth}")
            print()
        
        # Health Check
        print("🏥 HEALTH CHECK")
        print("-" * 80)
        
        # Check Q-table size
        if metrics['total_size_kb'] > 10000:  # > 10 MB
            print("  ⚠️  WARNING: Q-tables are large (> 10 MB)")
            print("      Consider state space reduction or function approximation")
        elif metrics['total_size_kb'] > 1000:  # > 1 MB
            print("  ⚡ Q-tables growing normally")
        else:
            print("  ✅ Q-tables at healthy size")
        
        # Check epsilon
        avg_epsilon = (meta.get("epsilon", 0) + sizing.get("epsilon", 0)) / 2
        if avg_epsilon > 0.5:
            print("  🔍 High exploration rate - still learning")
        elif avg_epsilon > 0.1:
            print("  ⚖️  Balanced exploration/exploitation")
        else:
            print("  🎯 Low exploration - mostly exploiting learned policy")
        
        # Check updates
        if metrics['total_updates'] < 100:
            print("  🌱 Early learning stage (< 100 updates)")
        elif metrics['total_updates'] < 1000:
            print("  📚 Active learning (100-1000 updates)")
        else:
            print("  🎓 Mature agent (> 1000 updates)")
        
        print()
        print("=" * 80)
        print()
    
    def monitor_continuous(self, interval_seconds: int = 60):
        """
        Continuously monitor and display dashboard.
        
        Args:
            interval_seconds: Update interval in seconds
        """
        print(f"\n🔄 Starting continuous monitoring (updates every {interval_seconds}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.print_dashboard()
                self.record_metrics()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n✋ Monitoring stopped by user")
            logger.info("[RL v2 Monitor] Continuous monitoring stopped")
    
    def get_growth_rate(self, window: int = 10) -> Dict[str, float]:
        """
        Calculate growth rates from historical metrics.
        
        Args:
            window: Number of historical points to consider
            
        Returns:
            Growth rates
        """
        if len(self.metrics_history) < 2:
            return {
                "meta_states_per_update": 0.0,
                "sizing_states_per_update": 0.0
            }
        
        # Get window of metrics
        recent = self.metrics_history[-window:] if len(self.metrics_history) >= window else self.metrics_history
        
        if len(recent) < 2:
            return {
                "meta_states_per_update": 0.0,
                "sizing_states_per_update": 0.0
            }
        
        # Calculate growth
        first = recent[0]
        last = recent[-1]
        
        meta_states_delta = last["meta_agent"].get("total_states", 0) - first["meta_agent"].get("total_states", 0)
        meta_updates_delta = last["meta_agent"].get("update_count", 0) - first["meta_agent"].get("update_count", 0)
        
        sizing_states_delta = last["sizing_agent"].get("total_states", 0) - first["sizing_agent"].get("total_states", 0)
        sizing_updates_delta = last["sizing_agent"].get("update_count", 0) - first["sizing_agent"].get("update_count", 0)
        
        return {
            "meta_states_per_update": meta_states_delta / meta_updates_delta if meta_updates_delta > 0 else 0.0,
            "sizing_states_per_update": sizing_states_delta / sizing_updates_delta if sizing_updates_delta > 0 else 0.0
        }


def main():
    """Run RL v2 monitoring dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL v2 Monitoring Dashboard")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous monitoring"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval for continuous monitoring (seconds)"
    )
    
    args = parser.parse_args()
    
    monitor = RLv2Monitor()
    
    if args.continuous:
        monitor.monitor_continuous(interval_seconds=args.interval)
    else:
        monitor.print_dashboard()
        monitor.record_metrics()


if __name__ == "__main__":
    main()
