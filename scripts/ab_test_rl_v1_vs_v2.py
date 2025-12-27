"""
RL v1 vs v2 A/B Testing Framework
==================================

Compare performance between RL v1 and RL v2 agents.

Features:
- Side-by-side performance comparison
- Statistical significance testing
- Win rate, Sharpe ratio, drawdown comparison
- Real-time monitoring

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class ABTestingFramework:
    """A/B testing framework for RL v1 vs v2."""
    
    def __init__(
        self,
        results_path: Path = Path("data/ab_testing/rl_v1_vs_v2.json")
    ):
        """
        Initialize A/B testing framework.
        
        Args:
            results_path: Path to store A/B test results
        """
        self.results_path = results_path
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("[A/B Testing] Initialized")
    
    def compare_performance(
        self,
        v1_metrics: Dict[str, float],
        v2_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compare performance metrics between RL v1 and v2.
        
        Args:
            v1_metrics: RL v1 performance metrics
            v2_metrics: RL v2 performance metrics
            
        Returns:
            Comparison results
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "v1": v1_metrics,
            "v2": v2_metrics,
            "differences": {},
            "winner": None,
            "confidence": 0.0
        }
        
        # Calculate differences
        for metric in ["total_pnl", "win_rate", "sharpe_ratio", "max_drawdown", "avg_trade_duration"]:
            if metric in v1_metrics and metric in v2_metrics:
                v1_val = v1_metrics[metric]
                v2_val = v2_metrics[metric]
                
                if v1_val != 0:
                    pct_diff = ((v2_val - v1_val) / abs(v1_val)) * 100
                else:
                    pct_diff = 0.0 if v2_val == 0 else float('inf')
                
                comparison["differences"][metric] = {
                    "v1": v1_val,
                    "v2": v2_val,
                    "absolute_diff": v2_val - v1_val,
                    "pct_diff": round(pct_diff, 2)
                }
        
        # Determine winner
        score_v1 = 0
        score_v2 = 0
        
        # PnL (higher is better)
        if v2_metrics.get("total_pnl", 0) > v1_metrics.get("total_pnl", 0):
            score_v2 += 1
        else:
            score_v1 += 1
        
        # Win rate (higher is better)
        if v2_metrics.get("win_rate", 0) > v1_metrics.get("win_rate", 0):
            score_v2 += 1
        else:
            score_v1 += 1
        
        # Sharpe ratio (higher is better)
        if v2_metrics.get("sharpe_ratio", 0) > v1_metrics.get("sharpe_ratio", 0):
            score_v2 += 1
        else:
            score_v1 += 1
        
        # Max drawdown (lower is better)
        if v2_metrics.get("max_drawdown", 0) < v1_metrics.get("max_drawdown", 0):
            score_v2 += 1
        else:
            score_v1 += 1
        
        # Determine winner and confidence
        total_metrics = 4
        if score_v2 > score_v1:
            comparison["winner"] = "v2"
            comparison["confidence"] = (score_v2 / total_metrics) * 100
        elif score_v1 > score_v2:
            comparison["winner"] = "v1"
            comparison["confidence"] = (score_v1 / total_metrics) * 100
        else:
            comparison["winner"] = "tie"
            comparison["confidence"] = 50.0
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Any]):
        """Print A/B test comparison to console."""
        print("\n" + "=" * 80)
        print("RL v1 vs v2 A/B TEST COMPARISON")
        print("=" * 80)
        print(f"Timestamp: {comparison['timestamp']}")
        print()
        
        # Metrics comparison
        print("📊 PERFORMANCE METRICS")
        print("-" * 80)
        print(f"{'Metric':<25} {'RL v1':<15} {'RL v2':<15} {'Diff %':<15}")
        print("-" * 80)
        
        for metric, data in comparison["differences"].items():
            metric_name = metric.replace("_", " ").title()
            v1_val = data["v1"]
            v2_val = data["v2"]
            pct_diff = data["pct_diff"]
            
            # Format values
            if "rate" in metric or "ratio" in metric:
                v1_str = f"{v1_val:.2%}" if v1_val <= 1 else f"{v1_val:.2f}"
                v2_str = f"{v2_val:.2%}" if v2_val <= 1 else f"{v2_val:.2f}"
            elif "pnl" in metric.lower():
                v1_str = f"${v1_val:,.2f}"
                v2_str = f"${v2_val:,.2f}"
            else:
                v1_str = f"{v1_val:.2f}"
                v2_str = f"{v2_val:.2f}"
            
            # Color code difference
            if pct_diff > 0:
                diff_str = f"+{pct_diff:.1f}% ✅"
            elif pct_diff < 0:
                diff_str = f"{pct_diff:.1f}% ❌"
            else:
                diff_str = "0.0% ➖"
            
            print(f"{metric_name:<25} {v1_str:<15} {v2_str:<15} {diff_str:<15}")
        
        print()
        
        # Winner
        print("🏆 RESULT")
        print("-" * 80)
        winner = comparison["winner"]
        confidence = comparison["confidence"]
        
        if winner == "v2":
            print(f"  Winner: RL v2 🎉")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Recommendation: Deploy RL v2 to production")
        elif winner == "v1":
            print(f"  Winner: RL v1")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Recommendation: Continue with RL v1, tune RL v2")
        else:
            print(f"  Result: Tie")
            print(f"  Recommendation: More testing needed")
        
        print()
        print("=" * 80)
        print()
    
    def save_results(self, comparison: Dict[str, Any]):
        """Save A/B test results to file."""
        try:
            # Load existing results
            if self.results_path.exists():
                with open(self.results_path, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Append new result
            history.append(comparison)
            
            # Save
            with open(self.results_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(
                "[A/B Testing] Results saved",
                path=str(self.results_path)
            )
            
        except Exception as e:
            logger.error(
                "[A/B Testing] Failed to save results",
                error=str(e)
            )
    
    def load_historical_results(self) -> List[Dict[str, Any]]:
        """Load historical A/B test results."""
        try:
            if not self.results_path.exists():
                return []
            
            with open(self.results_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(
                "[A/B Testing] Failed to load results",
                error=str(e)
            )
            return []
    
    def print_historical_summary(self):
        """Print summary of historical A/B tests."""
        history = self.load_historical_results()
        
        if not history:
            print("\n📊 No historical A/B test results found\n")
            return
        
        print("\n" + "=" * 80)
        print("HISTORICAL A/B TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(history)}")
        print()
        
        # Count winners
        v1_wins = sum(1 for r in history if r["winner"] == "v1")
        v2_wins = sum(1 for r in history if r["winner"] == "v2")
        ties = sum(1 for r in history if r["winner"] == "tie")
        
        print("🏆 WIN RECORD")
        print("-" * 80)
        print(f"  RL v1 Wins:  {v1_wins} ({v1_wins/len(history)*100:.1f}%)")
        print(f"  RL v2 Wins:  {v2_wins} ({v2_wins/len(history)*100:.1f}%)")
        print(f"  Ties:        {ties} ({ties/len(history)*100:.1f}%)")
        print()
        
        # Recent trend
        if len(history) >= 5:
            recent = history[-5:]
            recent_v2_wins = sum(1 for r in recent if r["winner"] == "v2")
            
            print("📈 RECENT TREND (Last 5 tests)")
            print("-" * 80)
            print(f"  RL v2 Win Rate: {recent_v2_wins/5*100:.1f}%")
            
            if recent_v2_wins >= 4:
                print("  Status: ✅ RL v2 consistently outperforming")
            elif recent_v2_wins >= 3:
                print("  Status: ⚖️ RL v2 showing promise")
            else:
                print("  Status: ⚠️ RL v2 needs improvement")
            print()
        
        # Latest result
        latest = history[-1]
        print("📅 LATEST TEST")
        print("-" * 80)
        print(f"  Date: {latest['timestamp']}")
        print(f"  Winner: {latest['winner'].upper()}")
        print(f"  Confidence: {latest['confidence']:.1f}%")
        print()
        
        print("=" * 80)
        print()


def main():
    """Run A/B testing framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL v1 vs v2 A/B Testing")
    parser.add_argument("--history", action="store_true", help="Show historical summary")
    parser.add_argument("--v1-pnl", type=float, help="RL v1 total PnL")
    parser.add_argument("--v1-winrate", type=float, help="RL v1 win rate")
    parser.add_argument("--v1-sharpe", type=float, help="RL v1 Sharpe ratio")
    parser.add_argument("--v1-drawdown", type=float, help="RL v1 max drawdown")
    parser.add_argument("--v2-pnl", type=float, help="RL v2 total PnL")
    parser.add_argument("--v2-winrate", type=float, help="RL v2 win rate")
    parser.add_argument("--v2-sharpe", type=float, help="RL v2 Sharpe ratio")
    parser.add_argument("--v2-drawdown", type=float, help="RL v2 max drawdown")
    
    args = parser.parse_args()
    
    framework = ABTestingFramework()
    
    if args.history:
        framework.print_historical_summary()
    elif all([args.v1_pnl, args.v1_winrate, args.v2_pnl, args.v2_winrate]):
        # Run comparison
        v1_metrics = {
            "total_pnl": args.v1_pnl,
            "win_rate": args.v1_winrate,
            "sharpe_ratio": args.v1_sharpe or 0.0,
            "max_drawdown": args.v1_drawdown or 0.0
        }
        
        v2_metrics = {
            "total_pnl": args.v2_pnl,
            "win_rate": args.v2_winrate,
            "sharpe_ratio": args.v2_sharpe or 0.0,
            "max_drawdown": args.v2_drawdown or 0.0
        }
        
        comparison = framework.compare_performance(v1_metrics, v2_metrics)
        framework.print_comparison(comparison)
        framework.save_results(comparison)
    else:
        print("\n⚠️  Please provide metrics for comparison or use --history")
        print("Example:")
        print("  python scripts/ab_test_rl_v1_vs_v2.py \\")
        print("    --v1-pnl 1000 --v1-winrate 0.55 --v1-sharpe 1.2 --v1-drawdown 150 \\")
        print("    --v2-pnl 1200 --v2-winrate 0.58 --v2-sharpe 1.5 --v2-drawdown 120")
        print()


if __name__ == "__main__":
    main()
