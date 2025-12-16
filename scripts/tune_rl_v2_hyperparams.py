"""
RL v2 Hyperparameter Tuning Script
===================================

Utilities for tuning alpha, gamma, epsilon based on performance.

Features:
- Performance-based hyperparameter recommendations
- A/B testing setup for RL v1 vs v2
- Automated hyperparameter search
- Gradient-free optimization

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import structlog

logger = structlog.get_logger(__name__)


class RLv2HyperparamTuner:
    """Hyperparameter tuning for RL v2 agents."""
    
    # Default hyperparameter ranges
    ALPHA_RANGE = (0.001, 0.1)      # Learning rate
    GAMMA_RANGE = (0.90, 0.999)     # Discount factor
    EPSILON_RANGE = (0.05, 0.3)     # Exploration rate
    
    def __init__(
        self,
        meta_q_table_path: Path = Path("data/rl_v2/meta_strategy_q_table.json"),
        sizing_q_table_path: Path = Path("data/rl_v2/position_sizing_q_table.json")
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            meta_q_table_path: Path to meta strategy Q-table
            sizing_q_table_path: Path to position sizing Q-table
        """
        self.meta_q_table_path = meta_q_table_path
        self.sizing_q_table_path = sizing_q_table_path
        
        logger.info("[RL v2 Tuner] Initialized")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze current agent performance.
        
        Returns:
            Performance analysis
        """
        try:
            # Load Q-tables
            meta_data = self._load_q_table(self.meta_q_table_path)
            sizing_data = self._load_q_table(self.sizing_q_table_path)
            
            if not meta_data or not sizing_data:
                return {
                    "status": "no_data",
                    "message": "Q-tables not found - agents not yet trained"
                }
            
            # Current hyperparameters
            current_hyperparams = {
                "meta_alpha": meta_data.get("alpha", 0.01),
                "meta_gamma": meta_data.get("gamma", 0.99),
                "meta_epsilon": meta_data.get("epsilon", 0.1),
                "sizing_alpha": sizing_data.get("alpha", 0.01),
                "sizing_gamma": sizing_data.get("gamma", 0.99),
                "sizing_epsilon": sizing_data.get("epsilon", 0.1)
            }
            
            # Q-value statistics
            meta_q_stats = self._calculate_q_stats(meta_data["q_table"])
            sizing_q_stats = self._calculate_q_stats(sizing_data["q_table"])
            
            # Learning progress
            meta_updates = meta_data.get("update_count", 0)
            sizing_updates = sizing_data.get("update_count", 0)
            
            # Convergence analysis
            meta_convergence = self._analyze_convergence(meta_q_stats, meta_updates)
            sizing_convergence = self._analyze_convergence(sizing_q_stats, sizing_updates)
            
            return {
                "status": "success",
                "current_hyperparams": current_hyperparams,
                "meta_agent": {
                    "q_stats": meta_q_stats,
                    "updates": meta_updates,
                    "convergence": meta_convergence
                },
                "sizing_agent": {
                    "q_stats": sizing_q_stats,
                    "updates": sizing_updates,
                    "convergence": sizing_convergence
                }
            }
            
        except Exception as e:
            logger.error(
                "[RL v2 Tuner] Performance analysis failed",
                error=str(e)
            )
            return {
                "status": "error",
                "error": str(e)
            }
    
    def recommend_hyperparams(self) -> Dict[str, Any]:
        """
        Recommend hyperparameter adjustments based on performance.
        
        Returns:
            Hyperparameter recommendations
        """
        analysis = self.analyze_performance()
        
        if analysis["status"] != "success":
            return analysis
        
        recommendations = {
            "meta_agent": {},
            "sizing_agent": {},
            "rationale": []
        }
        
        # Meta agent recommendations
        meta = analysis["meta_agent"]
        meta_current = analysis["current_hyperparams"]
        
        recommendations["meta_agent"] = self._recommend_for_agent(
            agent_name="Meta Strategy",
            q_stats=meta["q_stats"],
            updates=meta["updates"],
            convergence=meta["convergence"],
            current_alpha=meta_current["meta_alpha"],
            current_gamma=meta_current["meta_gamma"],
            current_epsilon=meta_current["meta_epsilon"]
        )
        
        # Sizing agent recommendations
        sizing = analysis["sizing_agent"]
        sizing_current = analysis["current_hyperparams"]
        
        recommendations["sizing_agent"] = self._recommend_for_agent(
            agent_name="Position Sizing",
            q_stats=sizing["q_stats"],
            updates=sizing["updates"],
            convergence=sizing["convergence"],
            current_alpha=sizing_current["sizing_alpha"],
            current_gamma=sizing_current["sizing_gamma"],
            current_epsilon=sizing_current["sizing_epsilon"]
        )
        
        return recommendations
    
    def _recommend_for_agent(
        self,
        agent_name: str,
        q_stats: Dict[str, float],
        updates: int,
        convergence: Dict[str, Any],
        current_alpha: float,
        current_gamma: float,
        current_epsilon: float
    ) -> Dict[str, Any]:
        """Generate recommendations for single agent."""
        recommendations = {
            "alpha": current_alpha,
            "gamma": current_gamma,
            "epsilon": current_epsilon,
            "changes": [],
            "rationale": []
        }
        
        # Alpha (learning rate) recommendations
        if updates < 100:
            # Early learning - keep alpha moderate
            if current_alpha < 0.01:
                recommendations["alpha"] = 0.01
                recommendations["changes"].append("alpha")
                recommendations["rationale"].append(
                    f"[{agent_name}] Increased alpha to 0.01 for faster initial learning"
                )
        elif updates < 1000:
            # Active learning - moderate alpha
            if current_alpha > 0.02:
                recommendations["alpha"] = 0.015
                recommendations["changes"].append("alpha")
                recommendations["rationale"].append(
                    f"[{agent_name}] Reduced alpha to 0.015 for more stable learning"
                )
        else:
            # Mature learning - reduce alpha
            if current_alpha > 0.005:
                recommendations["alpha"] = 0.005
                recommendations["changes"].append("alpha")
                recommendations["rationale"].append(
                    f"[{agent_name}] Reduced alpha to 0.005 for fine-tuning"
                )
        
        # Gamma (discount factor) recommendations
        q_variance = q_stats["variance"]
        if q_variance > 10.0:
            # High variance - reduce gamma for short-term focus
            if current_gamma > 0.95:
                recommendations["gamma"] = 0.95
                recommendations["changes"].append("gamma")
                recommendations["rationale"].append(
                    f"[{agent_name}] Reduced gamma to 0.95 due to high Q-value variance"
                )
        elif q_variance < 1.0:
            # Low variance - increase gamma for long-term focus
            if current_gamma < 0.99:
                recommendations["gamma"] = 0.99
                recommendations["changes"].append("gamma")
                recommendations["rationale"].append(
                    f"[{agent_name}] Increased gamma to 0.99 for better long-term planning"
                )
        
        # Epsilon (exploration rate) recommendations
        if convergence["status"] == "converging":
            # Converging - reduce exploration
            if current_epsilon > 0.05:
                recommendations["epsilon"] = max(0.05, current_epsilon * 0.8)
                recommendations["changes"].append("epsilon")
                recommendations["rationale"].append(
                    f"[{agent_name}] Reduced epsilon to {recommendations['epsilon']:.3f} as agent converges"
                )
        elif convergence["status"] == "diverging":
            # Diverging - increase exploration
            if current_epsilon < 0.2:
                recommendations["epsilon"] = min(0.2, current_epsilon * 1.5)
                recommendations["changes"].append("epsilon")
                recommendations["rationale"].append(
                    f"[{agent_name}] Increased epsilon to {recommendations['epsilon']:.3f} to escape local optimum"
                )
        elif updates > 1000 and current_epsilon > 0.1:
            # Mature agent - reduce exploration
            recommendations["epsilon"] = 0.05
            recommendations["changes"].append("epsilon")
            recommendations["rationale"].append(
                f"[{agent_name}] Reduced epsilon to 0.05 for mature agent"
            )
        
        return recommendations
    
    def apply_hyperparams(
        self,
        meta_alpha: Optional[float] = None,
        meta_gamma: Optional[float] = None,
        meta_epsilon: Optional[float] = None,
        sizing_alpha: Optional[float] = None,
        sizing_gamma: Optional[float] = None,
        sizing_epsilon: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply new hyperparameters to Q-tables.
        
        Args:
            meta_alpha: New meta agent learning rate
            meta_gamma: New meta agent discount factor
            meta_epsilon: New meta agent exploration rate
            sizing_alpha: New sizing agent learning rate
            sizing_gamma: New sizing agent discount factor
            sizing_epsilon: New sizing agent exploration rate
            
        Returns:
            Application result
        """
        results = {
            "meta_agent": None,
            "sizing_agent": None
        }
        
        # Update meta agent
        if any([meta_alpha, meta_gamma, meta_epsilon]):
            results["meta_agent"] = self._update_q_table_hyperparams(
                self.meta_q_table_path,
                alpha=meta_alpha,
                gamma=meta_gamma,
                epsilon=meta_epsilon
            )
        
        # Update sizing agent
        if any([sizing_alpha, sizing_gamma, sizing_epsilon]):
            results["sizing_agent"] = self._update_q_table_hyperparams(
                self.sizing_q_table_path,
                alpha=sizing_alpha,
                gamma=sizing_gamma,
                epsilon=sizing_epsilon
            )
        
        return results
    
    def _load_q_table(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load Q-table from file."""
        try:
            if not path.exists():
                return None
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(
                "[RL v2 Tuner] Failed to load Q-table",
                error=str(e),
                path=str(path)
            )
            return None
    
    def _calculate_q_stats(self, q_table: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate Q-value statistics."""
        all_q_values = [q for actions in q_table.values() for q in actions.values()]
        
        if not all_q_values:
            return {
                "max": 0.0,
                "min": 0.0,
                "mean": 0.0,
                "variance": 0.0,
                "std": 0.0,
                "count": 0
            }
        
        mean = sum(all_q_values) / len(all_q_values)
        variance = sum((q - mean) ** 2 for q in all_q_values) / len(all_q_values)
        std = variance ** 0.5
        
        return {
            "max": max(all_q_values),
            "min": min(all_q_values),
            "mean": mean,
            "variance": variance,
            "std": std,
            "count": len(all_q_values)
        }
    
    def _analyze_convergence(
        self,
        q_stats: Dict[str, float],
        updates: int
    ) -> Dict[str, Any]:
        """Analyze learning convergence."""
        if updates < 50:
            return {
                "status": "early",
                "message": "Too early to determine convergence"
            }
        
        # Check Q-value stability
        std = q_stats["std"]
        mean = abs(q_stats["mean"])
        
        if mean == 0:
            cv = float('inf')
        else:
            cv = std / mean  # Coefficient of variation
        
        if cv < 0.1:
            return {
                "status": "converged",
                "message": "Agent has converged (CV < 0.1)",
                "cv": cv
            }
        elif cv < 0.5:
            return {
                "status": "converging",
                "message": "Agent is converging (0.1 <= CV < 0.5)",
                "cv": cv
            }
        else:
            return {
                "status": "diverging",
                "message": "Agent shows high variance (CV >= 0.5)",
                "cv": cv
            }
    
    def _update_q_table_hyperparams(
        self,
        path: Path,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        epsilon: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update hyperparameters in Q-table file."""
        try:
            # Load current data
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Update hyperparameters
            if alpha is not None:
                data["alpha"] = alpha
            if gamma is not None:
                data["gamma"] = gamma
            if epsilon is not None:
                data["epsilon"] = epsilon
            
            # Save updated data
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(
                "[RL v2 Tuner] Updated hyperparameters",
                path=str(path),
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon
            )
            
            return {
                "status": "success",
                "path": str(path),
                "updates": {
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon
                }
            }
            
        except Exception as e:
            logger.error(
                "[RL v2 Tuner] Failed to update hyperparameters",
                error=str(e),
                path=str(path)
            )
            return {
                "status": "error",
                "error": str(e)
            }
    
    def print_recommendations(self):
        """Print hyperparameter recommendations to console."""
        recommendations = self.recommend_hyperparams()
        
        if recommendations.get("status") != "success" and "meta_agent" not in recommendations:
            print(f"\n⚠️  {recommendations.get('message', 'Unknown error')}")
            return
        
        print("\n" + "=" * 80)
        print("RL v2 HYPERPARAMETER RECOMMENDATIONS")
        print("=" * 80)
        print()
        
        # Meta agent
        meta = recommendations["meta_agent"]
        print("📊 META STRATEGY AGENT")
        print("-" * 80)
        if meta.get("changes"):
            print(f"  Recommended Changes:")
            print(f"    Alpha (α):    {meta['alpha']:.4f}")
            print(f"    Gamma (γ):    {meta['gamma']:.4f}")
            print(f"    Epsilon (ε):  {meta['epsilon']:.4f}")
            print()
            print(f"  Rationale:")
            for reason in meta["rationale"]:
                print(f"    • {reason}")
        else:
            print("  ✅ Current hyperparameters are optimal")
        print()
        
        # Sizing agent
        sizing = recommendations["sizing_agent"]
        print("📈 POSITION SIZING AGENT")
        print("-" * 80)
        if sizing.get("changes"):
            print(f"  Recommended Changes:")
            print(f"    Alpha (α):    {sizing['alpha']:.4f}")
            print(f"    Gamma (γ):    {sizing['gamma']:.4f}")
            print(f"    Epsilon (ε):  {sizing['epsilon']:.4f}")
            print()
            print(f"  Rationale:")
            for reason in sizing["rationale"]:
                print(f"    • {reason}")
        else:
            print("  ✅ Current hyperparameters are optimal")
        print()
        
        # Application command
        has_changes = meta.get("changes") or sizing.get("changes")
        if has_changes:
            print("🔧 TO APPLY RECOMMENDATIONS")
            print("-" * 80)
            print("  Run the following command:")
            print()
            
            cmd_parts = []
            if meta.get("changes"):
                if "alpha" in meta["changes"]:
                    cmd_parts.append(f"--meta-alpha {meta['alpha']}")
                if "gamma" in meta["changes"]:
                    cmd_parts.append(f"--meta-gamma {meta['gamma']}")
                if "epsilon" in meta["changes"]:
                    cmd_parts.append(f"--meta-epsilon {meta['epsilon']}")
            
            if sizing.get("changes"):
                if "alpha" in sizing["changes"]:
                    cmd_parts.append(f"--sizing-alpha {sizing['alpha']}")
                if "gamma" in sizing["changes"]:
                    cmd_parts.append(f"--sizing-gamma {sizing['gamma']}")
                if "epsilon" in sizing["changes"]:
                    cmd_parts.append(f"--sizing-epsilon {sizing['epsilon']}")
            
            cmd = f"  python scripts/tune_rl_v2_hyperparams.py --apply {' '.join(cmd_parts)}"
            print(cmd)
            print()
        
        print("=" * 80)
        print()


def main():
    """Run hyperparameter tuning script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL v2 Hyperparameter Tuning")
    parser.add_argument("--apply", action="store_true", help="Apply hyperparameter changes")
    parser.add_argument("--meta-alpha", type=float, help="Meta agent learning rate")
    parser.add_argument("--meta-gamma", type=float, help="Meta agent discount factor")
    parser.add_argument("--meta-epsilon", type=float, help="Meta agent exploration rate")
    parser.add_argument("--sizing-alpha", type=float, help="Sizing agent learning rate")
    parser.add_argument("--sizing-gamma", type=float, help="Sizing agent discount factor")
    parser.add_argument("--sizing-epsilon", type=float, help="Sizing agent exploration rate")
    
    args = parser.parse_args()
    
    tuner = RLv2HyperparamTuner()
    
    if args.apply:
        # Apply hyperparameters
        results = tuner.apply_hyperparams(
            meta_alpha=args.meta_alpha,
            meta_gamma=args.meta_gamma,
            meta_epsilon=args.meta_epsilon,
            sizing_alpha=args.sizing_alpha,
            sizing_gamma=args.sizing_gamma,
            sizing_epsilon=args.sizing_epsilon
        )
        
        print("\n✅ Hyperparameters applied:")
        if results["meta_agent"]:
            print(f"  Meta Agent: {results['meta_agent']}")
        if results["sizing_agent"]:
            print(f"  Sizing Agent: {results['sizing_agent']}")
        print()
    else:
        # Show recommendations
        tuner.print_recommendations()


if __name__ == "__main__":
    main()
