"""
Pattern Analyzer - Discovers patterns between strategies and results
"""
import numpy as np
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import structlog

logger = structlog.get_logger()


class PatternAnalyzer:
    """Analyzes patterns between policy, regime, and PnL performance"""
    
    def __init__(self):
        self.regime_performance = defaultdict(lambda: {
            "count": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "best_policy": None,
            "confidence": 0.0
        })
        logger.info("PatternAnalyzer initialized")
    
    def analyze(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in memory data
        
        Args:
            memory: Dictionary containing all strategic memory
        
        Returns:
            Analysis results with regime performance breakdown
        """
        try:
            # Reset statistics
            self.regime_performance.clear()
            
            # Analyze meta-regime stream
            self._analyze_regime_stream(memory.get("meta_stream", []))
            
            # Analyze PnL stream
            self._analyze_pnl_stream(memory.get("pnl_stream", []))
            
            # Analyze recent trades
            self._analyze_trades(memory.get("recent_trades", []))
            
            # Calculate final metrics
            self._calculate_metrics()
            
            # Find best regime
            best_regime = self._find_best_regime()
            
            # Build analysis result
            result = {
                "regimes": dict(self.regime_performance),
                "best_regime": best_regime,
                "total_samples": sum(r["count"] for r in self.regime_performance.values()),
                "current_policy": memory.get("portfolio_policy"),
                "preferred_regime": memory.get("preferred_regime"),
                "timestamp": self._get_timestamp()
            }
            
            logger.info(
                "Pattern analysis complete",
                best_regime=best_regime,
                total_samples=result["total_samples"],
                regimes_detected=len(self.regime_performance)
            )
            
            return result
            
        except Exception as e:
            logger.error("Pattern analysis failed", error=str(e))
            return self._empty_analysis()
    
    def _analyze_regime_stream(self, stream: List[tuple]) -> None:
        """Analyze meta-regime stream entries"""
        for entry_id, data in stream:
            try:
                # Decode bytes to dict
                decoded = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()
                }
                
                regime = decoded.get("regime", "UNKNOWN")
                pnl = float(decoded.get("pnl", 0.0))
                confidence = float(decoded.get("confidence", 0.0))
                
                stats = self.regime_performance[regime]
                stats["count"] += 1
                stats["total_pnl"] += pnl
                
                if pnl > 0:
                    stats["wins"] += 1
                elif pnl < 0:
                    stats["losses"] += 1
                
                # Update confidence (weighted average)
                if stats["count"] > 1:
                    stats["confidence"] = (
                        stats["confidence"] * (stats["count"] - 1) + confidence
                    ) / stats["count"]
                else:
                    stats["confidence"] = confidence
                    
            except Exception as e:
                logger.warning("Failed to analyze regime entry", error=str(e))
    
    def _analyze_pnl_stream(self, stream: List[tuple]) -> None:
        """Analyze portfolio PnL stream"""
        for entry_id, data in stream:
            try:
                decoded = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()
                }
                
                regime = decoded.get("regime", "UNKNOWN")
                pnl = float(decoded.get("pnl", 0.0))
                
                if regime in self.regime_performance:
                    stats = self.regime_performance[regime]
                    stats["count"] += 1
                    stats["total_pnl"] += pnl
                    
            except Exception as e:
                logger.warning("Failed to analyze PnL entry", error=str(e))
    
    def _analyze_trades(self, stream: List[tuple]) -> None:
        """Analyze recent trade results"""
        for entry_id, data in stream:
            try:
                decoded = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()
                }
                
                regime = decoded.get("market_regime", "UNKNOWN")
                pnl = float(decoded.get("realized_pnl", 0.0))
                policy = decoded.get("policy", None)
                
                stats = self.regime_performance[regime]
                stats["count"] += 1
                stats["total_pnl"] += pnl
                
                if pnl > 0:
                    stats["wins"] += 1
                    # Track which policy works best
                    if policy and (stats["best_policy"] is None or pnl > 0):
                        stats["best_policy"] = policy
                elif pnl < 0:
                    stats["losses"] += 1
                    
            except Exception as e:
                logger.warning("Failed to analyze trade", error=str(e))
    
    def _calculate_metrics(self) -> None:
        """Calculate final metrics for each regime"""
        for regime, stats in self.regime_performance.items():
            if stats["count"] > 0:
                stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
                
                # Calculate win rate
                total_outcomes = stats["wins"] + stats["losses"]
                if total_outcomes > 0:
                    stats["win_rate"] = stats["wins"] / total_outcomes
                else:
                    stats["win_rate"] = 0.0
    
    def _find_best_regime(self) -> Tuple[str, float]:
        """Find the best performing regime"""
        if not self.regime_performance:
            return ("UNKNOWN", 0.0)
        
        best = max(
            self.regime_performance.items(),
            key=lambda x: x[1]["avg_pnl"]
        )
        
        return (best[0], best[1]["avg_pnl"])
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            "regimes": {},
            "best_regime": ("UNKNOWN", 0.0),
            "total_samples": 0,
            "current_policy": None,
            "preferred_regime": None,
            "timestamp": self._get_timestamp()
        }
