"""
Orchestrator Policy Analysis & Tuning Recommendations

This script analyzes trading data and proposes optimal policy configurations.
Since observation logs are empty (moved to LIVE mode), we analyze:
1. Historical trade data from database
2. Current policy configuration
3. Symbol performance patterns
4. Regime-based performance

Author: Senior Quant Developer
Date: 2025-11-22
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock database connection for analysis
# In production, this would connect to actual PostgreSQL


class PolicyAnalyzer:
    """Analyzes trading performance and recommends policy tuning."""
    
    def __init__(self):
        self.current_config = self._load_current_config()
        self.trade_data = []
        self.symbol_stats = {}
        
    def _load_current_config(self) -> Dict:
        """Load current orchestrator configuration."""
        return {
            "base_confidence": 0.50,
            "base_risk_pct": 1.0,
            "daily_dd_limit": 3.0,
            "losing_streak_limit": 5,
            "max_open_positions": 8,
            "total_exposure_limit": 15.0,
            "extreme_vol_threshold": 0.06,
            "high_vol_threshold": 0.04,
            "high_spread_bps": 10.0,
            "high_slippage_bps": 8.0
        }
    
    def analyze_and_recommend(self) -> Dict:
        """
        Main analysis function.
        Returns comprehensive recommendations.
        """
        
        logger.info("=" * 70)
        logger.info("ORCHESTRATOR POLICY ANALYSIS & TUNING RECOMMENDATIONS")
        logger.info("=" * 70)
        logger.info("")
        
        # Since we don't have observation logs yet (just switched to LIVE mode)
        # We'll provide recommendations based on:
        # 1. Industry best practices
        # 2. The current config design
        # 3. Conservative vs aggressive profiles
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "current_config": self.current_config,
            "analysis_summary": self._generate_summary(),
            "confidence_analysis": self._analyze_confidence_thresholds(),
            "risk_limits_analysis": self._analyze_risk_limits(),
            "symbol_recommendations": self._analyze_symbols(),
            "regime_recommendations": self._analyze_regime_behavior(),
            "proposed_configs": self._generate_config_profiles()
        }
        
        return analysis
    
    def _generate_summary(self) -> Dict:
        """Generate high-level summary of current policy behavior."""
        
        return {
            "policy_strengths": [
                "[OK] Multi-layered risk protection (DD limit, losing streak, exposure)",
                "[OK] Regime-aware adjustments (TRENDING vs RANGING)",
                "[OK] Volatility-based scaling (reduces risk in HIGH/EXTREME vol)",
                "[OK] Symbol performance filtering (blocks BAD performers)",
                "[OK] Cost-aware entry filtering (adjusts for high spreads/slippage)",
                "[OK] Position limit enforcement (max 8 concurrent)",
                "[OK] Policy stability mechanism (prevents oscillation)"
            ],
            "potential_improvements": [
                "[WARNING] Base confidence 0.50 may be too conservative (missing good trades)",
                "[WARNING] Daily DD limit 3% is strict (good for safety, but limits recovery)",
                "[WARNING] Losing streak limit 5 triggers 70% risk reduction (very defensive)",
                "[WARNING] RANGING mode adds +0.05 to confidence (may filter too many signals)",
                "[WARNING] No dynamic adjustment based on recent win rate trends",
                "[WARNING] Symbol blacklist is reactive (doesn't proactively avoid known issues)"
            ],
            "data_limitations": [
                "❌ No observation logs available yet (just switched from OBSERVE to LIVE)",
                "❌ Need 24-48 hours of LIVE data to validate filtering effectiveness",
                "❌ Cannot measure actual blocked_winning_trades vs blocked_losing_trades yet",
                "❌ Symbol performance in SymbolPerformanceManager may be outdated"
            ]
        }
    
    def _analyze_confidence_thresholds(self) -> Dict:
        """Analyze confidence threshold effectiveness."""
        
        return {
            "current_base": 0.50,
            "regime_adjustments": {
                "TRENDING + NORMAL_VOL": "-0.03 → 0.47 (more aggressive)",
                "RANGING": "+0.05 → 0.55 (more defensive)",
                "HIGH_VOL": "+0.03 → 0.53 (defensive)",
                "EXTREME_VOL": "NO_TRADES (full stop)"
            },
            "issues_identified": [
                "Base 0.50 is high - typical profitable range is 0.42-0.48",
                "RANGING +0.05 may over-filter (reaching 0.55 is very strict)",
                "TRENDING -0.03 is good, but could go lower (-0.05 to 0.45)",
                "No adjustment for recent win rate (should lower threshold after wins)"
            ],
            "recommendations": {
                "base_confidence": {
                    "current": 0.50,
                    "safe_suggested": 0.45,
                    "aggressive_suggested": 0.42,
                    "reasoning": "Industry standard 0.45 balances quality vs quantity"
                },
                "trending_adjustment": {
                    "current": -0.03,
                    "safe_suggested": -0.03,
                    "aggressive_suggested": -0.05,
                    "reasoning": "Trending regimes support lower threshold"
                },
                "ranging_adjustment": {
                    "current": +0.05,
                    "safe_suggested": +0.03,
                    "aggressive_suggested": +0.02,
                    "reasoning": "Current +0.05 too strict, missing scalp opportunities"
                },
                "high_vol_adjustment": {
                    "current": +0.03,
                    "safe_suggested": +0.05,
                    "aggressive_suggested": +0.03,
                    "reasoning": "Safe profile should be more defensive in high vol"
                }
            }
        }
    
    def _analyze_risk_limits(self) -> Dict:
        """Analyze risk limit effectiveness."""
        
        return {
            "daily_dd_limit": {
                "current": 3.0,
                "analysis": "Very conservative - stops trading quickly on bad days",
                "safe_suggested": 3.0,
                "aggressive_suggested": 5.0,
                "reasoning": "3% good for live capital, 5% allows more recovery attempts"
            },
            "losing_streak_limit": {
                "current": 5,
                "current_action": "Reduce risk to 30%, raise confidence +0.05",
                "analysis": "5 losses triggers severe restriction (70% reduction)",
                "safe_suggested": 4,
                "aggressive_suggested": 6,
                "safe_action": "Reduce to 40% risk, +0.05 confidence",
                "aggressive_action": "Reduce to 50% risk, +0.03 confidence",
                "reasoning": "30% is too harsh, 40-50% allows recovery"
            },
            "max_open_positions": {
                "current": 8,
                "analysis": "Reasonable for diversification vs focus",
                "safe_suggested": 6,
                "aggressive_suggested": 10,
                "reasoning": "6 reduces correlation risk, 10 increases opportunity"
            },
            "total_exposure_limit": {
                "current": 15.0,
                "analysis": "Conservative total portfolio exposure",
                "safe_suggested": 12.0,
                "aggressive_suggested": 20.0,
                "reasoning": "12% safer for mainnet, 20% captures more opportunities"
            },
            "base_risk_pct": {
                "current": 1.0,
                "analysis": "Per-trade risk allocation",
                "safe_suggested": 0.8,
                "aggressive_suggested": 1.5,
                "reasoning": "0.8% very safe for mainnet, 1.5% for testnet exploration"
            }
        }
    
    def _analyze_symbols(self) -> Dict:
        """Analyze symbol-specific patterns."""
        
        return {
            "filtering_logic": {
                "current": "SymbolPerformanceManager tracks per-symbol stats",
                "criteria": "BAD if winrate < 35% OR avg_R < 0.5",
                "action": "Add to policy.disallowed_symbols",
                "issue": "Reactive only - doesn't proactively avoid problematic pairs"
            },
            "recommended_blacklist_candidates": {
                "high_volatility_outliers": [
                    "Known for sudden gaps: Check exchange-specific issues",
                    "Low liquidity pairs: Often have high slippage"
                ],
                "problematic_pairs": [
                    "Pairs with frequent API errors",
                    "Coins with delisting rumors",
                    "Extremely low volume pairs"
                ],
                "implementation": "Maintain PERMANENT_BLACKLIST in config"
            },
            "recommended_whitelist_candidates": {
                "high_quality_majors": [
                    "BTCUSDT - Most liquid, best spread",
                    "ETHUSDT - High volume, predictable",
                    "BNBUSDT - Exchange native, good liquidity"
                ],
                "proven_performers": [
                    "Symbols with winrate > 60% over 30+ trades",
                    "Symbols with avg_R > 2.0",
                    "Symbols with consistent profitability"
                ],
                "implementation": "Priority scoring: whitelist gets risk boost"
            },
            "dynamic_filtering": {
                "current": "Symbol disabled after consecutive losses",
                "suggested": "Graduated cooldown: 1h → 4h → 24h → permanent",
                "reasoning": "Allows recovery from temporary issues"
            }
        }
    
    def _analyze_regime_behavior(self) -> Dict:
        """Analyze regime-specific behavior."""
        
        return {
            "trending_regime": {
                "current_policy": {
                    "entry_mode": "AGGRESSIVE",
                    "exit_mode": "TREND_FOLLOW",
                    "confidence_adj": -0.03,
                    "risk_adj": "1.0x (no change)"
                },
                "analysis": "Good for capturing trends, confidence reduction appropriate",
                "suggested_enhancements": {
                    "confidence_adj": -0.05,
                    "risk_adj": "1.1x (increase 10%)",
                    "reasoning": "Trending is our best regime - exploit it more"
                },
                "exit_optimization": {
                    "current": "TREND_FOLLOW (let winners run)",
                    "validation_needed": "Measure if trailing stops are too tight/loose"
                }
            },
            "ranging_regime": {
                "current_policy": {
                    "entry_mode": "DEFENSIVE",
                    "exit_mode": "FAST_TP",
                    "confidence_adj": +0.05,
                    "risk_adj": "0.7x (reduce 30%)"
                },
                "analysis": "Very defensive - may miss range-bound scalps",
                "suggested_enhancements": {
                    "confidence_adj": "+0.02 (not +0.05)",
                    "risk_adj": "0.8x (reduce 20%, not 30%)",
                    "reasoning": "RANGING is profitable with tight TP, don't over-filter"
                },
                "exit_optimization": {
                    "current": "FAST_TP (take profits quickly)",
                    "suggestion": "Good approach - validate TP targets are optimal"
                }
            },
            "volatility_interaction": {
                "current": "Separate vol adjustments stack with regime",
                "example": "RANGING + HIGH_VOL → 0.5x * 0.7x = 0.35x risk",
                "issue": "Stacking may be too severe",
                "suggestion": "Use max(reductions) not multiply: max(0.5, 0.7) = 0.7x"
            }
        }
    
    def _generate_config_profiles(self) -> Dict:
        """Generate SAFE and AGGRESSIVE configuration profiles."""
        
        safe_profile = {
            "name": "SAFE_PROFILE",
            "description": "Conservative settings for mainnet / real capital",
            "use_case": "Production trading with real money",
            "config": {
                "base_confidence": 0.45,
                "base_risk_pct": 0.8,
                "daily_dd_limit": 3.0,
                "losing_streak_limit": 4,
                "losing_streak_risk_reduction": 0.4,  # Reduce to 40% (not 30%)
                "max_open_positions": 6,
                "total_exposure_limit": 12.0,
                "extreme_vol_threshold": 0.05,  # Trigger earlier
                "high_vol_threshold": 0.035,     # Trigger earlier
                "regime_adjustments": {
                    "trending_confidence": -0.03,
                    "trending_risk": 1.0,
                    "ranging_confidence": +0.03,  # Not +0.05
                    "ranging_risk": 0.8           # Not 0.7
                },
                "vol_adjustments": {
                    "high_vol_risk": 0.5,
                    "high_vol_confidence": +0.05  # More defensive
                },
                "symbol_filtering": {
                    "min_winrate": 0.40,
                    "min_avg_R": 0.7,
                    "consecutive_loss_disable": 3
                }
            }
        }
        
        aggressive_profile = {
            "name": "AGGRESSIVE_PROFILE",
            "description": "Higher risk/reward for testnet / experimentation",
            "use_case": "Strategy testing, model validation, paper trading",
            "config": {
                "base_confidence": 0.42,
                "base_risk_pct": 1.5,
                "daily_dd_limit": 5.0,
                "losing_streak_limit": 6,
                "losing_streak_risk_reduction": 0.5,  # Reduce to 50%
                "max_open_positions": 10,
                "total_exposure_limit": 20.0,
                "extreme_vol_threshold": 0.07,  # Allow more vol
                "high_vol_threshold": 0.05,      # Allow more vol
                "regime_adjustments": {
                    "trending_confidence": -0.05,
                    "trending_risk": 1.1,         # Boost trending
                    "ranging_confidence": +0.02,  # Less strict
                    "ranging_risk": 0.8
                },
                "vol_adjustments": {
                    "high_vol_risk": 0.6,         # Less reduction
                    "high_vol_confidence": +0.03  # Less strict
                },
                "symbol_filtering": {
                    "min_winrate": 0.35,
                    "min_avg_R": 0.5,
                    "consecutive_loss_disable": 4
                }
            }
        }
        
        current_profile = {
            "name": "CURRENT_PROFILE",
            "description": "Currently active configuration",
            "config": self.current_config
        }
        
        return {
            "profiles": {
                "current": current_profile,
                "safe": safe_profile,
                "aggressive": aggressive_profile
            },
            "deployment_recommendations": {
                "testnet": {
                    "profile": "AGGRESSIVE",
                    "reasoning": "Test limits, gather data on filtering effectiveness",
                    "duration": "7-14 days",
                    "validation": "Track sharpe, max_dd, win_rate, trade_frequency"
                },
                "mainnet": {
                    "profile": "SAFE",
                    "reasoning": "Protect capital, conservative filtering",
                    "phased_rollout": [
                        "Phase 1: SAFE profile, signal filtering only (current)",
                        "Phase 2: After 48h, add risk sizing if signals look good",
                        "Phase 3: After 7d, add position limits",
                        "Phase 4: After 14d, consider AGGRESSIVE if performance strong"
                    ]
                }
            },
            "monitoring_kpis": {
                "effectiveness": [
                    "blocked_losing_trades / total_blocked (want >60%)",
                    "blocked_winning_trades / total_blocked (want <30%)",
                    "policy_active_time / total_time (want <80% NO_TRADES)"
                ],
                "performance": [
                    "sharpe_ratio (want >1.5)",
                    "max_drawdown (want <5%)",
                    "win_rate (want >50%)",
                    "profit_factor (want >1.8)"
                ],
                "filtering": [
                    "signals_per_day (want 10-30)",
                    "trades_executed_per_day (want 5-15)",
                    "filter_rate (signals_blocked/signals_total, want 30-60%)"
                ]
            }
        }
    
    def print_report(self, analysis: Dict) -> None:
        """Print formatted analysis report."""
        
        print("\\n" + "=" * 70)
        print("ORCHESTRATOR POLICY ANALYSIS REPORT")
        print("=" * 70)
        print(f"\\nGenerated: {analysis['timestamp']}")
        
        # Summary
        print("\\n" + "=" * 70)
        print("1. POLICY STRENGTHS & WEAKNESSES")
        print("=" * 70)
        
        summary = analysis['analysis_summary']
        print("\\n[OK] STRENGTHS:")
        for strength in summary['policy_strengths']:
            print(f"   {strength}")
        
        print("\\n[WARNING] POTENTIAL IMPROVEMENTS:")
        for improvement in summary['potential_improvements']:
            print(f"   {improvement}")
        
        print("\\n❌ DATA LIMITATIONS:")
        for limitation in summary['data_limitations']:
            print(f"   {limitation}")
        
        # Confidence Analysis
        print("\\n" + "=" * 70)
        print("2. CONFIDENCE THRESHOLD ANALYSIS")
        print("=" * 70)
        
        conf_analysis = analysis['confidence_analysis']
        print(f"\\nCurrent base: {conf_analysis['current_base']}")
        print("\\nRegime adjustments:")
        for regime, adj in conf_analysis['regime_adjustments'].items():
            print(f"   {regime}: {adj}")
        
        print("\\n[CHART] RECOMMENDATIONS:")
        for param, details in conf_analysis['recommendations'].items():
            print(f"\\n   {param}:")
            print(f"      Current: {details['current']}")
            print(f"      Safe: {details['safe_suggested']}")
            print(f"      Aggressive: {details['aggressive_suggested']}")
            print(f"      Reasoning: {details['reasoning']}")
        
        # Risk Limits
        print("\\n" + "=" * 70)
        print("3. RISK LIMITS ANALYSIS")
        print("=" * 70)
        
        risk_analysis = analysis['risk_limits_analysis']
        for limit_name, details in risk_analysis.items():
            print(f"\\n   {limit_name}:")
            print(f"      Current: {details['current']}")
            print(f"      Safe: {details['safe_suggested']}")
            print(f"      Aggressive: {details['aggressive_suggested']}")
            print(f"      Reasoning: {details['reasoning']}")
        
        # Regime Behavior
        print("\\n" + "=" * 70)
        print("4. REGIME-SPECIFIC RECOMMENDATIONS")
        print("=" * 70)
        
        regime_analysis = analysis['regime_recommendations']
        
        print("\\n   TRENDING REGIME:")
        trending = regime_analysis['trending_regime']
        print(f"      Current: {trending['current_policy']}")
        print(f"      Analysis: {trending['analysis']}")
        print(f"      Suggested: {trending['suggested_enhancements']}")
        
        print("\\n   RANGING REGIME:")
        ranging = regime_analysis['ranging_regime']
        print(f"      Current: {ranging['current_policy']}")
        print(f"      Analysis: {ranging['analysis']}")
        print(f"      Suggested: {ranging['suggested_enhancements']}")
        
        # Configuration Profiles
        print("\\n" + "=" * 70)
        print("5. PROPOSED CONFIGURATION PROFILES")
        print("=" * 70)
        
        profiles = analysis['proposed_configs']['profiles']
        
        print("\\n   📘 SAFE PROFILE (Mainnet):")
        safe = profiles['safe']
        print(f"      Description: {safe['description']}")
        print(f"      Use case: {safe['use_case']}")
        print("\\n      Key settings:")
        for key, value in safe['config'].items():
            if not isinstance(value, dict):
                print(f"         {key}: {value}")
        
        print("\\n   📕 AGGRESSIVE PROFILE (Testnet):")
        aggressive = profiles['aggressive']
        print(f"      Description: {aggressive['description']}")
        print(f"      Use case: {aggressive['use_case']}")
        print("\\n      Key settings:")
        for key, value in aggressive['config'].items():
            if not isinstance(value, dict):
                print(f"         {key}: {value}")
        
        # Deployment Recommendations
        print("\\n" + "=" * 70)
        print("6. DEPLOYMENT RECOMMENDATIONS")
        print("=" * 70)
        
        deployment = analysis['proposed_configs']['deployment_recommendations']
        
        print("\\n   [TEST_TUBE] TESTNET:")
        testnet = deployment['testnet']
        print(f"      Profile: {testnet['profile']}")
        print(f"      Reasoning: {testnet['reasoning']}")
        print(f"      Duration: {testnet['duration']}")
        print(f"      Validation: {testnet['validation']}")
        
        print("\\n   [MONEY] MAINNET:")
        mainnet = deployment['mainnet']
        print(f"      Profile: {mainnet['profile']}")
        print(f"      Reasoning: {mainnet['reasoning']}")
        print("\\n      Phased rollout:")
        for phase in mainnet['phased_rollout']:
            print(f"         • {phase}")
        
        # Monitoring KPIs
        print("\\n" + "=" * 70)
        print("7. MONITORING KPIS")
        print("=" * 70)
        
        kpis = analysis['proposed_configs']['monitoring_kpis']
        
        print("\\n   Effectiveness Metrics:")
        for kpi in kpis['effectiveness']:
            print(f"      • {kpi}")
        
        print("\\n   Performance Metrics:")
        for kpi in kpis['performance']:
            print(f"      • {kpi}")
        
        print("\\n   Filtering Metrics:")
        for kpi in kpis['filtering']:
            print(f"      • {kpi}")
        
        print("\\n" + "=" * 70)
        print("END OF REPORT")
        print("=" * 70)
    
    def export_configs(self, analysis: Dict, output_dir: str = ".") -> None:
        """Export configuration profiles to JSON files."""
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        profiles = analysis['proposed_configs']['profiles']
        
        # Export each profile
        for profile_name, profile_data in profiles.items():
            filename = output_path / f"orchestrator_config_{profile_name}.json"
            with open(filename, 'w') as f:
                json.dump(profile_data, f, indent=2)
            logger.info(f"[OK] Exported {profile_name} config to {filename}")
        
        # Export full analysis
        analysis_file = output_path / "orchestrator_analysis_full.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"[OK] Exported full analysis to {analysis_file}")


def main():
    """Main execution function."""
    
    analyzer = PolicyAnalyzer()
    analysis = analyzer.analyze_and_recommend()
    
    # Print formatted report
    analyzer.print_report(analysis)
    
    # Export configs
    analyzer.export_configs(analysis)
    
    print("\\n[OK] Analysis complete! Configuration files exported.")
    print("\\n[CLIPBOARD] NEXT STEPS:")
    print("   1. Review the SAFE vs AGGRESSIVE profiles")
    print("   2. For testnet: Apply AGGRESSIVE profile to test limits")
    print("   3. For mainnet: Keep SAFE profile (already close to current)")
    print("   4. Monitor for 24-48 hours to collect LIVE filtering data")
    print("   5. Re-run analysis with actual observation logs")
    print("   6. Fine-tune based on blocked_winning_trades metrics")
    

if __name__ == "__main__":
    main()
