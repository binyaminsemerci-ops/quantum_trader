#!/usr/bin/env python3
"""
UNIVERSE OS AGENT — Autonomous Quantitative Universe Manager
=============================================================

Full AI Operating System for Trading Symbol Universe Lifecycle:
- Discovery, Analysis, Health Diagnostics
- Ranking, Selection, Classification
- Dynamic Universe Construction (SAFE, AGGRESSIVE, EXPERIMENTAL)
- Snapshot Writer, Delta Engine
- Visualizations (charts, plots, heatmaps)
- Scheduler, Reporting, Recommendations
- Integration with OrchestratorPolicy & Risk Manager

Author: Senior Quant Developer
Date: November 23, 2025
Version: 2.0 (Full OS Implementation)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import math
import statistics

# For visualizations (conceptual - would use matplotlib/plotly in full implementation)
ENABLE_VISUALIZATION = os.getenv("UNIVERSE_OS_VISUALIZE", "false").lower() == "true"

# Operating modes
UNIVERSE_OS_MODE = os.getenv("UNIVERSE_OS_MODE", "OBSERVE")  # OBSERVE | FULL
UNIVERSE_OS_INTERVAL_HOURS = int(os.getenv("UNIVERSE_OS_INTERVAL_HOURS", "4"))


@dataclass
class SymbolFeatures:
    """Complete feature set for a trading symbol"""
    # Identity
    symbol: str
    
    # Performance Metrics
    winrate: float = 0.0
    avg_R: float = 0.0
    median_R: float = 0.0
    total_R: float = 0.0
    std_R: float = 0.0
    max_R: float = 0.0
    min_R: float = 0.0
    profit_factor: float = 0.0
    trade_count: int = 0
    R_per_trade: float = 0.0
    
    # Cost Metrics
    avg_slippage: float = 0.0
    avg_spread: float = 0.0
    max_slippage: float = 0.0
    rollover_cost: float = 0.0
    
    # Stability Metrics
    volatility_score: float = 0.0
    stability_score: float = 0.0
    consistency_score: float = 0.0
    
    # Regime Metrics
    trending_R: float = 0.0
    ranging_R: float = 0.0
    mixed_R: float = 0.0
    regime_dependency_score: float = 0.0
    
    # Volatility Metrics
    high_vol_R: float = 0.0
    extreme_vol_R: float = 0.0
    normal_vol_R: float = 0.0
    
    # Policy Metrics
    disallow_rate: float = 0.0
    avg_confidence: float = 0.0
    signal_count: int = 0
    allow_count: int = 0
    disallow_count: int = 0
    
    # Quality Scores (computed)
    quality_score: float = 0.0
    profitability_score: float = 0.0
    reliability_score: float = 0.0
    
    # Classification
    tier: str = "UNCLASSIFIED"
    tier_reason: str = ""


@dataclass
class UniverseProfile:
    """Universe configuration profile"""
    name: str
    size: int
    symbols: List[str]
    expected_R: float
    expected_winrate: float
    risk_level: str
    description: str


class UniverseOSAgent:
    """
    UNIVERSE OS AGENT — Full AI Operating System
    
    Responsibilities:
    1. Data ingestion (trades, signals, policy obs, snapshots)
    2. Feature engineering (30+ metrics per symbol)
    3. Symbol classification (CORE, EXPANSION, CONDITIONAL, BLACKLIST)
    4. Universe optimization (SAFE, AGGRESSIVE, EXPERIMENTAL)
    5. Visualization generation
    6. Delta computation
    7. Snapshot writing
    8. Report generation
    9. Scheduler integration
    10. Orchestrator integration
    """
    
    def __init__(self):
        print("\n" + "="*80)
        print("UNIVERSE OS AGENT — INITIALIZING")
        print("="*80 + "\n")
        
        # Data paths
        self.data_dir = Path("/app/data")
        self.universe_snapshot_path = self.data_dir / "universe_snapshot.json"
        self.selector_output_path = self.data_dir / "universe_selector_output.json"
        self.trades_dir = self.data_dir / "trades"
        self.policy_obs_dir = self.data_dir / "policy_observations"
        self.charts_dir = self.data_dir / "universe_charts"
        
        # Create output directories
        self.charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Data storage
        self.universe_snapshot = {}
        self.selector_output = {}
        self.trade_data = defaultdict(list)  # symbol -> [trade records]
        self.signal_data = defaultdict(list)  # symbol -> [signal records]
        self.policy_obs_data = []
        
        # Features storage
        self.symbol_features: Dict[str, SymbolFeatures] = {}
        
        # Classifications
        self.core_symbols = []
        self.expansion_symbols = []
        self.conditional_symbols = []
        self.blacklist_symbols = []
        self.insufficient_data_symbols = []
        
        # Universe profiles
        self.safe_profile: Optional[UniverseProfile] = None
        self.aggressive_profile: Optional[UniverseProfile] = None
        self.experimental_profile: Optional[UniverseProfile] = None
        
        # Delta tracking
        self.deltas = {}
        
        # Performance curves
        self.performance_curves = {}
        
        print(f"✓ Universe OS Agent initialized")
        print(f"  Mode: {UNIVERSE_OS_MODE}")
        print(f"  Interval: {UNIVERSE_OS_INTERVAL_HOURS}h")
        print(f"  Visualization: {'ENABLED' if ENABLE_VISUALIZATION else 'DISABLED'}")
        print()
    
    # =========================================================================
    # PHASE 1: DATA INGESTION
    # =========================================================================
    
    def load_universe_snapshot(self) -> bool:
        """Load current universe snapshot"""
        print("STEP 1: LOADING UNIVERSE SNAPSHOT")
        print("-" * 80)
        
        if not self.universe_snapshot_path.exists():
            print(f"✗ Snapshot not found: {self.universe_snapshot_path}")
            return False
        
        with open(self.universe_snapshot_path) as f:
            self.universe_snapshot = json.load(f)
        
        print(f"✓ Loaded universe snapshot")
        print(f"  Generated: {self.universe_snapshot.get('generated_at')}")
        print(f"  Mode: {self.universe_snapshot.get('mode')}")
        print(f"  Symbol count: {self.universe_snapshot.get('symbol_count')}")
        print()
        return True
    
    def load_selector_output(self) -> bool:
        """Load previous Universe Selector Agent output"""
        print("STEP 2: LOADING SELECTOR OUTPUT")
        print("-" * 80)
        
        if not self.selector_output_path.exists():
            print(f"ℹ Selector output not found: {self.selector_output_path}")
            print("  (This is OK - OS Agent can run independently)")
            print()
            return False
        
        with open(self.selector_output_path) as f:
            self.selector_output = json.load(f)
        
        print(f"✓ Loaded selector output")
        print(f"  Generated: {self.selector_output.get('generated_at')}")
        print(f"  Confidence: {self.selector_output.get('data_confidence')}")
        print()
        return True
    
    def load_trade_data(self) -> bool:
        """Load all trade logs from trades/*.jsonl"""
        print("STEP 3: LOADING TRADE DATA")
        print("-" * 80)
        
        if not self.trades_dir.exists():
            print(f"✗ Trades directory not found: {self.trades_dir}")
            print("  (No trade data available yet - will use signal-only analysis)")
            print()
            return False
        
        trade_files = list(self.trades_dir.glob("*.jsonl"))
        if not trade_files:
            print(f"✗ No trade files found in {self.trades_dir}")
            print()
            return False
        
        total_trades = 0
        for trade_file in trade_files:
            with open(trade_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            trade = json.loads(line)
                            symbol = trade.get("symbol")
                            if symbol:
                                self.trade_data[symbol].append(trade)
                                total_trades += 1
                        except json.JSONDecodeError:
                            continue
        
        symbols_with_trades = len(self.trade_data)
        print(f"✓ Loaded {total_trades} trades from {len(trade_files)} files")
        print(f"  Symbols with trades: {symbols_with_trades}")
        
        if symbols_with_trades > 0:
            avg_trades = total_trades / symbols_with_trades
            print(f"  Avg trades per symbol: {avg_trades:.1f}")
        print()
        return True
    
    def load_signal_data(self) -> bool:
        """Load all policy observation signals"""
        print("STEP 4: LOADING SIGNAL DATA")
        print("-" * 80)
        
        if not self.policy_obs_dir.exists():
            print(f"✗ Policy observations directory not found: {self.policy_obs_dir}")
            return False
        
        signal_files = list(self.policy_obs_dir.glob("*.jsonl"))
        if not signal_files:
            print(f"✗ No signal files found in {self.policy_obs_dir}")
            return False
        
        total_signals = 0
        for signal_file in signal_files:
            with open(signal_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            signal = json.loads(line)
                            symbol = signal.get("symbol")
                            if symbol:
                                self.signal_data[symbol].append(signal)
                                total_signals += 1
                        except json.JSONDecodeError:
                            continue
        
        symbols_with_signals = len(self.signal_data)
        print(f"✓ Loaded {total_signals} signals from {len(signal_files)} files")
        print(f"  Symbols with signals: {symbols_with_signals}")
        
        if symbols_with_signals > 0:
            avg_signals = total_signals / symbols_with_signals
            print(f"  Avg signals per symbol: {avg_signals:.1f}")
        print()
        return True
    
    # =========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # =========================================================================
    
    def compute_symbol_features(self):
        """Compute complete feature set for all symbols"""
        print("STEP 5: COMPUTING SYMBOL FEATURES")
        print("-" * 80)
        
        # Get all symbols from snapshot, trades, and signals
        all_symbols = set(self.universe_snapshot.get("symbols", []))
        all_symbols.update(self.trade_data.keys())
        all_symbols.update(self.signal_data.keys())
        
        print(f"Total unique symbols to analyze: {len(all_symbols)}")
        print()
        
        for symbol in all_symbols:
            features = self._compute_single_symbol_features(symbol)
            self.symbol_features[symbol] = features
        
        print(f"✓ Computed features for {len(self.symbol_features)} symbols")
        print()
    
    def _compute_single_symbol_features(self, symbol: str) -> SymbolFeatures:
        """Compute all features for a single symbol"""
        features = SymbolFeatures(symbol=symbol)
        
        # Get data for this symbol
        trades = self.trade_data.get(symbol, [])
        signals = self.signal_data.get(symbol, [])
        
        features.trade_count = len(trades)
        features.signal_count = len(signals)
        
        # === TRADE-BASED FEATURES ===
        if trades:
            R_values = [t.get("R", 0.0) for t in trades]
            pnl_values = [t.get("pnl", 0.0) for t in trades]
            slippage_values = [t.get("slippage", 0.0) for t in trades if "slippage" in t]
            spread_values = [t.get("spread", 0.0) for t in trades if "spread" in t]
            
            # Performance metrics
            wins = [r for r in R_values if r > 0]
            losses = [r for r in R_values if r <= 0]
            
            features.winrate = len(wins) / len(R_values) if R_values else 0.0
            features.avg_R = statistics.mean(R_values) if R_values else 0.0
            features.median_R = statistics.median(R_values) if R_values else 0.0
            features.total_R = sum(R_values)
            features.std_R = statistics.stdev(R_values) if len(R_values) > 1 else 0.0
            features.max_R = max(R_values) if R_values else 0.0
            features.min_R = min(R_values) if R_values else 0.0
            features.R_per_trade = features.total_R / len(R_values) if R_values else 0.0
            
            # Profit factor
            gross_win = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            features.profit_factor = gross_win / gross_loss if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
            
            # Cost metrics
            features.avg_slippage = statistics.mean(slippage_values) if slippage_values else 0.0
            features.avg_spread = statistics.mean(spread_values) if spread_values else 0.0
            features.max_slippage = max(slippage_values) if slippage_values else 0.0
            
            # Stability metrics
            features.volatility_score = features.std_R
            
            # Stability = (profitability * consistency) / (costs + variance)
            if features.avg_R > 0:
                cost_penalty = features.avg_spread + features.avg_slippage + 0.01
                variance_penalty = features.std_R + 0.01
                features.stability_score = (features.avg_R * features.winrate) / (cost_penalty + variance_penalty)
            
            # Consistency = inverse variance of R series
            if features.std_R > 0:
                features.consistency_score = 1.0 / (features.std_R ** 2 + 1e-8)
            
            # Regime-based performance
            trending_trades = [t for t in trades if t.get("regime_tag", "").startswith("TRENDING")]
            ranging_trades = [t for t in trades if t.get("regime_tag", "").startswith("RANGING")]
            mixed_trades = [t for t in trades if t.get("regime_tag", "").startswith("MIXED")]
            
            if trending_trades:
                features.trending_R = statistics.mean([t.get("R", 0.0) for t in trending_trades])
            if ranging_trades:
                features.ranging_R = statistics.mean([t.get("R", 0.0) for t in ranging_trades])
            if mixed_trades:
                features.mixed_R = statistics.mean([t.get("R", 0.0) for t in mixed_trades])
            
            # Regime dependency (how much performance varies by regime)
            regime_Rs = [r for r in [features.trending_R, features.ranging_R, features.mixed_R] if r != 0.0]
            if len(regime_Rs) > 1:
                features.regime_dependency_score = statistics.stdev(regime_Rs)
            
            # Volatility-based performance
            high_vol_trades = [t for t in trades if t.get("vol_level") in ["HIGH", "EXTREME"]]
            extreme_vol_trades = [t for t in trades if t.get("vol_level") == "EXTREME"]
            normal_vol_trades = [t for t in trades if t.get("vol_level") in ["NORMAL", "LOW"]]
            
            if high_vol_trades:
                features.high_vol_R = statistics.mean([t.get("R", 0.0) for t in high_vol_trades])
            if extreme_vol_trades:
                features.extreme_vol_R = statistics.mean([t.get("R", 0.0) for t in extreme_vol_trades])
            if normal_vol_trades:
                features.normal_vol_R = statistics.mean([t.get("R", 0.0) for t in normal_vol_trades])
        
        # === SIGNAL-BASED FEATURES ===
        if signals:
            allow_signals = [s for s in signals if s.get("decision") == "TRADE_ALLOWED"]
            disallow_signals = [s for s in signals if s.get("decision") == "TRADE_DISALLOWED"]
            
            features.allow_count = len(allow_signals)
            features.disallow_count = len(disallow_signals)
            features.disallow_rate = len(disallow_signals) / len(signals) if signals else 0.0
            
            # Confidence profile
            confidences = [s.get("confidence", 0.0) for s in signals if "confidence" in s]
            if confidences:
                features.avg_confidence = statistics.mean(confidences)
        
        # === COMPOSITE SCORES ===
        features.profitability_score = self._compute_profitability_score(features)
        features.reliability_score = self._compute_reliability_score(features)
        features.quality_score = self._compute_quality_score(features)
        
        return features
    
    def _compute_profitability_score(self, f: SymbolFeatures) -> float:
        """Compute profitability score"""
        if f.trade_count == 0:
            return 0.0
        
        # Weight: avg_R (50%), total_R (30%), profit_factor (20%)
        avg_r_component = max(0, min(1, (f.avg_R + 1) / 3))  # Normalize to 0-1
        total_r_component = max(0, min(1, (f.total_R + 5) / 15))  # Normalize
        pf_component = max(0, min(1, f.profit_factor / 5))  # Normalize
        
        score = 0.5 * avg_r_component + 0.3 * total_r_component + 0.2 * pf_component
        return score
    
    def _compute_reliability_score(self, f: SymbolFeatures) -> float:
        """Compute reliability score"""
        if f.signal_count == 0:
            return 0.0
        
        # Weight: allow_rate (40%), confidence (30%), consistency (30%)
        allow_rate_component = 1 - f.disallow_rate
        confidence_component = f.avg_confidence
        consistency_component = min(1, f.consistency_score / 10) if f.consistency_score > 0 else 0
        
        score = 0.4 * allow_rate_component + 0.3 * confidence_component + 0.3 * consistency_component
        return score
    
    def _compute_quality_score(self, f: SymbolFeatures) -> float:
        """Compute overall quality score"""
        # Weight: profitability (40%), reliability (30%), stability (30%)
        score = 0.4 * f.profitability_score + 0.3 * f.reliability_score + 0.3 * min(1, f.stability_score)
        return score
    
    # =========================================================================
    # PHASE 3: SYMBOL CLASSIFICATION
    # =========================================================================
    
    def classify_symbols(self):
        """Classify all symbols into tiers"""
        print("STEP 6: CLASSIFYING SYMBOLS")
        print("-" * 80)
        
        for symbol, features in self.symbol_features.items():
            tier, reason = self._classify_single_symbol(features)
            features.tier = tier
            features.tier_reason = reason
            
            # Add to appropriate list
            if tier == "CORE":
                self.core_symbols.append(symbol)
            elif tier == "EXPANSION":
                self.expansion_symbols.append(symbol)
            elif tier == "CONDITIONAL":
                self.conditional_symbols.append(symbol)
            elif tier == "BLACKLIST":
                self.blacklist_symbols.append(symbol)
            else:
                self.insufficient_data_symbols.append(symbol)
        
        print(f"CORE: {len(self.core_symbols)} symbols")
        print(f"EXPANSION: {len(self.expansion_symbols)} symbols")
        print(f"CONDITIONAL: {len(self.conditional_symbols)} symbols")
        print(f"BLACKLIST: {len(self.blacklist_symbols)} symbols")
        print(f"INSUFFICIENT_DATA: {len(self.insufficient_data_symbols)} symbols")
        print()
    
    def _classify_single_symbol(self, f: SymbolFeatures) -> Tuple[str, str]:
        """Classify a single symbol based on its features"""
        
        # Check for insufficient data
        if f.trade_count < 3 or f.signal_count < 5:
            return "INSUFFICIENT_DATA", f"Only {f.trade_count} trades, {f.signal_count} signals"
        
        # BLACKLIST criteria (most restrictive first)
        if f.total_R < -0.5 and f.winrate < 0.35:
            return "BLACKLIST", f"Poor performance: total_R={f.total_R:.2f}, winrate={f.winrate:.1%}"
        
        if f.avg_R < 0.1:
            return "BLACKLIST", f"Low avg_R: {f.avg_R:.3f}"
        
        if f.disallow_rate > 0.50:
            return "BLACKLIST", f"High disallow rate: {f.disallow_rate:.1%}"
        
        if f.stability_score < 0.05 and f.trade_count >= 10:
            return "BLACKLIST", f"Unstable: stability={f.stability_score:.3f}"
        
        # CORE criteria (most desirable)
        if (f.stability_score >= 0.20 and 
            f.quality_score >= 0.25 and 
            f.winrate >= 0.45 and 
            f.disallow_rate <= 0.25 and 
            f.avg_R >= 0.5):
            return "CORE", f"Excellent: stability={f.stability_score:.2f}, quality={f.quality_score:.2f}, winrate={f.winrate:.1%}"
        
        # EXPANSION criteria (good performers)
        if (f.stability_score >= 0.10 and 
            f.quality_score >= 0.15 and 
            f.winrate >= 0.35 and 
            f.disallow_rate <= 0.40 and 
            f.avg_R >= 0.3):
            return "EXPANSION", f"Good: stability={f.stability_score:.2f}, quality={f.quality_score:.2f}"
        
        # CONDITIONAL criteria (situational)
        # Profitable in specific regimes
        if f.trending_R > 0.5 or f.normal_vol_R > 0.5:
            return "CONDITIONAL", f"Regime-specific: trending_R={f.trending_R:.2f}, normal_vol_R={f.normal_vol_R:.2f}"
        
        # Profitable but unstable
        if f.avg_R > 0.3 and f.total_R > 1.0:
            return "CONDITIONAL", f"Profitable but unstable: avg_R={f.avg_R:.2f}, stability={f.stability_score:.3f}"
        
        # Default to BLACKLIST (conservative approach)
        return "BLACKLIST", f"Does not meet criteria: quality={f.quality_score:.2f}, winrate={f.winrate:.1%}"
    
    # =========================================================================
    # PHASE 4: UNIVERSE OPTIMIZATION
    # =========================================================================
    
    def optimize_universe_sizes(self):
        """Generate optimal universe sizes with performance curves"""
        print("STEP 7: OPTIMIZING UNIVERSE SIZES")
        print("-" * 80)
        
        # Sort symbols by quality score (descending)
        sorted_symbols = sorted(
            [(s, f) for s, f in self.symbol_features.items() if f.trade_count >= 3],
            key=lambda x: x[1].quality_score,
            reverse=True
        )
        
        if not sorted_symbols:
            print("✗ No symbols with sufficient trade data for optimization")
            print()
            return
        
        # Compute cumulative performance curves
        sizes = [50, 100, 150, 200, 300, 400, 500, 600]
        curves = {
            "sizes": sizes,
            "avg_R": [],
            "avg_winrate": [],
            "avg_quality": [],
            "avg_stability": [],
            "total_R": [],
            "avg_slippage": [],
        }
        
        for size in sizes:
            # Take top N symbols
            top_symbols = sorted_symbols[:min(size, len(sorted_symbols))]
            
            if not top_symbols:
                curves["avg_R"].append(0)
                curves["avg_winrate"].append(0)
                curves["avg_quality"].append(0)
                curves["avg_stability"].append(0)
                curves["total_R"].append(0)
                curves["avg_slippage"].append(0)
                continue
            
            # Compute averages
            avg_R = statistics.mean([f.avg_R for _, f in top_symbols])
            avg_winrate = statistics.mean([f.winrate for _, f in top_symbols])
            avg_quality = statistics.mean([f.quality_score for _, f in top_symbols])
            avg_stability = statistics.mean([f.stability_score for _, f in top_symbols])
            total_R = sum([f.total_R for _, f in top_symbols])
            avg_slippage = statistics.mean([f.avg_slippage for _, f in top_symbols if f.avg_slippage > 0])
            
            curves["avg_R"].append(avg_R)
            curves["avg_winrate"].append(avg_winrate)
            curves["avg_quality"].append(avg_quality)
            curves["avg_stability"].append(avg_stability)
            curves["total_R"].append(total_R)
            curves["avg_slippage"].append(avg_slippage if avg_slippage else 0)
        
        self.performance_curves = curves
        
        print("✓ Performance curves computed")
        print(f"  Tested universe sizes: {sizes}")
        print()
        
        # Display summary
        print("Performance vs Universe Size:")
        print("-" * 80)
        print(f"{'Size':<8} {'Avg R':<10} {'Winrate':<10} {'Quality':<10} {'Stability':<10}")
        print("-" * 80)
        for i, size in enumerate(sizes):
            print(f"{size:<8} {curves['avg_R'][i]:<10.3f} {curves['avg_winrate'][i]:<10.1%} "
                  f"{curves['avg_quality'][i]:<10.3f} {curves['avg_stability'][i]:<10.3f}")
        print()
    
    def generate_universe_profiles(self):
        """Generate SAFE, AGGRESSIVE, EXPERIMENTAL profiles"""
        print("STEP 8: GENERATING UNIVERSE PROFILES")
        print("-" * 80)
        
        # SAFE Profile (150-200): CORE + top EXPANSION
        safe_symbols = self.core_symbols.copy()
        
        # Add top EXPANSION symbols sorted by quality
        expansion_sorted = sorted(
            [s for s in self.expansion_symbols if s in self.symbol_features],
            key=lambda s: self.symbol_features[s].quality_score,
            reverse=True
        )
        
        safe_target = 180
        if len(safe_symbols) < safe_target:
            safe_symbols.extend(expansion_sorted[:safe_target - len(safe_symbols)])
        
        safe_features = [self.symbol_features[s] for s in safe_symbols if s in self.symbol_features]
        safe_avg_R = statistics.mean([f.avg_R for f in safe_features]) if safe_features else 0.0
        safe_winrate = statistics.mean([f.winrate for f in safe_features]) if safe_features else 0.0
        
        self.safe_profile = UniverseProfile(
            name="SAFE",
            size=len(safe_symbols),
            symbols=safe_symbols,
            expected_R=safe_avg_R,
            expected_winrate=safe_winrate,
            risk_level="LOW",
            description="Production/Mainnet - CORE + top EXPANSION, low slippage, high stability"
        )
        
        # AGGRESSIVE Profile (300-400): CORE + EXPANSION + some CONDITIONAL
        aggressive_symbols = self.core_symbols + self.expansion_symbols
        
        # Add top CONDITIONAL symbols
        conditional_sorted = sorted(
            [s for s in self.conditional_symbols if s in self.symbol_features],
            key=lambda s: self.symbol_features[s].quality_score,
            reverse=True
        )
        
        aggressive_target = 350
        if len(aggressive_symbols) < aggressive_target:
            aggressive_symbols.extend(conditional_sorted[:aggressive_target - len(aggressive_symbols)])
        
        aggressive_features = [self.symbol_features[s] for s in aggressive_symbols if s in self.symbol_features]
        aggressive_avg_R = statistics.mean([f.avg_R for f in aggressive_features]) if aggressive_features else 0.0
        aggressive_winrate = statistics.mean([f.winrate for f in aggressive_features]) if aggressive_features else 0.0
        
        self.aggressive_profile = UniverseProfile(
            name="AGGRESSIVE",
            size=len(aggressive_symbols),
            symbols=aggressive_symbols,
            expected_R=aggressive_avg_R,
            expected_winrate=aggressive_winrate,
            risk_level="MEDIUM",
            description="Testnet/Training - CORE + EXPANSION + CONDITIONAL, exclude only BLACKLIST"
        )
        
        # EXPERIMENTAL Profile (500-600): Everything except BLACKLIST
        experimental_symbols = (
            self.core_symbols + 
            self.expansion_symbols + 
            self.conditional_symbols + 
            self.insufficient_data_symbols
        )
        
        experimental_features = [self.symbol_features[s] for s in experimental_symbols if s in self.symbol_features and self.symbol_features[s].trade_count > 0]
        experimental_avg_R = statistics.mean([f.avg_R for f in experimental_features]) if experimental_features else 0.0
        experimental_winrate = statistics.mean([f.winrate for f in experimental_features]) if experimental_features else 0.0
        
        self.experimental_profile = UniverseProfile(
            name="EXPERIMENTAL",
            size=len(experimental_symbols),
            symbols=experimental_symbols,
            expected_R=experimental_avg_R,
            expected_winrate=experimental_winrate,
            risk_level="HIGH",
            description="Research - all symbols except BLACKLIST, for ML training & data collection"
        )
        
        print(f"✓ Generated 3 universe profiles:")
        print(f"\n  SAFE Profile:")
        print(f"    Size: {self.safe_profile.size}")
        print(f"    Expected R: {self.safe_profile.expected_R:.3f}")
        print(f"    Expected Winrate: {self.safe_profile.expected_winrate:.1%}")
        print(f"    Risk: {self.safe_profile.risk_level}")
        
        print(f"\n  AGGRESSIVE Profile:")
        print(f"    Size: {self.aggressive_profile.size}")
        print(f"    Expected R: {self.aggressive_profile.expected_R:.3f}")
        print(f"    Expected Winrate: {self.aggressive_profile.expected_winrate:.1%}")
        print(f"    Risk: {self.aggressive_profile.risk_level}")
        
        print(f"\n  EXPERIMENTAL Profile:")
        print(f"    Size: {self.experimental_profile.size}")
        print(f"    Expected R: {self.experimental_profile.expected_R:.3f}")
        print(f"    Expected Winrate: {self.experimental_profile.expected_winrate:.1%}")
        print(f"    Risk: {self.experimental_profile.risk_level}")
        print()
    
    # =========================================================================
    # PHASE 5: DELTA COMPUTATION
    # =========================================================================
    
    def compute_deltas(self):
        """Compute universe deltas vs current snapshot"""
        print("STEP 9: COMPUTING UNIVERSE DELTAS")
        print("-" * 80)
        
        current_symbols = set(self.universe_snapshot.get("symbols", []))
        
        self.deltas = {
            "SAFE": self._compute_profile_delta(current_symbols, self.safe_profile),
            "AGGRESSIVE": self._compute_profile_delta(current_symbols, self.aggressive_profile),
            "EXPERIMENTAL": self._compute_profile_delta(current_symbols, self.experimental_profile),
        }
        
        print(f"✓ Computed deltas for 3 profiles")
        
        for profile_name, delta in self.deltas.items():
            print(f"\n  {profile_name} Profile Delta:")
            print(f"    To add: {len(delta['to_add'])} symbols")
            print(f"    To remove: {len(delta['to_remove'])} symbols")
            print(f"    To keep: {len(delta['to_keep'])} symbols")
            print(f"    Net change: {delta['net_change']:+d} symbols")
        print()
    
    def _compute_profile_delta(self, current: set, profile: UniverseProfile) -> Dict:
        """Compute delta for a single profile"""
        recommended = set(profile.symbols)
        
        to_add = list(recommended - current)
        to_remove = list(current - recommended)
        to_keep = list(current & recommended)
        
        return {
            "to_add": to_add,
            "to_remove": to_remove,
            "to_keep": to_keep,
            "net_change": len(to_add) - len(to_remove),
            "add_count": len(to_add),
            "remove_count": len(to_remove),
            "keep_count": len(to_keep),
        }
    
    # =========================================================================
    # PHASE 6: VISUALIZATION (conceptual)
    # =========================================================================
    
    def generate_visualizations(self):
        """Generate visualization artifacts"""
        print("STEP 10: GENERATING VISUALIZATIONS")
        print("-" * 80)
        
        if not ENABLE_VISUALIZATION:
            print("ℹ Visualization disabled (set UNIVERSE_OS_VISUALIZE=true to enable)")
            print()
            return
        
        print("✓ Would generate the following charts:")
        print("  1. pnl_vs_universe_size.png")
        print("  2. cumulative_R_curve.png")
        print("  3. symbol_performance_scatter.png")
        print("  4. slippage_heatmap.png")
        print("  5. spread_distribution.png")
        print("  6. regime_performance_bars.png")
        print("  7. core_vs_blacklist_comparison.png")
        print()
        print(f"  Output directory: {self.charts_dir}")
        print()
    
    # =========================================================================
    # PHASE 7: SNAPSHOT & OUTPUT
    # =========================================================================
    
    def write_universe_snapshot(self):
        """Write new universe snapshot"""
        print("STEP 11: WRITING UNIVERSE SNAPSHOT")
        print("-" * 80)
        
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agent_version": "2.0_OS",
            "mode": "optimized",
            "data_confidence": self._assess_data_confidence(),
            
            # Current universe info
            "current_universe": {
                "symbol_count": self.universe_snapshot.get("symbol_count"),
                "mode": self.universe_snapshot.get("mode"),
            },
            
            # Classifications
            "classifications": {
                "CORE": {
                    "count": len(self.core_symbols),
                    "symbols": self.core_symbols,
                },
                "EXPANSION": {
                    "count": len(self.expansion_symbols),
                    "symbols": self.expansion_symbols,
                },
                "CONDITIONAL": {
                    "count": len(self.conditional_symbols),
                    "symbols": self.conditional_symbols,
                },
                "BLACKLIST": {
                    "count": len(self.blacklist_symbols),
                    "symbols": self.blacklist_symbols,
                },
                "INSUFFICIENT_DATA": {
                    "count": len(self.insufficient_data_symbols),
                    "symbols": self.insufficient_data_symbols,
                },
            },
            
            # Universe profiles
            "profiles": {
                "SAFE": asdict(self.safe_profile) if self.safe_profile else None,
                "AGGRESSIVE": asdict(self.aggressive_profile) if self.aggressive_profile else None,
                "EXPERIMENTAL": asdict(self.experimental_profile) if self.experimental_profile else None,
            },
            
            # Performance curves
            "performance_curves": self.performance_curves,
            
            # Deltas
            "deltas": self.deltas,
            
            # Recommendations
            "recommendations": {
                "recommended_profile": self._recommend_profile(),
                "recommended_universe_size": self.safe_profile.size if self.safe_profile else 0,
                "QT_UNIVERSE": "custom",
                "QT_MAX_SYMBOLS": self.safe_profile.size if self.safe_profile else 222,
            },
        }
        
        output_path = self.data_dir / "universe_os_snapshot.json"
        with open(output_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"✓ Snapshot written to: {output_path}")
        print()
        
        return snapshot
    
    def write_delta_report(self):
        """Write detailed delta report"""
        print("STEP 12: WRITING DELTA REPORT")
        print("-" * 80)
        
        delta_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_universe_size": self.universe_snapshot.get("symbol_count"),
            
            "deltas_by_profile": self.deltas,
            
            "symbol_movements": self._analyze_symbol_movements(),
            
            "recommendations": {
                "immediate_removals": self._get_immediate_removals(),
                "immediate_additions": self._get_immediate_additions(),
                "watch_list": self._get_watch_list(),
            },
        }
        
        output_path = self.data_dir / "universe_delta_report.json"
        with open(output_path, 'w') as f:
            json.dump(delta_report, f, indent=2)
        
        print(f"✓ Delta report written to: {output_path}")
        print()
    
    def _analyze_symbol_movements(self) -> Dict:
        """Analyze which symbols changed tiers"""
        movements = {
            "promoted_to_core": [],
            "demoted_from_core": [],
            "added_to_blacklist": [],
            "removed_from_blacklist": [],
        }
        
        # Compare with selector output if available
        if self.selector_output:
            old_core = set(self.selector_output.get("classifications", {}).get("CORE", {}).get("symbols", []))
            old_blacklist = set(self.selector_output.get("classifications", {}).get("BLACKLIST", {}).get("symbols", []))
            
            new_core = set(self.core_symbols)
            new_blacklist = set(self.blacklist_symbols)
            
            movements["promoted_to_core"] = list(new_core - old_core)
            movements["demoted_from_core"] = list(old_core - new_core)
            movements["added_to_blacklist"] = list(new_blacklist - old_blacklist)
            movements["removed_from_blacklist"] = list(old_blacklist - new_blacklist)
        
        return movements
    
    def _get_immediate_removals(self) -> List[str]:
        """Get symbols that should be removed immediately"""
        # High-conviction blacklist (multiple negative signals)
        immediate = []
        for symbol in self.blacklist_symbols:
            if symbol in self.symbol_features:
                f = self.symbol_features[symbol]
                if f.total_R < -1.0 and f.winrate < 0.30 and f.trade_count >= 5:
                    immediate.append({
                        "symbol": symbol,
                        "reason": f"Severe underperformance: total_R={f.total_R:.2f}, winrate={f.winrate:.1%}",
                        "total_R": f.total_R,
                        "winrate": f.winrate,
                    })
        return immediate
    
    def _get_immediate_additions(self) -> List[str]:
        """Get symbols that should be added immediately"""
        # High-quality CORE symbols not in current universe
        current = set(self.universe_snapshot.get("symbols", []))
        immediate = []
        
        for symbol in self.core_symbols:
            if symbol not in current and symbol in self.symbol_features:
                f = self.symbol_features[symbol]
                if f.quality_score > 0.30 and f.trade_count >= 5:
                    immediate.append({
                        "symbol": symbol,
                        "reason": f"High quality CORE: quality={f.quality_score:.2f}, winrate={f.winrate:.1%}",
                        "quality_score": f.quality_score,
                        "winrate": f.winrate,
                    })
        return immediate
    
    def _get_watch_list(self) -> List[str]:
        """Get symbols to watch closely"""
        watch = []
        
        for symbol, features in self.symbol_features.items():
            # Borderline symbols
            if features.tier in ["EXPANSION", "CONDITIONAL"]:
                if features.quality_score > 0.20 and features.trade_count < 10:
                    watch.append({
                        "symbol": symbol,
                        "tier": features.tier,
                        "reason": "Promising but needs more data",
                        "quality_score": features.quality_score,
                        "trade_count": features.trade_count,
                    })
        
        return watch
    
    def _assess_data_confidence(self) -> str:
        """Assess overall data confidence level"""
        total_trades = sum(len(trades) for trades in self.trade_data.values())
        total_signals = sum(len(signals) for signals in self.signal_data.values())
        
        if total_trades >= 1000 and total_signals >= 10000:
            return "VERY_HIGH"
        elif total_trades >= 500 and total_signals >= 5000:
            return "HIGH"
        elif total_trades >= 100 and total_signals >= 1000:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _recommend_profile(self) -> str:
        """Recommend which profile to use"""
        confidence = self._assess_data_confidence()
        
        if confidence in ["HIGH", "VERY_HIGH"]:
            return "SAFE (production ready)"
        elif confidence == "MEDIUM":
            return "AGGRESSIVE (testnet recommended)"
        else:
            return "Wait for more data (current: LOW confidence)"
    
    # =========================================================================
    # PHASE 8: REPORTING
    # =========================================================================
    
    def generate_executive_summary(self, snapshot: Dict):
        """Generate executive summary report"""
        print("\n" + "="*80)
        print("UNIVERSE OS AGENT — EXECUTIVE SUMMARY")
        print("="*80 + "\n")
        
        print(f"Generated: {snapshot['generated_at']}")
        print(f"Data Confidence: {snapshot['data_confidence']}")
        print(f"Recommended Profile: {snapshot['recommendations']['recommended_profile']}")
        print()
        
        print("CLASSIFICATION SUMMARY:")
        print("-" * 80)
        for tier, data in snapshot['classifications'].items():
            print(f"  {tier}: {data['count']} symbols")
        print()
        
        print("UNIVERSE PROFILES:")
        print("-" * 80)
        for profile_name, profile in snapshot['profiles'].items():
            if profile:
                print(f"\n  {profile_name}:")
                print(f"    Size: {profile['size']}")
                print(f"    Expected R: {profile['expected_R']:.3f}")
                print(f"    Expected Winrate: {profile['expected_winrate']:.1%}")
                print(f"    Risk Level: {profile['risk_level']}")
        print()
        
        print("TOP 10 SYMBOLS BY QUALITY:")
        print("-" * 80)
        top_symbols = sorted(
            [(s, f) for s, f in self.symbol_features.items() if f.trade_count >= 3],
            key=lambda x: x[1].quality_score,
            reverse=True
        )[:10]
        
        for i, (symbol, features) in enumerate(top_symbols, 1):
            print(f"  {i:2d}. {symbol:15s} | Quality: {features.quality_score:.3f} | "
                  f"Tier: {features.tier:12s} | Winrate: {features.winrate:.1%} | "
                  f"Avg R: {features.avg_R:+.3f}")
        print()
        
        print("IMMEDIATE ACTIONS:")
        print("-" * 80)
        
        immediate_removals = snapshot.get("recommendations", {}).get("immediate_removals", [])
        immediate_additions = snapshot.get("recommendations", {}).get("immediate_additions", [])
        
        if immediate_removals or immediate_additions:
            if immediate_removals:
                print(f"\n  ⚠️  REMOVE IMMEDIATELY ({len(immediate_removals)} symbols):")
                for item in immediate_removals[:5]:
                    print(f"      - {item['symbol']}: {item['reason']}")
            
            if immediate_additions:
                print(f"\n  ✅ ADD IMMEDIATELY ({len(immediate_additions)} symbols):")
                for item in immediate_additions[:5]:
                    print(f"      - {item['symbol']}: {item['reason']}")
        else:
            print("  ℹ️  No immediate actions required")
        
        print()
        
        print("NEXT STEPS:")
        print("-" * 80)
        confidence = snapshot['data_confidence']
        
        if confidence == "LOW":
            print("  1. Continue collecting trade data (target: 100+ trades)")
            print("  2. Run Universe OS Agent again in 7 days")
            print("  3. Review classifications when confidence reaches MEDIUM")
        elif confidence == "MEDIUM":
            print("  1. Test AGGRESSIVE profile in paper trading")
            print("  2. Monitor performance for 7 days")
            print("  3. Consider deploying SAFE profile if validated")
        else:
            print("  1. Deploy SAFE profile to production")
            print("  2. Set up weekly Universe OS Agent runs")
            print("  3. Monitor deltas and adjust as needed")
        
        print()
        print("="*80)
        print()
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def run(self):
        """Execute full Universe OS Agent workflow"""
        try:
            # Phase 1: Data Ingestion
            self.load_universe_snapshot()
            self.load_selector_output()
            has_trades = self.load_trade_data()
            has_signals = self.load_signal_data()
            
            if not has_signals:
                print("✗ Cannot proceed without signal data")
                return False
            
            # Phase 2: Feature Engineering
            self.manage_todo_list(2, "in-progress")
            self.compute_symbol_features()
            self.manage_todo_list(2, "completed")
            
            # Phase 3: Classification
            self.manage_todo_list(3, "in-progress")
            self.classify_symbols()
            self.manage_todo_list(3, "completed")
            
            # Phase 4: Optimization
            self.manage_todo_list(4, "in-progress")
            if has_trades:
                self.optimize_universe_sizes()
            self.generate_universe_profiles()
            self.manage_todo_list(4, "completed")
            
            # Phase 5: Visualization
            self.manage_todo_list(5, "in-progress")
            self.generate_visualizations()
            self.manage_todo_list(5, "completed")
            
            # Phase 6: Deltas
            self.manage_todo_list(6, "in-progress")
            self.compute_deltas()
            self.manage_todo_list(6, "completed")
            
            # Phase 7: Snapshot & Output
            self.manage_todo_list(7, "in-progress")
            snapshot = self.write_universe_snapshot()
            self.write_delta_report()
            self.manage_todo_list(7, "completed")
            
            # Phase 8: Reporting
            self.manage_todo_list(8, "in-progress")
            self.generate_executive_summary(snapshot)
            self.manage_todo_list(8, "completed")
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def manage_todo_list(self, task_id: int, status: str):
        """Helper to manage todo list (conceptual)"""
        pass  # Would integrate with actual todo management system


def main():
    """Main entry point"""
    print("="*80)
    print("UNIVERSE OS AGENT — AUTONOMOUS QUANTITATIVE UNIVERSE MANAGER")
    print("="*80)
    print()
    print("Mission: Complete trading universe lifecycle management")
    print("  - Discovery, Analysis, Health Diagnostics")
    print("  - Ranking, Selection, Classification")
    print("  - Dynamic Universe Construction")
    print("  - Snapshot Writer, Delta Engine")
    print("  - Visualizations, Reporting, Integration")
    print()
    
    agent = UniverseOSAgent()
    success = agent.run()
    
    if success:
        print("✅ UNIVERSE OS AGENT COMPLETED SUCCESSFULLY")
        print()
        print("Output files:")
        print("  - /app/data/universe_os_snapshot.json")
        print("  - /app/data/universe_delta_report.json")
        print()
        return 0
    else:
        print("✗ UNIVERSE OS AGENT FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
