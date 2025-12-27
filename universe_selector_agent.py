"""
UNIVERSE SELECTOR AGENT — Autonomous Universe Optimization

Mission:
- Continuously evaluate symbol performance and stability
- Classify symbols into CORE, EXPANSION, CONDITIONAL, BLACKLIST
- Recommend optimal universe size for different risk profiles
- Generate actionable universe configuration recommendations

This agent does NOT modify code — it only analyzes and recommends.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import statistics
import math

# Data paths
UNIVERSE_SNAPSHOT = "/app/data/universe_snapshot.json"
POLICY_OBSERVATIONS = "/app/data/policy_observations"
TRADES_PATH = "/app/data/trades"
OUTPUT_FILE = "/app/data/universe_selector_output.json"

# Thresholds and weights
MIN_SIGNALS_FOR_CLASSIFICATION = 5
MIN_TRADES_FOR_PERFORMANCE = 3

# Stability score weights
STABILITY_WEIGHTS = {
    'avg_r': 0.3,
    'winrate': 0.25,
    'variance_penalty': 0.2,
    'cost_penalty': 0.15,
    'consistency': 0.1
}

# Quality score weights
QUALITY_WEIGHTS = {
    'stability': 0.3,
    'profitability': 0.25,
    'reliability': 0.2,
    'regime_adaptability': 0.15,
    'execution_quality': 0.1
}

# Classification thresholds
CORE_THRESHOLDS = {
    'min_stability': 0.20,
    'min_quality': 0.25,
    'min_winrate': 0.45,
    'max_disallow_rate': 0.25,
    'min_avg_r': 0.5
}

EXPANSION_THRESHOLDS = {
    'min_stability': 0.10,
    'min_quality': 0.15,
    'min_winrate': 0.35,
    'max_disallow_rate': 0.40,
    'min_avg_r': 0.3
}

BLACKLIST_THRESHOLDS = {
    'max_total_r': -0.5,
    'max_winrate': 0.35,
    'max_avg_r': 0.1,
    'min_disallow_rate': 0.50,
    'max_stability': 0.05
}


class UniverseSelectorAgent:
    """Autonomous agent for universe selection and optimization"""
    
    def __init__(self):
        self.universe_snapshot = {}
        self.current_symbols = []
        
        # Raw data storage
        self.signal_data = defaultdict(lambda: {
            'total_signals': 0,
            'allowed_signals': 0,
            'blocked_signals': 0,
            'confidences': [],
            'actions': Counter(),
            'regimes': Counter(),
            'vol_levels': Counter(),
            'disallow_reasons': Counter(),
        })
        
        self.trade_data = defaultdict(lambda: {
            'trades': [],
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'r_multiples': [],
            'pnls': [],
            'slippages': [],
            'spreads': [],
            'holding_times': [],
            'exit_reasons': Counter(),
            'regime_performance': defaultdict(list),
            'vol_performance': defaultdict(list),
        })
        
        # Computed features
        self.symbol_features = {}
        self.symbol_classifications = {
            'CORE': [],
            'EXPANSION': [],
            'CONDITIONAL': [],
            'BLACKLIST': [],
            'INSUFFICIENT_DATA': []
        }
        
        # Market state
        self.current_regime = "UNKNOWN"
        self.current_vol_level = "UNKNOWN"
        
    def load_universe_snapshot(self) -> bool:
        """Load current universe configuration"""
        print("\n" + "="*80)
        print("STEP 1: LOADING UNIVERSE SNAPSHOT")
        print("="*80)
        
        try:
            with open(UNIVERSE_SNAPSHOT, 'r') as f:
                self.universe_snapshot = json.load(f)
            
            self.current_symbols = self.universe_snapshot.get('symbols', [])
            
            print(f"✓ Loaded universe snapshot")
            print(f"  Mode: {self.universe_snapshot.get('mode')}")
            print(f"  Symbol count: {len(self.current_symbols)}")
            print(f"  Generated: {self.universe_snapshot.get('generated_at')}")
            
            return True
            
        except FileNotFoundError:
            print(f"✗ Universe snapshot not found at {UNIVERSE_SNAPSHOT}")
            return False
    
    def load_signal_data(self) -> bool:
        """Load and parse policy observation logs"""
        print("\n" + "="*80)
        print("STEP 2: LOADING SIGNAL DATA")
        print("="*80)
        
        signal_files = list(Path(POLICY_OBSERVATIONS).glob("signals_*.jsonl"))
        
        if not signal_files:
            print(f"✗ No signal files found in {POLICY_OBSERVATIONS}")
            return False
        
        total_records = 0
        
        for signal_file in signal_files:
            with open(signal_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        
                        if record.get('type') != 'signal_decision':
                            continue
                        
                        signal = record.get('signal', {})
                        symbol = signal.get('symbol')
                        
                        if not symbol or symbol not in self.current_symbols:
                            continue
                        
                        # Extract data
                        confidence = signal.get('confidence', 0)
                        action = signal.get('action')
                        actual_decision = record.get('actual_decision')
                        
                        # Store metrics
                        data = self.signal_data[symbol]
                        data['total_signals'] += 1
                        
                        if actual_decision == 'TRADE_ALLOWED':
                            data['allowed_signals'] += 1
                        else:
                            data['blocked_signals'] += 1
                        
                        data['confidences'].append(confidence)
                        data['actions'][action] += 1
                        
                        # Extract regime and vol if available (from policy features)
                        # Note: This may not be in signal logs, might need policy_obs logs
                        
                        total_records += 1
                        
                    except json.JSONDecodeError:
                        continue
        
        print(f"✓ Loaded {total_records} signal records")
        print(f"✓ Symbols with signals: {len(self.signal_data)}")
        
        # Try to load policy observation logs for regime/vol data
        self._load_policy_observations()
        
        return True
    
    def _load_policy_observations(self):
        """Load policy observation logs for regime and volatility data"""
        policy_files = list(Path(POLICY_OBSERVATIONS).glob("policy_obs_*.json*"))
        
        if not policy_files:
            print("  ℹ No policy observation files found")
            return
        
        for policy_file in policy_files:
            try:
                with open(policy_file, 'r') as f:
                    if policy_file.suffix == '.jsonl':
                        for line in f:
                            self._process_policy_record(json.loads(line.strip()))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for record in data:
                                self._process_policy_record(record)
                        else:
                            self._process_policy_record(data)
            except Exception as e:
                print(f"  Warning: Failed to load {policy_file.name}: {e}")
    
    def _process_policy_record(self, record: Dict):
        """Process a single policy observation record"""
        symbol = record.get('symbol')
        if not symbol or symbol not in self.current_symbols:
            return
        
        regime = record.get('regime_tag', record.get('regime'))
        vol = record.get('vol_level', record.get('volatility'))
        
        if regime:
            self.signal_data[symbol]['regimes'][regime] += 1
            self.current_regime = regime  # Update current regime
        
        if vol:
            self.signal_data[symbol]['vol_levels'][vol] += 1
            self.current_vol_level = vol  # Update current vol
    
    def load_trade_data(self) -> bool:
        """Load trade logs for performance analysis"""
        print("\n" + "="*80)
        print("STEP 3: LOADING TRADE DATA")
        print("="*80)
        
        # Check for trade data
        trades_path = Path(TRADES_PATH)
        if not trades_path.exists():
            print(f"✗ Trade data path not found: {TRADES_PATH}")
            print("  ℹ Proceeding with signal-only analysis")
            return False
        
        trade_files = list(trades_path.glob("*.jsonl")) + list(trades_path.glob("*.json"))
        
        if not trade_files:
            print(f"  ℹ No trade files found in {TRADES_PATH}")
            print("  ℹ Proceeding with signal-only analysis")
            return False
        
        total_trades = 0
        
        for trade_file in trade_files:
            try:
                with open(trade_file, 'r') as f:
                    if trade_file.suffix == '.jsonl':
                        for line in f:
                            if self._process_trade_record(json.loads(line.strip())):
                                total_trades += 1
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for record in data:
                                if self._process_trade_record(record):
                                    total_trades += 1
                        else:
                            if self._process_trade_record(data):
                                total_trades += 1
            except Exception as e:
                print(f"  Warning: Failed to load {trade_file.name}: {e}")
        
        if total_trades > 0:
            print(f"✓ Loaded {total_trades} trade records")
            print(f"✓ Symbols with trades: {len([s for s in self.trade_data.values() if s['total_trades'] > 0])}")
            return True
        else:
            print("  ℹ No trade data available")
            return False
    
    def _process_trade_record(self, record: Dict) -> bool:
        """Process a single trade record"""
        symbol = record.get('symbol')
        if not symbol or symbol not in self.current_symbols:
            return False
        
        data = self.trade_data[symbol]
        data['total_trades'] += 1
        data['trades'].append(record)
        
        # Extract metrics
        r_multiple = record.get('r_multiple', record.get('R', record.get('r')))
        pnl = record.get('pnl', record.get('net_pnl'))
        slippage = record.get('slippage', 0)
        spread = record.get('spread', 0)
        holding_time = record.get('holding_time', record.get('duration'))
        exit_reason = record.get('exit_reason', 'UNKNOWN')
        regime = record.get('regime', record.get('regime_tag'))
        vol = record.get('vol_level', record.get('volatility'))
        
        # Store metrics
        if r_multiple is not None:
            data['r_multiples'].append(float(r_multiple))
            if float(r_multiple) > 0:
                data['wins'] += 1
            else:
                data['losses'] += 1
        
        if pnl is not None:
            data['pnls'].append(float(pnl))
        
        if slippage:
            data['slippages'].append(abs(float(slippage)))
        
        if spread:
            data['spreads'].append(float(spread))
        
        if holding_time:
            data['holding_times'].append(float(holding_time))
        
        data['exit_reasons'][exit_reason] += 1
        
        # Regime-specific performance
        if regime and r_multiple is not None:
            data['regime_performance'][regime].append(float(r_multiple))
        
        # Volatility-specific performance
        if vol and r_multiple is not None:
            data['vol_performance'][vol].append(float(r_multiple))
        
        return True
    
    def compute_symbol_features(self):
        """Engineer features for each symbol"""
        print("\n" + "="*80)
        print("STEP 4: COMPUTING SYMBOL FEATURES")
        print("="*80)
        
        for symbol in self.current_symbols:
            signal_data = self.signal_data.get(symbol)
            trade_data = self.trade_data.get(symbol)
            
            features = {
                'symbol': symbol,
                'has_signal_data': signal_data and signal_data['total_signals'] > 0,
                'has_trade_data': trade_data and trade_data['total_trades'] > 0,
            }
            
            # Signal-based features
            if signal_data and signal_data['total_signals'] > 0:
                total = signal_data['total_signals']
                allowed = signal_data['allowed_signals']
                
                features['total_signals'] = total
                features['allow_rate'] = allowed / total
                features['disallow_rate'] = 1 - (allowed / total)
                features['avg_confidence'] = statistics.mean(signal_data['confidences']) if signal_data['confidences'] else 0
                features['confidence_std'] = statistics.stdev(signal_data['confidences']) if len(signal_data['confidences']) > 1 else 0
            else:
                features['total_signals'] = 0
                features['allow_rate'] = 0
                features['disallow_rate'] = 1.0
                features['avg_confidence'] = 0
                features['confidence_std'] = 0
            
            # Trade-based features
            if trade_data and trade_data['total_trades'] >= MIN_TRADES_FOR_PERFORMANCE:
                r_mults = trade_data['r_multiples']
                
                features['total_trades'] = trade_data['total_trades']
                features['wins'] = trade_data['wins']
                features['losses'] = trade_data['losses']
                features['winrate'] = trade_data['wins'] / trade_data['total_trades']
                
                features['avg_r'] = statistics.mean(r_mults)
                features['median_r'] = statistics.median(r_mults)
                features['total_r'] = sum(r_mults)
                features['r_std'] = statistics.stdev(r_mults) if len(r_mults) > 1 else 0
                
                features['avg_slippage'] = statistics.mean(trade_data['slippages']) if trade_data['slippages'] else 0
                features['avg_spread'] = statistics.mean(trade_data['spreads']) if trade_data['spreads'] else 0
                
                # Regime-specific performance
                features['regime_performance'] = {}
                for regime, r_list in trade_data['regime_performance'].items():
                    if r_list:
                        features['regime_performance'][regime] = {
                            'avg_r': statistics.mean(r_list),
                            'count': len(r_list)
                        }
                
                # Vol-specific performance
                features['vol_performance'] = {}
                for vol, r_list in trade_data['vol_performance'].items():
                    if r_list:
                        features['vol_performance'][vol] = {
                            'avg_r': statistics.mean(r_list),
                            'count': len(r_list)
                        }
                
            else:
                # No trade data
                features['total_trades'] = trade_data['total_trades'] if trade_data else 0
                features['wins'] = 0
                features['losses'] = 0
                features['winrate'] = 0
                features['avg_r'] = 0
                features['median_r'] = 0
                features['total_r'] = 0
                features['r_std'] = 0
                features['avg_slippage'] = 0
                features['avg_spread'] = 0
                features['regime_performance'] = {}
                features['vol_performance'] = {}
            
            # Compute composite scores
            features['stability_score'] = self._compute_stability_score(features)
            features['quality_score'] = self._compute_quality_score(features)
            features['profitability_score'] = self._compute_profitability_score(features)
            features['reliability_score'] = self._compute_reliability_score(features)
            
            self.symbol_features[symbol] = features
        
        print(f"✓ Computed features for {len(self.symbol_features)} symbols")
    
    def _compute_stability_score(self, features: Dict) -> float:
        """Compute stability score based on multiple factors"""
        if not features['has_trade_data'] or features['total_trades'] < MIN_TRADES_FOR_PERFORMANCE:
            return 0.0
        
        avg_r = features['avg_r']
        winrate = features['winrate']
        r_std = features['r_std']
        avg_slippage = features['avg_slippage']
        avg_spread = features['avg_spread']
        
        # Base stability: profitability adjusted by consistency
        if r_std > 0:
            variance_penalty = 1 / (1 + r_std)
        else:
            variance_penalty = 1.0
        
        # Cost penalty
        total_cost = avg_slippage + avg_spread
        cost_penalty = 1 / (1 + total_cost * 100)  # Scale costs
        
        # Consistency bonus for high winrate
        consistency = winrate
        
        # Weighted combination
        stability = (
            STABILITY_WEIGHTS['avg_r'] * max(0, avg_r) +
            STABILITY_WEIGHTS['winrate'] * winrate +
            STABILITY_WEIGHTS['variance_penalty'] * variance_penalty +
            STABILITY_WEIGHTS['cost_penalty'] * cost_penalty +
            STABILITY_WEIGHTS['consistency'] * consistency
        )
        
        return max(0, min(1, stability))
    
    def _compute_quality_score(self, features: Dict) -> float:
        """Compute overall quality score"""
        stability = features.get('stability_score', 0)
        profitability = features.get('profitability_score', 0)
        reliability = features.get('reliability_score', 0)
        
        # Regime adaptability: does it work across regimes?
        regime_perf = features.get('regime_performance', {})
        if len(regime_perf) >= 2:
            regime_adaptability = 1.0
        elif len(regime_perf) == 1:
            regime_adaptability = 0.5
        else:
            regime_adaptability = 0.0
        
        # Execution quality: low disallow rate + high confidence
        execution_quality = (
            features.get('allow_rate', 0) * 0.6 +
            (features.get('avg_confidence', 0) - 0.38) / 0.62 * 0.4  # Normalize confidence above threshold
        )
        execution_quality = max(0, min(1, execution_quality))
        
        quality = (
            QUALITY_WEIGHTS['stability'] * stability +
            QUALITY_WEIGHTS['profitability'] * profitability +
            QUALITY_WEIGHTS['reliability'] * reliability +
            QUALITY_WEIGHTS['regime_adaptability'] * regime_adaptability +
            QUALITY_WEIGHTS['execution_quality'] * execution_quality
        )
        
        return max(0, min(1, quality))
    
    def _compute_profitability_score(self, features: Dict) -> float:
        """Compute profitability score"""
        if not features['has_trade_data'] or features['total_trades'] < MIN_TRADES_FOR_PERFORMANCE:
            return 0.0
        
        avg_r = features['avg_r']
        total_r = features['total_r']
        
        # Normalize: avg_r of 1.0 = excellent, 0.5 = good, 0 = neutral, negative = bad
        avg_r_score = max(0, min(1, (avg_r + 0.5) / 1.5))
        
        # Total R: more is better
        total_r_score = max(0, min(1, (total_r + 2) / 6))  # Scale: -2 to +4 → 0 to 1
        
        profitability = (avg_r_score * 0.6 + total_r_score * 0.4)
        
        return max(0, min(1, profitability))
    
    def _compute_reliability_score(self, features: Dict) -> float:
        """Compute reliability score based on signal quality"""
        if not features['has_signal_data']:
            return 0.0
        
        allow_rate = features.get('allow_rate', 0)
        avg_confidence = features.get('avg_confidence', 0)
        confidence_std = features.get('confidence_std', 1)
        
        # High allow rate + high confidence + low variance = reliable
        confidence_consistency = 1 / (1 + confidence_std)
        
        reliability = (
            allow_rate * 0.4 +
            (avg_confidence - 0.38) / 0.62 * 0.4 +  # Normalize confidence
            confidence_consistency * 0.2
        )
        
        return max(0, min(1, reliability))
    
    def classify_symbols(self):
        """Classify symbols into CORE, EXPANSION, CONDITIONAL, BLACKLIST"""
        print("\n" + "="*80)
        print("STEP 5: CLASSIFYING SYMBOLS")
        print("="*80)
        
        for symbol, features in self.symbol_features.items():
            
            # Insufficient data
            if features['total_signals'] < MIN_SIGNALS_FOR_CLASSIFICATION:
                self.symbol_classifications['INSUFFICIENT_DATA'].append(symbol)
                continue
            
            # Extract key metrics
            stability = features['stability_score']
            quality = features['quality_score']
            winrate = features['winrate']
            disallow_rate = features['disallow_rate']
            avg_r = features['avg_r']
            total_r = features['total_r']
            
            # BLACKLIST: Clear losers
            if (
                (total_r < BLACKLIST_THRESHOLDS['max_total_r'] and 
                 winrate < BLACKLIST_THRESHOLDS['max_winrate']) or
                avg_r < BLACKLIST_THRESHOLDS['max_avg_r'] or
                disallow_rate > BLACKLIST_THRESHOLDS['min_disallow_rate'] or
                (features['has_trade_data'] and stability < BLACKLIST_THRESHOLDS['max_stability'])
            ):
                self.symbol_classifications['BLACKLIST'].append(symbol)
                continue
            
            # CORE: Top performers
            if (
                stability >= CORE_THRESHOLDS['min_stability'] and
                quality >= CORE_THRESHOLDS['min_quality'] and
                (not features['has_trade_data'] or winrate >= CORE_THRESHOLDS['min_winrate']) and
                disallow_rate <= CORE_THRESHOLDS['max_disallow_rate'] and
                (not features['has_trade_data'] or avg_r >= CORE_THRESHOLDS['min_avg_r'])
            ):
                self.symbol_classifications['CORE'].append(symbol)
                continue
            
            # EXPANSION: Decent performers
            if (
                stability >= EXPANSION_THRESHOLDS['min_stability'] and
                quality >= EXPANSION_THRESHOLDS['min_quality'] and
                (not features['has_trade_data'] or winrate >= EXPANSION_THRESHOLDS['min_winrate']) and
                disallow_rate <= EXPANSION_THRESHOLDS['max_disallow_rate'] and
                (not features['has_trade_data'] or avg_r >= EXPANSION_THRESHOLDS['min_avg_r'])
            ):
                self.symbol_classifications['EXPANSION'].append(symbol)
                continue
            
            # CONDITIONAL: Regime-specific or situational winners
            regime_perf = features.get('regime_performance', {})
            vol_perf = features.get('vol_performance', {})
            
            is_conditional = False
            
            # Good in trending but not ranging
            if regime_perf.get('TRENDING', {}).get('avg_r', 0) > 0.5 and avg_r > 0:
                is_conditional = True
            
            # Good in normal vol but not extreme
            if vol_perf.get('NORMAL', {}).get('avg_r', 0) > 0.5 and avg_r > 0:
                is_conditional = True
            
            # Profitable but inconsistent
            if total_r > 0 and avg_r > 0 and stability < EXPANSION_THRESHOLDS['min_stability']:
                is_conditional = True
            
            if is_conditional:
                self.symbol_classifications['CONDITIONAL'].append(symbol)
            else:
                # Default: insufficient data or marginal performance
                self.symbol_classifications['INSUFFICIENT_DATA'].append(symbol)
        
        # Print summary
        for category, symbols in self.symbol_classifications.items():
            print(f"  {category:20s}: {len(symbols):4d} symbols")
    
    def optimize_universe_size(self):
        """Determine optimal universe size for different profiles"""
        print("\n" + "="*80)
        print("STEP 6: OPTIMIZING UNIVERSE SIZE")
        print("="*80)
        
        # Sort symbols by quality score
        scored_symbols = [
            (symbol, features['quality_score'])
            for symbol, features in self.symbol_features.items()
            if features['quality_score'] > 0
        ]
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        
        # Compute curves
        test_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        
        print("\nPerformance vs Universe Size:")
        print(f"{'Size':>6} | {'Avg Quality':>12} | {'Avg Stability':>14} | {'CORE':>6} | {'EXP':>6} | {'COND':>6} | {'BL':>6}")
        print("-" * 90)
        
        curves = {
            'pnl_vs_size': [],
            'quality_vs_size': [],
            'stability_vs_size': [],
            'cumulative_r_vs_size': []
        }
        
        for size in test_sizes:
            if size > len(scored_symbols):
                break
            
            subset_symbols = [s for s, _ in scored_symbols[:size]]
            subset_features = [self.symbol_features[s] for s in subset_symbols]
            
            avg_quality = statistics.mean([f['quality_score'] for f in subset_features])
            avg_stability = statistics.mean([f['stability_score'] for f in subset_features])
            
            # Count classifications in subset
            cats = Counter()
            for s in subset_symbols:
                for cat, symbols in self.symbol_classifications.items():
                    if s in symbols:
                        cats[cat] += 1
                        break
            
            # Compute aggregate metrics
            total_r = sum([f['total_r'] for f in subset_features if f['has_trade_data']])
            
            print(f"{size:6d} | {avg_quality:12.4f} | {avg_stability:14.4f} | "
                  f"{cats['CORE']:6d} | {cats['EXPANSION']:6d} | "
                  f"{cats['CONDITIONAL']:6d} | {cats['BLACKLIST']:6d}")
            
            curves['quality_vs_size'].append({'size': size, 'avg_quality': avg_quality})
            curves['stability_vs_size'].append({'size': size, 'avg_stability': avg_stability})
            curves['cumulative_r_vs_size'].append({'size': size, 'total_r': total_r})
        
        return curves
    
    def generate_recommendations(self, curves: Dict) -> Dict:
        """Generate universe configuration recommendations"""
        print("\n" + "="*80)
        print("STEP 7: GENERATING RECOMMENDATIONS")
        print("="*80)
        
        core_count = len(self.symbol_classifications['CORE'])
        expansion_count = len(self.symbol_classifications['EXPANSION'])
        conditional_count = len(self.symbol_classifications['CONDITIONAL'])
        blacklist_count = len(self.symbol_classifications['BLACKLIST'])
        
        # SAFE profile
        safe_size = min(200, core_count + expansion_count // 2)
        safe_size = max(50, safe_size)  # At least 50
        
        # AGGRESSIVE profile
        aggressive_size = min(400, core_count + expansion_count + conditional_count // 2)
        aggressive_size = max(200, aggressive_size)  # At least 200
        
        # EXPERIMENTAL profile
        experimental_size = min(600, core_count + expansion_count + conditional_count)
        experimental_size = max(300, experimental_size)  # At least 300
        
        recommendations = {
            'SAFE': {
                'recommended_size': safe_size,
                'description': 'Conservative - Real money / Mainnet',
                'include': sorted(
                    self.symbol_classifications['CORE'] +
                    self.symbol_classifications['EXPANSION'][:max(0, safe_size - core_count)]
                ),
                'exclude': sorted(
                    self.symbol_classifications['BLACKLIST'] +
                    self.symbol_classifications['CONDITIONAL']
                ),
                'qt_universe': 'custom',
                'qt_max_symbols': safe_size,
            },
            'AGGRESSIVE': {
                'recommended_size': aggressive_size,
                'description': 'Balanced - Testnet / Training',
                'include': sorted(
                    self.symbol_classifications['CORE'] +
                    self.symbol_classifications['EXPANSION'] +
                    self.symbol_classifications['CONDITIONAL'][:max(0, aggressive_size - core_count - expansion_count)]
                ),
                'exclude': sorted(self.symbol_classifications['BLACKLIST']),
                'qt_universe': 'l1l2-top',
                'qt_max_symbols': aggressive_size,
            },
            'EXPERIMENTAL': {
                'recommended_size': experimental_size,
                'description': 'Aggressive - Maximum diversity',
                'include': sorted(
                    self.symbol_classifications['CORE'] +
                    self.symbol_classifications['EXPANSION'] +
                    self.symbol_classifications['CONDITIONAL']
                ),
                'exclude': sorted(self.symbol_classifications['BLACKLIST']),
                'qt_universe': 'all-usdt',
                'qt_max_symbols': experimental_size,
            }
        }
        
        print(f"\nRECOMMENDATIONS:")
        print(f"  SAFE:         {safe_size} symbols (CORE + top EXPANSION)")
        print(f"  AGGRESSIVE:   {aggressive_size} symbols (CORE + EXPANSION + some CONDITIONAL)")
        print(f"  EXPERIMENTAL: {experimental_size} symbols (All except BLACKLIST)")
        
        return recommendations
    
    def compute_deltas(self, recommendations: Dict) -> Dict:
        """Compute add/remove deltas relative to current universe"""
        print("\n" + "="*80)
        print("STEP 8: COMPUTING DELTAS")
        print("="*80)
        
        current_set = set(self.current_symbols)
        
        deltas = {}
        
        for profile_name, profile in recommendations.items():
            recommended_set = set(profile['include'])
            
            to_add = list(recommended_set - current_set)
            to_remove = list(current_set - recommended_set)
            to_keep = list(current_set & recommended_set)
            
            deltas[profile_name] = {
                'to_add': sorted(to_add),
                'to_remove': sorted(to_remove),
                'to_keep': sorted(to_keep),
                'add_count': len(to_add),
                'remove_count': len(to_remove),
                'keep_count': len(to_keep),
            }
            
            print(f"\n{profile_name}:")
            print(f"  Add:    {len(to_add):4d} symbols")
            print(f"  Remove: {len(to_remove):4d} symbols")
            print(f"  Keep:   {len(to_keep):4d} symbols")
        
        return deltas
    
    def generate_output(self, curves: Dict, recommendations: Dict, deltas: Dict):
        """Generate final output JSON"""
        print("\n" + "="*80)
        print("STEP 9: GENERATING OUTPUT")
        print("="*80)
        
        output = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'agent_version': '1.0',
            'data_confidence': self._assess_data_confidence(),
            
            'market_state': {
                'regime': self.current_regime,
                'volatility_level': self.current_vol_level,
            },
            
            'current_universe': {
                'mode': self.universe_snapshot.get('mode'),
                'symbol_count': len(self.current_symbols),
                'generated_at': self.universe_snapshot.get('generated_at'),
            },
            
            'classifications': {
                'CORE': {
                    'count': len(self.symbol_classifications['CORE']),
                    'symbols': sorted(self.symbol_classifications['CORE']),
                    'description': 'Top performers - stable, profitable, reliable'
                },
                'EXPANSION': {
                    'count': len(self.symbol_classifications['EXPANSION']),
                    'symbols': sorted(self.symbol_classifications['EXPANSION']),
                    'description': 'Good performers - profitable with acceptable stability'
                },
                'CONDITIONAL': {
                    'count': len(self.symbol_classifications['CONDITIONAL']),
                    'symbols': sorted(self.symbol_classifications['CONDITIONAL']),
                    'description': 'Situational - good in specific regimes/conditions'
                },
                'BLACKLIST': {
                    'count': len(self.symbol_classifications['BLACKLIST']),
                    'symbols': sorted(self.symbol_classifications['BLACKLIST']),
                    'description': 'Poor performers - exclude from trading'
                },
                'INSUFFICIENT_DATA': {
                    'count': len(self.symbol_classifications['INSUFFICIENT_DATA']),
                    'symbols': sorted(self.symbol_classifications['INSUFFICIENT_DATA']),
                    'description': 'Not enough data for classification'
                },
            },
            
            'recommendations': recommendations,
            
            'deltas': deltas,
            
            'performance_curves': curves,
            
            'symbol_scores': {
                symbol: {
                    'quality_score': round(features['quality_score'], 4),
                    'stability_score': round(features['stability_score'], 4),
                    'profitability_score': round(features['profitability_score'], 4),
                    'reliability_score': round(features['reliability_score'], 4),
                    'winrate': round(features['winrate'], 4),
                    'avg_r': round(features['avg_r'], 4),
                    'total_r': round(features['total_r'], 4),
                    'allow_rate': round(features['allow_rate'], 4),
                    'total_signals': features['total_signals'],
                    'total_trades': features['total_trades'],
                }
                for symbol, features in self.symbol_features.items()
                if features['total_signals'] >= MIN_SIGNALS_FOR_CLASSIFICATION
            },
            
            'summary': {
                'total_symbols_analyzed': len(self.current_symbols),
                'symbols_with_data': len([f for f in self.symbol_features.values() if f['has_signal_data']]),
                'symbols_with_trades': len([f for f in self.symbol_features.values() if f['has_trade_data']]),
                'core_count': len(self.symbol_classifications['CORE']),
                'expansion_count': len(self.symbol_classifications['EXPANSION']),
                'conditional_count': len(self.symbol_classifications['CONDITIONAL']),
                'blacklist_count': len(self.symbol_classifications['BLACKLIST']),
                'insufficient_data_count': len(self.symbol_classifications['INSUFFICIENT_DATA']),
            }
        }
        
        # Save to file
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Output saved to: {OUTPUT_FILE}")
        
        return output
    
    def _assess_data_confidence(self) -> str:
        """Assess overall confidence in data quality"""
        total_signals = sum([f['total_signals'] for f in self.symbol_features.values()])
        symbols_with_trades = len([f for f in self.symbol_features.values() if f['has_trade_data']])
        
        if total_signals < 1000 or symbols_with_trades < 20:
            return 'LOW'
        elif total_signals < 5000 or symbols_with_trades < 50:
            return 'MEDIUM'
        elif total_signals < 10000 or symbols_with_trades < 100:
            return 'HIGH'
        else:
            return 'VERY_HIGH'
    
    def print_summary(self, output: Dict):
        """Print executive summary"""
        print("\n" + "="*80)
        print("UNIVERSE SELECTOR AGENT — EXECUTIVE SUMMARY")
        print("="*80)
        
        print(f"\nData Confidence: {output['data_confidence']}")
        print(f"Market Regime: {output['market_state']['regime']}")
        print(f"Volatility Level: {output['market_state']['volatility_level']}")
        
        print(f"\nCurrent Universe: {output['current_universe']['symbol_count']} symbols")
        
        print(f"\nClassifications:")
        for cat, data in output['classifications'].items():
            print(f"  {cat:20s}: {data['count']:4d} symbols")
        
        print(f"\nRecommended Profiles:")
        for profile, rec in output['recommendations'].items():
            print(f"  {profile:15s}: {rec['recommended_size']:4d} symbols — {rec['description']}")
        
        print(f"\nTop 10 Symbols (by quality score):")
        sorted_symbols = sorted(
            output['symbol_scores'].items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )
        for i, (symbol, scores) in enumerate(sorted_symbols[:10], 1):
            print(f"  {i:2d}. {symbol:15s} | Quality: {scores['quality_score']:.4f} | "
                  f"Stability: {scores['stability_score']:.4f} | R: {scores['avg_r']:+.3f}")
        
        print(f"\nBottom 10 Symbols (by quality score):")
        for i, (symbol, scores) in enumerate(sorted_symbols[-10:], 1):
            print(f"  {i:2d}. {symbol:15s} | Quality: {scores['quality_score']:.4f} | "
                  f"Stability: {scores['stability_score']:.4f} | R: {scores['avg_r']:+.3f}")
        
        print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("UNIVERSE SELECTOR AGENT — AUTONOMOUS UNIVERSE OPTIMIZATION")
    print("="*80)
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    
    agent = UniverseSelectorAgent()
    
    # Step 1: Load data
    if not agent.load_universe_snapshot():
        print("\n✗ Failed to load universe snapshot. Exiting.")
        return 1
    
    if not agent.load_signal_data():
        print("\n✗ Failed to load signal data. Exiting.")
        return 1
    
    agent.load_trade_data()  # Optional - proceed even if no trade data
    
    # Step 2: Compute features
    agent.compute_symbol_features()
    
    # Step 3: Classify symbols
    agent.classify_symbols()
    
    # Step 4: Optimize universe size
    curves = agent.optimize_universe_size()
    
    # Step 5: Generate recommendations
    recommendations = agent.generate_recommendations(curves)
    
    # Step 6: Compute deltas
    deltas = agent.compute_deltas(recommendations)
    
    # Step 7: Generate output
    output = agent.generate_output(curves, recommendations, deltas)
    
    # Step 8: Print summary
    agent.print_summary(output)
    
    print(f"\n✓ Agent execution complete: {datetime.utcnow().isoformat()}Z")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
