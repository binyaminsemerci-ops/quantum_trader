"""
UNIVERSE ANALYZER — Optimal Trading Universe Size & Symbol Selection

Determines:
- Optimal universe size
- Symbol performance classification
- Which symbols to prioritize, exclude, or conditionally filter
- Recommended configurations for SAFE, AGGRESSIVE, and SCALP profiles
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any
import statistics

# Data paths
UNIVERSE_SNAPSHOT = "/app/data/universe_snapshot.json"
POLICY_OBSERVATIONS = "/app/data/policy_observations"
TRADES_PATH = "/app/data/trades"
OUTPUT_DIR = "/app/data/analysis"

# Major coins that should always be prioritized
MAJORS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "TRXUSDT"
}

# Known high-slippage or low-liquidity symbols
KNOWN_PROBLEMATIC = {
    "PERLAUSDT", "CTXCUSDT", "VITEUSDT", "PUNDIXUSDT", "REQUSDT",
    "DGBUSDT", "SCUSDT", "SYSUSDT"
}


class UniverseAnalyzer:
    """Comprehensive universe performance analysis"""
    
    def __init__(self):
        self.universe_config = {}
        self.symbols = []
        self.signal_data = defaultdict(lambda: {
            'total_signals': 0,
            'allowed_signals': 0,
            'blocked_signals': 0,
            'confidences': [],
            'actions': Counter(),
            'policy_verdicts': Counter(),
            'agreements': 0,
            'disagreements': 0,
            'consensus_types': Counter(),
        })
        self.symbol_metrics = {}
        
    def load_universe_snapshot(self):
        """Load universe configuration"""
        print("\n" + "="*80)
        print("STEP 1: LOADING UNIVERSE SNAPSHOT")
        print("="*80)
        
        try:
            with open(UNIVERSE_SNAPSHOT, 'r') as f:
                self.universe_config = json.load(f)
            
            self.symbols = self.universe_config.get('symbols', [])
            
            print(f"✓ Universe mode: {self.universe_config.get('mode')}")
            print(f"✓ Symbol count: {self.universe_config.get('symbol_count')}")
            print(f"✓ QT_MAX_SYMBOLS: {self.universe_config.get('qt_max_symbols')}")
            print(f"✓ Generated at: {self.universe_config.get('generated_at')}")
            
            # Count majors in universe
            majors_in_universe = [s for s in self.symbols if s in MAJORS]
            print(f"✓ Majors in universe: {len(majors_in_universe)}/{len(MAJORS)}")
            
            return True
            
        except FileNotFoundError:
            print(f"✗ Universe snapshot not found at {UNIVERSE_SNAPSHOT}")
            return False
    
    def load_signal_data(self):
        """Load and parse signal decision logs"""
        print("\n" + "="*80)
        print("STEP 2: LOADING SIGNAL DATA")
        print("="*80)
        
        signal_files = list(Path(POLICY_OBSERVATIONS).glob("signals_*.jsonl"))
        
        if not signal_files:
            print(f"✗ No signal files found in {POLICY_OBSERVATIONS}")
            return False
        
        total_records = 0
        
        for signal_file in signal_files:
            print(f"\nProcessing: {signal_file.name}")
            
            with open(signal_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        
                        if record.get('type') != 'signal_decision':
                            continue
                        
                        signal = record.get('signal', {})
                        symbol = signal.get('symbol')
                        
                        if not symbol or symbol not in self.symbols:
                            continue
                        
                        # Extract data
                        action = signal.get('action')
                        confidence = signal.get('confidence', 0)
                        model_info = signal.get('model', {})
                        consensus = model_info.get('consensus', 'unknown')
                        
                        actual_decision = record.get('actual_decision')
                        policy_verdict = record.get('policy_verdict')
                        agreement = record.get('agreement', False)
                        
                        # Store metrics
                        data = self.signal_data[symbol]
                        data['total_signals'] += 1
                        
                        if actual_decision == 'TRADE_ALLOWED':
                            data['allowed_signals'] += 1
                        else:
                            data['blocked_signals'] += 1
                        
                        data['confidences'].append(confidence)
                        data['actions'][action] += 1
                        data['policy_verdicts'][policy_verdict] += 1
                        data['consensus_types'][consensus] += 1
                        
                        if agreement:
                            data['agreements'] += 1
                        else:
                            data['disagreements'] += 1
                        
                        total_records += 1
                        
                    except json.JSONDecodeError:
                        print(f"  Warning: Failed to parse line {line_num}")
                        continue
            
            print(f"  ✓ Processed {total_records} signal records")
        
        print(f"\n✓ Total symbols with signals: {len(self.signal_data)}")
        print(f"✓ Total signal records: {total_records}")
        
        return True
    
    def compute_symbol_metrics(self):
        """Compute performance metrics for each symbol"""
        print("\n" + "="*80)
        print("STEP 3: COMPUTING SYMBOL METRICS")
        print("="*80)
        
        for symbol in self.symbols:
            data = self.signal_data.get(symbol)
            
            if not data or data['total_signals'] == 0:
                # No data for this symbol
                self.symbol_metrics[symbol] = {
                    'total_signals': 0,
                    'allowed_signals': 0,
                    'blocked_signals': 0,
                    'signal_rate': 0,
                    'allow_rate': 0,
                    'avg_confidence': 0,
                    'confidence_std': 0,
                    'consensus_quality': 0,
                    'agreement_rate': 0,
                    'stability_score': 0,
                    'trust_score': 0,
                    'category': 'NO_DATA',
                    'is_major': symbol in MAJORS,
                    'is_problematic': symbol in KNOWN_PROBLEMATIC,
                }
                continue
            
            # Calculate metrics
            confidences = data['confidences']
            avg_conf = statistics.mean(confidences) if confidences else 0
            std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0
            
            allow_rate = data['allowed_signals'] / data['total_signals']
            agreement_rate = data['agreements'] / data['total_signals']
            
            # Consensus quality: ratio of strong/weak consensus
            consensus_counts = data['consensus_types']
            strong_count = consensus_counts.get('strong', 0)
            weak_count = consensus_counts.get('weak', 0)
            total_consensus = strong_count + weak_count
            consensus_quality = strong_count / total_consensus if total_consensus > 0 else 0
            
            # Stability score: high confidence, low variance, high agreement
            stability = (avg_conf * (1 - std_conf) * agreement_rate) if std_conf < 1 else 0
            
            # Trust score: allow rate × consensus quality × stability
            trust_score = allow_rate * consensus_quality * stability
            
            self.symbol_metrics[symbol] = {
                'total_signals': data['total_signals'],
                'allowed_signals': data['allowed_signals'],
                'blocked_signals': data['blocked_signals'],
                'signal_rate': data['total_signals'],  # Relative activity
                'allow_rate': allow_rate,
                'avg_confidence': avg_conf,
                'confidence_std': std_conf,
                'consensus_quality': consensus_quality,
                'agreement_rate': agreement_rate,
                'stability_score': stability,
                'trust_score': trust_score,
                'is_major': symbol in MAJORS,
                'is_problematic': symbol in KNOWN_PROBLEMATIC,
                'category': 'PENDING',  # Will be classified later
            }
        
        print(f"✓ Computed metrics for {len(self.symbol_metrics)} symbols")
        
        # Show top performers
        sorted_by_trust = sorted(
            self.symbol_metrics.items(),
            key=lambda x: x[1]['trust_score'],
            reverse=True
        )
        
        print("\nTop 20 symbols by trust score:")
        for i, (symbol, metrics) in enumerate(sorted_by_trust[:20], 1):
            print(f"  {i:2d}. {symbol:15s} | Trust: {metrics['trust_score']:.4f} | "
                  f"Allow: {metrics['allow_rate']:.2%} | Conf: {metrics['avg_confidence']:.3f}")
        
        print("\nBottom 20 symbols by trust score:")
        for i, (symbol, metrics) in enumerate(sorted_by_trust[-20:], 1):
            print(f"  {i:2d}. {symbol:15s} | Trust: {metrics['trust_score']:.4f} | "
                  f"Allow: {metrics['allow_rate']:.2%} | Conf: {metrics['avg_confidence']:.3f}")
    
    def classify_symbols(self):
        """Classify symbols into performance buckets"""
        print("\n" + "="*80)
        print("STEP 4: SYMBOL CLASSIFICATION")
        print("="*80)
        
        classifications = {
            'CORE': [],           # High-quality, stable performers
            'EXPANSION': [],      # Profitable but less stable
            'CONDITIONAL': [],    # Situationally good
            'BLACKLIST': [],      # Poor performers
            'NO_DATA': [],        # Insufficient data
        }
        
        for symbol, metrics in self.symbol_metrics.items():
            
            # No data symbols
            if metrics['total_signals'] < 5:
                metrics['category'] = 'NO_DATA'
                classifications['NO_DATA'].append(symbol)
                continue
            
            # Known problematic symbols
            if metrics['is_problematic']:
                metrics['category'] = 'BLACKLIST'
                classifications['BLACKLIST'].append(symbol)
                continue
            
            trust = metrics['trust_score']
            allow_rate = metrics['allow_rate']
            confidence = metrics['avg_confidence']
            stability = metrics['stability_score']
            
            # CORE: High trust, high stability, decent activity
            if (trust > 0.15 and stability > 0.20 and 
                allow_rate > 0.5 and confidence > 0.48):
                metrics['category'] = 'CORE'
                classifications['CORE'].append(symbol)
            
            # EXPANSION: Decent performance but less stable
            elif (trust > 0.08 and allow_rate > 0.4 and confidence > 0.45):
                metrics['category'] = 'EXPANSION'
                classifications['EXPANSION'].append(symbol)
            
            # CONDITIONAL: Low trust but not terrible
            elif allow_rate > 0.3 and confidence > 0.42:
                metrics['category'] = 'CONDITIONAL'
                classifications['CONDITIONAL'].append(symbol)
            
            # BLACKLIST: Poor metrics
            else:
                metrics['category'] = 'BLACKLIST'
                classifications['BLACKLIST'].append(symbol)
        
        # Force majors into CORE (unless truly terrible)
        for symbol in MAJORS:
            if symbol in self.symbols:
                metrics = self.symbol_metrics[symbol]
                if metrics['category'] not in ['CORE', 'EXPANSION']:
                    if metrics['total_signals'] >= 5:
                        # Promote majors unless they're truly awful
                        if metrics['allow_rate'] > 0.2:
                            metrics['category'] = 'CORE'
                            if symbol in classifications['BLACKLIST']:
                                classifications['BLACKLIST'].remove(symbol)
                            if symbol in classifications['CONDITIONAL']:
                                classifications['CONDITIONAL'].remove(symbol)
                            if symbol not in classifications['CORE']:
                                classifications['CORE'].append(symbol)
        
        # Print classification summary
        for category, symbols in classifications.items():
            print(f"\n{category}: {len(symbols)} symbols")
            if symbols and len(symbols) <= 30:
                print(f"  {', '.join(sorted(symbols)[:30])}")
        
        return classifications
    
    def optimize_universe_size(self, classifications):
        """Determine optimal universe size through performance curves"""
        print("\n" + "="*80)
        print("STEP 5: UNIVERSE SIZE OPTIMIZATION")
        print("="*80)
        
        # Sort symbols by trust score
        sorted_symbols = sorted(
            self.symbol_metrics.items(),
            key=lambda x: x[1]['trust_score'],
            reverse=True
        )
        
        # Compute cumulative metrics at different universe sizes
        test_sizes = [50, 100, 150, 200, 250, 300, 400, 500]
        
        print("\nPerformance vs Universe Size:")
        print(f"{'Size':>6} | {'Avg Trust':>10} | {'Avg Allow':>10} | {'Avg Conf':>10} | {'CORE':>6} | {'EXP':>6} | {'COND':>6} | {'BL':>6}")
        print("-" * 90)
        
        results = {}
        
        for size in test_sizes:
            if size > len(sorted_symbols):
                break
            
            subset = sorted_symbols[:size]
            
            avg_trust = statistics.mean([m['trust_score'] for s, m in subset])
            avg_allow = statistics.mean([m['allow_rate'] for s, m in subset])
            avg_conf = statistics.mean([m['avg_confidence'] for s, m in subset])
            
            # Count categories
            category_counts = Counter([m['category'] for s, m in subset])
            
            print(f"{size:6d} | {avg_trust:10.4f} | {avg_allow:10.2%} | {avg_conf:10.3f} | "
                  f"{category_counts['CORE']:6d} | {category_counts['EXPANSION']:6d} | "
                  f"{category_counts['CONDITIONAL']:6d} | {category_counts['BLACKLIST']:6d}")
            
            results[size] = {
                'avg_trust': avg_trust,
                'avg_allow': avg_allow,
                'avg_conf': avg_conf,
                'categories': dict(category_counts),
            }
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        core_count = len(classifications['CORE'])
        expansion_count = len(classifications['EXPANSION'])
        
        print(f"\n1. SAFE PROFILE (Real Money / Mainnet):")
        safe_size = min(200, core_count + expansion_count)
        print(f"   → Universe size: {safe_size}")
        print(f"   → Include: CORE ({core_count}) + Top EXPANSION")
        print(f"   → Exclude: BLACKLIST, CONDITIONAL")
        
        print(f"\n2. AGGRESSIVE PROFILE (Testnet / High Risk):")
        aggressive_size = min(400, core_count + expansion_count + len(classifications['CONDITIONAL']))
        print(f"   → Universe size: {aggressive_size}")
        print(f"   → Include: CORE + EXPANSION + CONDITIONAL")
        print(f"   → Exclude: BLACKLIST only")
        
        print(f"\n3. SCALP PROFILE (Ultra-High Frequency):")
        scalp_size = len([s for s in MAJORS if s in self.symbols])
        print(f"   → Universe size: {scalp_size}")
        print(f"   → Include: MAJORS only")
        print(f"   → Focus: Highest liquidity")
        
        return results
    
    def generate_config(self, classifications):
        """Generate JSON configuration for deployment"""
        print("\n" + "="*80)
        print("STEP 6: GENERATING DEPLOYMENT CONFIG")
        print("="*80)
        
        config = {
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "source_universe": {
                "mode": self.universe_config.get('mode'),
                "symbol_count": self.universe_config.get('symbol_count'),
                "generated_at": self.universe_config.get('generated_at'),
            },
            "classifications": {
                category: sorted(symbols)
                for category, symbols in classifications.items()
            },
            "profiles": {
                "SAFE": {
                    "description": "Real money / Mainnet - Conservative, high-quality symbols",
                    "max_symbols": min(200, len(classifications['CORE']) + len(classifications['EXPANSION'])),
                    "include_categories": ["CORE", "EXPANSION"],
                    "exclude_categories": ["BLACKLIST", "CONDITIONAL", "NO_DATA"],
                    "whitelist": sorted(set(classifications['CORE'] + classifications['EXPANSION'][:100])),
                    "blacklist": sorted(classifications['BLACKLIST']),
                    "priority_majors": sorted([s for s in MAJORS if s in self.symbols]),
                },
                "AGGRESSIVE": {
                    "description": "Testnet / High risk - Maximum diversity",
                    "max_symbols": min(400, len(self.symbols) - len(classifications['BLACKLIST'])),
                    "include_categories": ["CORE", "EXPANSION", "CONDITIONAL"],
                    "exclude_categories": ["BLACKLIST"],
                    "whitelist": sorted(set(
                        classifications['CORE'] + 
                        classifications['EXPANSION'] + 
                        classifications['CONDITIONAL']
                    )),
                    "blacklist": sorted(classifications['BLACKLIST']),
                    "priority_majors": sorted([s for s in MAJORS if s in self.symbols]),
                },
                "SCALP": {
                    "description": "Ultra-high frequency - Majors only",
                    "max_symbols": len([s for s in MAJORS if s in self.symbols]),
                    "include_categories": ["CORE"],
                    "exclude_categories": ["EXPANSION", "CONDITIONAL", "BLACKLIST", "NO_DATA"],
                    "whitelist": sorted([s for s in MAJORS if s in self.symbols]),
                    "blacklist": [],
                    "priority_majors": sorted([s for s in MAJORS if s in self.symbols]),
                },
            },
            "symbol_details": {
                symbol: {
                    "category": metrics['category'],
                    "trust_score": round(metrics['trust_score'], 4),
                    "allow_rate": round(metrics['allow_rate'], 4),
                    "avg_confidence": round(metrics['avg_confidence'], 4),
                    "stability_score": round(metrics['stability_score'], 4),
                    "total_signals": metrics['total_signals'],
                    "is_major": metrics['is_major'],
                }
                for symbol, metrics in self.symbol_metrics.items()
            }
        }
        
        # Save to file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = Path(OUTPUT_DIR) / f"universe_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to: {output_file}")
        
        return config
    
    def print_summary(self, classifications, config):
        """Print executive summary"""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        total_symbols = len(self.symbols)
        symbols_with_data = len([m for m in self.symbol_metrics.values() if m['total_signals'] > 0])
        
        print(f"\nUniverse Statistics:")
        print(f"  Total symbols in universe: {total_symbols}")
        print(f"  Symbols with signal data: {symbols_with_data}")
        print(f"  Coverage: {symbols_with_data/total_symbols:.1%}")
        
        print(f"\nClassification Breakdown:")
        for category, symbols in classifications.items():
            pct = len(symbols) / total_symbols * 100
            print(f"  {category:15s}: {len(symbols):4d} ({pct:5.1f}%)")
        
        print(f"\nTop 10 Performers (by trust score):")
        sorted_by_trust = sorted(
            self.symbol_metrics.items(),
            key=lambda x: x[1]['trust_score'],
            reverse=True
        )
        for i, (symbol, metrics) in enumerate(sorted_by_trust[:10], 1):
            print(f"  {i:2d}. {symbol:15s} | {metrics['category']:12s} | "
                  f"Trust: {metrics['trust_score']:.4f} | Major: {metrics['is_major']}")
        
        print(f"\nRecommended Deployment:")
        for profile_name, profile in config['profiles'].items():
            print(f"\n  {profile_name}:")
            print(f"    Max symbols: {profile['max_symbols']}")
            print(f"    Whitelist: {len(profile['whitelist'])} symbols")
            print(f"    Blacklist: {len(profile['blacklist'])} symbols")
            print(f"    Priority majors: {len(profile['priority_majors'])}")
        
        print("\n" + "="*80)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*80)
        
        safe_profile = config['profiles']['SAFE']
        aggressive_profile = config['profiles']['AGGRESSIVE']
        
        print(f"\n🎯 FOR MAINNET (Real Money):")
        print(f"   Set: QT_UNIVERSE=custom")
        print(f"   Set: QT_MAX_SYMBOLS={safe_profile['max_symbols']}")
        print(f"   Use: {len(safe_profile['whitelist'])} CORE + EXPANSION symbols")
        print(f"   Avoid: {len(safe_profile['blacklist'])} blacklisted symbols")
        
        print(f"\n🚀 FOR TESTNET (Training & Testing):")
        print(f"   Set: QT_UNIVERSE=custom")
        print(f"   Set: QT_MAX_SYMBOLS={aggressive_profile['max_symbols']}")
        print(f"   Use: {len(aggressive_profile['whitelist'])} diverse symbols")
        print(f"   Avoid: {len(aggressive_profile['blacklist'])} blacklisted symbols")
        
        print(f"\n⚡ FOR SCALPING:")
        print(f"   Set: QT_UNIVERSE=megacap")
        print(f"   Set: QT_MAX_SYMBOLS=50")
        print(f"   Use: Major coins only")
        
        print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("QUANTUM TRADER — UNIVERSE ANALYZER")
    print("="*80)
    print(f"Analysis started: {datetime.utcnow().isoformat()}Z")
    
    analyzer = UniverseAnalyzer()
    
    # Step 1: Load universe
    if not analyzer.load_universe_snapshot():
        print("\n✗ Failed to load universe snapshot. Exiting.")
        return 1
    
    # Step 2: Load signal data
    if not analyzer.load_signal_data():
        print("\n✗ Failed to load signal data. Exiting.")
        return 1
    
    # Step 3: Compute metrics
    analyzer.compute_symbol_metrics()
    
    # Step 4: Classify symbols
    classifications = analyzer.classify_symbols()
    
    # Step 5: Optimize universe size
    size_analysis = analyzer.optimize_universe_size(classifications)
    
    # Step 6: Generate config
    config = analyzer.generate_config(classifications)
    
    # Step 7: Print summary
    analyzer.print_summary(classifications, config)
    
    print(f"\n✓ Analysis complete: {datetime.utcnow().isoformat()}Z")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
