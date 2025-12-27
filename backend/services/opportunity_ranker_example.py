"""
Example usage and fake implementations for OpportunityRanker.

Demonstrates the complete workflow with in-memory fake services.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from opportunity_ranker import OpportunityRanker, SymbolMetrics


# ============================================================================
# FAKE IMPLEMENTATIONS
# ============================================================================

class FakeMarketDataClient:
    """Fake market data client for testing."""
    
    def __init__(self):
        self.symbol_configs = {
            "BTCUSDT": {"volatility": "optimal", "trend": "strong_up", "liquidity": "high"},
            "ETHUSDT": {"volatility": "optimal", "trend": "up", "liquidity": "high"},
            "SOLUSDT": {"volatility": "high", "trend": "strong_up", "liquidity": "medium"},
            "AVAXUSDT": {"volatility": "optimal", "trend": "neutral", "liquidity": "medium"},
            "XRPUSDT": {"volatility": "low", "trend": "down", "liquidity": "low"},
        }
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Generate fake candle data based on symbol configuration."""
        config = self.symbol_configs.get(symbol, {"volatility": "optimal", "trend": "neutral", "liquidity": "medium"})
        
        # Base price
        base_price = 50000 if symbol == "BTCUSDT" else 3000 if symbol == "ETHUSDT" else 100
        
        # Generate timestamps
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=limit)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=limit)
        
        # Trend configuration
        if config["trend"] == "strong_up":
            trend = np.linspace(0, base_price * 0.15, limit)  # 15% uptrend
        elif config["trend"] == "up":
            trend = np.linspace(0, base_price * 0.08, limit)  # 8% uptrend
        elif config["trend"] == "down":
            trend = np.linspace(0, -base_price * 0.05, limit)  # 5% downtrend
        else:
            trend = np.zeros(limit)  # Neutral
        
        # Volatility configuration
        if config["volatility"] == "high":
            noise = np.random.normal(0, base_price * 0.04, limit)  # 4% std
        elif config["volatility"] == "low":
            noise = np.random.normal(0, base_price * 0.008, limit)  # 0.8% std
        else:
            noise = np.random.normal(0, base_price * 0.02, limit)  # 2% std (optimal)
        
        # Generate prices
        close = base_price + trend + noise
        
        # Generate OHLC with realistic relationships
        high = close + np.abs(np.random.normal(0, base_price * 0.005, limit))
        low = close - np.abs(np.random.normal(0, base_price * 0.005, limit))
        open_ = close + np.random.normal(0, base_price * 0.003, limit)
        
        # Volume based on liquidity
        if config["liquidity"] == "high":
            volume = np.random.uniform(1000, 5000, limit)
        elif config["liquidity"] == "medium":
            volume = np.random.uniform(300, 1500, limit)
        else:
            volume = np.random.uniform(50, 300, limit)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        return df
    
    def get_spread(self, symbol: str) -> float:
        """Return fake spread based on liquidity."""
        config = self.symbol_configs.get(symbol, {"liquidity": "medium"})
        
        if config["liquidity"] == "high":
            return 0.0003  # 3 basis points (excellent)
        elif config["liquidity"] == "medium":
            return 0.0008  # 8 basis points
        else:
            return 0.0025  # 25 basis points (poor)
    
    def get_liquidity(self, symbol: str) -> float:
        """Return fake 24h volume in USD."""
        config = self.symbol_configs.get(symbol, {"liquidity": "medium"})
        
        if config["liquidity"] == "high":
            return 500_000_000  # $500M
        elif config["liquidity"] == "medium":
            return 50_000_000  # $50M
        else:
            return 5_000_000  # $5M


class FakeTradeLogRepository:
    """Fake trade log repository for testing."""
    
    def __init__(self):
        self.symbol_winrates = {
            "BTCUSDT": 0.68,  # 68% winrate
            "ETHUSDT": 0.62,
            "SOLUSDT": 0.55,
            "AVAXUSDT": 0.48,
            "XRPUSDT": 0.42,
        }
    
    def get_symbol_winrate(self, symbol: str, last_n: int = 200) -> float:
        """Return fake historical winrate."""
        return self.symbol_winrates.get(symbol, 0.50)


class FakeRegimeDetector:
    """Fake regime detector for testing."""
    
    def __init__(self, regime: str = "BULL"):
        self.regime = regime
    
    def get_global_regime(self) -> str:
        """Return configured regime."""
        return self.regime


class InMemoryOpportunityStore:
    """In-memory opportunity store for testing."""
    
    def __init__(self):
        self.rankings: Dict[str, float] = {}
        self.last_update: datetime = None
    
    def update(self, rankings: Dict[str, float]) -> None:
        """Store rankings."""
        self.rankings = rankings
        self.last_update = datetime.utcnow()
        print(f"\n✅ Opportunity Store Updated at {self.last_update.strftime('%H:%M:%S')}")
        print(f"   Stored {len(rankings)} symbols")
    
    def get(self) -> Dict[str, float]:
        """Retrieve rankings."""
        return self.rankings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_basic_example():
    """Basic example: Compute rankings for 5 symbols."""
    print("\n" + "="*70)
    print("OPPORTUNITY RANKER - BASIC EXAMPLE")
    print("="*70)
    
    # Initialize fake dependencies
    market_data = FakeMarketDataClient()
    trade_logs = FakeTradeLogRepository()
    regime_detector = FakeRegimeDetector(regime="BULL")
    opportunity_store = InMemoryOpportunityStore()
    
    # Create OpportunityRanker
    ranker = OpportunityRanker(
        market_data=market_data,
        trade_logs=trade_logs,
        regime_detector=regime_detector,
        opportunity_store=opportunity_store,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"],
        timeframe="1h",
        candle_limit=200,
        min_score_threshold=0.0,  # Show all symbols
    )
    
    # Compute rankings
    rankings = ranker.update_rankings()
    
    # Display results
    print("\n📊 OPPORTUNITY RANKINGS:")
    print("-" * 70)
    for rank, (symbol, score) in enumerate(rankings.items(), 1):
        bar = "█" * int(score * 30)
        print(f"  {rank}. {symbol:12s} {score:.3f} {bar}")
    
    print("\n🏆 TOP 3 SYMBOLS:")
    top_3 = ranker.get_top_n(n=3)
    for i, symbol in enumerate(top_3, 1):
        print(f"  {i}. {symbol} (score: {rankings[symbol]:.3f})")
    
    return ranker, rankings


def run_detailed_analysis():
    """Detailed example: Show individual metric breakdown."""
    print("\n" + "="*70)
    print("OPPORTUNITY RANKER - DETAILED METRICS ANALYSIS")
    print("="*70)
    
    # Initialize
    market_data = FakeMarketDataClient()
    trade_logs = FakeTradeLogRepository()
    regime_detector = FakeRegimeDetector(regime="BULL")
    opportunity_store = InMemoryOpportunityStore()
    
    ranker = OpportunityRanker(
        market_data=market_data,
        trade_logs=trade_logs,
        regime_detector=regime_detector,
        opportunity_store=opportunity_store,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframe="1h",
        candle_limit=200,
    )
    
    # Compute detailed metrics
    metrics_dict = ranker.compute_symbol_scores()
    
    # Display detailed breakdown
    for symbol, metrics in sorted(metrics_dict.items(), key=lambda x: x[1].final_score, reverse=True):
        print(f"\n{'─'*70}")
        print(f"📈 {symbol}")
        print(f"{'─'*70}")
        print(f"  Trend Strength:       {metrics.trend_strength:.3f}  {'█' * int(metrics.trend_strength * 20)}")
        print(f"  Volatility Quality:   {metrics.volatility_quality:.3f}  {'█' * int(metrics.volatility_quality * 20)}")
        print(f"  Liquidity Score:      {metrics.liquidity_score:.3f}  {'█' * int(metrics.liquidity_score * 20)}")
        print(f"  Spread Score:         {metrics.spread_score:.3f}  {'█' * int(metrics.spread_score * 20)}")
        print(f"  Symbol Winrate:       {metrics.symbol_winrate_score:.3f}  {'█' * int(metrics.symbol_winrate_score * 20)}")
        print(f"  Regime Score:         {metrics.regime_score:.3f}  {'█' * int(metrics.regime_score * 20)}")
        print(f"  Noise Score:          {metrics.noise_score:.3f}  {'█' * int(metrics.noise_score * 20)}")
        print(f"  {'─'*70}")
        print(f"  FINAL SCORE:          {metrics.final_score:.3f}  {'█' * int(metrics.final_score * 20)}")
    
    return metrics_dict


def run_regime_comparison():
    """Compare rankings across different market regimes."""
    print("\n" + "="*70)
    print("OPPORTUNITY RANKER - REGIME COMPARISON")
    print("="*70)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]
    regimes = ["BULL", "BEAR", "CHOPPY"]
    
    results = {}
    
    for regime in regimes:
        print(f"\n🌍 Global Regime: {regime}")
        print("-" * 50)
        
        market_data = FakeMarketDataClient()
        trade_logs = FakeTradeLogRepository()
        regime_detector = FakeRegimeDetector(regime=regime)
        opportunity_store = InMemoryOpportunityStore()
        
        ranker = OpportunityRanker(
            market_data=market_data,
            trade_logs=trade_logs,
            regime_detector=regime_detector,
            opportunity_store=opportunity_store,
            symbols=symbols,
            min_score_threshold=0.0,
        )
        
        rankings = ranker.update_rankings()
        results[regime] = rankings
        
        # Show top 3
        for rank, (symbol, score) in enumerate(list(rankings.items())[:3], 1):
            print(f"  {rank}. {symbol:12s} {score:.3f}")
    
    # Show how rankings change
    print("\n📊 RANKING CHANGES BY REGIME:")
    print("-" * 70)
    print(f"{'Symbol':<12} {'BULL':<12} {'BEAR':<12} {'CHOPPY':<12}")
    print("-" * 70)
    for symbol in symbols:
        bull_score = results["BULL"][symbol]
        bear_score = results["BEAR"][symbol]
        choppy_score = results["CHOPPY"][symbol]
        print(f"{symbol:<12} {bull_score:.3f}        {bear_score:.3f}        {choppy_score:.3f}")
    
    return results


def run_custom_weights_example():
    """Example with custom metric weights."""
    print("\n" + "="*70)
    print("OPPORTUNITY RANKER - CUSTOM WEIGHTS")
    print("="*70)
    
    # Define custom weights (prioritize trend and liquidity)
    custom_weights = {
        'trend_strength': 0.40,      # Increased from 0.25
        'liquidity_score': 0.25,     # Increased from 0.15
        'volatility_quality': 0.15,  # Decreased from 0.20
        'regime_score': 0.10,        # Decreased from 0.15
        'symbol_winrate_score': 0.05,
        'spread_score': 0.03,
        'noise_score': 0.02,
    }
    
    print("\n⚖️  Custom Weights:")
    for metric, weight in custom_weights.items():
        print(f"  {metric:<25s} {weight:.2%}")
    
    # Compare default vs custom
    market_data = FakeMarketDataClient()
    trade_logs = FakeTradeLogRepository()
    regime_detector = FakeRegimeDetector()
    
    print("\n📊 DEFAULT WEIGHTS:")
    print("-" * 50)
    store_default = InMemoryOpportunityStore()
    ranker_default = OpportunityRanker(
        market_data=market_data,
        trade_logs=trade_logs,
        regime_detector=regime_detector,
        opportunity_store=store_default,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    rankings_default = ranker_default.update_rankings()
    for symbol, score in rankings_default.items():
        print(f"  {symbol:12s} {score:.3f}")
    
    print("\n📊 CUSTOM WEIGHTS:")
    print("-" * 50)
    store_custom = InMemoryOpportunityStore()
    ranker_custom = OpportunityRanker(
        market_data=market_data,
        trade_logs=trade_logs,
        regime_detector=regime_detector,
        opportunity_store=store_custom,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        weights=custom_weights,
    )
    rankings_custom = ranker_custom.update_rankings()
    for symbol, score in rankings_custom.items():
        print(f"  {symbol:12s} {score:.3f}")
    
    return rankings_default, rankings_custom


def run_threshold_filtering_example():
    """Example showing threshold filtering."""
    print("\n" + "="*70)
    print("OPPORTUNITY RANKER - THRESHOLD FILTERING")
    print("="*70)
    
    market_data = FakeMarketDataClient()
    trade_logs = FakeTradeLogRepository()
    regime_detector = FakeRegimeDetector()
    opportunity_store = InMemoryOpportunityStore()
    
    # Test different thresholds
    thresholds = [0.0, 0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        print(f"\n🎯 Minimum Score Threshold: {threshold:.1f}")
        print("-" * 50)
        
        ranker = OpportunityRanker(
            market_data=market_data,
            trade_logs=trade_logs,
            regime_detector=regime_detector,
            opportunity_store=opportunity_store,
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"],
            min_score_threshold=threshold,
        )
        
        rankings = ranker.update_rankings()
        
        if rankings:
            print(f"  Passed: {len(rankings)} symbols")
            for symbol, score in rankings.items():
                print(f"    {symbol:12s} {score:.3f}")
        else:
            print("  ❌ No symbols passed threshold")


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 QUANTUM TRADER - OPPORTUNITY RANKER EXAMPLES")
    print("="*70)
    
    # Run all examples
    print("\n\n[1/5] Running basic example...")
    ranker, rankings = run_basic_example()
    
    print("\n\n[2/5] Running detailed analysis...")
    metrics = run_detailed_analysis()
    
    print("\n\n[3/5] Running regime comparison...")
    regime_results = run_regime_comparison()
    
    print("\n\n[4/5] Running custom weights example...")
    default, custom = run_custom_weights_example()
    
    print("\n\n[5/5] Running threshold filtering...")
    run_threshold_filtering_example()
    
    print("\n\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nThe OpportunityRanker is ready for integration into Quantum Trader.")
    print("\nNext steps:")
    print("  1. Wire into your FastAPI backend")
    print("  2. Connect to real MarketDataClient (Binance/CCXT)")
    print("  3. Connect to your TradeLogRepository")
    print("  4. Connect to RegimeDetector service")
    print("  5. Implement OpportunityStore (Redis/PostgreSQL)")
    print("  6. Schedule periodic ranking updates (every 5-15 minutes)")
    print("  7. Expose rankings via REST API endpoint")
    print("  8. Integrate with Orchestrator and Strategy Engine")
