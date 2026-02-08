#!/usr/bin/env python3
"""
Generate Mock SimpleCLM Data for Calibration Testing

Creates realistic trade data with:
- Mix of wins/losses (realistic ~52% win rate)
- Varying confidence levels
- Model predictions
- Realistic PnL percentages
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def generate_mock_trade(trade_id: int, timestamp: datetime) -> dict:
    """
    Generate a single mock trade with realistic characteristics.
    
    Strategy:
    - 52% win rate (slightly profitable)
    - Confidence calibration bias: models slightly overconfident
    - Ensemble predictions from 4 models
    - Realistic PnL distribution
    """
    # Decide outcome (52% win rate)
    is_win = random.random() < 0.52
    
    # Generate confidence (models tend to be 5-10% overconfident)
    if is_win:
        # For wins, confidence should be 0.55-0.90
        base_confidence = random.uniform(0.55, 0.90)
        # Add overconfidence bias (5-10%)
        predicted_confidence = min(0.95, base_confidence + random.uniform(0.05, 0.10))
    else:
        # For losses, models were confident but wrong
        predicted_confidence = random.uniform(0.60, 0.85)
    
    # Generate entry/exit prices
    entry_price = random.uniform(40000, 45000)  # BTC price range
    
    if is_win:
        # Win: 0.5% to 2.5% gain
        pnl_percent = random.uniform(0.5, 2.5)
        exit_price = entry_price * (1 + pnl_percent / 100)
        outcome_label = "WIN"
    else:
        # Loss: -0.5% to -2.0% loss
        pnl_percent = random.uniform(-2.0, -0.5)
        exit_price = entry_price * (1 + pnl_percent / 100)
        outcome_label = "LOSS"
    
    # Generate model predictions (4 models)
    # Simulate that some models are better than others
    models = {
        'xgb': {
            'weight': 0.30,
            'accuracy': 0.55,  # Slightly below average
        },
        'lgbm': {
            'weight': 0.30,
            'accuracy': 0.58,  # Better performer
        },
        'nhits': {
            'weight': 0.20,
            'accuracy': 0.50,  # Average
        },
        'patchtst': {
            'weight': 0.20,
            'accuracy': 0.51,  # Slightly above average
        }
    }
    
    model_predictions = {}
    actions = ['BUY', 'SELL', 'HOLD']
    
    for model_name, model_info in models.items():
        # Model's action depends on its accuracy
        if random.random() < model_info['accuracy']:
            # Model predicts correctly
            action = 'BUY' if is_win else 'HOLD'
        else:
            # Model predicts incorrectly
            action = random.choice(['HOLD', 'SELL']) if is_win else 'BUY'
        
        model_conf = random.uniform(0.55, 0.85)
        model_predictions[model_name] = {
            'action': action,
            'confidence': round(model_conf, 3)
        }
    
    # Choose symbol (mostly BTC, some ETH)
    symbol = 'BTCUSDT' if random.random() < 0.8 else 'ETHUSDT'
    
    # Build trade record
    trade = {
        'trade_id': f'mock_trade_{trade_id:04d}',
        'timestamp': timestamp.isoformat(),
        'symbol': symbol,
        'action': 'BUY',  # SimpleCLM records executed trades
        'entry_price': round(entry_price, 2),
        'exit_price': round(exit_price, 2),
        'pnl_percent': round(pnl_percent, 3),
        'confidence': round(predicted_confidence, 3),
        'outcome_label': outcome_label,
        'duration_minutes': random.randint(15, 180),
        'model_predictions': model_predictions,
        'meta_agent_used': random.choice([True, False]),
        'arbiter_used': False,
        'market_regime': random.choice(['bullish', 'bearish', 'neutral', 'volatile'])
    }
    
    return trade


def generate_mock_clm_dataset(num_trades: int = 52, output_path: str = None) -> str:
    """
    Generate complete mock CLM dataset.
    
    Args:
        num_trades: Number of trades to generate (default 52, minimum for calibration)
        output_path: Where to save the file
    
    Returns:
        Path to generated file
    """
    if output_path is None:
        output_path = Path(__file__).parent / "mock_clm_trades.jsonl"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🔧 Generating {num_trades} mock trades...")
    
    # Generate trades over last 2 weeks
    end_time = datetime.now()
    start_time = end_time - timedelta(days=14)
    
    trades = []
    for i in range(num_trades):
        # Distribute trades evenly over time period
        trade_time = start_time + (end_time - start_time) * (i / num_trades)
        trade = generate_mock_trade(i + 1, trade_time)
        trades.append(trade)
    
    # Write to JSONL file
    with open(output_path, 'w') as f:
        for trade in trades:
            f.write(json.dumps(trade) + '\n')
    
    # Print statistics
    wins = sum(1 for t in trades if t['outcome_label'] == 'WIN')
    losses = num_trades - wins
    win_rate = wins / num_trades * 100
    avg_confidence = sum(t['confidence'] for t in trades) / num_trades
    avg_pnl = sum(t['pnl_percent'] for t in trades) / num_trades
    
    print(f"\n✅ Mock data generated: {output_path}")
    print(f"\n📊 Statistics:")
    print(f"   Total trades: {num_trades}")
    print(f"   Wins: {wins} ({win_rate:.1f}%)")
    print(f"   Losses: {losses} ({100-win_rate:.1f}%)")
    print(f"   Avg confidence: {avg_confidence:.3f}")
    print(f"   Avg PnL: {avg_pnl:+.2f}%")
    print(f"   Date range: {start_time.date()} to {end_time.date()}")
    print(f"\n🎯 Calibration traits:")
    print(f"   - Models ~7% overconfident (predicted {avg_confidence:.1%} vs {win_rate:.1%} actual)")
    print(f"   - LGBM performs slightly better (should get higher weight)")
    print(f"   - XGB slightly underperforms (should get lower weight)")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock SimpleCLM data for calibration testing"
    )
    parser.add_argument(
        '--count',
        type=int,
        default=52,
        help='Number of trades to generate (default: 52)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: tests/mock_clm_trades.jsonl)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    output_path = generate_mock_clm_dataset(args.count, args.output)
    
    print(f"\n🧪 Test with:")
    print(f"   export CLM_DATA_PATH={output_path}")
    print(f"   python microservices/learning/calibration_cli.py run --force")
    print()


if __name__ == '__main__':
    main()
