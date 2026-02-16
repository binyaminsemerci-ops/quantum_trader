#!/usr/bin/env python3
"""
Feature Calculation V6 - 49 Features
Canonical implementation matching FEATURES_V6 schema

Used by all v6 retraining scripts to ensure consistent feature engineering.
Schema aligned with ai_engine/common_features.py
"""
import numpy as np
import pandas as pd


def calculate_features_v6(df):
    """
    Calculate all 49 features from FEATURES_V6 schema.
    
    Input: DataFrame with OHLCV columns (open, high, low, close, volume)
    Output: DataFrame with 49 feature columns + original OHLCV
    
    Features:
    - Candlestick patterns (11): returns, log_returns, price_range, body_size,
      upper_wick, lower_wick, is_doji, is_hammer, is_engulfing, gap_up, gap_down
    - Oscillators (7): rsi, macd, macd_signal, macd_hist, stoch_k, stoch_d, roc
    - EMAs (8): ema_9, ema_9_dist, ema_21, ema_21_dist, ema_50, ema_50_dist,
      ema_200, ema_200_dist
    - SMAs (2): sma_20, sma_50
    - ADX (3): adx, plus_di, minus_di
    - Bollinger Bands (5): bb_middle, bb_upper, bb_lower, bb_width, bb_position
    - Volatility (3): atr, atr_pct, volatility
    - Volume (5): volume_sma, volume_ratio, obv, obv_ema, vpt
    - Momentum (5): momentum_5, momentum_10, momentum_20, acceleration, relative_spread
    """
    print("[FEATURES-V6] Calculating 49 technical features...")
    
    df = df.copy()
    
    # === CANDLESTICK FEATURES (11) ============================================
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
    
    # Candlestick patterns (binary)
    df['is_doji'] = (df['body_size'] < 0.001).astype(int)
    df['is_hammer'] = ((df['lower_wick'] > 2 * df['body_size']) & 
                       (df['upper_wick'] < df['body_size'])).astype(int)
    df['is_engulfing'] = (df['body_size'] > df['body_size'].shift(1) * 1.5).astype(int)
    
    # Gap detection
    prev_close = df['close'].shift(1)
    df['gap_up'] = ((df['open'] > prev_close) & ((df['open'] - prev_close) / prev_close > 0.002)).astype(int)
    df['gap_down'] = ((df['open'] < prev_close) & ((prev_close - df['open']) / prev_close > 0.002)).astype(int)
    
    # === OSCILLATORS (7) ======================================================
    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Stochastic Oscillator (14-period)
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Rate of Change (10-period)
    df['roc'] = df['close'].pct_change(periods=10) * 100
    
    # === EMAS (8) =============================================================
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # EMA distances (normalized)
    df['ema_9_dist'] = (df['close'] - df['ema_9']) / df['close']
    df['ema_21_dist'] = (df['close'] - df['ema_21']) / df['close']
    df['ema_50_dist'] = (df['close'] - df['ema_50']) / df['close']
    df['ema_200_dist'] = (df['close'] - df['ema_200']) / df['close']
    
    # === SMAS (2) =============================================================
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # === ADX (3) ==============================================================
    # ADX (14-period) - Simplified Wilder's calculation
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    
    # Directional movement
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm_14 = pd.Series(plus_dm, index=df.index).rolling(window=14).mean()
    minus_dm_14 = pd.Series(minus_dm, index=df.index).rolling(window=14).mean()
    
    df['plus_di'] = 100 * (plus_dm_14 / atr_14)
    df['minus_di'] = 100 * (minus_dm_14 / atr_14)
    
    # ADX calculation
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(window=14).mean()
    
    # === BOLLINGER BANDS (5) ==================================================
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # === VOLATILITY (3) =======================================================
    # ATR (14-period) - already calculated above
    df['atr'] = atr_14
    df['atr_pct'] = df['atr'] / df['close']
    
    # Historical volatility (20-period)
    df['volatility'] = df['close'].rolling(window=20).std() / df['close']
    
    # === VOLUME (5) ===========================================================
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
    
    # Volume Price Trend (VPT)
    df['vpt'] = (df['volume'] * df['returns']).cumsum()
    
    # === MOMENTUM (5) =========================================================
    df['momentum_5'] = df['close'].pct_change(periods=5)
    df['momentum_10'] = df['close'].pct_change(periods=10)
    df['momentum_20'] = df['close'].pct_change(periods=20)
    
    # Acceleration (change in momentum)
    df['acceleration'] = df['momentum_10'].diff()
    
    # Relative spread
    df['relative_spread'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    
    # === CLEANUP ==============================================================
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN (from rolling calculations)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"[FEATURES-V6] ✅ Calculated 49 features")
    print(f"[FEATURES-V6] Valid samples: {len(df)} (dropped {dropped_rows} due to NaN)")
    
    return df


def get_features_v6():
    """Return the canonical 49-feature list (same order as FEATURES_V6)"""
    return [
        # Candlestick (11)
        'returns', 'log_returns', 'price_range', 'body_size', 'upper_wick', 'lower_wick',
        'is_doji', 'is_hammer', 'is_engulfing', 'gap_up', 'gap_down',
        # Oscillators (7)
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'roc',
        # EMAs (8)
        'ema_9', 'ema_9_dist', 'ema_21', 'ema_21_dist', 'ema_50', 'ema_50_dist', 
        'ema_200', 'ema_200_dist',
        # SMAs (2)
        'sma_20', 'sma_50',
        # ADX (3)
        'adx', 'plus_di', 'minus_di',
        # Bollinger (5)
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        # Volatility (3)
        'atr', 'atr_pct', 'volatility',
        # Volume (5)
        'volume_sma', 'volume_ratio', 'obv', 'obv_ema', 'vpt',
        # Momentum (5)
        'momentum_5', 'momentum_10', 'momentum_20', 'acceleration', 'relative_spread'
    ]


def create_labels(df_with_features, df_original, threshold=0.015, lookahead=5):
    """
    Create labels based on future price movement.
    
    Labels:
        0 = SELL (price drops > threshold)
        1 = HOLD (price changes < threshold)
        2 = BUY (price rises > threshold)
    
    Args:
        df_with_features: DataFrame with calculated features
        df_original: Original DataFrame with OHLCV (for label calculation)
        threshold: Price change threshold (default 1.5%)
        lookahead: Periods to look ahead (default 5)
    
    Returns:
        (df_with_features, labels) - aligned DataFrame and label array
    """
    print(f"[LABELS] Creating labels (threshold={threshold*100}%, lookahead={lookahead})...")
    
    # Align indices
    df_original = df_original.loc[df_with_features.index]
    
    # Calculate future return
    future_return = df_original['close'].shift(-lookahead) / df_original['close'] - 1
    
    # Create labels
    labels = np.where(future_return > threshold, 2,       # BUY
                     np.where(future_return < -threshold, 0,  # SELL
                             1))                              # HOLD
    
    # Drop last lookahead rows (no future data)
    df_with_features = df_with_features.iloc[:-lookahead]
    labels = labels[:-lookahead]
    
    # Report distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    sell_count = label_dist.get(0, 0)
    hold_count = label_dist.get(1, 0)
    buy_count = label_dist.get(2, 0)
    
    print(f"[LABELS] Distribution: SELL={sell_count}, HOLD={hold_count}, BUY={buy_count}")
    print(f"[LABELS] Percentages: SELL={sell_count/len(labels)*100:.1f}%, "
          f"HOLD={hold_count/len(labels)*100:.1f}%, BUY={buy_count/len(labels)*100:.1f}%")
    
    return df_with_features, labels


if __name__ == "__main__":
    print("Feature Calculation V6 - 49 Features")
    print("=" * 70)
    print("This module provides feature calculation for v6 retraining scripts.")
    print("Import and use calculate_features_v6(df) in your training scripts.")
    print("=" * 70)
    print(f"\nTotal features: {len(get_features_v6())}")
    print("\nFeature groups:")
    print("  - Candlestick patterns: 11")
    print("  - Oscillators: 7")
    print("  - EMAs: 8")
    print("  - SMAs: 2")
    print("  - ADX: 3")
    print("  - Bollinger Bands: 5")
    print("  - Volatility: 3")
    print("  - Volume: 5")
    print("  - Momentum: 5")
