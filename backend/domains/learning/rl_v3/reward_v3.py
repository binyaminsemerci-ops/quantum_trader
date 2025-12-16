"""
Reward function for RL v3 trading environment.
"""


def compute_reward(
    pnl_delta: float,
    drawdown: float,
    position_size: float,
    regime_alignment: float,
    volatility: float = 0.02,
    tp_zone_accuracy: float = 0.0,
    tp_reward_weight: float = 1.0
) -> float:
    """
    Compute reward for trading action.
    
    Args:
        pnl_delta: Change in PnL (positive = profit)
        drawdown: Current drawdown as ratio (0.0 - 1.0)
        position_size: Absolute position size (0.0 - 1.0)
        regime_alignment: How well action aligns with regime (-1 to +1)
        volatility: Current market volatility
        tp_zone_accuracy: TP zone hit rate (0.0 - 1.0) [TP v3]
        tp_reward_weight: Weight for TP accuracy reward component (default 1.0) [CLM v3 TP feedback]
        
    Returns:
        Scalar reward
    """
    # Base reward: PnL
    reward = pnl_delta * 100.0  # Scale to reasonable range
    
    # Drawdown penalty (exponential to heavily punish large drawdowns)
    drawdown_penalty = (drawdown ** 2) * 50.0
    reward -= drawdown_penalty
    
    # Position size penalty in high volatility
    if volatility > 0.03:
        position_penalty = (position_size ** 2) * 10.0
        reward -= position_penalty
    
    # Regime alignment bonus (reward for trading with the trend)
    regime_bonus = regime_alignment * 2.0
    reward += regime_bonus
    
    # Small survival bonus (encourages not dying)
    survival_bonus = 0.1
    reward += survival_bonus
    
    # Overtrading penalty (discourage too many actions)
    # This is implicit in action costs, but can add explicit penalty
    if abs(pnl_delta) < 0.0001:  # Action but no meaningful PnL
        reward -= 0.5
    
    # [TP v3] TP zone accuracy bonus with configurable weight
    # CLM v3 adjusts weight based on production TP performance:
    # - High weight (e.g. 2.0) when hit rate is low → encourage better TP prediction
    # - Default weight (1.0) when performance is acceptable
    # - Lower weight (e.g. 0.5) when hit rate is high but R is low → deprioritize TP accuracy
    if tp_zone_accuracy > 0:
        tp_accuracy_bonus = tp_zone_accuracy * 5.0 * tp_reward_weight  # Weighted bonus
        reward += tp_accuracy_bonus
    
    return reward
