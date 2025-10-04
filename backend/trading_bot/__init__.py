"""
Trading Bot Module

This module implements the autonomous trading bot that connects AI signals to actual trade execution.
"""

from .autonomous_trader import AutonomousTradingBot, get_trading_bot
from .routes import router

__all__ = ['AutonomousTradingBot', 'get_trading_bot', 'router']