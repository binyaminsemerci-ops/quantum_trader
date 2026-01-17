"""
HarvestBrain: Profit Harvesting Microservice

Implements deterministic profit harvesting strategies:
- R-based ladder (0.5R, 1.0R, 1.5R partials)
- Break-even moves
- Trailing stop updates

Modes:
- SHADOW: Proposals only (no live intents)
- LIVE: Publish reduce-only intents to trade.intent stream
"""

__version__ = "1.0.0"
__author__ = "Quantum Trader System"
