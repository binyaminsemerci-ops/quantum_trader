"""
Portfolio Governance Agent
==========================
AI-drevet porteføljestyring med exposure memory og dynamisk policy-justering.

Formål:
- Samle PnL-historikk, risikoeksponeringer og AI-confidence data
- Bygge "exposure memory" database for langtidshukommelse
- Dynamisk justere posisjonsgrenser basert på performance
- Lære hvordan porteføljen reagerer på volatilitetsregimer
- Styre Portfolio Score som feedback til ExitBrain v3.5 og RL Sizing Agent

Komponenter:
- ExposureMemory: Lagrer og analyserer trade-historikk
- PortfolioGovernanceAgent: Policy controller og decision maker
- Redis Streams: Event-drevet arkitektur for real-time data
"""

__version__ = "1.0.0"
__author__ = "Quantum Trader AI Team"
