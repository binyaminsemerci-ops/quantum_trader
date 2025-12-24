# GAPS & FIXES BACKLOG
Audit Date: December 24, 2025

## P0 CRITICAL
P0-1: Consumer lag 10,014 events - Trading offline
P0-2: Nginx unhealthy - Access impaired

## P1 HIGH  
P1-1: ILF not consumed - Blocked by P0-1
P1-2: Regime detection not connected
P1-3: Risk Safety stub implementation

## P2 MEDIUM
P2-1: No git version control on VPS
P2-2: Manual orchestration (no compose)
P2-3: High resource usage (80% RAM, 74% disk)
P2-4: Recent service restarts pattern

## P3 LOW
P3-1: AI Engine 404s for some symbols
P3-2: Funding rate and exchange divergence always 0

See individual reports for details:
- SERVICE_CATALOG.md
- EVENT_FLOW_MAP.md
- ORDER_LIFECYCLE.md
- TP_SL_EXIT_AUDIT.md
- LEVERAGE_SIZING_AUDIT.md
- AI_MODULES_STATUS.md
