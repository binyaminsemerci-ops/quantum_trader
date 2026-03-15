"""exit_management_agent — shadow-only position evaluation service (PATCH-1).

Contract:
  - Reads:  quantum:state:positions:* (SCAN + HGETALL)
  - Writes: quantum:stream:exit.audit
            quantum:stream:exit.metrics
            quantum:exit_agent:heartbeat  (SET with TTL)
  - NEVER writes to: apply.plan, trade.intent, harvest.intent, exit.intent live
  - DRY_RUN is enforced in code regardless of environment variables.
"""
__version__ = "0.1.0"
