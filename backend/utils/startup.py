import asyncio
import logging

logger = logging.getLogger(__name__)


async def log_startup_info() -> None:
    """Minimal startup logging used in tests/worktrees.

    Keep it non-blocking and lightweight.
    """
    # small sleep to simulate async non-blocking startup tasks in production
    await asyncio.sleep(0)
    logger.debug("startup: backend initialized (test shim)")
