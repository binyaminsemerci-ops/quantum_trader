#!/usr/bin/env python3
"""Remove event-driven exit listener code from autonomous_trader.py"""

filepath = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(filepath, "r") as f:
    content = f.read()

# Remove 1: self._exit_listener_task declaration (line 81)
content = content.replace(
    "        self._exit_listener_task: Optional[asyncio.Task] = None\n",
    ""
)

# Remove 2: task creation in start() (line 134)
content = content.replace(
    "        self._exit_listener_task = asyncio.create_task(self._exit_event_listener())\n",
    ""
)

# Remove 3: task cancellation in stop() - the multi-line block
old_cancel_block = """        if self._exit_listener_task:
            self._exit_listener_task.cancel()
            try:
                await self._exit_listener_task
            except asyncio.CancelledError:
                pass
"""
content = content.replace(old_cancel_block, "")

# Remove 4: Both methods _exit_event_listener and _handle_exit_event
# Find the start of _exit_event_listener and remove everything until the end marker

import re

# Pattern to match from _exit_event_listener to end of _handle_exit_event
# This captures both methods which were added at the end before main()
pattern = r'\n    async def _exit_event_listener\(self\):.*?logger\.error\(f"\[EXIT-EVENTS\] Failed to handle exit event: \{e\}"\)\n'
content = re.sub(pattern, "\n", content, flags=re.DOTALL)

with open(filepath, "w") as f:
    f.write(content)

print("Removed event-driven exit listener code")
print("Remaining: local R-based fallback in exit_manager.py")
