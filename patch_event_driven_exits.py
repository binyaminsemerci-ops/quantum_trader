#!/usr/bin/env python3
"""
Patch autonomous_trader.py to add event-driven exit listener
Listens to quantum:stream:ai.exit.decision and executes exits without HTTP calls
"""

filepath = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(filepath, "r") as f:
    content = f.read()

# Check if already patched
if "exit_event_listener" in content:
    print("SKIP: Exit event listener already exists")
    exit(0)

# Add import for redis stream reading (already has redis.asyncio)
# Find the class variable section and add a new task variable
old_init = """        self._main_loop_task: Optional[asyncio.Task] = None"""
new_init = """        self._main_loop_task: Optional[asyncio.Task] = None
        self._exit_listener_task: Optional[asyncio.Task] = None
        self._exit_stream_consumer_id = f"autonomous-trader-{os.getpid()}"
        self._exit_stream_group = "autonomous-trader:exit-listeners\""""

content = content.replace(old_init, new_init)

# Add start of exit listener in start() method
old_start = """        # Start main loop
        self._main_loop_task = asyncio.create_task(self._main_loop())

        logger.info("✅ Autonomous Trader STARTED")"""
        
new_start = """        # Start main loop
        self._main_loop_task = asyncio.create_task(self._main_loop())
        
        # Start exit event listener (event-driven exits from AI Engine)
        self._exit_listener_task = asyncio.create_task(self._exit_event_listener())

        logger.info("✅ Autonomous Trader STARTED (with event-driven exits)")"""

content = content.replace(old_start, new_start)

# Add stop of exit listener in stop() method
old_stop = """        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass"""
                
new_stop = """        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        
        if self._exit_listener_task:
            self._exit_listener_task.cancel()
            try:
                await self._exit_listener_task
            except asyncio.CancelledError:
                pass"""

content = content.replace(old_stop, new_stop)

# Add the exit event listener method before the last method (or at end of class)
exit_listener_method = '''
    async def _exit_event_listener(self):
        """
        🔥 EVENT-DRIVEN EXITS: Listen to AI Engine exit decisions via Redis stream
        Bypasses HTTP timeout issues by consuming events directly
        """
        stream_name = "quantum:stream:ai.exit.decision"
        group_name = self._exit_stream_group
        consumer_id = self._exit_stream_consumer_id
        
        logger.info(f"[EXIT-EVENTS] Starting exit event listener on {stream_name}")
        
        # Create consumer group if not exists
        try:
            await self.redis.xgroup_create(stream_name, group_name, id="0", mkstream=True)
            logger.info(f"[EXIT-EVENTS] Created consumer group: {group_name}")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"[EXIT-EVENTS] Consumer group already exists: {group_name}")
            else:
                logger.warning(f"[EXIT-EVENTS] Failed to create group: {e}")
        
        while self._running:
            try:
                # Read from stream with 5s timeout
                messages = await self.redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_id,
                    streams={stream_name: ">"},
                    count=10,
                    block=5000  # 5 second timeout
                )
                
                if not messages:
                    continue
                
                for stream, msg_list in messages:
                    for msg_id, data in msg_list:
                        await self._handle_exit_event(msg_id, data)
                        # Acknowledge message
                        await self.redis.xack(stream_name, group_name, msg_id)
                        
            except asyncio.CancelledError:
                logger.info("[EXIT-EVENTS] Exit listener cancelled")
                break
            except Exception as e:
                logger.warning(f"[EXIT-EVENTS] Error reading stream: {e}")
                await asyncio.sleep(1)
    
    async def _handle_exit_event(self, msg_id: str, data: dict):
        """Process AI exit decision event"""
        try:
            symbol = data.get("symbol", "")
            action = data.get("action", "HOLD")
            percentage = float(data.get("percentage", "0"))
            reason = data.get("reason", "ai_event")
            exit_score = int(data.get("exit_score", "0"))
            hold_score = int(data.get("hold_score", "0"))
            
            # Skip HOLD decisions
            if action == "HOLD":
                return
            
            # Check if we have this position
            position = self.position_tracker.get_position(symbol)
            if not position:
                logger.debug(f"[EXIT-EVENTS] No position for {symbol}, skipping")
                return
            
            # Only act if exit_score > hold_score (AI recommends exit)
            if exit_score <= hold_score:
                logger.debug(f"[EXIT-EVENTS] {symbol}: hold_score({hold_score}) >= exit_score({exit_score}), HOLD")
                return
            
            logger.info(
                f"[EXIT-EVENTS] 🎯 Executing AI exit: {symbol} {action} {int(percentage*100)}% "
                f"(exit={exit_score} > hold={hold_score}) - {reason}"
            )
            
            # Calculate exit quantity
            exit_qty = position.position_qty * percentage
            
            # Execute exit via harvest intent
            await self._execute_exit(
                position=position,
                action=action,
                percentage=percentage,
                reason=f"ai_event:{reason}"
            )
            
        except Exception as e:
            logger.error(f"[EXIT-EVENTS] Failed to handle exit event: {e}")

'''

# Find a good place to insert - before the last method or at end of class
# Look for the _execute_exit method and add after it
insert_marker = "    async def _execute_exit("
if insert_marker in content:
    # Find the end of _execute_exit method
    idx = content.find(insert_marker)
    # Find the next method after _execute_exit
    next_method_patterns = ["    async def ", "    def ", "\nif __name__"]
    end_idx = len(content)
    for pattern in next_method_patterns:
        search_start = idx + len(insert_marker)
        found_idx = content.find(pattern, search_start)
        if found_idx != -1 and found_idx < end_idx:
            end_idx = found_idx
    
    # Insert before the next method or at end
    content = content[:end_idx] + exit_listener_method + content[end_idx:]
else:
    # Just append before if __name__
    if "if __name__" in content:
        idx = content.find("if __name__")
        content = content[:idx] + exit_listener_method + "\n" + content[idx:]
    else:
        content += exit_listener_method

# Write back
with open(filepath, "w") as f:
    f.write(content)

print("SUCCESS: Added event-driven exit listener to autonomous_trader.py")
print("Stream: quantum:stream:ai.exit.decision")
print("Consumer group: autonomous-trader:exit-listeners")
