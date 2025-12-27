#!/usr/bin/env python3
"""
Quick patch for ai_engine to initialize Redis client before EventBus.
Run this to update microservices/ai_engine/service.py
"""

import os

service_file = os.path.join(os.path.dirname(__file__), "microservices/ai_engine/service.py")

# Read file
with open(service_file, 'r') as f:
    content = f.read()

# Find the start() method and patch EventBus initialization
old_code = """            # Initialize EventBus
            logger.info("[AI-ENGINE] Initializing EventBus...")
            self.event_bus = EventBus()"""

new_code = """            # Initialize Redis client
            import redis.asyncio as redis
            logger.info(f"[AI-ENGINE] Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}...")
            redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False
            )
            # Test connection
            await redis_client.ping()
            logger.info("[AI-ENGINE] ✅ Redis connected")
            
            # Initialize EventBus with Redis client
            logger.info("[AI-ENGINE] Initializing EventBus...")
            self.event_bus = EventBus(redis_client=redis_client, service_name="ai-engine")"""

if old_code in content:
    content = content.replace(old_code, new_code)
    
    # Write back
    with open(service_file, 'w') as f:
        f.write(content)
    
    print("✅ Patched microservices/ai_engine/service.py")
    print("   - Added Redis client initialization")
    print("   - Fixed EventBus constructor call")
else:
    print("⚠️  Could not find exact match in service.py")
    print("    You may need to manually add Redis client initialization")
