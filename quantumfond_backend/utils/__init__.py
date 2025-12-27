"""
Real-time communication utilities via Redis pub/sub
"""
import redis
import json
import os

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

async def push(channel: str, data: dict):
    """
    Push data to Redis channel for WebSocket broadcasting
    """
    try:
        message = json.dumps(data)
        redis_client.publish(f"quantumfond:{channel}", message)
        return True
    except Exception as e:
        print(f"Redis push error: {e}")
        return False

def subscribe(channel: str):
    """
    Subscribe to Redis channel
    """
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"quantumfond:{channel}")
    return pubsub
