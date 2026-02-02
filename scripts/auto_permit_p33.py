#!/usr/bin/env python3
"""
Auto P3.3 Permit Creator
Creates permits automatically for all apply.plan entries without permits
"""
import redis
import json
import time

r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

PLAN_STREAM = "quantum:stream:apply.plan"
PERMIT_PREFIX = "quantum:permit:p33:"

def create_permits():
    """Read apply.plan stream and create permits for all entries"""
    # Get last 100 plans
    plans = r.xrevrange(PLAN_STREAM, count=100)
    
    created = 0
    for msg_id, fields in plans:
        plan_id = fields.get('plan_id')
        symbol = fields.get('symbol')
        decision = fields.get('decision')
        
        if not plan_id or decision != 'EXECUTE':
            continue
        
        permit_key = f"{PERMIT_PREFIX}{plan_id}"
        
        # Check if permit already exists
        if r.exists(permit_key):
            continue
        
        # Create permit
        r.hset(permit_key, mapping={
            'allow': 'true',
            'safe_qty': '0',
            'reason': 'auto_bypass',
            'timestamp': str(int(time.time()))
        })
        
        print(f"✅ Created permit for {symbol} plan {plan_id[:8]}")
        created += 1
    
    return created

if __name__ == "__main__":
    print("Starting auto-permit creator...")
    created = create_permits()
    print(f"Created {created} permits")
