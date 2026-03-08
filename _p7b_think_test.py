#!/usr/bin/env python3
"""Test qwen3:0.6b with think=False to measure latency without thinking tokens."""
import json, urllib.request, time

USER = '{"symbol":"ETHUSDT","side":"LONG","R_net":-0.0988,"age_sec":120.0,"age_fraction":0.0033,"giveback_pct":0.0,"distance_to_sl_pct":0.0833,"leverage":1.0,"exit_score":0.0198,"d_r_loss":0.0659,"d_r_gain":0.0,"d_giveback":0.0,"d_time":0.0002,"d_sl_proximity":0.0,"formula_suggestion":{"action":"HOLD","urgency":"LOW","confidence":0.0198,"reason":"Score=0.020"}}'

SYSTEM = "You are a trading risk advisor. Reply ONLY with a JSON object. Allowed actions: HOLD, PARTIAL_CLOSE_25, FULL_CLOSE, TIME_STOP_EXIT. Format: {\"action\":\"<action>\",\"confidence\":<0.0-1.0>,\"reason\":\"<max 80 chars>\"}"

for label, opts in [
    ("with think=False num_ctx=512", {"think": False, "num_ctx": 512}),
    ("with think=False num_ctx=1024", {"think": False, "num_ctx": 1024}),
    ("with think=True  num_ctx=512",  {"think": True,  "num_ctx": 512}),
]:
    body = json.dumps({
        "model": "qwen3:0.6b",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER},
        ],
        "stream": False,
        "format": "json",
        "options": opts,
    }).encode()

    t = time.time()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        wall = int((time.time() - t) * 1000)
        total_ms = r.get("total_duration", 0) // 1_000_000
        eval_count = r.get("eval_count", "?")
        content = r.get("message", {}).get("content", "")[:120]
        print(f"\n[{label}]")
        print(f"  wall_ms={wall}  total_ms={total_ms}  eval_count={eval_count}")
        print(f"  content={content}")
    except Exception as e:
        wall = int((time.time() - t) * 1000)
        print(f"\n[{label}]  ERROR after {wall}ms: {e}")
