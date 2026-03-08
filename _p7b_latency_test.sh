#!/usr/bin/env bash
# Measure actual qwen3:0.6b latency on VPS using the exact same request
# structure that qwen3_layer.py sends via _post()

SYSTEM_PROMPT='You are a constrained trading risk advisor operating on Binance testnet futures.
You receive a JSON object describing one open position and the formula engine'\''s
exit recommendation. Your only job is to select an exit action.

Allowed actions (you MUST use exactly one of these strings):
  HOLD
  PARTIAL_CLOSE_25
  FULL_CLOSE
  TIME_STOP_EXIT

Rules:
- Emergency stops are already handled upstream. You will never see urgency=EMERGENCY.
- TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN are not available to you.
- If uncertain, prefer HOLD (defer to the formula recommendation).
- formula_suggestion is advisory. You may agree or override it.

You MUST respond with ONLY a JSON object — no markdown, no explanation, no extra text:
{"action": "<one of the 4 actions>", "confidence": <0.0-1.0>, "reason": "<max 120 chars>"}'

USER_CONTENT='{"symbol":"ETHUSDT","side":"LONG","R_net":-0.0988,"age_sec":120.0,"age_fraction":0.0033,"giveback_pct":0.0,"distance_to_sl_pct":0.0833,"leverage":1.0,"exit_score":0.0198,"d_r_loss":0.0659,"d_r_gain":0.0,"d_giveback":0.0,"d_time":0.0002,"d_sl_proximity":0.0,"formula_suggestion":{"action":"HOLD","urgency":"LOW","confidence":0.0198,"reason":"Score=0.020 \u2014 no exit criteria met"}}'

REQUEST_BODY=$(python3 -c "
import json, sys
body = {
    'model': 'qwen3:0.6b',
    'messages': [
        {'role': 'system', 'content': sys.argv[1]},
        {'role': 'user', 'content': sys.argv[2]},
    ],
    'stream': False,
    'format': 'json',
}
print(json.dumps(body))
" "$SYSTEM_PROMPT" "$USER_CONTENT")

echo "=== QWEN3:0.6b REALISTIC LATENCY TEST ==="
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Sending /api/chat request..."
echo "(First call may include model load time)"
echo ""

START=$(date +%s%3N)
RESPONSE=$(curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY" 2>&1)
END=$(date +%s%3N)
ELAPSED=$((END - START))

echo "Raw response (trimmed):"
echo "$RESPONSE" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    msg = r.get('message', {}).get('content', '')
    total_ns = r.get('total_duration', 0)
    eval_dur_ns = r.get('eval_duration', 0)
    eval_count = r.get('eval_count', 0)
    prompt_eval = r.get('prompt_eval_duration', 0)
    load_dur = r.get('load_duration', 0)
    print(f'  content: {msg[:200]}')
    print(f'  total_duration_ms: {total_ns/1e6:.0f}')
    print(f'  load_duration_ms: {load_dur/1e6:.0f}')
    print(f'  prompt_eval_ms: {prompt_eval/1e6:.0f}')
    print(f'  eval_count (output tokens): {eval_count}')
    print(f'  eval_duration_ms: {eval_dur_ns/1e6:.0f}')
    if eval_count > 0:
        ms_per_tok = (eval_dur_ns/1e6) / eval_count
        print(f'  ms_per_token: {ms_per_tok:.1f}')
except Exception as e:
    print(f'  parse error: {e}')
    print(sys.stdin.read()[:300])
"
echo ""
echo "  wall_clock_ms: $ELAPSED"
echo ""
echo "=== SECOND CALL (warm, no load time) ==="
START2=$(date +%s%3N)
RESPONSE2=$(curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY" 2>&1)
END2=$(date +%s%3N)
ELAPSED2=$((END2 - START2))

echo "$RESPONSE2" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    msg = r.get('message', {}).get('content', '')
    total_ns = r.get('total_duration', 0)
    eval_count = r.get('eval_count', 0)
    print(f'  content: {msg[:200]}')
    print(f'  total_duration_ms: {total_ns/1e6:.0f}')
    print(f'  eval_count (output tokens): {eval_count}')
except Exception as e:
    print(f'  parse error: {e}')
"
echo "  wall_clock_ms: $ELAPSED2"
