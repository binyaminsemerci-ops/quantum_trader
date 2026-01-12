#!/usr/bin/env python3
"""
Prove hardcoded 0.5/0.75 values in telemetry (QSC-Compliant)

Samples Redis stream and extracts EXACT confidence values
to prove root cause of dead zone trap.

Usage: ops/run.sh ai-engine ops/model_safety/prove_hardcoded_values.py
"""
import sys
import json
import redis
from collections import Counter

def main():
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        r.ping()
        
        # Sample 200 events
        raw_events = r.xrevrange('quantum:stream:trade.intent', count=200)
        
        print("="*80)
        print("PROVING HARDCODED VALUES IN TELEMETRY")
        print("="*80)
        print()
        
        # Collect exact confidence values
        xgb_confs = []
        patchtst_confs = []
        lgbm_confs = []
        
        for event_id, fields in raw_events:
            decoded_fields = {}
            for key, value in fields.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                decoded_fields[key_str] = value_str
            
            if decoded_fields.get('event_type') != 'trade.intent':
                continue
            
            payload_json = decoded_fields.get('payload', '{}')
            try:
                payload = json.loads(payload_json)
            except:
                continue
            
            breakdown = payload.get('model_breakdown', {})
            
            if 'xgb' in breakdown:
                conf = breakdown['xgb'].get('confidence')
                if conf is not None:
                    xgb_confs.append(float(conf))
            
            if 'patchtst' in breakdown:
                conf = breakdown['patchtst'].get('confidence')
                if conf is not None:
                    patchtst_confs.append(float(conf))
            
            if 'lgbm' in breakdown:
                conf = breakdown['lgbm'].get('confidence')
                if conf is not None:
                    lgbm_confs.append(float(conf))
        
        print(f"Analyzed {len(raw_events)} events\n")
        
        # Count EXACT values
        print("### XGBoost Confidence Values (sample 200):")
        xgb_counter = Counter(xgb_confs)
        for conf, count in xgb_counter.most_common(10):
            pct = count / len(xgb_confs) * 100 if xgb_confs else 0
            marker = "🔴 HARDCODED" if conf == 0.5 else ""
            print(f"  {conf:.4f}: {count} ({pct:.1f}%) {marker}")
        
        print(f"\n### PatchTST Confidence Values (sample 200):")
        patchtst_counter = Counter(patchtst_confs)
        for conf, count in patchtst_counter.most_common(10):
            pct = count / len(patchtst_confs) * 100 if patchtst_confs else 0
            marker = "🔴 HARDCODED" if conf == 0.5 else ""
            print(f"  {conf:.4f}: {count} ({pct:.1f}%) {marker}")
        
        print(f"\n### LightGBM Confidence Values (sample 200):")
        lgbm_counter = Counter(lgbm_confs)
        for conf, count in lgbm_counter.most_common(10):
            pct = count / len(lgbm_confs) * 100 if lgbm_confs else 0
            marker = "🔴 HARDCODED" if conf in [0.75, 0.68, 0.72] else ""
            print(f"  {conf:.4f}: {count} ({pct:.1f}%) {marker}")
        
        print("\n" + "="*80)
        print("EVIDENCE SUMMARY:")
        
        xgb_050 = sum(1 for c in xgb_confs if abs(c - 0.5) < 0.0001)
        patchtst_050 = sum(1 for c in patchtst_confs if abs(c - 0.5) < 0.0001)
        lgbm_075 = sum(1 for c in lgbm_confs if abs(c - 0.75) < 0.0001)
        
        if xgb_050 > 0:
            pct = xgb_050 / len(xgb_confs) * 100
            print(f"  🔴 XGBoost: {xgb_050}/{len(xgb_confs)} ({pct:.1f}%) outputs EXACTLY 0.5000")
            print(f"     Code: ai_engine/agents/xgb_agent.py:294,397 ('HOLD', 0.50, ...)")
        
        if patchtst_050 > 0:
            pct = patchtst_050 / len(patchtst_confs) * 100
            print(f"  🔴 PatchTST: {patchtst_050}/{len(patchtst_confs)} ({pct:.1f}%) outputs EXACTLY 0.5000")
            print(f"     Code: ai_engine/agents/patchtst_agent.py:364 (confidence = 0.5)")
        
        if lgbm_075 > 0:
            pct = lgbm_075 / len(lgbm_confs) * 100
            print(f"  🔴 LightGBM: {lgbm_075}/{len(lgbm_confs)} ({pct:.1f}%) outputs capped at 0.75")
            print(f"     Code: ai_engine/agents/lgbm_agent.py:227-240 (min(0.75, ...))")
        
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
