#!/bin/bash
set -euo pipefail
cd /home/qt/quantum_trader

# 0) ENV (idempotent)
grep -q '^RL_INFLUENCE_ENABLED=' /etc/quantum/ai-engine.env || cat >> /etc/quantum/ai-engine.env <<'EOF'

# RL Bootstrap v2 - shadow_gated
RL_INFLUENCE_ENABLED=true
RL_INFLUENCE_WEIGHT=0.05
RL_INFLUENCE_MAX_WEIGHT=0.10
RL_INFLUENCE_MIN_CONF=0.65
RL_INFLUENCE_COOLDOWN_SEC=120
RL_INFLUENCE_KILL_SWITCH=false
RL_INFLUENCE_MODE=shadow_gated
RL_POLICY_REDIS_PREFIX=quantum:rl:policy:
RL_POLICY_MAX_AGE_SEC=600
EOF

# 1) NEW MODULE: rl_influence.py
cat > microservices/ai_engine/rl_influence.py <<'PY'
import os, time, json, asyncio
from typing import Optional, Dict, Tuple
def _b(k,d="false"): return os.getenv(k,d).lower()=="true"
class RLInfluenceV2:
    def __init__(self, redis_client, logger):
        self.r=redis_client; self.log=logger
        self.enabled=_b("RL_INFLUENCE_ENABLED","false"); self.kill=_b("RL_INFLUENCE_KILL_SWITCH","false")
        self.w=float(os.getenv("RL_INFLUENCE_WEIGHT","0.05")); self.wmax=float(os.getenv("RL_INFLUENCE_MAX_WEIGHT","0.10"))
        self.min=float(os.getenv("RL_INFLUENCE_MIN_CONF","0.65")); self.cool=int(os.getenv("RL_INFLUENCE_COOLDOWN_SEC","120"))
        self.mode=os.getenv("RL_INFLUENCE_MODE","shadow_gated")
        self.pref=os.getenv("RL_POLICY_REDIS_PREFIX","quantum:rl:policy:"); self.max_age=int(os.getenv("RL_POLICY_MAX_AGE_SEC","600"))
        self._cool: Dict[str,float]={}; self._last=0.0
    async def fetch(self,sym:str)->Optional[Dict]:
        if (not self.enabled) or self.kill: return None
        k=f"{self.pref}{sym}"
        try:
            try: raw=await asyncio.wait_for(self.r.get(k),timeout=0.15)
            except asyncio.TimeoutError: return None
            if not raw: return None
            if isinstance(raw,(bytes,bytearray)): raw=raw.decode("utf-8","ignore")
            p=json.loads(raw); a=p.get("action","HOLD"); c=float(p.get("confidence",0.0)); ts=int(p.get("timestamp",0) or 0)
            age=int(time.time()-ts) if ts else 999999
            return {"rl_action":a,"rl_confidence":c,"rl_version":p.get("version","v2.0"),"rl_policy_age_sec":age,"rl_reason":p.get("reason","rl_policy")}
        except Exception: return None
    def gate(self,sym:str,ens:float,rl:Optional[Dict])->Tuple[bool,str]:
        if self.kill: return (False,"kill_switch_active")
        if not self.enabled: return (False,"rl_disabled")
        if not rl: return (False,"no_rl_data")
        if rl.get("rl_policy_age_sec",99999)>self.max_age: return (False,"policy_stale")
        if rl.get("rl_confidence",0.0)<self.min: return (False,"rl_conf_low")
        if ens<0.55: return (False,"ensemble_conf_low")
        last=self._cool.get(sym,0.0)
        if (time.time()-last)<self.cool: return (False,"cooldown_active")
        return (True,"pass")
    def apply_shadow(self,sym:str,action:str,ens:float,rl:Optional[Dict])->Tuple[str,Dict]:
        m={"rl_influence_enabled": self.enabled and (not self.kill),"rl_gate_pass":False,"rl_gate_reason":"not_attempted",
           "rl_action":None,"rl_confidence":0.0,"rl_version":None,"rl_policy_age_sec":None,"rl_weight_effective":0.0,"rl_effect":"none"}
        ok,reason=self.gate(sym,ens,rl); m["rl_gate_pass"]=ok; m["rl_gate_reason"]=reason
        if not ok: return (action,m)
        m["rl_action"]=rl.get("rl_action"); m["rl_confidence"]=float(rl.get("rl_confidence",0.0))
        m["rl_version"]=rl.get("rl_version"); m["rl_policy_age_sec"]=rl.get("rl_policy_age_sec"); m["rl_weight_effective"]=float(min(self.w,self.wmax))
        ra=m["rl_action"] or "HOLD"; rc=m["rl_confidence"]
        if action=="HOLD": m["rl_effect"]="none"
        elif ra==action: m["rl_effect"]="reinforce"; self._cool[sym]=time.time()
        else:
            if ens>=0.75: m["rl_effect"]="would_flip"
            elif 0.55<=ens<0.75 and rc>=0.80: m["rl_effect"]="would_flip"; self._cool[sym]=time.time()
        now=time.time()
        if (now-self._last)>=30:
            self._last=now; self.log.info(f"[AI-ENGINE] RL_SHADOW symbol={sym} gate={reason} effect={m['rl_effect']} ensemble={action}({ens:.2f}) rl={ra}({rc:.2f})")
        return (action,m)
PY
python3 -m py_compile microservices/ai_engine/rl_influence.py

# 2) PATCH service.py (minimal, idempotent)
python3 <<'PY'
import re,sys,pathlib
p=pathlib.Path("microservices/ai_engine/service.py"); s=p.read_text(encoding="utf-8")
if "from microservices.ai_engine.rl_influence import RLInfluenceV2" not in s:
    s=re.sub(r"(\nfrom typing import .*?\n)", r"\1from microservices.ai_engine.rl_influence import RLInfluenceV2\n", s, count=1, flags=re.DOTALL)
if "self.rl_influence = RLInfluenceV2" not in s:
    s=re.sub(r"(self\.redis_client\s*=\s*.*?\n)", r"\1        self.rl_influence = RLInfluenceV2(self.redis_client, logger)\n", s, count=1, flags=re.DOTALL)
m=re.search(r"\n(\s*)trade_intent_payload\s*=\s*\{", s)
if not m: print("ERR: trade_intent_payload not found"); sys.exit(1)
indent=m.group(1)
if "RL Bootstrap v2 (shadow_gated)" not in s:
    block=f"""{indent}# RL Bootstrap v2 (shadow_gated)\n{indent}rl_meta={{}}\n{indent}try:\n{indent}    rl_data = await self.rl_influence.fetch(symbol) if getattr(self,'rl_influence',None) else None\n{indent}    action, rl_meta = self.rl_influence.apply_shadow(symbol, action, float(ensemble_confidence), rl_data) if getattr(self,'rl_influence',None) else (action, {{}})\n{indent}except Exception:\n{indent}    rl_meta={{}}\n\n"""
    s=s[:m.start()]+"\n"+block+s[m.start():]
if "**(rl_meta" not in s:
    mm=re.search(r"(trade_intent_payload\s*=\s*\{\n)", s)
    if not mm: print("ERR: payload open not found"); sys.exit(1)
    s=s[:mm.end()]+f"{indent}    **(rl_meta if isinstance(rl_meta, dict) else {{}}),\n"+s[mm.end():]
p.write_text(s,encoding="utf-8"); print("OK")
PY
python3 -m py_compile microservices/ai_engine/service.py

# 3) Seed test policies
redis-cli SET quantum:rl:policy:BTCUSDT "{\"action\":\"BUY\",\"confidence\":0.72,\"version\":\"v2.0\",\"timestamp\":$(date +%s),\"reason\":\"rl_test\"}" >/dev/null
redis-cli SET quantum:rl:policy:ETHUSDT "{\"action\":\"SELL\",\"confidence\":0.85,\"version\":\"v2.0\",\"timestamp\":$(date +%s),\"reason\":\"rl_test\"}" >/dev/null
redis-cli SET quantum:rl:policy:SOLUSDT "{\"action\":\"BUY\",\"confidence\":0.68,\"version\":\"v2.0\",\"timestamp\":$(date +%s),\"reason\":\"rl_test\"}" >/dev/null

# 4) Restart + proof
systemctl restart quantum-ai-engine.service
sleep 3
echo "HEAD=$(git rev-parse --short HEAD)"
echo "ACTIVE=$(systemctl is-active quantum-ai-engine.service)"
echo "RL_POL_BTC=$(redis-cli GET quantum:rl:policy:BTCUSDT)"
journalctl -u quantum-ai-engine.service --since "2 minutes ago" --no-pager | grep -E "RL_SHADOW" | tail -5 || true
journalctl -u quantum-ai-engine.service --since "2 minutes ago" --no-pager | grep -E "trade\.intent|rl_gate_reason|rl_effect" | tail -1 || echo "NO_TRADE_INTENT_YET"

# 5) Commit+push
git add microservices/ai_engine/rl_influence.py microservices/ai_engine/service.py /etc/quantum/ai-engine.env || true
git commit -m "feat(ai-engine): RL Bootstrap v2 shadow_gated (redis policy + attribution)" || true
git push origin main || true
