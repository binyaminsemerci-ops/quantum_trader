import os, time, json, asyncio
from typing import Optional, Dict, Tuple

def _b(k: str, d: str = "false") -> bool:
    return os.getenv(k, d).lower() == "true"

class RLInfluenceV2:
    def __init__(self, redis_client, logger):
        self.r = redis_client
        self.log = logger
        self.enabled = _b("RL_INFLUENCE_ENABLED", "false")
        self.kill = _b("RL_INFLUENCE_KILL_SWITCH", "false")
        self.w = float(os.getenv("RL_INFLUENCE_WEIGHT", "0.05"))
        self.wmax = float(os.getenv("RL_INFLUENCE_MAX_WEIGHT", "0.10"))
        self.min_conf = float(os.getenv("RL_INFLUENCE_MIN_CONF", "0.65"))
        self.cool = int(os.getenv("RL_INFLUENCE_COOLDOWN_SEC", "120"))
        self.mode = os.getenv("RL_INFLUENCE_MODE", "shadow_gated")
        self.pref = os.getenv("RL_POLICY_REDIS_PREFIX", "quantum:rl:policy:")
        self.max_age = int(os.getenv("RL_POLICY_MAX_AGE_SEC", "600"))
        self._cool: Dict[str, float] = {}
        self._last_log = 0.0

    async def fetch(self, sym: str) -> Optional[Dict]:
        if (not self.enabled) or self.kill:
            return None
        k = f"{self.pref}{sym}"
        try:
            try:
                raw = await asyncio.wait_for(self.r.get(k), timeout=0.15)
            except asyncio.TimeoutError:
                return None
            if not raw:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "ignore")
            p = json.loads(raw)
            a = p.get("action", "HOLD")
            c = float(p.get("confidence", 0.0))
            ts = int(p.get("timestamp", 0) or 0)
            age = int(time.time() - ts) if ts else 999999
            return {
                "rl_action": a,
                "rl_confidence": c,
                "rl_version": p.get("version", "v2.0"),
                "rl_policy_age_sec": age,
                "rl_reason": p.get("reason", "rl_policy"),
            }
        except Exception:
            return None

    def gate(self, sym: str, ens_conf: float, rl: Optional[Dict]) -> Tuple[bool, str]:
        if self.kill:
            return (False, "kill_switch_active")
        if not self.enabled:
            return (False, "rl_disabled")
        if not rl:
            return (False, "no_rl_data")
        if rl.get("rl_policy_age_sec", 99999) > self.max_age:
            return (False, "policy_stale")
        if rl.get("rl_confidence", 0.0) < self.min_conf:
            return (False, "rl_conf_low")
        if ens_conf < 0.55:
            return (False, "ensemble_conf_low")
        last = self._cool.get(sym, 0.0)
        if (time.time() - last) < self.cool:
            return (False, "cooldown_active")
        return (True, "pass")

    def apply_shadow(self, sym: str, action: str, ens_conf: float, rl: Optional[Dict]) -> Tuple[str, Dict]:
        m = {
            "rl_influence_enabled": bool(self.enabled and (not self.kill)),
            "rl_gate_pass": False,
            "rl_gate_reason": "not_attempted",
            "rl_action": None,
            "rl_confidence": 0.0,
            "rl_version": None,
            "rl_policy_age_sec": None,
            "rl_weight_effective": 0.0,
            "rl_effect": "none",
        }

        ok, reason = self.gate(sym, ens_conf, rl)
        m["rl_gate_pass"] = ok
        m["rl_gate_reason"] = reason
        if not ok:
            return (action, m)

        m["rl_action"] = rl.get("rl_action")
        m["rl_confidence"] = float(rl.get("rl_confidence", 0.0))
        m["rl_version"] = rl.get("rl_version")
        m["rl_policy_age_sec"] = rl.get("rl_policy_age_sec")
        m["rl_weight_effective"] = float(min(self.w, self.wmax))

        ra = m["rl_action"] or "HOLD"
        rc = float(m["rl_confidence"] or 0.0)

        if action == "HOLD":
            m["rl_effect"] = "none"
        elif ra == action:
            m["rl_effect"] = "reinforce"
            self._cool[sym] = time.time()
        else:
            if ens_conf >= 0.75:
                m["rl_effect"] = "would_flip"
            elif 0.55 <= ens_conf < 0.75 and rc >= 0.80:
                m["rl_effect"] = "would_flip"
                self._cool[sym] = time.time()

        now = time.time()
        if (now - self._last_log) >= 30:
            self._last_log = now
            self.log.info(
                f"[AI-ENGINE] RL_SHADOW symbol={sym} gate={reason} effect={m['rl_effect']} "
                f"ensemble={action}({ens_conf:.2f}) rl={ra}({rc:.2f})"
            )

        return (action, m)
