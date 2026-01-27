#!/usr/bin/env python3
import sys, json, re

# Metric name mappings (old Docker -> new systemd)
METRIC_MAP = {
    # RL Shadow metrics
    "rl_shadow_gate_pass_rate_by_symbol": "quantum_rl_gate_pass_rate",
    "rl_shadow_eligible_rate": "quantum_rl_eligible_rate", 
    "rl_shadow_total_intents_analyzed": "quantum_rl_intents_analyzed_total",
    "rl_shadow_would_flip_rate": "quantum_rl_would_flip_rate",
    "rl_shadow_cooldown_blocking_rate": "quantum_rl_cooldown_blocking_rate",
    
    # Safety metrics
    "quantum_safety_mode_status": "quantum_safety_safe_mode",
    "safety_safe_mode": "quantum_safety_safe_mode",
    "safety_ttl": "quantum_safety_safe_mode_ttl_seconds",
    "safety_faults": "quantum_safety_faults_last_1h",
    
    # Redis metrics  
    "redis_up": "redis_uptime_in_seconds",
}

def fix_expr(expr):
    """Replace old metric names with new ones"""
    if not expr:
        return expr
    
    for old, new in METRIC_MAP.items():
        expr = expr.replace(old, new)
    
    # Fix specific patterns
    expr = re.sub(r'redis_up\{[^}]*\}', 'redis_uptime_in_seconds > 0', expr)
    
    return expr

def fix_dashboard(dash_json):
    """Fix all metric references in dashboard"""
    dash = json.loads(dash_json)
    
    if "dashboard" in dash:
        dash_obj = dash["dashboard"]
    else:
        dash_obj = dash
    
    fixed_count = 0
    
    for panel in dash_obj.get("panels", []):
        if "targets" in panel:
            for target in panel["targets"]:
                old_expr = target.get("expr", "")
                if old_expr:
                    new_expr = fix_expr(old_expr)
                    if new_expr != old_expr:
                        target["expr"] = new_expr
                        fixed_count += 1
    
    return json.dumps({"dashboard": dash_obj, "overwrite": True}), fixed_count

if __name__ == "__main__":
    dash_json = sys.stdin.read()
    fixed, count = fix_dashboard(dash_json)
    print(fixed)
    sys.stderr.write(f"Fixed {count} queries\n")
