import pathlib, sys

BASE = pathlib.Path("/opt/quantum/microservices/exit_management_agent")

# ─── 1. models.py ────────────────────────────────────────────────────
f = BASE / "models.py"
src = f.read_text()
old = "    r_effective_t1: float\n    r_effective_lock: float"
new = (
    "    r_effective_t1: float\n"
    "    r_effective_t2: float      # 4R/sqrt(lev) -> PARTIAL_CLOSE_50\n"
    "    r_effective_t3: float      # 6R/sqrt(lev) -> PARTIAL_CLOSE_75\n"
    "    r_effective_lock: float"
)
assert old in src, f"models.py: marker not found"
f.write_text(src.replace(old, new, 1))
print("  [OK] models.py")

# ─── 2. perception.py ────────────────────────────────────────────────
f = BASE / "perception.py"
src = f.read_text()

# 2a: _get_r_targets returns 4 values instead of 2
old2 = "    if _RISK_SETTINGS_AVAILABLE:\n        targets = compute_harvest_r_targets(leverage, DEFAULT_SETTINGS)\n        return targets[\"T1_R\"], targets[\"lock_R\"]\n\n    # Fallback: same formula, same defaults as DEFAULT_SETTINGS\n    scale = math.sqrt(max(float(leverage), 1.0))\n    return 2.0 / scale, 1.5 / scale"
new2 = (
    "    if _RISK_SETTINGS_AVAILABLE:\n"
    "        targets = compute_harvest_r_targets(leverage, DEFAULT_SETTINGS)\n"
    "        t1 = targets.get(\"T1_R\", targets.get(\"T1_R\", 2.0))\n"
    "        t2 = targets.get(\"T2_R\", t1 * 2.0)\n"
    "        t3 = targets.get(\"T3_R\", t1 * 3.0)\n"
    "        lock = targets[\"lock_R\"]\n"
    "        return t1, t2, t3, lock\n"
    "    # Fallback: same formula, same defaults as DEFAULT_SETTINGS\n"
    "    scale = math.sqrt(max(float(leverage), 1.0))\n"
    "    return 2.0 / scale, 4.0 / scale, 6.0 / scale, 1.5 / scale"
)
assert old2 in src, f"perception.py: _get_r_targets body not found"
src = src.replace(old2, new2, 1)

# 2b: unpack 4 values in PerceptionEngine.compute()
old3 = "        r_t1, r_lock = _get_r_targets(snapshot.leverage)"
new3 = "        r_t1, r_t2, r_t3, r_lock = _get_r_targets(snapshot.leverage)"
assert old3 in src, f"perception.py: unpack line not found"
src = src.replace(old3, new3, 1)

# 2c: add t2/t3 to PerceptionResult constructor
old4 = (
    "            r_effective_t1=r_t1,\n"
    "            r_effective_lock=r_lock,"
)
new4 = (
    "            r_effective_t1=r_t1,\n"
    "            r_effective_t2=r_t2,\n"
    "            r_effective_t3=r_t3,\n"
    "            r_effective_lock=r_lock,"
)
assert old4 in src, f"perception.py: PerceptionResult constructor not found"
f.write_text(src.replace(old4, new4, 1))
print("  [OK] perception.py")

# ─── 3. decision_engine.py ───────────────────────────────────────────
f = BASE / "decision_engine.py"
src = f.read_text()

# Add PARTIAL_CLOSE_50/75 constants after PARTIAL_CLOSE_25
old5 = 'PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"\nFULL_CLOSE = "FULL_CLOSE"'
new5 = (
    'PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"\n'
    'PARTIAL_CLOSE_50 = "PARTIAL_CLOSE_50"\n'
    'PARTIAL_CLOSE_75 = "PARTIAL_CLOSE_75"\n'
    'FULL_CLOSE = "FULL_CLOSE"'
)
assert old5 in src, f"decision_engine.py: action constants not found"
src = src.replace(old5, new5, 1)

# Add Rule 4b (T3 at 6R -> PARTIAL_75) and 4c (T2 at 4R -> PARTIAL_50) BEFORE Rule 4 (T1)
old6 = (
    "        # ── Rule 4: Partial harvest at T1 ──────────────────────────────────\n"
    "        if p.R_net >= p.r_effective_t1:"
)
new6 = (
    "        # ── Rule 4c: Partial harvest at T3 (6R) ──────────────────────────\n"
    "        if p.R_net >= p.r_effective_t3:\n"
    "            return _decision(\n"
    "                snap=snap,\n"
    "                action=PARTIAL_CLOSE_75,\n"
    "                reason=(\n"
    "                    f\"Harvest T3: R_net={p.R_net:.2f} >= T3={p.r_effective_t3:.2f}R \"\n"
    "                    f\"(leverage={snap.leverage:.0f}x scaled) -- take 75%\"\n"
    "                ),\n"
    "                urgency=URGENCY_MEDIUM,\n"
    "                r_net=p.R_net,\n"
    "                confidence=0.80,\n"
    "                suggested_qty_fraction=0.75,\n"
    "                dry_run=dry_run,\n"
    "            )\n\n"
    "        # ── Rule 4b: Partial harvest at T2 (4R) ──────────────────────────\n"
    "        if p.R_net >= p.r_effective_t2:\n"
    "            return _decision(\n"
    "                snap=snap,\n"
    "                action=PARTIAL_CLOSE_50,\n"
    "                reason=(\n"
    "                    f\"Harvest T2: R_net={p.R_net:.2f} >= T2={p.r_effective_t2:.2f}R \"\n"
    "                    f\"(leverage={snap.leverage:.0f}x scaled) -- take 50%\"\n"
    "                ),\n"
    "                urgency=URGENCY_MEDIUM,\n"
    "                r_net=p.R_net,\n"
    "                confidence=0.77,\n"
    "                suggested_qty_fraction=0.50,\n"
    "                dry_run=dry_run,\n"
    "            )\n\n"
    "        # ── Rule 4: Partial harvest at T1 (2R) ────────────────────────────\n"
    "        if p.R_net >= p.r_effective_t1:"
)
assert old6 in src, f"decision_engine.py: Rule 4 block not found"
f.write_text(src.replace(old6, new6, 1))
print("  [OK] decision_engine.py")

# ─── 4. scoring_engine.py ────────────────────────────────────────────
f = BASE / "scoring_engine.py"
src = f.read_text()

# Add PARTIAL_CLOSE_50/75 constants
old7 = 'PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"\nFULL_CLOSE = "FULL_CLOSE"'
new7 = (
    'PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"\n'
    'PARTIAL_CLOSE_50 = "PARTIAL_CLOSE_50"\n'
    'PARTIAL_CLOSE_75 = "PARTIAL_CLOSE_75"\n'
    'FULL_CLOSE = "FULL_CLOSE"'
)
assert old7 in src, f"scoring_engine.py: action constants not found"
src = src.replace(old7, new7, 1)

# Add PARTIAL_CLOSE_50/75 to FORMULA_QTY_MAP
old8 = "    PARTIAL_CLOSE_25:  0.25,"
new8 = "    PARTIAL_CLOSE_25:  0.25,\n    PARTIAL_CLOSE_50:  0.50,\n    PARTIAL_CLOSE_75:  0.75,"
assert old8 in src, f"scoring_engine.py: FORMULA_QTY_MAP not found"
f.write_text(src.replace(old8, new8, 1))
print("  [OK] scoring_engine.py")

# ─── 5. validator.py ─────────────────────────────────────────────────
f = BASE / "validator.py"
src = f.read_text()
old9 = (
    '        "FULL_CLOSE",\n'
    '        "PARTIAL_CLOSE_25",\n'
    '        "TIME_STOP_EXIT",\n'
    '    }'
)
new9 = (
    '        "FULL_CLOSE",\n'
    '        "PARTIAL_CLOSE_25",\n'
    '        "PARTIAL_CLOSE_50",\n'
    '        "PARTIAL_CLOSE_75",\n'
    '        "TIME_STOP_EXIT",\n'
    '    }'
)
assert old9 in src, f"validator.py: LIVE_ACTION_WHITELIST not found"
f.write_text(src.replace(old9, new9, 1))
print("  [OK] validator.py")

print("All patches applied.")
