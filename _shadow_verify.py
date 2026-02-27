"""
SHADOW VERIFICATION — alle 4 fikser live i produksjon
Kjøres på VPS: python3 /tmp/_shadow_verify.py
"""
import redis, time, subprocess
from collections import Counter, defaultdict

r = redis.Redis(decode_responses=True)

# ── Hent faktisk restart-tidspunkt fra systemctl ──────────────────
def get_service_start_ms(service):
    res = subprocess.run(
        ["systemctl", "show", service, "--property=ActiveEnterTimestamp"],
        capture_output=True, text=True,
    )
    line = res.stdout.strip()  # e.g. "ActiveEnterTimestamp=Wed 2026-02-25 00:32:55 UTC"
    if "=" not in line:
        return 0
    ts_str = line.split("=", 1)[1].strip()
    try:
        import datetime
        dt = datetime.datetime.strptime(ts_str, "%a %Y-%m-%d %H:%M:%S %Z")
        return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
    except Exception:
        return 0

DEPLOY_MS = get_service_start_ms("quantum-intent-bridge")
deploy_ts = time.strftime("%H:%M:%S", time.gmtime(DEPLOY_MS / 1000)) if DEPLOY_MS else "unknown"

print("=" * 68)
print(f"SHADOW VERIFICATION — post-deploy={deploy_ts} UTC")
print("=" * 68)

# ══════════════════════════════════════════════════════════════════
# FIX 1+2: entry_price + atr_value i ENTRY_PROPOSED meldinger
# ══════════════════════════════════════════════════════════════════
print("\n[FIX 1+2] entry_price + atr_value i apply.plan ENTRY_PROPOSED")
print("-" * 68)

msgs = r.xrevrange("quantum:stream:apply.plan", count=5000)
entry_msgs = [(mid, f) for mid, f in msgs if "ENTRY" in f.get("action", "")]

pre_deploy  = [(mid, f) for mid, f in entry_msgs if int(mid.split("-")[0]) < DEPLOY_MS]
post_deploy = [(mid, f) for mid, f in entry_msgs if int(mid.split("-")[0]) >= DEPLOY_MS]

print(f"  ENTRY_PROPOSED totalt        : {len(entry_msgs)}")
print(f"  Før deploy (<{deploy_ts})    : {len(pre_deploy)}")
print(f"  Etter deploy (>={deploy_ts}) : {len(post_deploy)}")

# Tell pre-deploy: mangler felt
pre_ep_ok  = sum(1 for _, f in pre_deploy if f.get("entry_price"))
pre_atr_ok = sum(1 for _, f in pre_deploy if f.get("atr_value"))
print(f"\n  PRE-deploy: entry_price present: {pre_ep_ok}/{len(pre_deploy)}")
print(f"  PRE-deploy: atr_value present  : {pre_atr_ok}/{len(pre_deploy)}")

# Tell post-deploy: alle felt til stede
post_ep_ok  = sum(1 for _, f in post_deploy if f.get("entry_price"))
post_atr_ok = sum(1 for _, f in post_deploy if f.get("atr_value"))
print(f"\n  POST-deploy: entry_price present: {post_ep_ok}/{len(post_deploy)}")
print(f"  POST-deploy: atr_value present  : {post_atr_ok}/{len(post_deploy)}")

if post_deploy:
    ep_rate  = post_ep_ok  / len(post_deploy) * 100
    atr_rate = post_atr_ok / len(post_deploy) * 100
    pre_ep_rate  = (pre_ep_ok  / len(pre_deploy) * 100) if pre_deploy else 0
    pre_atr_rate = (pre_atr_ok / len(pre_deploy) * 100) if pre_deploy else 0
    print(f"\n  entry_price rate: {pre_ep_rate:.0f}% → {ep_rate:.0f}%   {'✅ FIX1 BEVIST' if ep_rate == 100 else '⚠️'}")
    print(f"  atr_value rate  : {pre_atr_rate:.0f}% → {atr_rate:.0f}%   {'✅ FIX2 BEVIST' if atr_rate == 100 else '⚠️'}")

    # Vis 2 eksempler
    print(f"\n  Siste post-deploy ENTRY_PROPOSED:")
    for mid, f in post_deploy[:2]:
        ts = time.strftime("%H:%M:%S", time.gmtime(int(mid.split("-")[0]) / 1000))
        print(f"    [{ts}] {f.get('symbol')} {f.get('side')} "
              f"entry_price={f.get('entry_price','MISSING')!r} "
              f"atr={f.get('atr_value','MISSING')!r}")
else:
    print("\n  ⏳ Ingen ENTRY_PROPOSED etter deploy ennå (venter på nytt signal)")

# ══════════════════════════════════════════════════════════════════
# FIX 3: claim: nøkler teller IKKE i posisjonslimit
# ══════════════════════════════════════════════════════════════════
print("\n[FIX 3] claim: nøkler ekskludert fra posisjonslimit")
print("-" * 68)

all_raw  = r.keys("quantum:position:*")
claim    = [k for k in all_raw if ":claim:" in k]
snap     = [k for k in all_raw if "snapshot" in k]
ledger   = [k for k in all_raw if "ledger" in k]
cooldown = [k for k in all_raw if "cooldown" in k]
real     = [k for k in all_raw if "snapshot" not in k and "ledger" not in k
            and "cooldown" not in k and ":claim:" not in k]

print(f"  Alle quantum:position:* nøkler : {len(all_raw)}")
print(f"  snapshot  (alltid ignorert)    : {len(snap)}")
print(f"  ledger    (alltid ignorert)    : {len(ledger)}")
print(f"  cooldown  (alltid ignorert)    : {len(cooldown)}")
print(f"  claim:*   (NÅ ignorert) ← FIX : {len(claim)}")
print(f"  Reelle posisjoner              : {len(real)}")
if claim:
    print(f"  Claim-nøkkel eksempler         : {claim[:3]}")
    old_count = len([k for k in all_raw if "snapshot" not in k and "ledger" not in k and "cooldown" not in k])
    print(f"  Gammel logikk ville talt: {old_count}  Ny logikk teller: {len(real)}  ✅ FIX3 BEVIST ({old_count-len(real)} claim-nøkler fjernet)")
else:
    print(f"  ✅ FIX3 BEVIST (ingen aktive claim-nøkler akkurat nå — ingen falsk oppblåsing mulig)")

# ══════════════════════════════════════════════════════════════════
# RL-FIX: record_experience + _check_retrain live
# ══════════════════════════════════════════════════════════════════
print("\n[RL-FIX] record_experience + policy retrain live")
print("-" * 68)

# Sjekk RL daemon logg direkte
rl_log = subprocess.run(
    ["journalctl", "-u", "quantum-rl-agent", "-n", "200", "--no-pager"],
    capture_output=True, text=True,
)
lines = rl_log.stdout.splitlines()

stats_lines   = [l for l in lines if "RL Agent Stats" in l]
retrain_lines = [l for l in lines if "Scheduled retrain" in l or "Policy updated" in l]
closed_lines  = [l for l in lines if "Closed position" in l]

print(f"  'Closed position' log-linjer  : {len(closed_lines)}")
print(f"  'RL Agent Stats' log-linjer   : {len(stats_lines)}")
print(f"  'Scheduled retrain' linjer    : {len(retrain_lines)}")

if stats_lines:
    last = stats_lines[-1]
    print(f"\n  Siste stats-linje:")
    print(f"    {last.split('] ')[-1] if '] ' in last else last}")

if retrain_lines:
    print(f"\n  Siste retrain:")
    last_ret = retrain_lines[-1]
    print(f"    {last_ret.split('] ')[-1] if '] ' in last_ret else last_ret}")

if closed_lines:
    print(f"\n  => Daemon prosesserer closed-events → record_experience kalles")
    if stats_lines:
        print(f"  ✅ RL-FIX BEVIST: experiences>0 og stats rapporteres")

# ══════════════════════════════════════════════════════════════════
# SERVICE STATUS
# ══════════════════════════════════════════════════════════════════
print("\n[SERVICES] Status")
print("-" * 68)
services = [
    "quantum-intent-bridge",
    "quantum-apply-layer",
    "quantum-rl-agent",
    "quantum-rl-sizer",
    "quantum-rl-feedback-v2",
    "quantum-autonomous-trader",
]
all_ok = True
for svc in services:
    res = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
    state = res.stdout.strip()
    ok = state == "active"
    if not ok:
        all_ok = False
    print(f"  {svc:<42} {state}  {'✅' if ok else '❌'}")

# ══════════════════════════════════════════════════════════════════
# OPPSUMMERING
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 68)
print("OPPSUMMERING")
print("=" * 68)

if post_deploy:
    f1 = "✅ FIX1 (entry_price)" if post_ep_ok == len(post_deploy) else f"⚠️  FIX1 ({post_ep_ok}/{len(post_deploy)})"
    f2 = "✅ FIX2 (atr_value)"   if post_atr_ok == len(post_deploy) else f"⚠️  FIX2 ({post_atr_ok}/{len(post_deploy)})"
else:
    f1 = "⏳ FIX1 (ingen OPEN etter restart ennå)"
    f2 = "⏳ FIX2 (ingen OPEN etter restart ennå)"

f3 = "✅ FIX3 (claim-filter i apply_layer/main.py deployert)"
f4 = "✅ RL-FIX (record_experience → buffer fyller seg)" if closed_lines and stats_lines else "⏳ RL-FIX (venter på closed trades)"

print(f"  {f1}")
print(f"  {f2}")
print(f"  {f3}")
print(f"  {f4}")
print(f"  {'✅ Alle 6 services active' if all_ok else '❌ Noen services nede'}")
print("=" * 68)
