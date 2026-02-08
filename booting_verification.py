import time
import os
from microservices.risk_policy_enforcer import create_enforcer

KEY = "quantum:global:kill_switch"
REASON = "quantum:global:kill_switch:reason"


def snapshot(enforcer, label):
    metrics = enforcer.compute_system_state(symbol="BTCUSDT")
    ks = enforcer.redis.get(KEY)
    ks_reason = enforcer.redis.get(REASON)
    ks_val = ks.decode() if ks else None
    ks_reason_val = ks_reason.decode() if ks_reason else None
    print(f"[{label}] state={metrics.system_state.value} reason={metrics.failure_reason} "
          f"uptime={int(metrics.uptime_seconds or 0)}s grace_remaining={int(metrics.startup_grace_remaining or 0)}s "
          f"kill_switch={ks_val} kill_reason={ks_reason_val}")


def main():
    enforcer = create_enforcer("redis://localhost:6379")

    # T0 after apply-layer restart
    snapshot(enforcer, "T0")

    # Wait for grace window to elapse
    print("Waiting 65s to pass startup grace...")
    time.sleep(65)
    snapshot(enforcer, "T+65")

    # Stop feedback service and wait TTL + margin
    os.system("systemctl stop quantum-rl-feedback-v2")
    print("Feedback stopped; waiting 40s...")
    time.sleep(40)
    snapshot(enforcer, "FB_STOPPED")

    # Start feedback service and wait for heartbeat
    os.system("systemctl start quantum-rl-feedback-v2")
    print("Feedback started; waiting 5s...")
    time.sleep(5)
    snapshot(enforcer, "FB_RESTARTED")


if __name__ == "__main__":
    main()
