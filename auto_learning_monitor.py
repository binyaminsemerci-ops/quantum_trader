"""
Learning Cadence Monitor - AUTOMATIC MODE

Runs continuously in background, checking if learning conditions are met.
When ready: automatically runs calibration and deploys if safe.

Safety limits:
- Max risk score: 5% (auto-approve)
- Max calibrations per day: 2
- Requires validation pass
- Logs all actions for audit
"""

import asyncio
import logging
import sys
import os
import subprocess
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .cadence_policy import LearningCadencePolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/qt/quantum_trader/logs/learning_cadence.log')
    ]
)

logger = logging.getLogger(__name__)

# Auto-learning configuration
AUTO_LEARNING_ENABLED = True
MAX_RISK_SCORE_AUTO_APPROVE = 5.0  # Auto-approve if risk < 5%
MAX_CALIBRATIONS_PER_DAY = 2
MIN_IMPROVEMENT_PCT = 5.0  # Require at least 5% improvement
COOLDOWN_HOURS = 12  # Min hours between auto-calibrations

# State tracking
state_file = Path("/home/qt/quantum_trader/data/auto_learning_state.json")


def load_state():
    """Load auto-learning state"""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "last_auto_calibration": None,
        "calibrations_today": 0,
        "last_calibration_date": None,
        "total_auto_calibrations": 0,
        "last_job_id": None
    }


def save_state(state):
    """Save auto-learning state"""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def can_auto_calibrate(state) -> tuple[bool, str]:
    """Check if auto-calibration is allowed"""
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    
    # Reset daily counter
    if state.get("last_calibration_date") != today:
        state["calibrations_today"] = 0
        state["last_calibration_date"] = today
        save_state(state)
    
    # Check daily limit
    if state["calibrations_today"] >= MAX_CALIBRATIONS_PER_DAY:
        return False, f"Daily limit reached ({MAX_CALIBRATIONS_PER_DAY}/day)"
    
    # Check cooldown
    if state.get("last_auto_calibration"):
        last = datetime.fromisoformat(state["last_auto_calibration"])
        elapsed = (now - last).total_seconds() / 3600
        if elapsed < COOLDOWN_HOURS:
            return False, f"Cooldown active ({COOLDOWN_HOURS - elapsed:.1f}h remaining)"
    
    return True, "OK"


async def run_calibration():
    """Run calibration analysis and return result"""
    try:
        cmd = [
            "/opt/quantum/venvs/ai-engine/bin/python",
            "-m", "microservices.learning.calibration_cli",
            "run"
        ]
        
        result = subprocess.run(
            cmd,
            cwd="/home/qt/quantum_trader",
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse output for job ID and status
        output = result.stdout + result.stderr
        
        # Find job ID
        job_id = None
        for line in output.split("\n"):
            if "Job ID:" in line:
                job_id = line.split("Job ID:")[-1].strip()
                break
            if "cal_" in line and "pending_approval" in line.lower():
                # Extract job ID from status line
                import re
                match = re.search(r'cal_\d{8}_\d{6}', line)
                if match:
                    job_id = match.group()
                    break
        
        # Find risk score
        risk_score = None
        for line in output.split("\n"):
            if "Risk:" in line or "risk=" in line.lower():
                import re
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    risk_score = float(match.group(1))
                    break
        
        # Find improvement
        improvement = None
        for line in output.split("\n"):
            if "Improvement:" in line or "improvement_pct" in line:
                import re
                match = re.search(r'[+]?(\d+\.?\d*)%', line)
                if match:
                    improvement = float(match.group(1))
                    break
        
        return {
            "success": result.returncode == 0,
            "job_id": job_id,
            "risk_score": risk_score,
            "improvement": improvement,
            "output": output[-2000:]  # Last 2000 chars
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Calibration timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def approve_calibration(job_id: str):
    """Approve and deploy calibration"""
    try:
        # Copy JSON to config
        json_path = f"/tmp/calibration_{job_id}.json"
        config_path = "/home/qt/quantum_trader/config/calibration.json"
        
        # Backup current
        backup_cmd = f"cp {config_path} {config_path}.backup.$(date +%Y%m%d_%H%M%S)"
        subprocess.run(backup_cmd, shell=True, check=False)
        
        # Deploy new
        deploy_cmd = f"cp {json_path} {config_path}"
        result = subprocess.run(deploy_cmd, shell=True, capture_output=True)
        
        if result.returncode != 0:
            return {"success": False, "error": "Failed to copy calibration"}
        
        # Mark training complete via API
        import requests
        response = requests.post(
            "http://localhost:8003/training/completed",
            json={"action": "calibration", "notes": f"Auto-approved {job_id}"},
            timeout=10
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "api_response": response.json() if response.ok else response.text
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


async def check_readiness_once(policy: LearningCadencePolicy):
    """Single readiness check with detailed logging"""
    try:
        result = policy.evaluate_learning_readiness()

        # Build status message
        status_emoji = "🟢" if result["ready"] else "⏸️"

        logger.info(f"{status_emoji} LEARNING READINESS CHECK")
        logger.info(f"  Gate: {'✅ PASSED' if result['gate_passed'] else '❌ FAILED'} - {result['gate_reason']}")
        logger.info(f"  Trigger: {'🔥 FIRED' if result['trigger_fired'] else '⏳ WAITING'} - {result['trigger_reason']} (type={result['trigger_type']})")
        logger.info(f"  Authorization: {result['allowed_actions'] if result['allowed_actions'] else 'None'}")
        logger.info(f"  Stats: {result['stats']['total_trades']} trades ({result['stats']['new_trades']} new), "
                   f"{result['stats']['time_span_days']:.1f} days span, "
                   f"WR={result['stats']['win_rate']:.1%}, LR={result['stats']['loss_rate']:.1%}")
        logger.info(f"  Last training: {result['stats']['last_training']}, Total trainings: {result['stats']['total_trainings']}")

        return result

    except Exception as e:
        logger.error(f"❌ Readiness check failed: {e}", exc_info=True)
        return None


async def auto_learning_cycle(policy: LearningCadencePolicy, state: dict):
    """Execute automatic learning cycle if conditions are met"""
    
    # Check readiness
    result = await check_readiness_once(policy)
    
    if not result or not result.get("ready"):
        return False, "Not ready for learning"
    
    # Check if auto-calibration is allowed
    can_run, reason = can_auto_calibrate(state)
    if not can_run:
        logger.info(f"⏸️ Auto-calibration blocked: {reason}")
        return False, reason
    
    # Check if calibration is in allowed actions
    if "calibration" not in result.get("allowed_actions", []):
        logger.info("⏸️ Calibration not in allowed actions")
        return False, "Calibration not authorized"
    
    logger.warning("🚀 AUTO-LEARNING TRIGGERED!")
    logger.info("  Running calibration analysis...")
    
    # Run calibration
    cal_result = await run_calibration()
    
    if not cal_result.get("success"):
        logger.error(f"❌ Calibration failed: {cal_result.get('error', 'Unknown')}")
        return False, f"Calibration failed: {cal_result.get('error')}"
    
    job_id = cal_result.get("job_id")
    risk_score = cal_result.get("risk_score", 100)
    improvement = cal_result.get("improvement", 0)
    
    logger.info(f"  Job ID: {job_id}")
    logger.info(f"  Risk Score: {risk_score}%")
    logger.info(f"  Improvement: {improvement}%")
    
    # Safety checks for auto-approve
    if risk_score > MAX_RISK_SCORE_AUTO_APPROVE:
        logger.warning(f"⚠️ Risk too high for auto-approve ({risk_score}% > {MAX_RISK_SCORE_AUTO_APPROVE}%)")
        logger.warning(f"   Manual approval required: calibration_cli.py approve {job_id}")
        return False, f"Risk too high ({risk_score}%)"
    
    if improvement < MIN_IMPROVEMENT_PCT:
        logger.warning(f"⚠️ Improvement too small ({improvement}% < {MIN_IMPROVEMENT_PCT}%)")
        logger.warning(f"   Skipping auto-deploy")
        return False, f"Improvement too small ({improvement}%)"
    
    # Auto-approve!
    logger.warning(f"✅ AUTO-APPROVING calibration {job_id}")
    logger.info(f"  Risk: {risk_score}% (max: {MAX_RISK_SCORE_AUTO_APPROVE}%)")
    logger.info(f"  Improvement: {improvement}% (min: {MIN_IMPROVEMENT_PCT}%)")
    
    approve_result = await approve_calibration(job_id)
    
    if approve_result.get("success"):
        # Update state
        state["last_auto_calibration"] = datetime.now(timezone.utc).isoformat()
        state["calibrations_today"] += 1
        state["total_auto_calibrations"] += 1
        state["last_job_id"] = job_id
        save_state(state)
        
        logger.warning(f"🎉 AUTO-LEARNING COMPLETE!")
        logger.info(f"  Deployed: {job_id}")
        logger.info(f"  Total auto-calibrations: {state['total_auto_calibrations']}")
        return True, f"Deployed {job_id}"
    else:
        logger.error(f"❌ Auto-approve failed: {approve_result.get('error')}")
        return False, f"Approve failed: {approve_result.get('error')}"


async def monitor_loop(check_interval_seconds: int = 300):
    """
    Main monitoring loop with automatic learning.

    Args:
        check_interval_seconds: Time between checks (default 5 minutes)
    """
    policy = LearningCadencePolicy()
    state = load_state()

    logger.info("=" * 80)
    logger.info("🎓 LEARNING CADENCE MONITOR STARTED")
    logger.info(f"   Mode: AUTOMATIC LEARNING ENABLED")
    logger.info(f"   Auto-approve max risk: {MAX_RISK_SCORE_AUTO_APPROVE}%")
    logger.info(f"   Min improvement: {MIN_IMPROVEMENT_PCT}%")
    logger.info(f"   Max per day: {MAX_CALIBRATIONS_PER_DAY}")
    logger.info(f"   Cooldown: {COOLDOWN_HOURS}h")
    logger.info(f"   Check interval: {check_interval_seconds}s ({check_interval_seconds/60:.1f} minutes)")
    logger.info(f"   CLM storage: {policy.clm_path}")
    logger.info(f"   State file: {state_file}")
    logger.info("=" * 80)

    iteration = 0

    while True:
        iteration += 1
        logger.info(f"--- Check #{iteration} at {datetime.now(timezone.utc).isoformat()} ---")
        
        if AUTO_LEARNING_ENABLED:
            success, message = await auto_learning_cycle(policy, state)
            if success:
                logger.info(f"✅ Auto-learning completed: {message}")
            else:
                logger.debug(f"ℹ️ No auto-learning: {message}")
        else:
            await check_readiness_once(policy)

        logger.info(f"⏰ Next check in {check_interval_seconds}s")
        await asyncio.sleep(check_interval_seconds)


def main():
    """Entry point for monitor service"""
    import argparse

    parser = argparse.ArgumentParser(description="Learning Cadence Monitor - AUTO MODE")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Disable automatic learning (logging only)"
    )

    args = parser.parse_args()
    
    global AUTO_LEARNING_ENABLED
    if args.manual:
        AUTO_LEARNING_ENABLED = False
        logger.info("Running in MANUAL mode (logging only)")

    try:
        asyncio.run(monitor_loop(check_interval_seconds=args.interval))
    except KeyboardInterrupt:
        logger.info("🛑 Monitor stopped by user")
    except Exception as e:
        logger.error(f"💥 Monitor crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
