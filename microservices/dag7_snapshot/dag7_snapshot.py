#!/usr/bin/env python3
"""
dag7_snapshot.py — Immutable System Snapshot

Produces a point-in-time snapshot of the running Quantum OS:

  1. Redis BGSAVE — flushes current state to disk (dump.rdb)
  2. File checksums — SHA256 of every deployed microservice .py file
  3. Service manifest — systemd unit name + active/sub state for all quantum-* services
  4. Config snapshot — all quantum:config:* Redis hashes
  5. Health truth — current runtime truth snapshot
  6. Equity state — current equity / peak / drawdown

Output:
  /opt/quantum/snapshots/<TAG>/MANIFEST.json    (on VPS disk)
  quantum:dag7:snapshot:latest                  (Redis hash, no TTL)
  quantum:dag7:snapshot:history                 (Redis list, capped 10)

The snapshot is immutable: once written it is never modified by this script.
Re-running creates a new tag.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import logging

import redis as redis_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s snap %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dag7")

REDIS_HOST   = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))
SNAPSHOT_DIR = "/opt/quantum/snapshots"
MICROSERVICE_ROOT = "/opt/quantum/microservices"

STATE_KEY   = "quantum:dag7:snapshot:latest"
HISTORY_KEY = "quantum:dag7:snapshot:history"


# ── Helpers ───────────────────────────────────────────────────────────────

def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v is not None else ""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ── Steps ─────────────────────────────────────────────────────────────────

def step_redis_bgsave(r: redis_lib.Redis) -> dict:
    logger.info("[DAG7] Step 1: Redis BGSAVE ...")
    r.bgsave()
    # Poll for completion (max 30s)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        info = r.info("persistence")
        if info.get("rdb_bgsave_in_progress", 1) == 0:
            last_save_ts = info.get("rdb_last_save_time", 0)
            last_save_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_save_ts))
            logger.info("[DAG7] BGSAVE complete — last_save=%s", last_save_iso)

            # Copy dump.rdb to snapshot dir (done later after dir created)
            rdb_dir      = _run(["redis-cli", "CONFIG", "GET", "dir"]).split()
            rdb_filename = _run(["redis-cli", "CONFIG", "GET", "dbfilename"]).split()
            rdb_path = ""
            if len(rdb_dir) >= 2 and len(rdb_filename) >= 2:
                rdb_path = os.path.join(rdb_dir[1], rdb_filename[1])

            return {
                "status":         "OK",
                "last_save_ts":   last_save_ts,
                "last_save_iso":  last_save_iso,
                "rdb_path":       rdb_path,
                "rdb_size_bytes": os.path.getsize(rdb_path) if rdb_path and os.path.isfile(rdb_path) else 0,
            }
        time.sleep(1)
    return {"status": "TIMEOUT", "note": "BGSAVE did not complete within 30s"}


def step_file_checksums() -> list[dict]:
    logger.info("[DAG7] Step 2: Computing file checksums ...")
    files = []
    for root, _, fnames in os.walk(MICROSERVICE_ROOT):
        for fname in sorted(fnames):
            if not fname.endswith(".py"):
                continue
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, MICROSERVICE_ROOT)
            try:
                sha  = _sha256(abs_path)
                size = os.path.getsize(abs_path)
                mtime = int(os.path.getmtime(abs_path))
                files.append({
                    "path":   rel_path,
                    "sha256": sha,
                    "size":   size,
                    "mtime":  mtime,
                })
            except Exception as e:
                files.append({"path": rel_path, "error": str(e)})
    logger.info("[DAG7] Checksummed %d files", len(files))
    return files


def step_service_manifest() -> list[dict]:
    logger.info("[DAG7] Step 3: Collecting service manifest ...")
    out = _run(["systemctl", "list-units", "quantum-*.service",
                "--no-pager", "--no-legend", "--all"], timeout=15)
    services = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            services.append({
                "unit":   parts[0],
                "load":   parts[1],
                "active": parts[2],
                "sub":    parts[3],
                "ok":     parts[2] == "active" and parts[3] == "running",
            })
    running   = sum(1 for s in services if s.get("ok"))
    degraded  = len(services) - running
    logger.info("[DAG7] Services: %d total, %d running, %d degraded/other",
                len(services), running, degraded)
    return services


def step_config_snapshot(r: redis_lib.Redis) -> dict:
    logger.info("[DAG7] Step 4: Snapshotting quantum:config:* ...")
    configs = {}
    for key in sorted(r.keys("quantum:config:*")):
        key_str  = _decode(key)
        key_type = _decode(r.type(key))
        if key_type == "hash":
            raw = r.hgetall(key)
            configs[key_str] = {_decode(k): _decode(v) for k, v in raw.items()}
        elif key_type == "string":
            configs[key_str] = _decode(r.get(key))
        else:
            configs[key_str] = f"<type:{key_type}>"
    logger.info("[DAG7] Captured %d config keys", len(configs))
    return configs


def step_health_snapshot(r: redis_lib.Redis) -> dict:
    logger.info("[DAG7] Step 5: Capturing health truth ...")
    raw = r.hgetall("quantum:health:truth:latest")
    return {_decode(k): _decode(v) for k, v in raw.items()}


def step_equity_snapshot(r: redis_lib.Redis) -> dict:
    logger.info("[DAG7] Step 6: Capturing equity state ...")
    raw = r.hgetall("quantum:equity:current")
    return {_decode(k): _decode(v) for k, v in raw.items()}


def step_dag_states(r: redis_lib.Redis) -> dict:
    logger.info("[DAG7] Step 7: Capturing DAG service states ...")
    dag_keys = [
        "quantum:dag3:hw_stops:latest",
        "quantum:dag4:deadlock_guard:latest",
        "quantum:dag5:lockdown_guard:latest",
        "quantum:dag6:chaos_test:latest",
    ]
    states = {}
    for key in dag_keys:
        raw = r.hgetall(key)
        if raw:
            states[key] = {_decode(k): _decode(v) for k, v in raw.items()}
        else:
            states[key] = None
    return states


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    tag = time.strftime("v%Y-%m-%d-hardening")
    snap_dir = os.path.join(SNAPSHOT_DIR, tag)

    logger.info("[DAG7] Immutable Snapshot — tag=%s", tag)

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
        logger.info("[DAG7] Redis OK")
    except redis_lib.ConnectionError as e:
        logger.error("[DAG7] Redis FAILED: %s", e)
        sys.exit(1)

    # Check if snapshot already exists
    existing = r.hget(STATE_KEY, "tag")
    if existing and _decode(existing) == tag:
        logger.warning("[DAG7] Snapshot '%s' already exists — creating incremental", tag)
        # Append timestamp suffix to make unique
        tag = time.strftime("v%Y-%m-%d-%H%M%S-hardening")
        snap_dir = os.path.join(SNAPSHOT_DIR, tag)

    os.makedirs(snap_dir, exist_ok=True)
    logger.info("[DAG7] Snapshot directory: %s", snap_dir)

    # Run all steps
    bgsave   = step_redis_bgsave(r)
    files    = step_file_checksums()
    services = step_service_manifest()
    configs  = step_config_snapshot(r)
    health   = step_health_snapshot(r)
    equity   = step_equity_snapshot(r)
    dag_st   = step_dag_states(r)

    # Copy RDB if found
    rdb_path = bgsave.get("rdb_path", "")
    rdb_snap = ""
    if rdb_path and os.path.isfile(rdb_path):
        rdb_snap = os.path.join(snap_dir, "dump.rdb")
        import shutil
        shutil.copy2(rdb_path, rdb_snap)
        logger.info("[DAG7] Copied %s → %s (%d bytes)",
                    rdb_path, rdb_snap, os.path.getsize(rdb_snap))

    # Summary stats
    total_svcs   = len(services)
    running_svcs = sum(1 for s in services if s.get("ok"))
    total_files  = len(files)
    total_bytes  = sum(f.get("size", 0) for f in files)

    manifest = {
        "tag":              tag,
        "created_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_mode":      _decode(r.get("quantum:system:mode") or b""),
        "summary": {
            "services_total":   total_svcs,
            "services_running": running_svcs,
            "services_degraded": total_svcs - running_svcs,
            "py_files_total":   total_files,
            "py_files_bytes":   total_bytes,
            "config_keys":      len(configs),
            "rdb_snapshot":     rdb_snap or "not_copied",
            "rdb_size_bytes":   bgsave.get("rdb_size_bytes", 0),
        },
        "equity":           equity,
        "health_truth":     health,
        "dag_states":       dag_st,
        "bgsave":           bgsave,
        "services":         services,
        "configs":          configs,
        "files":            files,
    }

    # Write MANIFEST.json
    manifest_path = os.path.join(snap_dir, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    manifest_size = os.path.getsize(manifest_path)
    manifest_sha  = _sha256(manifest_path)

    logger.info("[DAG7] MANIFEST.json written: %s (%d bytes, sha256=%s...)",
                manifest_path, manifest_size, manifest_sha[:16])

    # Write immutable seal file
    seal = {
        "tag":            tag,
        "manifest_sha256": manifest_sha,
        "manifest_size":  manifest_size,
        "sealed_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sealed_by":      "dag7_snapshot",
    }
    with open(os.path.join(snap_dir, "SEAL"), "w") as f:
        json.dump(seal, f, indent=2)

    # Publish to Redis
    redis_summary = {
        "tag":               tag,
        "created_at":        manifest["created_at"],
        "system_mode":       manifest["system_mode"],
        "services_total":    str(total_svcs),
        "services_running":  str(running_svcs),
        "services_degraded": str(total_svcs - running_svcs),
        "py_files_total":    str(total_files),
        "py_files_bytes":    str(total_bytes),
        "manifest_path":     manifest_path,
        "manifest_sha256":   manifest_sha,
        "rdb_snapshot":      rdb_snap or "not_copied",
        "equity":            equity.get("equity", "?"),
        "peak":              equity.get("peak", "?"),
        "drawdown_pct":      str(round(
            max(0, (float(equity.get("peak", 0) or 0) - float(equity.get("equity", 0) or 0))
                / max(1, float(equity.get("peak", 1) or 1)) * 100), 2)),
        "health_overall":    health.get("overall_health", "?"),
        "bgsave_status":     bgsave.get("status", "?"),
        "status":            "OK",
    }
    r.hset(STATE_KEY, mapping=redis_summary)
    r.lpush(HISTORY_KEY, json.dumps({"tag": tag, "ts": manifest["created_at"],
                                      "services_running": running_svcs,
                                      "manifest_sha256": manifest_sha}))
    r.ltrim(HISTORY_KEY, 0, 9)

    # Final report
    logger.info("")
    logger.info("=" * 60)
    logger.info("[DAG7] SNAPSHOT COMPLETE")
    logger.info("  tag              : %s", tag)
    logger.info("  manifest         : %s", manifest_path)
    logger.info("  manifest_sha256  : %s", manifest_sha)
    logger.info("  services running : %d / %d", running_svcs, total_svcs)
    logger.info("  py files         : %d  (%d bytes)", total_files, total_bytes)
    logger.info("  system mode      : %s", manifest["system_mode"])
    logger.info("  health overall   : %s", health.get("overall_health", "?"))
    logger.info("  equity           : %s  peak=%s  dd=%s%%",
                equity.get("equity", "?"), equity.get("peak", "?"),
                redis_summary["drawdown_pct"])
    logger.info("  redis bgsave     : %s  rdb=%s",
                bgsave.get("status"), rdb_snap or "n/a")
    logger.info("=" * 60)

    # Count actually failed/errored (not just inactive from --all)
    failed_out = _run(["systemctl", "list-units", "quantum-*.service",
                        "--no-pager", "--no-legend",
                        "--state=failed", "--state=activating"], timeout=10)
    truly_failed = len([l for l in failed_out.splitlines() if l.strip()])
    if truly_failed > 3:
        logger.warning("[DAG7] WARNING: %d services failed/activating — review before release",
                       truly_failed)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
