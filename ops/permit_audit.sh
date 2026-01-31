#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# SAFE: READ-ONLY - Quantum Trader 3-Permit Gate Infrastructure Audit
# ═══════════════════════════════════════════════════════════════════════════
# Audits all three permit sources: Governor, P3.3 Position Gate, P2.6 Portfolio Gate
# Exit codes: 0=all present, 2=missing permits, 1=error

set -euo pipefail

# Defaults
REMOTE_MODE=false
SSH_HOST="root@46.224.116.254"
SSH_KEY="$HOME/.ssh/hetzner_fresh"
SAMPLE_COUNT=3
OUTPUT_FORMAT="human"

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Audit Quantum Trader 3-permit gate infrastructure (read-only).

OPTIONS:
    --remote           Execute via SSH (default: local execution)
    --host <host>      SSH host (only with --remote, default: $SSH_HOST)
    --key <path>       SSH key path (only with --remote, default: $SSH_KEY)
    --sample <N>       Sample keys per type (default: $SAMPLE_COUNT)
    --json             Output JSON format
    -h, --help         Show this help

EXIT CODES:
    0 = All three permit sources present and active
    2 = One or more permit sources missing (infrastructure gap)
    1 = Runtime error or invalid usage

EXAMPLES:
    # Local execution (on VPS)
    $0
    $0 --json
    $0 --sample 5
    
    # Remote execution (from workstation)
    $0 --remote
    $0 --remote --host root@myserver --key ~/.ssh/mykey

PERMIT PATTERNS:
    Governor:       quantum:permit:{plan_id}
    P2.6 Portfolio: quantum:permit:p26:{plan_id}
    P3.3 Position:  quantum:permit:p33:{plan_id}
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE_MODE=true; shift ;;
        --host) SSH_HOST="$2"; shift 2 ;;
        --key) SSH_KEY="$2"; shift 2 ;;
        --sample) SAMPLE_COUNT="$2"; shift 2 ;;
        --json) OUTPUT_FORMAT="json"; shift ;;
        -h|--help) usage ;;
        *) echo "Error: Unknown option $1" >&2; usage ;;
    esac
done

# Command wrapper (local or remote)
exec_cmd() {
    if [[ "$REMOTE_MODE" == true ]]; then
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SSH_HOST" "$@"
    else
        bash -c "$@"
    fi
}

# Main audit logic
main() {exec_cmd
    local exit_code=0
    
    # Fetch all permits
    local all_permits
    if ! all_permits=$(ssh_exec "redis-cli --scan --pattern 'quantum:permit:*' 2>/dev/null"); then
        echo "Error: Failed to connect to Redis" >&2
        exit 1
    fi
    
    # Count by type
    local p26_count=$(echo "$all_permits" | grep -c "p26:" || true)
    local p33_count=$(echo "$all_permits" | grep -c "p33:" || true)
    local gov_count=$(echo "$all_permits" | grep -v "p26:\|p33:" | grep -c "quantum:permit:" || true)
    
    # Check for missing permits
    if [[ $gov_count -eq 0 ]] || [[ $p26_count -eq 0 ]] || [[ $p33_count -eq 0 ]]; then
        exit_code=2
    fi
    
    # Sample keys
    local p26_samples=$(echo "$all_permits" | grep "p26:" | head -n "$SAMPLE_COUNT")
    local p33_samples=$(echo "$all_permits" | grep "p33:" | head -n "$SAMPLE_COUNT")
    local gov_samples=$(echo "$all_permits" | grep -v "p26:\|p33:" | head -n "$SAMPLE_COUNT")
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        output_json "$gov_count" "$p26_count" "$p33_count" "$gov_samples" "$p26_samples" "$p33_samples" "$exit_code"
    else
        output_human "$gov_count" "$p26_count" "$p33_count" "$gov_samples" "$p26_samples" "$p33_samples" "$exit_code"
    fi
    
    exit "$exit_code"
}

# Human-readable output
output_human() {
    local gov_count=$1 p26_count=$2 p33_count=$3
    local gov_samples=$4 p26_samples=$5 p33_samples=$6
    local exit_code=$7
    
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  QUANTUM TRADER 3-PERMIT GATE AUDIT                       ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    
    # Status indicators
    local gov_status="✅" p26_status="✅" p33_status="✅"
    [[ $gov_count -eq 0 ]] && gov_status="❌"
    [[ $p26_count -eq 0 ]] && p26_status="❌"
    [[ $p33_count -eq 0 ]] && p33_status="❌"
    
    echo "PERMIT COUNTS:"
    echo "  $gov_status Governor:                $gov_count permits"
    echo "  $p26_status P2.6 Portfolio Gate:     $p26_count permits"
    echo "  $p33_status P3.3 Position Gate:      $p33_count permits"
    echo ""
    
    echo "PATTERN CLARIFICATION:"
    echo "  Governor:       quantum:permit:{plan_id}"
    echo "  P2.6 Portfolio: quantum:permit:p26:{plan_id}"
    echo "  P3.3 Position:  quantum:permit:p33:{plan_id}"
    echo ""
    
    # Sample detailsexec_cmd "redis-cli TTL '$key' 2>/dev/null" || echo "-1")
            local value=$(exec_cmd "redis-cli GET '$key' 2>/dev/null | head -c 300" || echo "")
            echo "  Key: $key"
            echo "  TTL: ${ttl}s"
            echo "  Val: ${value:0:100}..."
            echo ""
        done <<< "$gov_samples"
    fi
    
    if [[ $p26_count -gt 0 ]]; then
        echo "P2.6 SAMPLES (up to $SAMPLE_COUNT):"
        while IFS= read -r key; do
            [[ -z "$key" ]] && continue
            local ttl=$(exec_cmd "redis-cli TTL '$key' 2>/dev/null" || echo "-1")
            echo "  $key: TTL=${ttl}s"
        done <<< "$p26_samples"
        echo ""
    fi
    
    if [[ $p33_count -gt 0 ]]; then
        echo "P3.3 SAMPLES (up to $SAMPLE_COUNT):"
        while IFS= read -r key; do
            [[ -z "$key" ]] && continue
            local ttl=$(exec_cmd
    if [[ $p33_count -gt 0 ]]; then
        echo "P3.3 SAMPLES (up to $SAMPLE_COUNT):"
        while IFS= read -r key; do
            [[ -z "$key" ]] && continue
            local ttl=$(ssh_exec "redis-cli TTL '$key' 2>/dev/null" || echo "-1")
            echo "  $key: TTL=${ttl}s"
        done <<< "$p33_samples"
        echo ""
    fi
    
    echo "═══════════════════════════════════════════════════════════"
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ INFRASTRUCTURE: ALL PERMIT SOURCES ACTIVE"
    else
        echo "❌ INFRASTRUCTURE: MISSING PERMIT SOURCES DETECTED"
        [[ $gov_count -eq 0 ]] && echo "   - Governor permits: MISSING"
        [[ $p26_count -eq 0 ]] && echo "   - P2.6 permits: MISSING"
        [[ $p33_count -eq 0 ]] && echo "   - P3.3 permits: MISSING"
    fi
    echo "═══════════════════════════════════════════════════════════"
}

# JSON output
output_json() {
    local gov_count=$1 p26_count=$2 p33_count=$3
    local gov_samples=$4 p26_samples=$5 p33_samples=$6
    local exit_code=$7
    
    cat << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "infrastructure_status": $([ $exit_code -eq 0 ] && echo "\"operational\"" || echo "\"degraded\""),
  "exit_code": $exit_code,
  "permits": {
    "governor": {
      "count": $gov_count,
      "pattern": "quantum:permit:{plan_id}",
      "status": $([ $gov_count -gt 0 ] && echo "\"active\"" || echo "\"missing\"")
    },
    "p26_portfolio_gate": {
      "count": $p26_count,
      "pattern": "quantum:permit:p26:{plan_id}",
      "status": $([ $p26_count -gt 0 ] && echo "\"active\"" || echo "\"missing\"")
    },
    "p33_position_gate": {
      "count": $p33_count,
      "pattern": "quantum:permit:p33:{plan_id}",
      "status": $([ $p33_count -gt 0 ] && echo "\"active\"" || echo "\"missing\"")
    }
  },
  "sample_keys": {
    "governor": $(echo "$gov_samples" | head -n 3 | jq -R . | jq -s . 2>/dev/null || echo "[]"),
    "p26": $(echo "$p26_samples" | head -n 3 | jq -R . | jq -s . 2>/dev/null || echo "[]"),
    "p33": $(echo "$p33_samples" | head -n 3 | jq -R . | jq -s . 2>/dev/null || echo "[]")
  }
}
EOF
}

main
