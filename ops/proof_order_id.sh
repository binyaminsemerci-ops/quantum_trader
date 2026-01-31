#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# SAFE: READ-ONLY - Quantum Trader Order Execution Proof
# ═══════════════════════════════════════════════════════════════════════════
# Searches apply.result stream for executed=True + order_id evidence
# Exit codes: 0=proof found, 3=no proof, 1=error

set -euo pipefail

# Defaults
SSH_HOST="root@46.224.116.254"
SSH_KEY="$HOME/.ssh/hetzner_fresh"
RESULT_COUNT=400
FILTER_SYMBOL=""
FILTER_PLAN_ID=""
FOLLOW_MODE=false
TIMEOUT=120
OUTPUT_FORMAT="human"

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Search for trade execution proof (executed=True + order_id) in apply.result stream.

OPTIONS:
    --host <host>      SSH host (default: $SSH_HOST)
    --key <path>       SSH key path (default: $SSH_KEY)
    --count <N>        Apply.result entries to check (default: $RESULT_COUNT)
    --symbol <SYM>     Filter by symbol (e.g., TRXUSDT)
    --plan_id <ID>     Filter by specific plan_id
    --follow           Poll every 3s until proof found or timeout
    --timeout <SEC>    Timeout for --follow mode (default: $TIMEOUT)
    --json             Output JSON format
    -h, --help         Show this help

EXIT CODES:
    0 = Execution proof found (executed=True + order_id present)
    3 = No proof found (infrastructure OK but no executed trades)
    1 = Runtime error or invalid usage

EXAMPLES:
    $0
    $0 --count 200
    $0 --symbol TRXUSDT
    $0 --follow --symbol TRXUSDT --timeout 180
    $0 --json
    $0 --plan_id a2c7d3839f66267a

PROOF CRITERIA:
    - executed field must be "true" or "True"
    - order_id field must exist and be non-empty
    - If filters provided, match must satisfy all filters
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) SSH_HOST="$2"; shift 2 ;;
        --key) SSH_KEY="$2"; shift 2 ;;
        --count) RESULT_COUNT="$2"; shift 2 ;;
        --symbol) FILTER_SYMBOL="$2"; shift 2 ;;
        --plan_id) FILTER_PLAN_ID="$2"; shift 2 ;;
        --follow) FOLLOW_MODE=true; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --json) OUTPUT_FORMAT="json"; shift ;;
        -h|--help) usage ;;
        *) echo "Error: Unknown option $1" >&2; usage ;;
    esac
done

# SSH command wrapper
ssh_exec() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SSH_HOST" "$@"
}

# Search for execution proof
search_proof() {
    local results
    if ! results=$(ssh_exec "redis-cli XREVRANGE quantum:stream:apply.result + - COUNT $RESULT_COUNT 2>/dev/null"); then
        echo "Error: Failed to fetch apply.result stream" >&2
        return 1
    fi
    
    # Parse results for executed=true + order_id
    local current_entry=""
    local plan_id=""
    local symbol=""
    local executed=""
    local order_id=""
    local matches=0
    local match_details=""
    
    while IFS= read -r line; do
        # Detect entry boundaries (timestamp lines like "1769838...")
        if [[ "$line" =~ ^[0-9]+-[0-9]+$ ]]; then
            # Process previous entry if complete
            if [[ -n "$plan_id" ]]; then
                # Check if it matches filters
                local filter_match=true
                [[ -n "$FILTER_SYMBOL" && "$symbol" != "$FILTER_SYMBOL" ]] && filter_match=false
                [[ -n "$FILTER_PLAN_ID" && "$plan_id" != "$FILTER_PLAN_ID" ]] && filter_match=false
                
                # Check for proof criteria
                if [[ "$filter_match" == true ]] && \
                   [[ "$executed" =~ ^[Tt]rue$ ]] && \
                   [[ -n "$order_id" && "$order_id" != "null" ]]; then
                    ((matches++))
                    match_details="plan_id=$plan_id symbol=$symbol order_id=$order_id"
                fi
            fi
            
            # Reset for new entry
            plan_id=""
            symbol=""
            executed=""
            order_id=""
        fi
        
        # Parse fields
        [[ "$line" == "plan_id" ]] && { read -r plan_id; continue; }
        [[ "$line" == "symbol" ]] && { read -r symbol; continue; }
        [[ "$line" == "executed" ]] && { read -r executed; continue; }
        [[ "$line" == "order_id" ]] && { read -r order_id; continue; }
    done <<< "$results"
    
    # Check last entry
    if [[ -n "$plan_id" ]]; then
        local filter_match=true
        [[ -n "$FILTER_SYMBOL" && "$symbol" != "$FILTER_SYMBOL" ]] && filter_match=false
        [[ -n "$FILTER_PLAN_ID" && "$plan_id" != "$FILTER_PLAN_ID" ]] && filter_match=false
        
        if [[ "$filter_match" == true ]] && \
           [[ "$executed" =~ ^[Tt]rue$ ]] && \
           [[ -n "$order_id" && "$order_id" != "null" ]]; then
            ((matches++))
            match_details="plan_id=$plan_id symbol=$symbol order_id=$order_id"
        fi
    fi
    
    # Output results
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        cat << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "proof_found": $([ $matches -gt 0 ] && echo "true" || echo "false"),
  "executed_true_count": $matches,
  "filters": {
    "symbol": "${FILTER_SYMBOL:-null}",
    "plan_id": "${FILTER_PLAN_ID:-null}"
  },
  "search_params": {
    "entries_checked": $RESULT_COUNT
  },
  "last_match": $([ -n "$match_details" ] && echo "\"$match_details\"" || echo "null")
}
EOF
    else
        echo "╔═══════════════════════════════════════════════════════════╗"
        echo "║  EXECUTION PROOF SEARCH                                   ║"
        echo "╚═══════════════════════════════════════════════════════════╝"
        echo ""
        echo "Search parameters:"
        echo "  Entries checked: $RESULT_COUNT"
        [[ -n "$FILTER_SYMBOL" ]] && echo "  Symbol filter:   $FILTER_SYMBOL"
        [[ -n "$FILTER_PLAN_ID" ]] && echo "  Plan ID filter:  $FILTER_PLAN_ID"
        echo ""
        
        if [[ $matches -gt 0 ]]; then
            echo "✅ PROOF FOUND: $matches execution(s) with order_id"
            echo ""
            echo "Latest match:"
            echo "  $match_details"
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "✅ END-TO-END EXECUTION VERIFIED"
            echo "═══════════════════════════════════════════════════════════"
        else
            echo "⏳ NO PROOF FOUND"
            echo ""
            echo "No entries found with:"
            echo "  - executed=True"
            echo "  - order_id present"
            [[ -n "$FILTER_SYMBOL" ]] && echo "  - symbol=$FILTER_SYMBOL"
            [[ -n "$FILTER_PLAN_ID" ]] && echo "  - plan_id=$FILTER_PLAN_ID"
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "⏳ INFRASTRUCTURE OK, WAITING FOR EXECUTION"
            echo "═══════════════════════════════════════════════════════════"
        fi
    fi
    
    return $([ $matches -gt 0 ] && echo 0 || echo 3)
}

# Main execution
main() {
    if [[ "$FOLLOW_MODE" == true ]]; then
        local start_time=$(date +%s)
        local elapsed=0
        
        echo "Following apply.result stream (polling every 3s, timeout ${TIMEOUT}s)..."
        echo ""
        
        while [[ $elapsed -lt $TIMEOUT ]]; do
            if search_proof; then
                exit 0
            fi
            
            echo "No proof yet (${elapsed}s elapsed)... retrying in 3s"
            sleep 3
            elapsed=$(( $(date +%s) - start_time ))
        done
        
        echo ""
        echo "Timeout reached (${TIMEOUT}s) - no proof found"
        exit 3
    else
        search_proof
        exit $?
    fi
}

main
