#!/bin/bash
# Shadow Validation Monitor - 24-48 Hour Performance Tracking
# Monitors: WIN rate, PNL, confidence, governance, model performance

LOG_FILE="shadow_validation_$(date +%Y%m%d_%H%M%S).log"
INTERVAL=1800  # 30 minutes
DURATION=$((48*3600))  # 48 hours
START_TIME=$(date +%s)

echo "🚀 Starting Shadow Validation Monitor"
echo "Duration: 48 hours | Interval: 30 min | Log: $LOG_FILE"
echo "Started at: $(date)"
echo "================================================"

# Function: Get ensemble metrics
get_ensemble_metrics() {
    echo ""
    echo "📊 ENSEMBLE PERFORMANCE"
    echo "========================"
    
    # Recent predictions
    echo "Last 10 predictions:"
    docker logs quantum_ai_engine 2>&1 | \
        grep '\[CHART\] ENSEMBLE' | \
        tail -10 | \
        sed 's/.*\[CHART\] //'
    
    # Action distribution (last 50)
    echo ""
    echo "Action distribution (last 50):"
    docker logs quantum_ai_engine 2>&1 | \
        grep 'Ensemble:.*\(BUY\|SELL\|HOLD\)' | \
        tail -50 | \
        grep -oP '(BUY|SELL|HOLD)' | \
        sort | uniq -c
    
    # Average confidence
    echo ""
    echo "Average ensemble confidence:"
    docker logs quantum_ai_engine 2>&1 | \
        grep '🎯 Ensemble returned:' | \
        tail -50 | \
        grep -oP 'confidence=\K[0-9.]+' | \
        awk '{sum+=$1; count++} END {if(count>0) printf "%.4f (n=%d)\n", sum/count, count; else print "No data"}'
}

# Function: Get PNL metrics
get_pnl_metrics() {
    echo ""
    echo "💰 PNL METRICS"
    echo "=============="
    
    # Check if stream exists
    stream_len=$(docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.pnl 2>/dev/null || echo "0")
    echo "ExitBrain PNL stream length: $stream_len events"
    
    if [ "$stream_len" -gt "0" ]; then
        # Total PNL from ExitBrain stream
        echo ""
        echo "Total PNL (from stream):"
        docker exec quantum_redis redis-cli XREVRANGE quantum:stream:exitbrain.pnl + - COUNT 100 2>/dev/null | \
            awk '
                /^[0-9]+-[0-9]+$/ {entry_id=$0}
                /^pnl_trend$/ {getline; pnl=$0; total+=pnl; count++; if(pnl>0) wins++; else losses++}
                END {
                    if(count>0) {
                        printf "Total: %.2f | Events: %d | Wins: %d | Losses: %d | Win Rate: %.1f%%\n", 
                            total, count, wins, losses, (wins*100/count)
                    } else {
                        print "No PNL data available"
                    }
                }
            '
        
        # Recent PNL events
        echo ""
        echo "Recent PNL events (last 10):"
        docker exec quantum_redis redis-cli XREVRANGE quantum:stream:exitbrain.pnl + - COUNT 10 2>/dev/null | \
            awk '
                /^[0-9]+-[0-9]+$/ {if(entry!="") printf "%-12s PNL: %8s  Conf: %s\n", symbol, pnl, conf; entry=$0; symbol="?"; pnl="?"; conf="?"}
                /^symbol$/ {getline; symbol=$0}
                /^pnl_trend$/ {getline; pnl=$0}
                /^confidence$/ {getline; conf=$0}
                END {if(entry!="") printf "%-12s PNL: %8s  Conf: %s\n", symbol, pnl, conf}
            '
    else
        echo "No PNL events in stream yet"
    fi
    
    # Closed trades
    echo ""
    closed_len=$(docker exec quantum_redis redis-cli XLEN quantum:stream:trade.closed 2>/dev/null || echo "0")
    echo "Closed trades stream length: $closed_len"
    
    if [ "$closed_len" -gt "0" ]; then
        echo "Recent closed trades (last 5):"
        docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.closed + - COUNT 5 2>/dev/null | \
            awk '
                /^[0-9]+-[0-9]+$/ {if(entry!="") print entry; entry=$0; symbol="?"; pnl="?"}
                /^symbol$/ {getline; symbol=$0}
                /^pnl$/ {getline; pnl=$0}
                END {if(entry!="") printf "%-12s PNL: %s\n", symbol, pnl}
            '
    fi
}

# Function: Get governance metrics
get_governance_metrics() {
    echo ""
    echo "⚖️ GOVERNANCE METRICS"
    echo "===================="
    
    # Current weights
    echo "Current model weights (latest cycle):"
    docker logs quantum_ai_engine 2>&1 | \
        grep 'Cycle complete.*Weights:' | \
        tail -1 | \
        sed 's/.*Weights: //'
    
    # Weight adjustments over time
    echo ""
    echo "Weight evolution (last 5 cycles per symbol):"
    docker logs quantum_ai_engine 2>&1 | \
        grep 'Cycle complete' | \
        tail -20 | \
        awk -F'for ' '{symbol=$2; sub(/ -.*/,"",symbol); weights=$2; sub(/.*Weights: /,"",weights); print symbol, weights}' | \
        column -t
    
    # Weight change rate
    echo ""
    echo "Weight volatility (std dev of changes):"
    docker logs quantum_ai_engine 2>&1 | \
        grep 'Adjusted weights' | \
        tail -50 | \
        wc -l | \
        awk '{printf "Adjustments in last 50 logs: %d\n", $1}'
}

# Function: Get model health
get_model_health() {
    echo ""
    echo "🏥 MODEL HEALTH"
    echo "==============="
    
    # Individual model predictions
    echo "Model breakdown (last 10):"
    docker logs quantum_ai_engine 2>&1 | \
        grep '\[CHART\] ENSEMBLE' | \
        tail -10 | \
        grep -oP '(XGB|LGBM|NH|PT):[A-Z]+/[0-9.]+' | \
        awk '{model=substr($1,1,index($1,":")-1); rest=substr($1,index($1,":")+1); action=substr(rest,1,index(rest,"/")-1); conf=substr(rest,index(rest,"/")+1); count[model"_"action]++; sum_conf[model]+=conf; count_conf[model]++} END {for(m in count_conf) {avg=sum_conf[m]/count_conf[m]; print m, "avg_conf:", avg; for(a in count) if(index(a,m"_")>0) print "  ", a, count[a]}}' | \
        sort
    
    # Model errors
    echo ""
    echo "Recent model errors:"
    docker logs quantum_ai_engine 2>&1 | \
        grep -E '(ERROR|Failed|Exception)' | \
        tail -10
}

# Function: Get confidence distribution
get_confidence_metrics() {
    echo ""
    echo "🎯 CONFIDENCE METRICS"
    echo "===================="
    
    # Confidence histogram
    echo "Confidence distribution (last 100):"
    docker logs quantum_ai_engine 2>&1 | \
        grep '🎯 Ensemble returned:' | \
        tail -100 | \
        grep -oP 'confidence=\K[0-9.]+' | \
        awk '{
            if($1<0.3) low++;
            else if($1<0.5) mid_low++;
            else if($1<0.7) mid++;
            else if($1<0.9) mid_high++;
            else high++;
            total++;
        } END {
            printf "  0.0-0.3 (low):      %d (%.1f%%)\n", low, low/total*100;
            printf "  0.3-0.5 (mid-low):  %d (%.1f%%)\n", mid_low, mid_low/total*100;
            printf "  0.5-0.7 (mid):      %d (%.1f%%)\n", mid, mid/total*100;
            printf "  0.7-0.9 (mid-high): %d (%.1f%%)\n", mid_high, mid_high/total*100;
            printf "  0.9-1.0 (high):     %d (%.1f%%)\n", high, high/total*100;
        }'
    
    # Confidence vs action
    echo ""
    echo "Confidence by action type:"
    docker logs quantum_ai_engine 2>&1 | \
        tail -200 | \
        grep -E '(Ensemble:.*confidence=)' | \
        awk '{action=""; conf=""; for(i=1;i<=NF;i++) {if($i~/BUY|SELL|HOLD/) action=$i; if($i~/confidence=/) {split($i,a,"="); conf=a[2]; gsub(/[,)]/,"",conf)}} if(action!="" && conf!="") {sum[action]+=conf; count[action]++}} END {for(a in sum) printf "%s: %.4f (n=%d)\n", a, sum[a]/count[a], count[a]}' | \
        sort
}

# Function: System health
get_system_health() {
    echo ""
    echo "🔧 SYSTEM HEALTH"
    echo "================"
    
    # Container status
    echo "AI Engine container:"
    docker ps --filter name=quantum_ai_engine --format "table {{.Status}}\t{{.Names}}"
    
    # Memory usage
    echo ""
    echo "Memory usage:"
    docker stats quantum_ai_engine --no-stream --format "CPU: {{.CPUPerc}} | MEM: {{.MemUsage}}"
    
    # Redis connection
    echo ""
    echo "Redis health:"
    docker exec quantum_redis redis-cli PING 2>/dev/null || echo "Redis not reachable"
    
    # Stream lengths
    echo ""
    echo "Stream sizes:"
    docker exec quantum_redis redis-cli XLEN quantum:stream:ai.decision.made 2>/dev/null | awk '{print "AI decisions: "$1}'
    docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.pnl 2>/dev/null | awk '{print "ExitBrain PNL: "$1}'
    docker exec quantum_redis redis-cli XLEN quantum:stream:trade.closed 2>/dev/null | awk '{print "Closed trades: "$1}'
}

# Main monitoring loop
echo "Starting monitoring loop..."
echo ""

iteration=0
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    # Check if duration exceeded
    if [ $ELAPSED -ge $DURATION ]; then
        echo ""
        echo "✅ 48-hour monitoring period complete!"
        echo "Total duration: $((ELAPSED/3600)) hours"
        echo "Final report saved to: $LOG_FILE"
        break
    fi
    
    iteration=$((iteration + 1))
    REMAINING=$((DURATION - ELAPSED))
    
    # Timestamp and iteration
    {
        echo ""
        echo "================================================"
        echo "📍 ITERATION $iteration | $(date)"
        echo "Elapsed: $((ELAPSED/3600))h ${ELAPSED%3600}m | Remaining: $((REMAINING/3600))h $((REMAINING%3600))m"
        echo "================================================"
        
        # Run all metric collectors
        get_system_health
        get_ensemble_metrics
        get_pnl_metrics
        get_governance_metrics
        get_model_health
        get_confidence_metrics
        
        echo ""
        echo "Next check in 30 minutes..."
        echo "================================================"
    } | tee -a "$LOG_FILE"
    
    # Sleep until next iteration
    sleep $INTERVAL
done

echo ""
echo "📊 Monitoring complete. Review $LOG_FILE for full history."
