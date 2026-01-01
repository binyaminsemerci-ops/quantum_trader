#!/bin/bash
# Smart Disk Manager v2 - NOW WITH BUILD CACHE CLEANUP

REDIS_HOST="localhost"
REDIS_PORT="6379"
DISK_THRESHOLD_WARN=70
DISK_THRESHOLD_SAFE=80
DISK_THRESHOLD_EMERGENCY=90

get_disk_usage() {
    df / | awk 'NR==2 {print $5}' | sed 's/%//'
}

log_event() {
    local level=$1
    local message=$2
    local timestamp=$(date +%s)
    docker exec quantum_redis redis-cli XADD "quantum:system:disk_alerts" "*" timestamp "$timestamp" level "$level" usage "$(get_disk_usage)%" message "$message" > /dev/null 2>&1
}

safe_cleanup() {
    echo "🟡 [SAFE CLEANUP] Starting intelligent cleanup..."
    
    # 1. Remove dangling images
    DANGLING=$(docker images -f "dangling=true" -q | wc -l)
    if [ "$DANGLING" -gt 0 ]; then
        echo "   Removing $DANGLING dangling images..."
        docker image prune -f
    fi

    # 2. Remove stopped containers older than 7 days
    docker container prune -f --filter "until=168h" > /dev/null

    # 3. Remove unused networks
    docker network prune -f > /dev/null

    # 4. Clean build cache older than 7 days (KEY FIX!)
    echo "   Pruning build cache >7 days..."
    docker builder prune -af --filter "until=168h" --force > /dev/null

    # 5. Rotate system logs >30 days
    find /var/log -name "*.log.*" -mtime +30 -delete 2>/dev/null
    find /var/log -name "*.gz" -mtime +30 -delete 2>/dev/null

    # 6. Clean old Redis backups >14 days
    find /mnt/redis-backups -name "*.rdb" -mtime +14 -delete 2>/dev/null

    FREED=$(df / | awk 'NR==2 {print $4}')
    echo "✅ Safe cleanup complete. Available: $FREED"
    log_event "INFO" "Safe cleanup freed space"
}

aggressive_cleanup() {
    echo "🟠 [AGGRESSIVE CLEANUP] Disk critically full..."

    # 1. CRITICAL: Prune ALL build cache (not just old)
    echo "   Pruning ALL build cache..."
    CACHE_BEFORE=$(docker system df | awk 'NR==5 {print $4}')
    docker builder prune --all --force
    CACHE_AFTER=$(docker system df | awk 'NR==5 {print $4}')
    echo "   Build cache: $CACHE_BEFORE → $CACHE_AFTER"

    # 2. Remove old image versions (keep latest 2 per repo)
    REPOS=$(docker images --format "{{.Repository}}" | sort -u | grep -v "<none>")
    for repo in $REPOS; do
        COUNT=$(docker images "$repo" --format "{{.Tag}}" | grep -v "<none>" | wc -l)
        if [ "$COUNT" -gt 2 ]; then
            echo "   $repo has $COUNT versions, keeping latest 2..."
            OLD_IMAGES=$(docker images "$repo" --format "{{.ID}} {{.CreatedAt}}" | sort -k2 -r | tail -n +3 | awk '{print $1}')
            for img in $OLD_IMAGES; do
                IN_USE=$(docker ps -a --filter "ancestor=$img" -q | wc -l)
                if [ "$IN_USE" -eq 0 ]; then
                    docker rmi -f "$img" 2>/dev/null
                fi
            done
        fi
    done

    # 3. Also run safe cleanup
    safe_cleanup

    FREED=$(df / | awk 'NR==2 {print $4}')
    echo "✅ Aggressive cleanup complete. Available: $FREED"
    log_event "WARNING" "Aggressive cleanup performed"
}

emergency_cleanup() {
    echo "🔴 [EMERGENCY] Disk >90% full!"

    # 1. Stop non-critical retraining workers
    docker stop quantum_retraining_worker 2>/dev/null

    # 2. Aggressive cleanup (includes build cache)
    aggressive_cleanup

    # 3. Clear APT cache
    apt-get clean 2>/dev/null

    FREED=$(df / | awk 'NR==2 {print $4}')
    echo "🚨 Emergency cleanup complete. Available: $FREED"
    log_event "ERROR" "Emergency cleanup triggered! Disk was >90%"
}

main() {
    USAGE=$(get_disk_usage)
    echo "📊 Disk usage: ${USAGE}%"

    if [ "$USAGE" -ge "$DISK_THRESHOLD_EMERGENCY" ]; then
        echo "🔴 EMERGENCY: Disk usage ${USAGE}% >= ${DISK_THRESHOLD_EMERGENCY}%"
        emergency_cleanup

    elif [ "$USAGE" -ge "$DISK_THRESHOLD_SAFE" ]; then
        echo "🟠 WARNING: Disk usage ${USAGE}% >= ${DISK_THRESHOLD_SAFE}%"
        aggressive_cleanup

    elif [ "$USAGE" -ge "$DISK_THRESHOLD_WARN" ]; then
        echo "🟡 NOTICE: Disk usage ${USAGE}% >= ${DISK_THRESHOLD_WARN}%"
        safe_cleanup

    else
        echo "✅ Disk usage healthy: ${USAGE}%"
        log_event "INFO" "Disk usage healthy at ${USAGE}%"
    fi

    FINAL_USAGE=$(get_disk_usage)
    echo "📈 Final disk usage: ${FINAL_USAGE}%"
    echo ""
    echo "📦 Top 5 space consumers:"
    docker images --format "table {{.Repository}}\t{{.Size}}" | sort -k2 -h -r | head -6
}

main
