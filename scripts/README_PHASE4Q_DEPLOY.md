# Phase 4Q Deployment Scripts

## Overview

Deployment og validering scripts for Portfolio Governance Agent (Phase 4Q).

## Files

### 1. deploy_phase4q.sh (Linux/VPS)
Bash script for deployment på VPS/Linux server.

**Usage:**
```bash
cd /home/qt/quantum_trader/scripts
chmod +x deploy_phase4q.sh
./deploy_phase4q.sh
```

### 2. deploy_phase4q.ps1 (Windows)
PowerShell script for deployment på Windows.

**Usage:**
```powershell
cd C:\quantum_trader\scripts
.\deploy_phase4q.ps1
```

## What These Scripts Do

1. **Pull Latest Code** - Updates from GitHub
2. **Build Docker Image** - Builds portfolio_governance service
3. **Start Service** - Launches container with docker-compose
4. **Validate Running** - Checks container status
5. **Health Check** - Tests AI Engine integration
6. **Redis Validation** - Verifies streams and keys
7. **Simulate Data** - Adds test PnL events
8. **Check Logs** - Displays recent container logs
9. **Summary Report** - Shows deployment status

## Expected Output

```
🎯 PHASE 4Q+ DEPLOYMENT SUCCESSFULLY COMPLETED!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Service Status:
   • Container: quantum_portfolio_governance (RUNNING)
   • Memory Stream: quantum:stream:portfolio.memory (2 samples)
   • Current Policy: BALANCED
   • Portfolio Score: 0.0

📡 Integration Points:
   • AI Engine Health: http://localhost:8001/health
   • Redis Policy Key: quantum:governance:policy
   • Redis Score Key: quantum:governance:score
```

## Prerequisites

### Linux/VPS
- Docker & docker-compose installed
- Git configured
- curl and jq (optional but recommended)
- Access to port 8001 (AI Engine)

### Windows
- Docker Desktop running
- PowerShell 5.1 or higher
- Git configured

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs quantum_portfolio_governance

# Rebuild image
docker-compose -f docker-compose.vps.yml build --no-cache portfolio-governance

# Restart service
docker-compose -f docker-compose.vps.yml restart portfolio-governance
```

### Redis Connection Failed
```bash
# Check Redis is running
docker ps | grep redis

# Test Redis connection
docker exec redis redis-cli ping
# Expected: PONG
```

### Health Endpoint Not Available
```bash
# Check AI Engine is running
docker ps | grep ai_engine

# Check AI Engine logs
docker logs quantum_ai_engine | grep -i portfolio

# Test endpoint manually
curl http://localhost:8001/health
```

## Continuous Monitoring

### Watch Policy Changes
```bash
# Linux
watch -n 15 'docker exec redis redis-cli GET quantum:governance:policy'

# Windows
while ($true) { docker exec redis redis-cli GET quantum:governance:policy; Start-Sleep 15 }
```

### Monitor Logs
```bash
docker logs -f quantum_portfolio_governance
```

### Check Memory Stream
```bash
# Get stream length
docker exec redis redis-cli XLEN quantum:stream:portfolio.memory

# View recent events
docker exec redis redis-cli XREVRANGE quantum:stream:portfolio.memory + - COUNT 10
```

## Manual Validation Commands

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}" | grep portfolio

# Get current policy
docker exec redis redis-cli GET quantum:governance:policy

# Get portfolio score
docker exec redis redis-cli GET quantum:governance:score

# Get policy parameters
docker exec redis redis-cli GET quantum:governance:params | jq '.'

# Check AI Engine integration
curl -s http://localhost:8001/health | jq '.metrics.portfolio_governance'
```

## Post-Deployment

After successful deployment:

1. ✅ Monitor for 30 minutes - Ensure service stays healthy
2. ✅ Check policy updates - Should see BALANCED initially
3. ✅ Verify memory accumulation - Stream should grow with trades
4. ✅ Test integration - ExitBrain and RL Agent should read policy
5. ✅ Monitor logs - No errors or warnings

## Integration Testing

### Test ExitBrain Integration
```bash
# ExitBrain should read policy
docker logs quantum_position_monitor | grep -i governance
```

### Test RL Agent Integration
```bash
# RL Agent should read parameters
docker logs quantum_ai_engine | grep -i "governance"
```

## Rollback Procedure

If deployment fails:

```bash
# Stop service
docker-compose -f docker-compose.vps.yml stop portfolio-governance

# Remove container
docker-compose -f docker-compose.vps.yml rm -f portfolio-governance

# Clear Redis keys (optional)
docker exec redis redis-cli DEL quantum:governance:policy
docker exec redis redis-cli DEL quantum:governance:score

# Restart other services
docker-compose -f docker-compose.vps.yml restart ai-engine
```

## Support

For issues or questions:
1. Check logs: `docker logs quantum_portfolio_governance`
2. Review documentation: [AI_PORTFOLIO_GOVERNANCE_VALIDATION.md](../AI_PORTFOLIO_GOVERNANCE_VALIDATION.md)
3. Test Redis manually: `docker exec redis redis-cli ping`
4. Verify AI Engine health: `curl http://localhost:8001/health`

---

**Version:** 1.0.0  
**Phase:** 4Q  
**Updated:** 2025-12-21
