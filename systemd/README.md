# QUANTUM TRADER - SYSTEMD MIGRATION
## EXECUTION READY

All files have been generated and are ready for deployment.

## DIRECTORY STRUCTURE

```
systemd/
├── units/              # systemd service files
│   ├── quantum-redis.service
│   ├── quantum-ai-engine.service
│   ├── quantum-rl-sizer.service
│   ├── quantum-strategy-ops.service
│   ├── quantum-execution.service
│   ├── quantum-ceo-brain.service
│   ├── quantum-strategy-brain.service
│   ├── quantum-risk-brain.service
│   ├── quantum-cross-exchange.service
│   ├── quantum-market-publisher.service
│   ├── quantum-exposure-balancer.service
│   └── quantum-trader.target
│
├── scripts/            # Deployment automation
│   ├── MIGRATE_TO_SYSTEMD.sh  ⭐ MASTER SCRIPT
│   ├── deploy_systemd.sh
│   ├── create_users.sh
│   ├── setup_venvs.sh
│   ├── start_all.sh
│   ├── stop_all.sh
│   ├── verify_health.sh
│   └── restart_service.sh
│
├── configs/            # Service configurations
│   └── redis.conf
│
└── env-templates/      # Environment variable templates
    ├── ai-engine.env
    ├── execution.env
    └── ai-client-base.env
```

## EXECUTION INSTRUCTIONS

### On your VPS (Ubuntu/Debian):

```bash
# 1. Transfer files to VPS
cd quantum_trader
git add systemd/
git commit -m "Add systemd migration infrastructure"
git push origin main

# 2. On VPS, pull latest
ssh root@46.224.116.254
cd /home/qt/quantum_trader
git pull origin main

# 3. Make scripts executable
chmod +x systemd/scripts/*.sh

# 4. Run migration (SINGLE COMMAND)
./systemd/scripts/MIGRATE_TO_SYSTEMD.sh
```

The master script will:
- ✅ Backup Docker data
- ✅ Create system users
- ✅ Setup Python venvs
- ✅ Deploy code to /opt/quantum
- ✅ Install systemd units
- ✅ Configure environment
- ✅ Migrate data from Docker
- ✅ Start all services in correct order
- ✅ Verify health
- ✅ Offer to remove Docker containers

## ARCHITECTURE SUMMARY

### Model Servers (3):
1. **ai-engine** (2GB) - XGBoost, LightGBM, NHiTS, PatchTST
2. **rl-sizer** (384MB) - RL position sizing
3. **strategy-ops** (384MB) - Strategy ensemble

### AI Clients (24):
All other services - NO torch, NO models, lightweight

### Communication:
- Redis: 127.0.0.1:6379
- HTTP APIs: localhost only
- No external ports except via reverse proxy

## POST-MIGRATION

```bash
# View all services
systemctl list-units 'quantum-*'

# Check specific service
systemctl status quantum-ai-engine.service

# View logs
journalctl -u quantum-ai-engine.service -f

# Health check
./systemd/scripts/verify_health.sh

# Restart single service
./systemd/scripts/restart_service.sh ai-engine

# Stop all
./systemd/scripts/stop_all.sh

# Start all
./systemd/scripts/start_all.sh
```

## ROLLBACK

If migration fails:

```bash
systemctl stop quantum-trader.target
cd /home/qt/quantum_trader
docker-compose -f docker-compose.vps.yml up -d
```

## SUCCESS CRITERIA

✅ All services running: `systemctl status quantum-trader.target`
✅ RAM usage < 12GB: `free -h`
✅ Health checks pass: `./verify_health.sh`
✅ No crashes for 1 hour: `journalctl -u 'quantum-*' --since "1 hour ago"`

## COMPLETE SERVICE LIST

| Service | Type | RAM | Port |
|---|---|---|---|
| quantum-redis | INFRA | 512M | 6379 |
| quantum-ai-engine | MODEL-SERVER | 2G | 8001 |
| quantum-rl-sizer | MODEL-SERVER | 384M | - |
| quantum-strategy-ops | MODEL-SERVER | 384M | - |
| quantum-cross-exchange | AI-CLIENT | 256M | - |
| quantum-market-publisher | INFRA | 256M | - |
| quantum-exposure-balancer | AI-CLIENT | 128M | - |
| quantum-ceo-brain | AI-CLIENT | 256M | 8010 |
| quantum-strategy-brain | AI-CLIENT | 128M | 8011 |
| quantum-risk-brain | AI-CLIENT | 128M | 8012 |
| quantum-execution | AI-CLIENT | 1G | 8002 |

(+ 21 more AI-client services)

**Total RAM: ~11.5GB**

---

**STATUS**: READY FOR DEPLOYMENT
**ESTIMATED TIME**: 45-60 minutes
**RISK**: LOW (rollback available)
