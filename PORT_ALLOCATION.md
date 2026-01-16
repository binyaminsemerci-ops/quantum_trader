# Port Allocation - Quantum Trader VPS

## Summary
✅ **No port conflicts detected**  
All services running on unique ports with proper isolation.

## Port Mapping

| Port | Service | Container | Type | Purpose |
|------|---------|-----------|------|---------|
| **6379** | Redis | quantum_redis | Internal/External | Cache & Pub/Sub |
| **8001** | AI Engine | quantum_ai_engine | External | AI Model Inference |
| **8002** | Portfolio Governance | quantum_portfolio_governance | External | Position Management |
| **8006** | Universe OS | quantum_universe_os | External | System Orchestration |
| **8007** | Model Supervisor | quantum_model_supervisor | External | Model Monitoring |
| **8013** | PIL | quantum_pil | External | Position Intent Logger |
| **8020** | Model Federation | quantum_model_federation | External | Multi-Model Ensemble |
| **8025** | Dashboard Backend | quantum_dashboard_backend | External | FastAPI Dashboard API |
| **8026** | RL Dashboard | quantum_rl_dashboard | External | Flask RL Monitoring |
| **8889** | Dashboard Frontend | quantum_dashboard_frontend | External | React Frontend (Nginx) |

## Internal Services (No External Ports)
- quantum_auto_executor
- quantum_binance_pnl_tracker (NEW - polls Binance for PnL)
- quantum_ceo_brain
- quantum_cross_exchange
- quantum_market_publisher (internal 8001)
- quantum_meta_regime
- quantum_position_monitor
- quantum_risk_brain
- quantum_rl_calibrator
- quantum_rl_feedback_v2
- quantum_rl_monitor (NEW - logs RL rewards)
- quantum_rl_sizing_agent
- quantum_strategic_evolution
- quantum_strategic_memory
- quantum_strategy_brain
- quantum_strategy_ops

## Port Availability Check
```bash
# Check all listening ports
netstat -tlnp | grep LISTEN | grep -E ":(6379|8001|8002|8006|8007|8013|8020|8025|8026|8889)"
```

## Recent Changes (Jan 1, 2026)

### 1. Binance PnL Tracker (NEW)
- **Service**: binance_pnl_tracker
- **Port**: None (internal only)
- **Purpose**: Poll Binance /fapi/v2/account every 15s for unrealized PnL + /fapi/v1/income for realized PnL
- **Output**: Writes to Redis keys `quantum:rl:reward:{symbol}`

### 2. Dashboard Backend Update
- **Port**: 8025
- **API**: `/rl-dashboard/`
- **New fields**:
  - `unrealized_pnl` - Current position PnL (USDT)
  - `realized_pnl` - Closed trades PnL (USDT)
  - `total_pnl` - Combined PnL (USDT)
  - `unrealized_pct` - Position PnL percentage
  - `realized_pct` - Realized PnL percentage
  - `realized_trades` - Number of closed trades

### 3. RL Dashboard (Flask - port 8026)
- **Status**: Active but not primary dashboard
- **Purpose**: Internal RL monitoring (reads from exitbrain.pnl stream)
- **Note**: Different from main dashboard (port 8025)

## Port Conflicts History
- **None detected** - All services use unique ports
- quantum_dashboard_backend and quantum_rl_dashboard use different ports (8025 vs 8026)

## Firewall Status
```bash
# Check UFW rules
ufw status numbered

# Test external access
curl -I http://46.224.116.254:8025/rl-dashboard/
```

## Access URLs
- **Main Dashboard**: https://app.quantumfond.com (NGINX proxy to port 8889)
- **Dashboard API**: https://app.quantumfond.com/api/rl-dashboard/ (proxy to port 8025)
- **RL Dashboard**: http://46.224.116.254:8026 (requires SSH tunnel or firewall rule)
- **AI Engine**: http://46.224.116.254:8001/health
- **Redis**: redis://46.224.116.254:6379 (protected by firewall)

## Recommendations
1. ✅ All critical ports (8001, 8002, 8025, 8889) are open and working
2. ✅ Port 8026 (RL Dashboard) blocked by Hetzner Cloud Firewall - use SSH tunnel if needed
3. ✅ Redis port 6379 accessible (consider restricting to internal network only)

## SSH Tunnel for RL Dashboard
```powershell
# Windows/WSL
wsl ssh -i ~/.ssh/hetzner_fresh -L 8026:localhost:8026 root@46.224.116.254 -N

# Then access: http://localhost:8026
```

