#!/bin/bash
# PHASE 4 Complete Stack Monitoring Script
# Usage: ./monitor_phase4.sh

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         PHASE 4 COMPLETE STACK - SYSTEM STATUS                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# System Overview
echo "📊 SYSTEM OVERVIEW"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8001/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d['metrics']
print(f'Models Active:      {m[\"models_loaded\"]}')
print(f'Governance Active:  {\"✅\" if m[\"governance_active\"] else \"❌\"}')
print(f'Retrainer Active:   {\"✅\" if m[\"adaptive_retrainer\"][\"enabled\"] else \"❌\"}')
print(f'Validator Active:   {\"✅\" if m[\"model_validator\"][\"enabled\"] else \"❌\"}')
"
echo ""

# Governance Status (Phase 4D+4E)
echo "🛡️  PHASE 4D+4E: GOVERNANCE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8001/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
g = d['metrics']['governance']
print(f'Registered Models:  {len(g[\"models\"])}')
print(f'Drift Threshold:    {g[\"drift_threshold\"]*100}%')
print('')
print('Model Weights:')
for name, data in g['models'].items():
    print(f'  • {name:12s}: {data[\"weight\"]:.3f}')
"
echo ""

# Retraining Status (Phase 4F)
echo "🔄 PHASE 4F: ADAPTIVE RETRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8001/health | python3 -c "
import sys, json
from datetime import datetime, timedelta
d = json.load(sys.stdin)
r = d['metrics']['adaptive_retrainer']
print(f'Interval:           {r[\"retrain_interval_seconds\"]}s (4 hours)')
print(f'Retrain Count:      {r[\"retrain_count\"]}')
print(f'Last Retrain:       {r[\"last_retrain\"][:19]}')

# Calculate time until next
secs = r['time_until_next_seconds']
hours = secs // 3600
mins = (secs % 3600) // 60
secs_remaining = secs % 60
print(f'Next Retrain:       {hours}h {mins}m {secs_remaining}s')
print('')
print('Models:')
for name, path in r['model_paths'].items():
    print(f'  • {name:12s}: {path}')
"
echo ""

# Validation Status (Phase 4G)
echo "🧪 PHASE 4G: MODEL VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8001/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
v = d['metrics']['model_validator']
print(f'Criteria:')
print(f'  • MAPE improvement: ≥{v[\"criteria\"][\"mape_improvement_required\"]}')
print(f'  • Sharpe improvement: {\"Required\" if v[\"criteria\"][\"sharpe_improvement_required\"] else \"Not Required\"}')
print('')
print(f'Validation Log:     {v[\"validation_log_path\"]}')
print(f'Recent Validations: {len(v[\"recent_validations\"])}')
if v['recent_validations']:
    print('')
    for entry in v['recent_validations'][-5:]:
        print(f'  {entry}')
else:
    print('  (No validations yet - waiting for first retraining cycle)')
"
echo ""

# Model Files
echo "📁 MODEL FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker exec quantum_ai_engine ls -lh /app/models/ 2>/dev/null | grep -E "patchtst|nhits" || echo "  (Model directory not yet accessible)"
echo ""

# Recent Logs
echo "📝 RECENT LOGS (Last 10 Phase 4 events)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker logs quantum_ai_engine --tail 500 2>/dev/null | grep -E "PHASE 4|Validator|Retrainer|Governance" | tail -10 || echo "  (Logs not available)"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                        SUMMARY                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
curl -s http://localhost:8001/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d['metrics']
r = m['adaptive_retrainer']
secs = r['time_until_next_seconds']
hours = secs // 3600
mins = (secs % 3600) // 60

status = '✅ ALL SYSTEMS OPERATIONAL' if all([
    m['models_loaded'] == 12,
    m['governance_active'],
    r['enabled'],
    m['model_validator']['enabled']
]) else '⚠️  SYSTEM DEGRADED'

print(status)
print('')
print(f'Next milestone: Retraining cycle in {hours}h {mins}m')
print(f'Following that: First validation (~5 min after retraining)')
print('')
print('All Phase 4 components active and ready.')
"
echo ""
echo "Generated: $(date)"
