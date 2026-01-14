#!/usr/bin/env python3
"""
Ensemble v5 Real-Time Monitor Dashboard
---------------------------------------
Live monitoring of all agents, risk metrics, and trading decisions.

Displays:
- Active models status (XGB, LGBM, PatchTST, N-HiTS, Meta)
- Signal variety and confidence distribution
- Meta override ratio
- Governer approvals/rejections
- Risk metrics (drawdown, exposure, win rate)
- Recent trading decisions

Usage:
    python3 ops/monitor_ensemble_v5.py
    python3 ops/monitor_ensemble_v5.py --web  # Web dashboard on port 8050
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available - using text-only mode")

# ---------- LOG PARSERS ----------
def parse_agent_logs(log_file: str, minutes: int = 5) -> Dict:
    """Parse agent logs for recent activity"""
    if not Path(log_file).exists():
        return {}
    
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    agents = defaultdict(list)
    
    try:
        with open(log_file, 'r') as f:
            for line in f.readlines()[-1000:]:  # Last 1000 lines
                if '→' not in line:
                    continue
                
                # Parse timestamp
                try:
                    ts_str = line.split('|')[0].strip()
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    if ts < cutoff:
                        continue
                except:
                    continue
                
                # Parse agent and prediction
                for agent in ['XGB-Agent', 'LGBM-Agent', 'PatchTST-Agent', 'NHiTS-Agent', 'Meta-Agent']:
                    if agent in line:
                        parts = line.split('→')
                        if len(parts) == 2:
                            symbol = parts[0].split()[-1]
                            action = parts[1].split()[0]
                            agents[agent].append({'symbol': symbol, 'action': action, 'time': ts})
                        break
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return dict(agents)

def parse_governer_state(state_file: str) -> Dict:
    """Load governer state"""
    if not Path(state_file).exists():
        return {}
    
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading governer state: {e}")
        return {}

# ---------- TEXT DASHBOARD ----------
def print_text_dashboard():
    """Print text-based dashboard to terminal"""
    print("\n" + "=" * 70)
    print("📊 QUANTUM TRADER v5 - REAL-TIME MONITOR")
    print("=" * 70)
    print(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("-" * 70)
    
    # Parse logs
    log_dir = Path("/var/log/quantum")
    agents_data = {}
    
    for agent_name in ['xgb-agent', 'lgbm-agent', 'patchtst-agent', 'nhits-agent', 'meta-agent']:
        log_file = log_dir / f"{agent_name}.log"
        if log_file.exists():
            data = parse_agent_logs(str(log_file), minutes=5)
            agents_data.update(data)
    
    # Active Models
    print("\n🤖 ACTIVE MODELS (last 5 min):")
    if agents_data:
        for agent, predictions in agents_data.items():
            actions = Counter([p['action'] for p in predictions])
            status = "🟢" if predictions else "🔴"
            print(f"  {status} {agent:20s} | Predictions: {len(predictions):3d} | {dict(actions)}")
    else:
        print("  ⚠️  No recent agent activity detected")
    
    # Signal Variety
    print("\n✨ SIGNAL VARIETY:")
    all_actions = []
    for predictions in agents_data.values():
        all_actions.extend([p['action'] for p in predictions])
    
    if all_actions:
        action_counts = Counter(all_actions)
        total = len(all_actions)
        for action, count in action_counts.most_common():
            pct = (count / total) * 100
            print(f"  {action:6s} | {'█' * int(pct/2):20s} {pct:5.1f}% ({count}/{total})")
    else:
        print("  No signals recorded")
    
    # Governer Stats
    print("\n🛡️  GOVERNER RISK MANAGEMENT:")
    governer_state = parse_governer_state("/app/data/governer_state.json")
    
    if governer_state:
        balance = governer_state.get('current_balance', 0)
        peak = governer_state.get('peak_balance', balance)
        drawdown = ((peak - balance) / peak * 100) if peak > 0 else 0
        
        print(f"  Balance:        ${balance:,.2f}")
        print(f"  Peak Balance:   ${peak:,.2f}")
        print(f"  Drawdown:       {drawdown:.2f}%")
        
        trades = governer_state.get('trade_history', [])
        if trades:
            recent = trades[-20:]
            wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
            win_rate = (wins / len(recent)) * 100
            total_pnl = sum(t.get('pnl', 0) for t in recent)
            
            print(f"  Total Trades:   {len(trades)}")
            print(f"  Recent Win Rate: {win_rate:.1f}% ({wins}/{len(recent)})")
            print(f"  Recent PnL:     ${total_pnl:,.2f}")
    else:
        print("  ⚠️  Governer state not found")
    
    print("\n" + "=" * 70)

# ---------- WEB DASHBOARD ----------
def create_web_dashboard():
    """Create interactive web dashboard with Plotly Dash"""
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not installed. Install with: pip install plotly dash")
        return
    
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        print("❌ Dash not installed. Install with: pip install dash")
        return
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("🤖 Quantum Trader v5 - Live Monitor", style={'textAlign': 'center'}),
        html.Div(id='live-time', style={'textAlign': 'center', 'fontSize': 18}),
        
        dcc.Graph(id='agent-activity'),
        dcc.Graph(id='signal-distribution'),
        dcc.Graph(id='risk-metrics'),
        
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # Update every 5 seconds
            n_intervals=0
        )
    ])
    
    @app.callback(
        [Output('live-time', 'children'),
         Output('agent-activity', 'figure'),
         Output('signal-distribution', 'figure'),
         Output('risk-metrics', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        current_time = f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        # Parse logs
        log_dir = Path("/var/log/quantum")
        agents_data = {}
        
        for agent_name in ['xgb-agent', 'lgbm-agent', 'patchtst-agent', 'nhits-agent', 'meta-agent']:
            log_file = log_dir / f"{agent_name}.log"
            if log_file.exists():
                data = parse_agent_logs(str(log_file), minutes=5)
                agents_data.update(data)
        
        # Agent Activity Chart
        agent_names = list(agents_data.keys())
        prediction_counts = [len(agents_data[a]) for a in agent_names]
        
        fig1 = go.Figure(data=[
            go.Bar(x=agent_names, y=prediction_counts, marker_color='lightblue')
        ])
        fig1.update_layout(
            title="Agent Activity (Last 5 Minutes)",
            xaxis_title="Agent",
            yaxis_title="Predictions"
        )
        
        # Signal Distribution
        all_actions = []
        for predictions in agents_data.values():
            all_actions.extend([p['action'] for p in predictions])
        
        action_counts = Counter(all_actions)
        
        fig2 = go.Figure(data=[
            go.Pie(labels=list(action_counts.keys()), values=list(action_counts.values()))
        ])
        fig2.update_layout(title="Signal Distribution")
        
        # Risk Metrics
        governer_state = parse_governer_state("/app/data/governer_state.json")
        
        if governer_state:
            balance = governer_state.get('current_balance', 0)
            peak = governer_state.get('peak_balance', balance)
            drawdown = ((peak - balance) / peak * 100) if peak > 0 else 0
            
            trades = governer_state.get('trade_history', [])
            recent = trades[-20:] if trades else []
            wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
            win_rate = (wins / len(recent) * 100) if recent else 0
            
            fig3 = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Balance", "Drawdown %", "Win Rate %")
            )
            
            fig3.add_trace(
                go.Indicator(mode="gauge+number", value=balance, title="USD"),
                row=1, col=1
            )
            fig3.add_trace(
                go.Indicator(mode="gauge+number", value=drawdown, title="%"),
                row=1, col=2
            )
            fig3.add_trace(
                go.Indicator(mode="gauge+number", value=win_rate, title="%"),
                row=1, col=3
            )
            
            fig3.update_layout(title="Governer Risk Metrics")
        else:
            fig3 = go.Figure()
            fig3.update_layout(title="Governer Metrics Unavailable")
        
        return current_time, fig1, fig2, fig3
    
    print("🚀 Starting web dashboard on http://localhost:8050")
    app.run_server(debug=False, host='0.0.0.0', port=8050)

# ---------- MAIN ----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble v5 Monitor")
    parser.add_argument('--web', action='store_true', help='Launch web dashboard')
    parser.add_argument('--continuous', action='store_true', help='Continuous text updates')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    if args.web:
        create_web_dashboard()
    elif args.continuous:
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                print_text_dashboard()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped")
    else:
        print_text_dashboard()
