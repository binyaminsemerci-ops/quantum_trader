import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';
import StatusBanner from '../components/StatusBanner';
import { safeCurrency } from '../utils/formatters';

export default function Overview() {
  const [dashboard, setDashboard] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/overview/dashboard')
      .then(res => res.json())
      .then(data => setDashboard(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Overview</h1>
        <div className="text-sm text-gray-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      <StatusBanner 
        type="success" 
        message="All systems operational • Live trading active" 
      />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="Daily P&L"
          value={safeCurrency(dashboard?.trading_status?.daily_pnl, 2)}
          change={dashboard?.trading_status?.pnl_percentage || 0}
          trend={dashboard?.trading_status?.pnl_percentage > 0 ? 'up' : 'down'}
          icon="💰"
        />
        <InsightCard
          title="Active Positions"
          value={dashboard?.trading_status?.active_positions || 0}
          icon="📊"
        />
        <InsightCard
          title="System Health"
          value={dashboard?.system_health?.status || 'Loading...'}
          icon="⚡"
        />
        <InsightCard
          title="AI Models Active"
          value={dashboard?.ai_status?.models_active || 0}
          icon="🤖"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-white">Risk Summary</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Portfolio VaR</span>
              <span className="text-white font-semibold">
                ${(dashboard?.risk_summary?.portfolio_var || 0).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Drawdown</span>
              <span className="text-white font-semibold">
                {dashboard?.risk_summary?.max_drawdown || 0}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Sharpe Ratio</span>
              <span className="text-white font-semibold">
                {dashboard?.risk_summary?.sharpe_ratio || 0}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-white">Recent Alerts</h2>
          <div className="space-y-2">
            {dashboard?.alerts?.map((alert: any, idx: number) => (
              <div key={idx} className="flex items-start gap-3 p-3 bg-gray-800 rounded">
                <span className={`text-xl ${
                  alert.severity === 'warning' ? '⚠️' : 'ℹ️'
                }`}>
                  {alert.severity === 'warning' ? '⚠️' : 'ℹ️'}
                </span>
                <div>
                  <p className="text-sm text-gray-300">{alert.message}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
