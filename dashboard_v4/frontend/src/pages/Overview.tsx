import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface OverviewData {
  portfolio: { pnl: number; exposure: number; positions: number };
  ai: { accuracy: number; sharpe: number };
  risk: { var: number; regime: string };
  system: { cpu: number; ram: number; containers: number };
}

export default function Overview() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchOverview = async () => {
      try {
        const [portfolio, ai, risk, system] = await Promise.all([
          fetch(`${API_BASE_URL}/portfolio/status`).then(r => r.json()),
          fetch(`${API_BASE_URL}/ai/status`).then(r => r.json()),
          fetch(`${API_BASE_URL}/risk/metrics`).then(r => r.json()),
          fetch(`${API_BASE_URL}/system/health`).then(r => r.json())
        ]);
        setData({ portfolio, ai, risk, system });
        setLoading(false);
      } catch (err) {
        console.error('Failed to load overview:', err);
        setLoading(false);
      }
    };

    fetchOverview();
    const interval = setInterval(fetchOverview, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading system overview...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load overview data</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-green-400">System Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="Portfolio PnL"
          value={`$${data.portfolio.pnl.toLocaleString()}`}
          subtitle={`${data.portfolio.positions} active positions`}
          color="green"
        />
        
        <InsightCard
          title="AI Accuracy"
          value={`${(data.ai.accuracy * 100).toFixed(1)}%`}
          subtitle={`Sharpe: ${data.ai.sharpe.toFixed(2)}`}
          color="blue"
        />
        
        <InsightCard
          title="Risk (VaR 95%)"
          value={`${(data.risk.var * 100).toFixed(2)}%`}
          subtitle={`Regime: ${data.risk.regime}`}
          color="yellow"
        />
        
        <InsightCard
          title="System Load"
          value={`${data.system.cpu.toFixed(1)}%`}
          subtitle={`RAM: ${data.system.ram.toFixed(1)}% | ${data.system.containers} containers`}
          color="purple"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Portfolio Exposure</h2>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Current Exposure</span>
                <span className="text-white font-bold">
                  {(data.portfolio.exposure * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${data.portfolio.exposure * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Quick Stats</h2>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Active Positions:</span>
              <span className="text-white font-bold">{data.portfolio.positions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">AI Models:</span>
              <span className="text-white font-bold">4</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Market Regime:</span>
              <span className="text-white font-bold">{data.risk.regime}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
