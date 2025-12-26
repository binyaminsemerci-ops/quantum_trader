import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';
import { safeInt } from '../utils/formatters';

export default function Risk() {
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/risk/metrics')
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Risk Management</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <InsightCard
          title="Portfolio VaR"
          value={`$${safeInt(metrics?.portfolio_var?.value)}`}
          icon="📉"
        />
        <InsightCard
          title="Max Drawdown"
          value={`${metrics?.drawdown?.max || 0}%`}
          icon="⬇️"
        />
        <InsightCard
          title="Leverage"
          value={`${metrics?.exposure?.leverage || 0}x`}
          icon="⚖️"
        />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-white">Risk Metrics</h2>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm text-gray-400 mb-3">Exposure</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-300">Gross Exposure</span>
                <span className="text-white font-semibold">
                  ${safeInt(metrics?.exposure?.gross_exposure)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Net Exposure</span>
                <span className="text-white font-semibold">
                  ${safeInt(metrics?.exposure?.net_exposure)}
                </span>
              </div>
            </div>
          </div>
          <div>
            <h3 className="text-sm text-gray-400 mb-3">Concentration</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-300">Top Position</span>
                <span className="text-white font-semibold">
                  {metrics?.concentration?.top_position_percentage || 0}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Top 3 Positions</span>
                <span className="text-white font-semibold">
                  {metrics?.concentration?.top_3_positions_percentage || 0}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
