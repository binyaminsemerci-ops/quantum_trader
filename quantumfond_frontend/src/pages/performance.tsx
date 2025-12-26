import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';
import { safePercent, safeNum } from '../utils/formatters';

export default function Performance() {
  const [summary, setSummary] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/performance/summary')
      .then(res => res.json())
      .then(data => setSummary(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Performance Analytics</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <InsightCard
          title="Daily Return"
          value={`${summary?.total_return?.daily || 0}%`}
          change={summary?.total_return?.daily || 0}
          trend={summary?.total_return?.daily > 0 ? 'up' : 'down'}
          icon="📊"
        />
        <InsightCard
          title="Sharpe Ratio"
          value={summary?.sharpe_ratio || 0}
          icon="📈"
        />
        <InsightCard
          title="Win Rate"
          value={safePercent(summary?.win_rate, 1)}
          icon="🎯"
        />
        <InsightCard
          title="Total Trades"
          value={summary?.total_trades || 0}
          icon="🔄"
        />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-white">Performance Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-gray-400 mb-1">Max Drawdown</p>
            <p className="text-2xl font-bold text-white">{summary?.max_drawdown || 0}%</p>
          </div>
          <div>
            <p className="text-sm text-gray-400 mb-1">Profit Factor</p>
            <p className="text-2xl font-bold text-white">{summary?.profit_factor || 0}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400 mb-1">Avg Win</p>
            <p className="text-2xl font-bold text-green-400">
              ${safeNum(summary?.average_win)}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-400 mb-1">Avg Loss</p>
            <p className="text-2xl font-bold text-red-400">
              ${safeNum(summary?.average_loss)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
