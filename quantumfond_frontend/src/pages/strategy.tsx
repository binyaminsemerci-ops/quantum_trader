import { useEffect, useState } from 'react';
import { safeNum, safePercent } from '../utils/formatters';

export default function Strategy() {
  const [strategies, setStrategies] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/strategy/active')
      .then(res => res.json())
      .then(data => setStrategies(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Strategy Management</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {strategies?.strategies?.map((strategy: any) => (
          <div key={strategy.id} className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-white">{strategy.name}</h3>
                <p className="text-sm text-gray-400">Allocation: {strategy.allocation}%</p>
              </div>
              <span className="px-3 py-1 rounded text-xs font-semibold bg-green-900/30 text-green-400">
                {strategy.status}
              </span>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Daily P&L</span>
                <span className={`font-semibold ${
                  strategy.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  ${safeNum(strategy.daily_pnl)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate</span>
                <span className="text-white font-semibold">
                  {safePercent(strategy.win_rate, 1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Trades Today</span>
                <span className="text-white font-semibold">{strategy.trades_today}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
