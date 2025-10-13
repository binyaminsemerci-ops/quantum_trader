import { useMemo } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';

export default function StatsCard(): JSX.Element {
  const { data } = useDashboardData();
  const derived = useMemo(() => {
    const s: any = data?.stats || {};
    return {
      totalValue: s.total_equity || s.portfolio_value || null,
      pnl: s.pnl ?? null,
      trades: s.total_trades || s.analytics?.trades_count || null,
      winRate: s.analytics?.win_rate ?? null,
      sharpe: s.analytics?.sharpe_ratio ?? s.sharpe_ratio ?? null
    };
  }, [data]);

  const StatItem = ({ icon, label, value, change, isPositive }: {
    icon: string;
    label: string;
    value: string | number;
    change?: number;
    isPositive?: boolean;
  }) => (
    <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 p-4 rounded-lg border dark:border-gray-600 hover:shadow-md transition-all">
      <div className="flex items-center justify-between mb-2">
        <span className="text-2xl">{icon}</span>
        {change !== undefined && (
          <span className={`text-sm font-medium px-2 py-1 rounded-full ${
            isPositive ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200' : 
            'bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200'
          }`}>
            {isPositive ? '+' : ''}{change.toFixed(2)}%
          </span>
        )}
      </div>
      <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">{label}</div>
      <div className="text-xl font-bold text-gray-800 dark:text-white">
        {typeof value === 'number' && label.toLowerCase().includes('value') ? 
          (Math.abs(value) >= 1000000 ? `$${(value/1000000).toFixed(2)}M` : 
           Math.abs(value) >= 1000 ? `$${(value/1000).toFixed(0)}K` : `$${value.toLocaleString()}`) :
         typeof value === 'number' && (label.toLowerCase().includes('pnl') || label.toLowerCase().includes('trade')) ? 
          (Math.abs(value) >= 1000000 ? `$${(value/1000000).toFixed(2)}M` :
           Math.abs(value) >= 1000 ? `$${(value/1000).toFixed(0)}K` : `$${value.toFixed(2)}`) :
         typeof value === 'number' && label.toLowerCase().includes('rate') ? `${value.toFixed(1)}%` :
         typeof value === 'number' && label.toLowerCase().includes('ratio') ? value.toFixed(2) :
         typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    </div>
  );

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">📊 Portfolio Overview</h3>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-500 dark:text-gray-400">Live</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
        {derived.totalValue != null && <StatItem icon="💰" label="Total Value" value={derived.totalValue} />}
        {derived.pnl != null && <StatItem icon="📈" label="Total P&L" value={derived.pnl} isPositive={(derived.pnl as number) >= 0} />}
        {derived.trades != null && <StatItem icon="🔢" label="Total Trades" value={derived.trades} />}
        {derived.winRate != null && <StatItem icon="🎯" label="Win Rate" value={derived.winRate} />}
        {derived.sharpe != null && <StatItem icon="⚡" label="Sharpe Ratio" value={derived.sharpe} />}
      </div>
    </div>
  );
}
