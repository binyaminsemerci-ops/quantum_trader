import { useEffect, useState } from 'react';

type StatsData = {
  totalValue: number;
  pnl: number;
  pnlPercent: number;
  totalTrades: number;
  winRate: number;
  dailyPnl: number;
  weeklyPnl: number;
  monthlyPnl: number;
  avgTrade: number;
  bestTrade: number;
  worstTrade: number;
  sharpeRatio: number;
};

export default function StatsCard(): JSX.Element {
  const [stats, setStats] = useState<StatsData>({
    totalValue: 10000,
    pnl: 1250.50,
    pnlPercent: 12.51,
    totalTrades: 145,
    winRate: 68.5,
    dailyPnl: 85.30,
    weeklyPnl: 345.20,
    monthlyPnl: 1250.50,
    avgTrade: 8.62,
    bestTrade: 245.80,
    worstTrade: -89.50,
    sharpeRatio: 1.85
  });

  // Mock data loading - replace with real API call
  useEffect(() => {
    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        pnl: prev.pnl + (Math.random() - 0.5) * 10,
        dailyPnl: prev.dailyPnl + (Math.random() - 0.5) * 5,
        totalValue: prev.totalValue + (Math.random() - 0.5) * 50,
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

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
        {typeof value === 'number' && label.toLowerCase().includes('value') ? `$${value.toLocaleString()}` :
         typeof value === 'number' && (label.toLowerCase().includes('pnl') || label.toLowerCase().includes('trade')) ? `$${value.toFixed(2)}` :
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
        <StatItem 
          icon="💰" 
          label="Total Value" 
          value={stats.totalValue} 
          change={stats.pnlPercent}
          isPositive={stats.pnlPercent >= 0}
        />
        <StatItem 
          icon="📈" 
          label="Total P&L" 
          value={stats.pnl}
          change={stats.pnlPercent}
          isPositive={stats.pnl >= 0}
        />
        <StatItem 
          icon="🎯" 
          label="Win Rate" 
          value={stats.winRate}
        />
        <StatItem 
          icon="🔢" 
          label="Total Trades" 
          value={stats.totalTrades}
        />
        <StatItem 
          icon="📅" 
          label="Daily P&L" 
          value={stats.dailyPnl}
          isPositive={stats.dailyPnl >= 0}
        />
        <StatItem 
          icon="📊" 
          label="Weekly P&L" 
          value={stats.weeklyPnl}
          isPositive={stats.weeklyPnl >= 0}
        />
        <StatItem 
          icon="💹" 
          label="Avg Trade" 
          value={stats.avgTrade}
          isPositive={stats.avgTrade >= 0}
        />
        <StatItem 
          icon="⚡" 
          label="Sharpe Ratio" 
          value={stats.sharpeRatio}
          isPositive={stats.sharpeRatio >= 1}
        />
      </div>
    </div>
  );
}
