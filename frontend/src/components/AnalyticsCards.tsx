import { useState, useEffect } from 'react';

export default function AnalyticsCards(): JSX.Element {
  const [analytics, setAnalytics] = useState({
    profitFactor: 1.85,
    maxConsecutiveWins: 8,
    maxConsecutiveLosses: 3,
    avgWinningTrade: 125.40,
    avgLosingTrade: -67.80,
    largestWin: 892.50,
    largestLoss: -234.10,
    expectedValue: 45.20,
    kellyCriterion: 0.25,
    calmarRatio: 2.1
  });

  // Mock real-time analytics updates
  useEffect(() => {
    const interval = setInterval(() => {
      setAnalytics(prev => ({
        ...prev,
        profitFactor: Math.max(0.1, prev.profitFactor + (Math.random() - 0.5) * 0.1),
        expectedValue: prev.expectedValue + (Math.random() - 0.5) * 5,
        calmarRatio: Math.max(0.1, prev.calmarRatio + (Math.random() - 0.5) * 0.2)
      }));
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const AnalyticsItem = ({ 
    icon, 
    title, 
    value, 
    unit = '', 
    trend,
    color = 'blue'
  }: {
    icon: string;
    title: string;
    value: number | string;
    unit?: string;
    trend?: 'up' | 'down' | 'neutral';
    color?: 'blue' | 'green' | 'red' | 'purple' | 'orange';
  }) => {
    const colorClasses = {
      blue: 'bg-gradient-to-br from-blue-500 to-blue-600',
      green: 'bg-gradient-to-br from-green-500 to-green-600',
      red: 'bg-gradient-to-br from-red-500 to-red-600',
      purple: 'bg-gradient-to-br from-purple-500 to-purple-600',
      orange: 'bg-gradient-to-br from-orange-500 to-orange-600'
    };

    const trendIcons = {
      up: '📈',
      down: '📉', 
      neutral: '➡️'
    };

    return (
      <div className={`${colorClasses[color]} p-4 rounded-lg text-white shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-1`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-2xl">{icon}</span>
          {trend && <span className="text-lg">{trendIcons[trend]}</span>}
        </div>
        <div className="text-sm opacity-90 mb-1">{title}</div>
        <div className="text-xl font-bold">
          {typeof value === 'number' ? 
            (title.toLowerCase().includes('ratio') || title.toLowerCase().includes('factor') ? 
              value.toFixed(2) : 
              title.toLowerCase().includes('win') || title.toLowerCase().includes('loss') ? 
                `$${value.toFixed(2)}` :
                value.toFixed(1)) 
            : value}{unit}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3">
        <AnalyticsItem
          icon="🎯"
          title="Profit Factor"
          value={analytics.profitFactor}
          trend={analytics.profitFactor >= 1.5 ? 'up' : analytics.profitFactor >= 1.2 ? 'neutral' : 'down'}
          color="green"
        />
        
        <AnalyticsItem
          icon="💡"
          title="Expected Value"
          value={analytics.expectedValue}
          unit="$"
          trend={analytics.expectedValue >= 40 ? 'up' : analytics.expectedValue >= 20 ? 'neutral' : 'down'}
          color="blue"
        />
        
        <AnalyticsItem
          icon="🏆"
          title="Largest Win"
          value={analytics.largestWin}
          unit="$"
          color="green"
        />
        
        <AnalyticsItem
          icon="📊"
          title="Calmar Ratio"
          value={analytics.calmarRatio}
          trend={analytics.calmarRatio >= 2.0 ? 'up' : analytics.calmarRatio >= 1.5 ? 'neutral' : 'down'}
          color="purple"
        />
        
        <AnalyticsItem
          icon="🎲"
          title="Kelly Criterion"
          value={(analytics.kellyCriterion * 100)}
          unit="%"
          color="orange"
        />
      </div>
    </div>
  );
}
