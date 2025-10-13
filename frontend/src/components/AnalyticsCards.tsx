import { useMemo } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';

export default function AnalyticsCards(): JSX.Element {
  const { data } = useDashboardData();
  const analytics = useMemo(() => {
    const a: any = data?.stats?.analytics || {};
    return {
      profitFactor: a.profit_factor,
      expectedValue: a.expected_value,
      largestWin: a.largest_win,
      largestLoss: a.largest_loss,
      calmarRatio: a.calmar_ratio,
      kellyCriterion: a.kelly_criterion
    };
  }, [data]);

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
        {analytics.profitFactor != null && <AnalyticsItem icon="🎯" title="Profit Factor" value={analytics.profitFactor} color="green" />}
        {analytics.expectedValue != null && <AnalyticsItem icon="💡" title="Expected Value" value={analytics.expectedValue} unit="$" color="blue" />}
        {analytics.largestWin != null && <AnalyticsItem icon="🏆" title="Largest Win" value={analytics.largestWin} unit="$" color="green" />}
        {analytics.calmarRatio != null && <AnalyticsItem icon="📊" title="Calmar Ratio" value={analytics.calmarRatio} color="purple" />}
        {analytics.kellyCriterion != null && <AnalyticsItem icon="🎲" title="Kelly Criterion" value={(analytics.kellyCriterion * 100)} unit="%" color="orange" />}
      </div>
    </div>
  );
}
