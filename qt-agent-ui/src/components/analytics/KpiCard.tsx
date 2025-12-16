import { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface KpiCardProps {
  title: string;
  value: string;
  change?: number;
  changeLabel?: string;
  subtitle?: string;
  trend: 'up' | 'down' | 'neutral';
  icon?: ReactNode;
}

export default function KpiCard({
  title,
  value,
  change,
  changeLabel,
  subtitle,
  trend,
  icon
}: KpiCardProps) {
  const trendColors = {
    up: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20',
    down: 'text-red-500 bg-red-500/10 border-red-500/20',
    neutral: 'text-slate-500 bg-slate-500/10 border-slate-500/20'
  };

  const trendIcons = {
    up: <TrendingUp className="w-4 h-4" />,
    down: <TrendingDown className="w-4 h-4" />,
    neutral: <Minus className="w-4 h-4" />
  };

  return (
    <div className="relative overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 transition-all hover:shadow-lg hover:shadow-slate-200/50 dark:hover:shadow-slate-800/50">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-50 to-transparent dark:from-slate-800/50 dark:to-transparent opacity-50" />
      
      <div className="relative">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-slate-600 dark:text-slate-400">{title}</span>
          {icon && (
            <div className="text-slate-400 dark:text-slate-600">
              {icon}
            </div>
          )}
        </div>

        {/* Value */}
        <div className="text-3xl font-bold mb-2">{value}</div>

        {/* Change or Subtitle */}
        {change !== undefined ? (
          <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-sm font-medium ${trendColors[trend]}`}>
            {trendIcons[trend]}
            <span>{change > 0 ? '+' : ''}{change.toFixed(2)}%</span>
            {changeLabel && <span className="text-xs opacity-75 ml-1">{changeLabel}</span>}
          </div>
        ) : subtitle ? (
          <div className="text-sm text-slate-500 dark:text-slate-400">{subtitle}</div>
        ) : null}
      </div>
    </div>
  );
}
