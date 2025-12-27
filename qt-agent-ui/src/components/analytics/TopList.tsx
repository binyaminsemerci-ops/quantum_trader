import { TrendingUp, TrendingDown } from 'lucide-react';

interface TopListItem {
  name: string;
  value: number;
  subtitle?: string;
  trend: 'up' | 'down';
}

interface TopListProps {
  title: string;
  items: TopListItem[];
  layout?: 'vertical' | 'horizontal';
}

export default function TopList({ title, items, layout = 'vertical' }: TopListProps) {
  if (!items || items.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
        <h3 className="text-lg font-semibold mb-4">{title}</h3>
        <div className="text-sm text-slate-500 dark:text-slate-400">No data available</div>
      </div>
    );
  }

  const containerClass = layout === 'horizontal' 
    ? 'flex gap-3 overflow-x-auto pb-2' 
    : 'space-y-3';

  const itemClass = layout === 'horizontal'
    ? 'flex-shrink-0 w-48'
    : '';

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      
      <div className={containerClass}>
        {items.map((item, index) => (
          <div 
            key={index}
            className={`${itemClass} flex items-center justify-between p-3 rounded-xl bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors`}
          >
            <div className="flex items-center gap-3 min-w-0">
              {/* Rank badge */}
              <div className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-xs font-semibold">
                {index + 1}
              </div>
              
              {/* Name & subtitle */}
              <div className="min-w-0">
                <div className="font-medium truncate">{item.name}</div>
                {item.subtitle && (
                  <div className="text-xs text-slate-500 dark:text-slate-400 truncate">
                    {item.subtitle}
                  </div>
                )}
              </div>
            </div>

            {/* Value & trend */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className={`font-semibold ${item.trend === 'up' ? 'text-emerald-500' : 'text-red-500'}`}>
                {(item.value ?? 0) >= 0 ? '+' : ''}${(item.value ?? 0).toFixed(2)}
              </div>
              {item.trend === 'up' ? (
                <TrendingUp className="w-4 h-4 text-emerald-500" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500" />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
