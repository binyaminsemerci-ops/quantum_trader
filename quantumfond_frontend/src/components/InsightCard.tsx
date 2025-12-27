interface InsightCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export default function InsightCard({ 
  title, 
  value, 
  change, 
  icon, 
  trend = 'neutral' 
}: InsightCardProps) {
  const getTrendColor = () => {
    if (trend === 'up') return 'text-green-400';
    if (trend === 'down') return 'text-red-400';
    return 'text-gray-400';
  };

  const getTrendIcon = () => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  // Defensive rendering for all value types
  const displayValue = (() => {
    if (typeof value === 'number') {
      return isFinite(value) ? value.toFixed(2) : '0.00';
    }
    if (typeof value === 'string') {
      return value || '—';
    }
    return '—';
  })();

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 hover:border-gray-700 transition-colors">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm text-gray-400 font-medium">{title}</h3>
        {icon && <span className="text-2xl">{icon}</span>}
      </div>
      
      <div className="flex items-end justify-between">
        <div className="text-3xl font-bold text-white">
          {displayValue}
        </div>
        
        {change !== undefined && (
          <div className={`flex items-center gap-1 text-sm font-semibold ${getTrendColor()}`}>
            <span>{getTrendIcon()}</span>
            <span>{Math.abs(change)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}
