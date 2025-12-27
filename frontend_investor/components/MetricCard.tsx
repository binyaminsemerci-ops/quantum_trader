// components/MetricCard.tsx
interface MetricCardProps {
  label: string;
  value: number | string | null | undefined;
  format?: 'number' | 'percentage' | 'currency';
  trend?: 'up' | 'down' | 'neutral';
  icon?: string;
}

export default function MetricCard({ 
  label, 
  value, 
  format = 'number',
  trend,
  icon 
}: MetricCardProps) {
  const formatValue = (val: number | string | null | undefined): string => {
    if (val === null || val === undefined || val === '') return 'N/A';
    
    const numVal = typeof val === 'string' ? parseFloat(val) : val;
    if (isNaN(numVal)) return 'N/A';

    switch (format) {
      case 'percentage':
        return `${(numVal * 100).toFixed(2)}%`;
      case 'currency':
        return `$${numVal.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      default:
        return numVal.toFixed(2);
    }
  };

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'text-green-400';
      case 'down':
        return 'text-red-400';
      default:
        return 'text-quantum-text';
    }
  };

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return '↑';
      case 'down':
        return '↓';
      default:
        return '';
    }
  };

  return (
    <div className="bg-quantum-card border border-quantum-border rounded-lg p-5 hover:border-quantum-accent transition">
      <div className="flex items-start justify-between mb-3">
        <div className="text-quantum-muted text-sm font-medium">
          {label}
        </div>
        {icon && <span className="text-xl">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold ${getTrendColor()}`}>
        {formatValue(value)}
        {trend && (
          <span className="ml-2 text-lg">
            {getTrendIcon()}
          </span>
        )}
      </div>
    </div>
  );
}
