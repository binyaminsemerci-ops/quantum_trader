interface InsightCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  color?: 'green' | 'blue' | 'yellow' | 'purple' | 'red';
}

export default function InsightCard({ title, value, subtitle, color = 'blue' }: InsightCardProps) {
  const colorClasses = {
    green: 'border-green-500 bg-green-500/10',
    blue: 'border-blue-500 bg-blue-500/10',
    yellow: 'border-yellow-500 bg-yellow-500/10',
    purple: 'border-purple-500 bg-purple-500/10',
    red: 'border-red-500 bg-red-500/10'
  };

  const valueColorClasses = {
    green: 'text-green-400',
    blue: 'text-blue-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400',
    red: 'text-red-400'
  };

  return (
    <div className={`${colorClasses[color]} border-l-4 rounded-lg p-6 hover:shadow-lg transition-shadow duration-200 bg-gray-800`}>
      <div className="text-sm text-gray-400 mb-2">{title}</div>
      <div className={`text-3xl font-bold mb-2 ${valueColorClasses[color]}`}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{subtitle}</div>
    </div>
  );
}
