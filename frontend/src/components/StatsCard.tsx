type StatsCardProps = {
  title?: string;
  value?: string | number;
  delta?: number | null;
};

export default function StatsCard({ title = 'Stats', value = '-', delta }: StatsCardProps): JSX.Element {
  const deltaClass = delta == null ? '' : delta >= 0 ? 'text-green-500' : 'text-red-500';
  return (
    <div className="bg-white dark:bg-gray-800 p-4 rounded shadow">
      <h4 className="text-sm text-gray-500">{title}</h4>
      <div className="text-2xl font-bold">{value}</div>
      {delta != null && <div className={`text-sm ${deltaClass}`}>{delta >= 0 ? `+${delta}` : delta}</div>}
    </div>
  );
}
