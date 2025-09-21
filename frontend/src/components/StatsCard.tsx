// using the automatic JSX runtime; no default React import required

type StatsProps = {
  title?: string;
  value?: number | string;
};

export default function StatsCard({ title = 'Stats', value = '-' }: StatsProps): JSX.Element {
  return (
    <div className="p-4 bg-gray-900 text-white rounded shadow">
      <h3 className="text-sm font-semibold">{title}</h3>
      <div className="text-2xl font-mono">{value}</div>
    </div>
  );
}
