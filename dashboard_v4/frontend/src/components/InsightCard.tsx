type Props = {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
};

export default function InsightCard({ title, value, subtitle, color = "text-green-400" }: Props) {
  return (
    <div className="p-4 bg-gray-800 rounded-xl shadow-lg border border-gray-700 hover:shadow-2xl transition">
      <h3 className={`text-lg font-semibold ${color}`}>{title}</h3>
      <p className="text-3xl font-bold">{value}</p>
      {subtitle && <p className="text-sm text-gray-400 mt-1">{subtitle}</p>}
    </div>
  );
}
