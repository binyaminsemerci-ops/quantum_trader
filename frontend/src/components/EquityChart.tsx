// Minimal, conservative TSX for EquityChart to avoid heavy external typing (recharts)
import { useMemo } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';

type ChartPoint = { timestamp?: string; equity?: number };

export default function EquityChart(): JSX.Element {
  const { data } = useDashboardData();
  const chart: ChartPoint[] = data?.chart || [];

  const summary = useMemo(() => {
    if (!chart || chart.length === 0) return { points: 0, first: null, last: null };
    return { points: chart.length, first: chart[0]?.equity ?? null, last: chart[chart.length - 1]?.equity ?? null };
  }, [chart]);

  if (summary.points === 0) {
    return (
      <div className="p-4 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-40 mb-4"></div>
        <div className="h-48 bg-gray-200 rounded"></div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">📈 Equity Curve</h2>
      </div>
      <div className="text-sm text-gray-600">Data points: {summary.points}</div>
      <div className="text-sm text-gray-600">Start equity: {String(summary.first)}</div>
      <div className="text-sm text-gray-600">End equity: {String(summary.last)}</div>
    </div>
  );
}
