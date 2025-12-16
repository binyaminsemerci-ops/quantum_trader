// TP Summary Tiles - displays aggregate stats from best/worst configurations
import type { TPDashboardSummary } from '@/lib/tpDashboardTypes';
import DashboardCard from '../DashboardCard';

interface TpSummaryTilesProps {
  summary: TPDashboardSummary | null;
  loading: boolean;
}

export default function TpSummaryTiles({ summary, loading }: TpSummaryTilesProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {[1, 2, 3, 4].map((i) => (
          <DashboardCard key={i} title="">
            <div className="animate-pulse h-16 bg-gray-200 dark:bg-slate-700 rounded"></div>
          </DashboardCard>
        ))}
      </div>
    );
  }

  if (!summary) {
    return null;
  }

  const bestCount = summary.best?.length || 0;
  const worstCount = summary.worst?.length || 0;

  const avgHitRateBest =
    bestCount > 0
      ? summary.best.reduce((sum, entry) => sum + entry.metrics.tp_hit_rate, 0) / bestCount
      : 0;

  const avgHitRateWorst =
    worstCount > 0
      ? summary.worst.reduce((sum, entry) => sum + entry.metrics.tp_hit_rate, 0) / worstCount
      : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      <DashboardCard title="Best Configs">
        <div className="text-3xl font-bold text-success">{bestCount}</div>
        <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          High performers
        </div>
      </DashboardCard>

      <DashboardCard title="Worst Configs">
        <div className="text-3xl font-bold text-danger">{worstCount}</div>
        <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Need optimization
        </div>
      </DashboardCard>

      <DashboardCard title="Avg Hit Rate (Best)">
        <div className="text-3xl font-bold text-success">
          {(avgHitRateBest * 100).toFixed(1)}%
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Top performers
        </div>
      </DashboardCard>

      <DashboardCard title="Avg Hit Rate (Worst)">
        <div className="text-3xl font-bold text-danger">
          {(avgHitRateWorst * 100).toFixed(1)}%
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Bottom performers
        </div>
      </DashboardCard>
    </div>
  );
}
