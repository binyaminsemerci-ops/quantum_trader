/**
 * StrategyPanel Component
 * Sprint 4 Del 2: Display active strategy, regime, and ensemble model scores
 * Sprint 4 Del 3: Improved UX with DashboardCard and Norwegian titles
 */

import { DashboardStrategy, MarketRegime } from '@/lib/types';
import { getRegimeColor, getRegimeBadgeClass } from '@/lib/utils';
import DashboardCard from '../DashboardCard';

interface StrategyPanelProps {
  strategy?: DashboardStrategy;
  loading?: boolean;
}

export default function StrategyPanel({ strategy, loading }: StrategyPanelProps) {
  if (loading) {
    return (
      <DashboardCard title="Strategi & RL">
        <div className="space-y-4">
          <div className="h-6 bg-gray-200 rounded animate-pulse" />
          <div className="h-6 bg-gray-200 rounded animate-pulse" />
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-4 bg-gray-200 rounded animate-pulse" />
            ))}
          </div>
        </div>
      </DashboardCard>
    );
  }

  if (!strategy) {
    return (
      <DashboardCard title="Strategi & RL">
        <p className="text-gray-500 text-sm">Strategi-data utilgjengelig</p>
      </DashboardCard>
    );
  }

  const { active_strategy, regime, ensemble_scores } = strategy;

  // Get regime badge color
  const regimeColorClass = getRegimeColor(regime);
  const regimeBadgeClass = getRegimeBadgeClass(regime);

  return (
    <DashboardCard title="Strategi & RL">
      {/* Active Strategy */}
      <div className="mb-4">
        <div className="dashboard-label">Aktiv strategi</div>
        <div className="text-xl font-bold text-primary">{active_strategy}</div>
      </div>

      {/* Market Regime */}
      <div className="mb-4">
        <div className="dashboard-label" title="Current market regime classification">
          Markedsregime
        </div>
        <span className={`badge ${regimeBadgeClass}`}>
          {regime.replace(/_/g, ' ')}
        </span>
      </div>

      {/* Ensemble Model Scores */}
      <div>
        <div className="dashboard-label mb-2" title="Confidence scores from each AI model in the ensemble">
          Ensemble-modeller
        </div>
        <div className="space-y-2">
          {Object.entries(ensemble_scores)
            .sort(([, a], [, b]) => b - a) // Sort by score descending
            .map(([model, score]) => (
              <div key={model} className="flex items-center gap-3">
                <div className="w-20 text-sm font-medium text-gray-700 uppercase">
                  {model}
                </div>
                <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      score >= 0.8
                        ? 'bg-success'
                        : score >= 0.7
                        ? 'bg-primary'
                        : score >= 0.6
                        ? 'bg-warning'
                        : 'bg-danger'
                    }`}
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
                <div className="w-12 text-sm font-semibold text-right">
                  {(score * 100).toFixed(0)}%
                </div>
              </div>
            ))}
        </div>
      </div>
    </DashboardCard>
  );
}
