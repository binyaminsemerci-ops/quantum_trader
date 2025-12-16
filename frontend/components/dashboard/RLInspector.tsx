/**
 * RLInspector Component
 * Sprint 4 Del 2: Display RL position sizing decisions (proposed vs capped)
 */

import { DashboardRLSizing } from '@/lib/types';
import { getVolatilityBucketColor } from '@/lib/utils';

interface RLInspectorProps {
  rlSizing?: DashboardRLSizing;
  loading?: boolean;
}

export default function RLInspector({ rlSizing, loading }: RLInspectorProps) {
  if (loading) {
    return (
      <div className="dashboard-card p-6">
        <h2 className="text-lg font-semibold mb-4">RL Position Sizing</h2>
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-12 bg-gray-200 rounded animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (!rlSizing) {
    return (
      <div className="dashboard-card p-6">
        <h2 className="text-lg font-semibold mb-4">RL Position Sizing</h2>
        <p className="text-gray-500 text-sm">No recent RL sizing decision</p>
      </div>
    );
  }

  const {
    symbol,
    proposed_risk_pct,
    capped_risk_pct,
    proposed_leverage,
    capped_leverage,
    volatility_bucket,
  } = rlSizing;

  const volBucketColor = getVolatilityBucketColor(volatility_bucket);

  // Determine if capping occurred
  const riskCapped = proposed_risk_pct !== capped_risk_pct;
  const leverageCapped = proposed_leverage !== capped_leverage;

  return (
    <div className="dashboard-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">RL Position Sizing</h2>
        <span className={`badge ${volBucketColor}`}>{volatility_bucket} VOL</span>
      </div>

      {/* Symbol */}
      <div className="mb-4">
        <div className="dashboard-label">Symbol</div>
        <div className="text-xl font-bold">{symbol}</div>
      </div>

      {/* Risk % - Proposed vs Capped */}
      <div className="mb-3">
        <div className="dashboard-label">Risk %</div>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="text-sm text-gray-600 mb-1">Proposed</div>
            <div className={`dashboard-stat ${riskCapped ? 'text-warning' : 'text-gray-900'}`}>
              {proposed_risk_pct.toFixed(2)}%
            </div>
          </div>
          <div className="text-gray-400">→</div>
          <div className="flex-1">
            <div className="text-sm text-gray-600 mb-1">Capped</div>
            <div className={`dashboard-stat ${riskCapped ? 'text-success' : 'text-gray-900'}`}>
              {capped_risk_pct.toFixed(2)}%
            </div>
          </div>
        </div>
        {riskCapped && (
          <div className="mt-1 text-xs text-warning">
            ⚠️ Risk reduced by policy constraints
          </div>
        )}
      </div>

      {/* Leverage - Proposed vs Capped */}
      <div>
        <div className="dashboard-label">Leverage</div>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="text-sm text-gray-600 mb-1">Proposed</div>
            <div className={`dashboard-stat ${leverageCapped ? 'text-warning' : 'text-gray-900'}`}>
              {proposed_leverage.toFixed(1)}x
            </div>
          </div>
          <div className="text-gray-400">→</div>
          <div className="flex-1">
            <div className="text-sm text-gray-600 mb-1">Capped</div>
            <div className={`dashboard-stat ${leverageCapped ? 'text-success' : 'text-gray-900'}`}>
              {capped_leverage.toFixed(1)}x
            </div>
          </div>
        </div>
        {leverageCapped && (
          <div className="mt-1 text-xs text-warning">
            ⚠️ Leverage reduced by policy constraints
          </div>
        )}
      </div>
    </div>
  );
}
