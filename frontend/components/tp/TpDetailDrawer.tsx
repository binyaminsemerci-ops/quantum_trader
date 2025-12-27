// TP Detail Drawer - displays detailed TP profile, metrics, and recommendations
import type { TPDashboardEntry } from '@/lib/tpDashboardTypes';
import { useEffect } from 'react';
import { safeNum, safePercent } from '@/lib/formatters';

interface TpDetailDrawerProps {
  entry: TPDashboardEntry | null;
  open: boolean;
  onClose: () => void;
  loading: boolean;
  error?: string;
}

export default function TpDetailDrawer({
  entry,
  open,
  onClose,
  loading,
  error,
}: TpDetailDrawerProps) {
  // Close drawer on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={onClose}
      ></div>

      {/* Drawer */}
      <div className="fixed right-0 top-0 bottom-0 w-full md:w-2/3 lg:w-1/2 bg-white dark:bg-slate-800 shadow-xl z-50 overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              {entry && (
                <>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                    {entry.key.strategy_id}
                  </h2>
                  <p className="text-gray-500 dark:text-gray-400">
                    {entry.key.symbol}
                  </p>
                </>
              )}
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
            >
              <svg
                className="w-6 h-6 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {/* Loading state */}
          {loading && (
            <div className="space-y-4">
              <div className="animate-pulse">
                <div className="h-8 bg-gray-200 dark:bg-slate-700 rounded w-1/3 mb-4"></div>
                <div className="space-y-3">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="h-4 bg-gray-200 dark:bg-slate-700 rounded"></div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Error state */}
          {error && !loading && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <p className="text-red-700 dark:text-red-400">{error}</p>
            </div>
          )}

          {/* Content */}
          {entry && !loading && !error && (
            <div className="space-y-6">
              {/* TP Profile Section */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  TP Profile
                </h3>
                <div className="bg-gray-50 dark:bg-slate-900 rounded-lg p-4 border border-gray-200 dark:border-slate-700">
                  <div className="mb-3">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Profile ID:
                    </span>{' '}
                    <span className="text-sm text-gray-900 dark:text-gray-100">
                      {entry.profile.profile_id}
                    </span>
                  </div>

                  {entry.profile.trailing_profile_id && (
                    <div className="mb-3">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Trailing Profile:
                      </span>{' '}
                      <span className="text-sm text-gray-900 dark:text-gray-100">
                        {entry.profile.trailing_profile_id}
                      </span>
                    </div>
                  )}

                  <div className="mt-4">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      TP Legs:
                    </p>
                    <div className="space-y-2">
                      {entry.profile.legs.map((leg, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between bg-white dark:bg-slate-800 p-3 rounded border border-gray-200 dark:border-slate-700"
                        >
                          <div>
                            <span className="font-medium text-gray-900 dark:text-gray-100">
                              {leg.label}
                            </span>
                            <span className="ml-2 text-xs px-2 py-1 bg-gray-200 dark:bg-slate-700 rounded">
                              {leg.kind}
                            </span>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                              {safeNum(leg.r_multiple, 2)}R
                            </div>
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              {safeNum(leg.size_fraction * 100, 0)}% size
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Metrics Section */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Performance Metrics
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <MetricCard
                    label="TP Hit Rate"
                    value={safePercent(entry.metrics.tp_hit_rate * 100, 1)}
                    color={entry.metrics.tp_hit_rate > 0.6 ? 'success' : entry.metrics.tp_hit_rate > 0.4 ? 'warning' : 'danger'}
                  />
                  <MetricCard
                    label="Attempts"
                    value={entry.metrics.tp_attempts.toString()}
                  />
                  <MetricCard
                    label="Hits"
                    value={entry.metrics.tp_hits.toString()}
                    color="success"
                  />
                  <MetricCard
                    label="Misses"
                    value={entry.metrics.tp_misses.toString()}
                    color="danger"
                  />

                  {entry.metrics.avg_slippage_pct !== undefined && (
                    <MetricCard
                      label="Avg Slippage"
                      value={safePercent(entry.metrics.avg_slippage_pct * 100, 2)}
                    />
                  )}

                  {entry.metrics.max_slippage_pct !== undefined && (
                    <MetricCard
                      label="Max Slippage"
                      value={safePercent(entry.metrics.max_slippage_pct * 100, 2)}
                    />
                  )}

                  {entry.metrics.avg_time_to_tp_minutes !== undefined && (
                    <MetricCard
                      label="Avg Time to TP"
                      value={`${safeNum(entry.metrics.avg_time_to_tp_minutes, 1)} min`}
                    />
                  )}

                  {entry.metrics.total_tp_profit_usd !== undefined && (
                    <MetricCard
                      label="Total TP Profit"
                      value={`$${entry.metrics.total_tp_profit_usd.toFixed(2)}`}
                      color="success"
                    />
                  )}

                  {entry.metrics.avg_tp_profit_usd !== undefined && (
                    <MetricCard
                      label="Avg TP Profit"
                      value={`$${entry.metrics.avg_tp_profit_usd.toFixed(2)}`}
                    />
                  )}

                  {entry.metrics.premature_exits !== undefined && (
                    <MetricCard
                      label="Premature Exits"
                      value={entry.metrics.premature_exits.toString()}
                      color="warning"
                    />
                  )}

                  {entry.metrics.missed_opportunities_usd !== undefined && (
                    <MetricCard
                      label="Missed Opportunities"
                      value={`$${entry.metrics.missed_opportunities_usd.toFixed(2)}`}
                      color="danger"
                    />
                  )}
                </div>
              </div>

              {/* Recommendation Section */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Optimization Recommendation
                </h3>
                {entry.recommendation.has_recommendation ? (
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <svg
                        className="w-6 h-6 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                          {entry.recommendation.reason}
                        </p>
                        {entry.recommendation.suggested_scale_factor !== null && entry.recommendation.suggested_scale_factor !== undefined && (
                          <p className="text-sm text-blue-700 dark:text-blue-300">
                            Suggested scale factor:{' '}
                            <span className="font-semibold">
                              {entry.recommendation.suggested_scale_factor.toFixed(2)}x
                            </span>
                          </p>
                        )}
                        {entry.recommendation.profile_id && (
                          <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                            Recommended profile:{' '}
                            <span className="font-semibold">
                              {entry.recommendation.profile_id}
                            </span>
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-700 rounded-lg p-4">
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      No optimization recommendation at this time. Current configuration appears optimal.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// Helper component for metric cards
function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: 'success' | 'danger' | 'warning';
}) {
  const colorClasses = {
    success: 'text-success',
    danger: 'text-danger',
    warning: 'text-warning',
  };

  return (
    <div className="bg-gray-50 dark:bg-slate-900 rounded-lg p-4 border border-gray-200 dark:border-slate-700">
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">{label}</p>
      <p
        className={`text-lg font-semibold ${
          color ? colorClasses[color] : 'text-gray-900 dark:text-gray-100'
        }`}
      >
        {value}
      </p>
    </div>
  );
}
