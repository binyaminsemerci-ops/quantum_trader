// Risk Panel - displays ESS state, drawdown, exposure
import type { DashboardRisk } from '@/lib/types';
import { formatPercent, getESSStateColor, formatCompact } from '@/lib/utils';
import DashboardCard from './DashboardCard';

interface RiskPanelProps {
  risk?: DashboardRisk;
}

export default function RiskPanel({ risk }: RiskPanelProps) {
  if (!risk) {
    return (
      <DashboardCard title="Risikobilde">
        <div className="text-center text-gray-500 py-8">Loading risk data...</div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard title="Risikobilde">
      <div className="space-y-4">
        {/* ESS State */}
        <div>
          <div className="dashboard-label mb-2">Emergency Stop System</div>
          <div className="flex items-center justify-between">
            <span className={`badge text-base px-4 py-2 ${getESSStateColor(risk.ess_state)}`}>
              {risk.ess_state}
            </span>
            {risk.ess_reason && (
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {risk.ess_reason}
              </span>
            )}
          </div>
        </div>

        {/* Daily PnL % (Sprint 4 Del 2) */}
        <div>
          <div className="dashboard-label mb-2" title="Current daily profit/loss as percentage of equity">
            Daily PnL
          </div>
          <div className={`text-2xl font-bold ${
            risk.daily_pnl_pct > 0 ? 'text-success' :
            risk.daily_pnl_pct < 0 ? 'text-danger' :
            'text-gray-700 dark:text-gray-300'
          }`}>
            {formatPercent(risk.daily_pnl_pct)}
          </div>
          {risk.max_allowed_dd_pct && (
            <div className="text-xs text-gray-500 dark:text-gray-500">
              Max allowed: {formatPercent(risk.max_allowed_dd_pct)}
            </div>
          )}
        </div>

        {/* Drawdown */}
        <div>
          <div className="dashboard-label mb-2" title="Drawdown measures peak-to-trough decline">
            Drawdown
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Daily</div>
              <div className={`text-lg font-semibold ${
                risk.daily_drawdown_pct < -3 ? 'text-danger' :
                risk.daily_drawdown_pct < -1 ? 'text-warning' : 
                'text-gray-700 dark:text-gray-300'
              }`}>
                {formatPercent(risk.daily_drawdown_pct)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Weekly</div>
              <div className={`text-lg font-semibold ${
                risk.weekly_drawdown_pct < -5 ? 'text-danger' :
                risk.weekly_drawdown_pct < -2 ? 'text-warning' : 
                'text-gray-700 dark:text-gray-300'
              }`}>
                {formatPercent(risk.weekly_drawdown_pct)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Max</div>
              <div className={`text-lg font-semibold ${
                risk.max_drawdown_pct < -10 ? 'text-danger' :
                risk.max_drawdown_pct < -5 ? 'text-warning' : 
                'text-gray-700 dark:text-gray-300'
              }`}>
                {formatPercent(risk.max_drawdown_pct)}
              </div>
            </div>
          </div>
        </div>

        {/* Open Risk % (Sprint 4 Del 2) */}
        <div>
          <div className="dashboard-label mb-2" title="Total risk exposure from all open positions">
            Open Risk
          </div>
          <div className="flex items-center justify-between">
            <div className={`text-xl font-bold ${
              risk.open_risk_pct > 5 ? 'text-danger' :
              risk.open_risk_pct > 3 ? 'text-warning' :
              'text-gray-700 dark:text-gray-300'
            }`}>
              {risk.open_risk_pct.toFixed(2)}%
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              Max per trade: {risk.max_risk_per_trade_pct.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Exposure */}
        <div>
          <div className="dashboard-label mb-2">Exposure</div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Long</div>
              <div className="text-sm font-semibold text-success">
                {formatCompact(risk.exposure_long)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Short</div>
              <div className="text-sm font-semibold text-danger">
                {formatCompact(risk.exposure_short)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Net</div>
              <div className="text-sm font-semibold">
                {formatCompact(risk.exposure_net)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-500">Total</div>
              <div className="text-sm font-semibold">
                {formatCompact(risk.exposure_total)}
              </div>
            </div>
          </div>
        </div>

        {/* Risk Limit */}
        <div>
          <div className="dashboard-label mb-2" title="Percentage of maximum daily loss limit used">
            Risk Limit Used
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  risk.risk_limit_used_pct > 80 ? 'bg-danger' :
                  risk.risk_limit_used_pct > 60 ? 'bg-warning' : 
                  'bg-success'
                }`}
                style={{ width: `${Math.min(risk.risk_limit_used_pct, 100)}%` }}
              />
            </div>
            <span className="text-sm font-medium w-12 text-right">
              {risk.risk_limit_used_pct.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
}
