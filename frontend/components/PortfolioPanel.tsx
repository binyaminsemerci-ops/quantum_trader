// Portfolio Summary Panel - displays equity, PnL, margin
import type { DashboardPortfolio } from '@/lib/types';
import { formatCurrency, formatPercent, formatCompact, getPnLColorClass } from '@/lib/utils';

interface PortfolioPanelProps {
  portfolio: DashboardPortfolio;
}

export default function PortfolioPanel({ portfolio }: PortfolioPanelProps) {
  return (
    <div className="dashboard-card">
      <h2 className="text-lg font-semibold mb-4">Portfolio</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Equity */}
        <div>
          <div className="dashboard-label">Equity</div>
          <div className="dashboard-stat">{formatCompact(portfolio.equity)}</div>
        </div>

        {/* Daily PnL */}
        <div>
          <div className="dashboard-label">Daily PnL</div>
          <div className={`dashboard-stat ${getPnLColorClass(portfolio.daily_pnl)}`}>
            {formatCompact(portfolio.daily_pnl)}
          </div>
          <div className={`text-sm ${getPnLColorClass(portfolio.daily_pnl_pct)}`}>
            {formatPercent(portfolio.daily_pnl_pct)}
          </div>
        </div>

        {/* Total PnL */}
        <div>
          <div className="dashboard-label">Total PnL</div>
          <div className={`dashboard-stat ${getPnLColorClass(portfolio.total_pnl)}`}>
            {formatCompact(portfolio.total_pnl)}
          </div>
        </div>

        {/* Positions */}
        <div>
          <div className="dashboard-label">Positions</div>
          <div className="dashboard-stat">{portfolio.position_count}</div>
        </div>

        {/* Cash */}
        <div>
          <div className="dashboard-label">Cash</div>
          <div className="text-lg font-semibold">{formatCompact(portfolio.cash)}</div>
        </div>

        {/* Margin Used */}
        <div>
          <div className="dashboard-label">Margin Used</div>
          <div className="text-lg font-semibold">{formatCompact(portfolio.margin_used)}</div>
        </div>

        {/* Margin Available */}
        <div>
          <div className="dashboard-label">Margin Avail.</div>
          <div className="text-lg font-semibold">{formatCompact(portfolio.margin_available)}</div>
        </div>

        {/* Weekly PnL */}
        <div>
          <div className="dashboard-label">Weekly PnL</div>
          <div className={`text-lg font-semibold ${getPnLColorClass(portfolio.weekly_pnl)}`}>
            {formatCompact(portfolio.weekly_pnl)}
          </div>
        </div>
      </div>
    </div>
  );
}
