// Positions Panel - displays open positions with live PnL
import type { DashboardPosition } from '@/lib/types';
import { formatCurrency, formatPercent, getPnLColorClass } from '@/lib/utils';
import DashboardCard from './DashboardCard';

interface PositionsPanelProps {
  positions: DashboardPosition[];
}

export default function PositionsPanel({ positions }: PositionsPanelProps) {
  if (positions.length === 0) {
    return (
      <DashboardCard title="Åpne posisjoner" fullHeight>
        <div className="text-center text-gray-500 py-8">
          Ingen åpne posisjoner
        </div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard 
      title="Åpne posisjoner"
      rightSlot={<span className="text-sm text-gray-600 dark:text-gray-400">({positions.length})</span>}
      fullHeight
    >
      <div className="overflow-auto scrollbar-thin">
        <table className="w-full text-sm">
          <thead className="border-b border-gray-200 dark:border-slate-700">
            <tr>
              <th className="text-left py-2 px-2 font-medium">Symbol</th>
              <th className="text-left py-2 px-2 font-medium">Side</th>
              <th className="text-right py-2 px-2 font-medium">Size</th>
              <th className="text-right py-2 px-2 font-medium">Entry</th>
              <th className="text-right py-2 px-2 font-medium">Current</th>
              <th className="text-right py-2 px-2 font-medium">PnL</th>
              <th className="text-right py-2 px-2 font-medium">PnL%</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position, idx) => (
              <tr
                key={`${position.symbol}-${idx}`}
                className="border-b border-gray-100 dark:border-slate-800 hover:bg-gray-50 dark:hover:bg-slate-700/50"
              >
                <td className="py-2 px-2 font-medium">{position.symbol}</td>
                <td className="py-2 px-2">
                  <span
                    className={`badge ${
                      position.side === 'LONG'
                        ? 'bg-success/20 text-success'
                        : 'bg-danger/20 text-danger'
                    }`}
                  >
                    {position.side}
                  </span>
                </td>
                <td className="py-2 px-2 text-right">{position.size.toFixed(4)}</td>
                <td className="py-2 px-2 text-right">
                  {formatCurrency(position.entry_price)}
                </td>
                <td className="py-2 px-2 text-right">
                  {formatCurrency(position.current_price)}
                </td>
                <td className={`py-2 px-2 text-right font-medium ${getPnLColorClass(position.unrealized_pnl)}`}>
                  {formatCurrency(position.unrealized_pnl)}
                </td>
                <td className={`py-2 px-2 text-right font-medium ${getPnLColorClass(position.unrealized_pnl_pct)}`}>
                  {formatPercent(position.unrealized_pnl_pct)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </DashboardCard>
  );
}
