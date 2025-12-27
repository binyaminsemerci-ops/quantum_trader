// Signals Panel - displays recent AI signals
import type { DashboardSignal } from '@/lib/types';
import { formatTime } from '@/lib/utils';
import { safeNum, safePercent } from '@/lib/formatters';
import DashboardCard from './DashboardCard';

interface SignalsPanelProps {
  signals: DashboardSignal[];
}

export default function SignalsPanel({ signals }: SignalsPanelProps) {
  if (signals.length === 0) {
    return (
      <DashboardCard title="Siste signaler" fullHeight>
        <div className="text-center text-gray-500 py-8">
          Ingen nylige signaler
        </div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard 
      title="Siste signaler"
      rightSlot={<span className="text-sm text-gray-600 dark:text-gray-400">({signals.length})</span>}
      fullHeight
    >
      <div className="space-y-2">
        {signals.map((signal, idx) => (
          <div
            key={`${signal.symbol}-${signal.timestamp}-${idx}`}
            className="p-3 rounded bg-gray-50 dark:bg-slate-700/50 hover:bg-gray-100 dark:hover:bg-slate-700"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-medium">{signal.symbol}</span>
              <span
                className={`badge ${
                  signal.direction === 'BUY'
                    ? 'bg-success/20 text-success'
                    : signal.direction === 'SELL'
                    ? 'bg-danger/20 text-danger'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {signal.direction}
              </span>
            </div>
            
            <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
              <span>
                Confidence: <span className="font-medium">{safeNum(signal.confidence * 100, 1)}%</span>
              </span>
              <span className="text-xs">{formatTime(signal.timestamp)}</span>
            </div>
            
            <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
              Strategy: {signal.strategy}
              {signal.target_size && ` • Size: ${safeNum(signal.target_size, 4)}`}
            </div>
          </div>
        ))}
      </div>
    </DashboardCard>
  );
}
