import { useDashboardWs } from '../hooks/useDashboardWs';
import { useEffect, useState } from 'react';
import { formatNumber, formatPercent } from '../utils/format';

export default function StatsWidget() {
  const { data, status } = useDashboardWs({ enabled: true });
  const [ts, setTs] = useState<number>(Date.now());
  useEffect(()=>{ if (data) setTs(Date.now()); }, [data]);
  const stats = data?.stats || {};
  return (
    <div className="h-full flex flex-col text-xs space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-semibold">Stats {status==='open'?'(live)':'(init)'}</span>
        <span className="text-gray-500">{new Date(ts).toLocaleTimeString()}</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
  <Metric label="Trades" value={formatNumber(stats.total_trades)} />
  <Metric label="Active Symbols" value={formatNumber(stats.active_symbols)} />
  <Metric label="Avg Price" value={formatNumber(stats.avg_price, { maximumFractionDigits: 2 })} />
  <Metric label="PnL" value={formatNumber(stats.pnl, { maximumFractionDigits: 2 })} highlight />
  <Metric label="Win%" value={stats.analytics?.win_rate != null ? formatPercent(stats.analytics.win_rate,1) : '—'} />
  <Metric label="Sharpe" value={stats.analytics?.sharpe_ratio != null ? Number(stats.analytics.sharpe_ratio).toFixed(2) : '—'} />
      </div>
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: any; highlight?: boolean }) {
  const cls = highlight ? 'text-green-600 dark:text-green-400' : 'text-gray-900 dark:text-gray-100';
  return (
    <div className="p-2 rounded bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</span>
      <span className={`text-sm font-semibold ${cls}`}>{value ?? '—'}</span>
    </div>
  );
}
