import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';

import { safeJson } from '../utils/api';

type EquityPoint = {
  timestamp?: string;
  date?: string;
  equity?: number;
  signal?: number;
};

type Metrics = {
  rmse?: number;
  mae?: number;
  directional_accuracy?: number;
};

type BacktestSummary = {
  final_equity?: number;
  pnl?: number;
  trades?: number;
  win_rate?: number;
  max_drawdown?: number;
  equity_curve?: EquityPoint[];
};

type BacktestResult = {
  symbol?: string;
  mode?: string;
  source?: string;
  num_samples?: number;
  metrics?: Metrics;
  backtest?: BacktestSummary;
  equity_curve?: EquityPoint[];
  pnl?: number;
  final_equity?: number;
  trades?: number;
  win_rate?: number;
  max_drawdown?: number;
  report?: Record<string, unknown>;
  error?: string;
};

const formatNumber = (value?: number, fractionDigits = 2): string => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '–';
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: fractionDigits,
  });
};

const formatPercent = (value?: number, fractionDigits = 1): string => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '–';
  }
  return `${(value * 100).toFixed(fractionDigits)}%`;
};

const buildQuery = (symbol: string, days: number, entryThreshold: string) => {
  const params = new URLSearchParams({ symbol, days: String(days) });
  if (entryThreshold.trim().length) {
    params.set('entry_threshold', entryThreshold);
  }
  return params.toString();
};

export default function Backtest(): JSX.Element {
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  const [days, setDays] = useState<number>(30);
  const [entryThreshold, setEntryThreshold] = useState<string>('0');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const equitySeries = useMemo(() => {
    const curve = result?.backtest?.equity_curve ?? result?.equity_curve ?? [];
    return curve.map((point, idx) => ({
      timestamp: point.timestamp ?? point.date ?? `t${idx}`,
      equity: point.equity ?? null,
      signal: point.signal ?? null,
    }));
  }, [result]);

  const metrics = result?.metrics ?? {};
  const summary = result?.backtest ?? {};

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/backtest?${buildQuery(symbol.trim(), days, entryThreshold)}`);
      const dataRaw = (await safeJson(res)) as BacktestResult | null;
      if (!dataRaw || typeof dataRaw !== 'object') {
        setResult(null);
        setError('Backtest returned an unexpected payload.');
        return;
      }

      const equityCurve = (dataRaw.backtest?.equity_curve ?? dataRaw.equity_curve ?? []).map(
        (point, idx) => ({
          timestamp: point.timestamp ?? point.date ?? `t${idx}`,
          equity: point.equity ?? null,
          signal: point.signal ?? null,
        })
      );

      setResult({
        ...dataRaw,
        backtest: dataRaw.backtest ? { ...dataRaw.backtest, equity_curve: equityCurve } : undefined,
        equity_curve: equityCurve,
        pnl: dataRaw.pnl ?? dataRaw.backtest?.pnl,
        final_equity: dataRaw.final_equity ?? dataRaw.backtest?.final_equity,
        trades: dataRaw.trades ?? dataRaw.backtest?.trades,
        win_rate: dataRaw.win_rate ?? dataRaw.backtest?.win_rate,
        max_drawdown: dataRaw.max_drawdown ?? dataRaw.backtest?.max_drawdown,
      });
    } catch (err) {
      console.error('Backtest failed', err);
      setError('Backtest failed. Check server logs for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 dark:bg-gray-900 dark:text-white min-h-screen">
      <h1 className="text-2xl font-bold">Backtesting</h1>

      <div className="bg-white dark:bg-gray-800 p-4 rounded shadow space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <label className="flex flex-col text-sm">
            Symbol
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="border rounded p-2 text-black"
              placeholder="BTCUSDT"
            />
          </label>
          <label className="flex flex-col text-sm">
            Lookback (days)
            <input
              type="number"
              min={5}
              value={days}
              onChange={(e) => setDays(Math.max(5, Number(e.target.value) || 5))}
              className="border rounded p-2 text-black"
            />
          </label>
          <label className="flex flex-col text-sm">
            Entry threshold
            <input
              value={entryThreshold}
              onChange={(e) => setEntryThreshold(e.target.value)}
              className="border rounded p-2 text-black"
              placeholder="0"
            />
            <span className="text-xs text-slate-500 mt-1">Minimum predicted return before taking a trade.</span>
          </label>
          <div className="flex items-end">
            <button
              onClick={runBacktest}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg shadow disabled:opacity-60"
              disabled={loading}
            >
              {loading ? 'Running…' : 'Run backtest'}
            </button>
          </div>
        </div>
        {result?.source === 'database' && (
          <div className="text-xs text-amber-500">
            Model artifacts were unavailable; falling back to the legacy SQLite simulator.
          </div>
        )}
      </div>

      {loading && <div className="p-4">Running backtest…</div>}
      {error && <div className="p-4 bg-red-600 text-white rounded">{error}</div>}

      {result && !result.error && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Symbol</h3>
              <p className="text-xl font-bold">{result.symbol}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Mode</h3>
              <p className="text-xl font-bold">{result.mode ?? '—'}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">PnL</h3>
              <p className={`text-xl font-bold ${((result.pnl ?? 0) >= 0) ? 'text-emerald-500' : 'text-rose-500'}`}>
                {formatNumber(result.pnl)}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Final equity</h3>
              <p className="text-xl font-bold">{formatNumber(result.final_equity)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Trades</h3>
              <p className="text-xl font-bold">{formatNumber(result.trades, 0)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Win rate</h3>
              <p className="text-xl font-bold">{formatPercent(result.win_rate)}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Directional accuracy</h3>
              <p className="text-xl font-bold">{formatPercent(metrics.directional_accuracy)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">RMSE</h3>
              <p className="text-xl font-bold">{formatNumber(metrics.rmse)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">MAE</h3>
              <p className="text-xl font-bold">{formatNumber(metrics.mae)}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3 className="text-xs uppercase tracking-wide text-slate-500">Max drawdown</h3>
              <p className="text-xl font-bold">{formatPercent(result.max_drawdown)}</p>
            </div>
          </div>

          {result.num_samples !== undefined && (
            <div className="text-sm text-slate-500">
              {formatNumber(result.num_samples, 0)} samples used in the training dataset.
            </div>
          )}

          <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
            <h2 className="text-xl font-bold mb-4">Equity curve</h2>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={equitySeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" minTickGap={24} />
                <YAxis domain={[dataMin => (typeof dataMin === 'number' ? dataMin * 0.99 : dataMin), dataMax => (typeof dataMax === 'number' ? dataMax * 1.01 : dataMax)]} />
                <Tooltip formatter={(value: number) => formatNumber(value)} labelFormatter={(label: string) => label} />
                <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {result?.error && (
        <div className="p-4 bg-rose-600 text-white rounded">{result.error}</div>
      )}
    </div>
  );
}
