import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

type BacktestResult = {
  symbol?: string;
  mode?: string;
  pnl?: number;
  sharpe_ratio?: number;
  equity_curve?: Array<{ date?: string; equity?: number }>;
  error?: string;
};

export default function Backtest(): JSX.Element {
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const runBacktest = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/backtest?symbol=${encodeURIComponent(symbol)}&days=30`);
      const data = await res.json();
      setResult(data as BacktestResult);
    } catch (err) {
      console.error('Backtest failed', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 dark:bg-gray-900 dark:text-white min-h-screen">
      <h1 className="text-2xl font-bold">🔮 Backtesting</h1>

      {/* Input */}
      <div className="bg-white dark:bg-gray-800 p-4 rounded shadow space-y-4">
        <div className="flex space-x-4">
          <div>
            <label className="block text-sm">Symbol</label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="border rounded p-2 text-black"
            />
          </div>
          <button
            onClick={runBacktest}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow self-end"
          >
            Run
          </button>
        </div>
      </div>

      {loading && <div className="p-4">Running backtest...</div>}

      {result && !result.error && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3>Symbol</h3>
              <p className="text-xl font-bold">{result.symbol}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3>Mode</h3>
              <p className="text-xl font-bold">{result.mode === 'trades' ? 'Based on Trades' : 'Based on Candles'}</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3>PnL</h3>
              <p className={`text-xl font-bold ${result.pnl && result.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {result.pnl}
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
              <h3>Sharpe Ratio</h3>
              <p className="text-xl font-bold">{result.sharpe_ratio}</p>
            </div>
          </div>

          {/* Equity curve */}
          <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg mt-6">
            <h2 className="text-xl font-bold mb-4">Equity Curve</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={result.equity_curve ?? []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {result?.error && <div className="p-4 bg-red-600 text-white rounded">{result.error}</div>}
    </div>
  );
}
