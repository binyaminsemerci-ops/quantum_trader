import { useState, useEffect } from 'react';

interface AISignal {
  id?: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  status: string;
  reason: string;
  timestamp: string;
}

export default function AITradingMonitor(): JSX.Element {
  const [stats, setStats] = useState<any>(null);
  const [signals, setSignals] = useState<AISignal[]>([]);
  const [trades, setTrades] = useState<AISignal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAll() {
      setLoading(true);
      try {
        const statsRes = await fetch('/stats');
        const statsJson = await statsRes.json();
        setStats(statsJson);

    // Use lightweight /trades/recent endpoint (limit param for count)
    const tradesRes = await fetch('/trades/recent?limit=8');
        const tradesJson = await tradesRes.json();
        setTrades(tradesJson);

        const logsRes = await fetch('/trade_logs/recent?limit=5');
        const logsJson = await logsRes.json();
        setSignals(logsJson.logs);
      } catch (e) {
        setStats(null);
        setTrades([]);
        setSignals([]);
      }
      setLoading(false);
    }
    fetchAll();
    const interval = setInterval(fetchAll, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white flex items-center">
          🤖 AI Trading Monitor
        </h3>
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${stats ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
          <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
            {stats ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
      {loading ? (
        <div className="text-center py-8 text-gray-500">Laster AI-data ...</div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gradient-to-br from-green-500 to-green-600 p-3 rounded-lg text-white">
              <div className="text-sm opacity-90">Total Trades</div>
              <div className="text-xl font-bold">{stats?.total_trades ?? '-'}</div>
            </div>
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-3 rounded-lg text-white">
              <div className="text-sm opacity-90">Total P&L</div>
              <div className="text-xl font-bold">${stats?.pnl?.toFixed(2) ?? '-'}</div>
            </div>
            <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-3 rounded-lg text-white">
              <div className="text-sm opacity-90">Win Rate</div>
              <div className="text-xl font-bold">{stats?.analytics?.win_rate ? `${stats.analytics.win_rate.toFixed(1)}%` : '-'}</div>
            </div>
            <div className="bg-gradient-to-br from-orange-500 to-orange-600 p-3 rounded-lg text-white">
              <div className="text-sm opacity-90">Sharpe Ratio</div>
              <div className="text-xl font-bold">{stats?.analytics?.sharpe_ratio ?? '-'}</div>
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* AI Signals (from trade_logs) */}
            <div>
              <h4 className="text-md font-semibold mb-3 text-gray-700 dark:text-gray-300">🎯 AI Signals</h4>
              <div className="space-y-3">
                {signals.map((signal, idx) => (
                  <div key={signal.timestamp + signal.symbol + idx} className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-800 dark:text-white">{signal.symbol}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          signal.side === 'BUY' ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200' :
                          signal.side === 'SELL' ? 'bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200' :
                          'bg-yellow-100 text-yellow-700 dark:bg-yellow-800 dark:text-yellow-200'
                        }`}>
                          {signal.side}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {signal.status}
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {signal.qty} @ ${signal.price} • {signal.reason}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            {/* Recent AI Trades (from /trades/recent) */}
            <div>
              <h4 className="text-md font-semibold mb-3 text-gray-700 dark:text-gray-300">📊 Recent AI Trades</h4>
              <div className="space-y-3">
                {trades.map((trade, idx) => (
                  <div key={trade.timestamp + trade.symbol + idx} className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-800 dark:text-white">{trade.symbol}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          trade.side === 'BUY' ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200' :
                          trade.side === 'SELL' ? 'bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200' :
                          'bg-yellow-100 text-yellow-700 dark:bg-yellow-800 dark:text-yellow-200'
                        }`}>
                          {trade.side}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {trade.status}
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {trade.qty} @ ${trade.price} • {trade.reason}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}