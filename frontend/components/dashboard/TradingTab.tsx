/**
 * Trading Tab Component
 * DASHBOARD-V3-001: Full Visual UI
 * 
 * Displays:
 * - Open positions table
 * - Recent orders table
 * - Recent signals list
 * - Strategies per account
 */

import { useEffect, useState } from 'react';
import DashboardCard from '../DashboardCard';
import { safeNum, safeCurrency, safePercent } from '@/lib/formatters';

interface TradingData {
  timestamp: string;
  open_positions: any[];
  recent_orders: any[];
  recent_signals: any[];
  strategies_per_account: any[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function TradingTab() {
  const [data, setData] = useState<TradingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTradingData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard/trading`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trading data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTradingData();
    // Poll every 3 seconds for active trading data
    const interval = setInterval(fetchTradingData, 3000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        {[1, 2].map(i => (
          <div key={i} className="dashboard-card h-64 animate-pulse bg-gray-200" />
        ))}
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <p className="text-danger text-lg">⚠️ {error || 'No data'}</p>
        <button onClick={fetchTradingData} className="mt-4 px-4 py-2 bg-primary text-white rounded-lg">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Open Positions Table */}
      <DashboardCard title={`Open Positions (${data.open_positions.length})`}>
        <div className="overflow-x-auto">
          {data.open_positions.length > 0 ? (
            <table className="w-full text-sm">
              <thead className="bg-gray-100 dark:bg-slate-800">
                <tr>
                  <th className="px-4 py-2 text-left font-semibold">Symbol</th>
                  <th className="px-4 py-2 text-left font-semibold">Side</th>
                  <th className="px-4 py-2 text-right font-semibold">Size</th>
                  <th className="px-4 py-2 text-right font-semibold">Entry</th>
                  <th className="px-4 py-2 text-right font-semibold">Current</th>
                  <th className="px-4 py-2 text-right font-semibold">PnL</th>
                  <th className="px-4 py-2 text-left font-semibold">Exchange</th>
                </tr>
              </thead>
              <tbody>
                {data.open_positions.map((pos, idx) => (
                  <tr key={idx} className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-slate-800">
                    <td className="px-4 py-3 font-mono font-semibold">{pos.symbol}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        pos.side === 'LONG' ? 'bg-success text-white' : 'bg-danger text-white'
                      }`}>
                        {pos.side}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">{safeNum(pos.size, 4)}</td>
                    <td className="px-4 py-3 text-right font-mono">{safeCurrency(pos.entry_price)}</td>
                    <td className="px-4 py-3 text-right font-mono">{safeCurrency(pos.current_price)}</td>
                    <td className={`px-4 py-3 text-right font-bold ${
                      (pos.pnl || 0) >= 0 ? 'text-success' : 'text-danger'
                    }`}>
                      {safeCurrency(pos.pnl)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">{pos.exchange || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-center text-gray-600 dark:text-gray-400 py-8">
              No open positions
            </p>
          )}
        </div>
      </DashboardCard>

      {/* Recent Orders Table */}
      <DashboardCard title={`Recent Orders (Last 50)`}>
        <div className="overflow-x-auto max-h-96">
          {data.recent_orders.length > 0 ? (
            <table className="w-full text-sm">
              <thead className="bg-gray-100 dark:bg-slate-800 sticky top-0">
                <tr>
                  <th className="px-4 py-2 text-left font-semibold">Time</th>
                  <th className="px-4 py-2 text-left font-semibold">Symbol</th>
                  <th className="px-4 py-2 text-left font-semibold">Side</th>
                  <th className="px-4 py-2 text-right font-semibold">Size</th>
                  <th className="px-4 py-2 text-right font-semibold">Price</th>
                  <th className="px-4 py-2 text-left font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_orders.map((order, idx) => (
                  <tr key={idx} className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-slate-800">
                    <td className="px-4 py-2 text-xs text-gray-600">
                      {new Date(order.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-4 py-2 font-mono">{order.symbol}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded text-xs ${
                        order.side === 'BUY' ? 'bg-success text-white' : 'bg-danger text-white'
                      }`}>
                        {order.side}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right">{safeNum(order.size, 4)}</td>
                    <td className="px-4 py-2 text-right font-mono">{safeCurrency(order.price)}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded text-xs ${
                        order.status === 'FILLED' ? 'bg-success text-white' :
                        order.status === 'PENDING' ? 'bg-warning text-white' :
                        'bg-gray-500 text-white'
                      }`}>
                        {order.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-center text-gray-600 dark:text-gray-400 py-8">
              No recent orders
            </p>
          )}
        </div>
      </DashboardCard>

      {/* Recent Signals */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title={`Recent Signals (Last 20)`}>
          <div className="space-y-2 max-h-96 overflow-y-auto p-4">
            {data.recent_signals.length > 0 ? (
              data.recent_signals.slice(0, 20).map((signal, idx) => (
                <div key={idx} className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      signal.direction === 'LONG' ? 'bg-success text-white' : 'bg-danger text-white'
                    }`}>
                      {signal.direction}
                    </span>
                    <span className="font-mono font-semibold">{signal.symbol}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold">
                      Conf: {safeNum(signal.confidence * 100, 0)}%
                    </div>
                    <div className="text-xs text-gray-600">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-gray-600 dark:text-gray-400 py-8">
                No recent signals
              </p>
            )}
          </div>
        </DashboardCard>

        {/* Strategies per Account */}
        <DashboardCard title="Active Strategies">
          <div className="p-4 space-y-3">
            {data.strategies_per_account.length > 0 ? (
              data.strategies_per_account.map((strategy, idx) => (
                <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {strategy.account || 'Default'}
                    </span>
                    <span className="text-xs bg-primary text-white px-2 py-1 rounded">
                      {strategy.strategy_name}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm text-gray-600 dark:text-gray-400">
                    <div>Positions: {strategy.position_count || 0}</div>
                    <div>Win Rate: {safePercent(strategy.win_rate)}</div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-gray-600 dark:text-gray-400 py-8">
                No active strategies
              </p>
            )}
          </div>
        </DashboardCard>
      </div>

      {/* Update Timestamp */}
      <div className="text-center text-xs text-gray-500 dark:text-gray-400">
        Last updated: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
