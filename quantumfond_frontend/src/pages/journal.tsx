import { useEffect, useState } from "react";
import InsightCard from "../components/InsightCard";
import { safeNum, safePct } from '../utils/formatters';

interface Trade {
  id: number;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price?: number;
  pnl?: number;
  tp: number;
  sl: number;
  trailing: number;
  confidence: number;
  model: string;
  features?: any;
  policy_state?: any;
  exit_reason: string;
  timestamp: string;
}

interface JournalStats {
  total: number;
  closed: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_pnl: number;
  total_pnl: number;
}

export default function Journal() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [stats, setStats] = useState<JournalStats | null>(null);
  const [filter, setFilter] = useState({ symbol: "", direction: "" });
  const [loading, setLoading] = useState(true);

  const apiUrl = (import.meta as any).env?.VITE_API_URL || "http://localhost:8026";

  useEffect(() => {
    fetchJournal();
    fetchStats();
  }, []);

  async function fetchJournal() {
    try {
      const params = new URLSearchParams();
      if (filter.symbol) params.append("symbol", filter.symbol);
      if (filter.direction) params.append("direction", filter.direction);
      
      const res = await fetch(`${apiUrl}/journal/history?${params}`);
      const data = await res.json();
      setTrades(data);
    } catch (err) {
      console.error("Failed to fetch journal:", err);
    } finally {
      setLoading(false);
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch(`${apiUrl}/journal/stats`);
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  }

  function handleFilter() {
    setLoading(true);
    fetchJournal();
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Trade Journal</h1>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-4">
          <InsightCard 
            title="Total Trades" 
            value={stats.total}
            icon="📊"
          />
          <InsightCard 
            title="Win Rate" 
            value={safePct(stats.win_rate, 1)}
            change={stats.win_rate || 0}
            trend={stats.win_rate >= 50 ? "up" : "down"}
          />
          <InsightCard 
            title="Avg PnL" 
            value={`$${safeNum(stats.avg_pnl)}`}
            trend={stats.avg_pnl >= 0 ? "up" : "down"}
          />
          <InsightCard 
            title="Total PnL" 
            value={`$${safeNum(stats.total_pnl)}`}
            trend={stats.total_pnl >= 0 ? "up" : "down"}
          />
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-4 p-4 bg-gray-900 border border-gray-800 rounded-lg">
        <input
          type="text"
          placeholder="Symbol (e.g., BTCUSDT)"
          value={filter.symbol}
          onChange={(e) => setFilter({ ...filter, symbol: e.target.value })}
          className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white"
        />
        <select
          value={filter.direction}
          onChange={(e) => setFilter({ ...filter, direction: e.target.value })}
          className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white"
        >
          <option value="">All Directions</option>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
        </select>
        <button
          onClick={handleFilter}
          className="px-4 py-2 bg-green-700 hover:bg-green-600 text-white rounded font-semibold"
        >
          Apply Filter
        </button>
        <button
          onClick={() => {
            setFilter({ symbol: "", direction: "" });
            fetchJournal();
          }}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded"
        >
          Clear
        </button>
      </div>

      {/* Trade Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-400">Loading trades...</div>
        ) : trades.length === 0 ? (
          <div className="p-8 text-center text-gray-400">No trades found</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-950">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Symbol</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Dir</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Entry</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">TP</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">SL</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Conf</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Model</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">PnL</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Exit</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {trades.map((t) => (
                <tr key={t.id} className="hover:bg-gray-800/50">
                  <td className="px-4 py-3 text-white font-mono">#{t.id}</td>
                  <td className="px-4 py-3 text-white font-medium">{t.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      t.direction === "BUY" 
                        ? "bg-green-900/30 text-green-400" 
                        : "bg-red-900/30 text-red-400"
                    }`}>
                      {t.direction}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-300">${safeNum(t.entry_price)}</td>
                  <td className="px-4 py-3 text-green-400">${safeNum(t.tp)}</td>
                  <td className="px-4 py-3 text-red-400">${safeNum(t.sl)}</td>
                  <td className="px-4 py-3 text-gray-300">{safePct(t.confidence, 0)}</td>
                  <td className="px-4 py-3 text-blue-400">{t.model}</td>
                  <td className="px-4 py-3">
                    {t.pnl ? (
                      <span className={t.pnl >= 0 ? "text-green-400" : "text-red-400"}>
                        ${safeNum(t.pnl)}
                      </span>
                    ) : (
                      <span className="text-gray-500">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-400">{t.exit_reason}</td>
                  <td className="px-4 py-3">
                    <a
                      href={`/replay?id=${t.id}`}
                      className="text-blue-400 hover:text-blue-300 underline text-xs"
                    >
                      Replay
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
