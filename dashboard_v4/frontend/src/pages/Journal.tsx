import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = '/api';

interface JournalEntry {
  id: number;
  trade_symbol: string;
  trade_side: string | null;
  entry_price: number | null;
  exit_price: number | null;
  pnl: number | null;
  strategy_tag: string | null;
  notes: string | null;
  rating: number | null;
  mistakes: string | null;
  lessons: string | null;
  created_by: string;
  created_at: string;
}

interface JournalStats {
  total_entries: number;
  avg_rating: number | null;
  avg_pnl: number | null;
  top_strategies: { tag: string; count: number }[];
}

export default function Journal({ token }: { token: string | null }) {
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [stats, setStats] = useState<JournalStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterSymbol, setFilterSymbol] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  // Form state
  const [form, setForm] = useState({
    trade_symbol: '', trade_side: 'BUY', entry_price: '', exit_price: '',
    pnl: '', strategy_tag: '', notes: '', rating: '', mistakes: '', lessons: '',
  });

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const fetchData = async () => {
    try {
      const params = filterSymbol ? `?symbol=${filterSymbol}` : '';
      const [entriesRes, statsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/journal/entries${params}`),
        fetch(`${API_BASE_URL}/journal/stats`),
      ]);
      if (entriesRes.ok) setEntries(await entriesRes.json());
      if (statsRes.ok) setStats(await statsRes.json());
    } catch (err) {
      console.error('Failed to load journal:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [filterSymbol]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return alert('Login required');
    try {
      const body = {
        trade_symbol: form.trade_symbol,
        trade_side: form.trade_side || null,
        entry_price: form.entry_price ? parseFloat(form.entry_price) : null,
        exit_price: form.exit_price ? parseFloat(form.exit_price) : null,
        pnl: form.pnl ? parseFloat(form.pnl) : null,
        strategy_tag: form.strategy_tag || null,
        notes: form.notes || null,
        rating: form.rating ? parseInt(form.rating) : null,
        mistakes: form.mistakes || null,
        lessons: form.lessons || null,
      };
      const res = await fetch(`${API_BASE_URL}/journal/entries`, {
        method: 'POST', headers, body: JSON.stringify(body),
      });
      if (res.ok) {
        setShowForm(false);
        setForm({ trade_symbol: '', trade_side: 'BUY', entry_price: '', exit_price: '', pnl: '', strategy_tag: '', notes: '', rating: '', mistakes: '', lessons: '' });
        fetchData();
      }
    } catch (err) {
      console.error('Failed to create entry:', err);
    }
  };

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading journal...</div></div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-green-400">Trade Journal</h1>
        <div className="flex gap-3">
          <input
            type="text"
            placeholder="Filter by symbol..."
            value={filterSymbol}
            onChange={(e) => setFilterSymbol(e.target.value.toUpperCase())}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-green-500"
          />
          {token && (
            <button onClick={() => setShowForm(!showForm)}
              className="px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium transition">
              {showForm ? 'Cancel' : '+ New Entry'}
            </button>
          )}
        </div>
      </div>

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <InsightCard title="Total Entries" value={stats.total_entries} />
          <InsightCard title="Avg Rating" value={stats.avg_rating?.toFixed(1) ?? 'N/A'} color="text-yellow-400" />
          <InsightCard title="Avg PnL" value={stats.avg_pnl != null ? `$${stats.avg_pnl.toFixed(2)}` : 'N/A'}
            color={stats.avg_pnl != null && stats.avg_pnl >= 0 ? 'text-green-400' : 'text-red-400'} />
          <InsightCard title="Top Strategy" value={stats.top_strategies[0]?.tag ?? 'N/A'}
            subtitle={stats.top_strategies[0] ? `${stats.top_strategies[0].count} trades` : undefined} color="text-blue-400" />
        </div>
      )}

      {/* New entry form */}
      {showForm && (
        <form onSubmit={handleCreate} className="bg-gray-800 p-6 rounded-xl border border-gray-700 space-y-4">
          <h2 className="text-lg font-semibold text-green-400">New Journal Entry</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <input placeholder="Symbol *" required value={form.trade_symbol} onChange={(e) => setForm({ ...form, trade_symbol: e.target.value.toUpperCase() })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
            <select title="Trade side" value={form.trade_side} onChange={(e) => setForm({ ...form, trade_side: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm">
              <option value="BUY">BUY</option><option value="SELL">SELL</option>
            </select>
            <input placeholder="Entry price" type="number" step="any" value={form.entry_price} onChange={(e) => setForm({ ...form, entry_price: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
            <input placeholder="Exit price" type="number" step="any" value={form.exit_price} onChange={(e) => setForm({ ...form, exit_price: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
            <input placeholder="PnL ($)" type="number" step="any" value={form.pnl} onChange={(e) => setForm({ ...form, pnl: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
            <input placeholder="Strategy tag" value={form.strategy_tag} onChange={(e) => setForm({ ...form, strategy_tag: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
            <input placeholder="Rating (1-5)" type="number" min="1" max="5" value={form.rating} onChange={(e) => setForm({ ...form, rating: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
          </div>
          <textarea placeholder="Notes" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm h-20" />
          <div className="grid grid-cols-2 gap-3">
            <textarea placeholder="Mistakes / what went wrong" value={form.mistakes} onChange={(e) => setForm({ ...form, mistakes: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm h-16" />
            <textarea placeholder="Lessons / what to improve" value={form.lessons} onChange={(e) => setForm({ ...form, lessons: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm h-16" />
          </div>
          <button type="submit" className="px-6 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium transition">
            Save Entry
          </button>
        </form>
      )}

      {/* Entries table */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-750 border-b border-gray-700">
            <tr>
              <th className="px-4 py-3 text-left text-gray-400">Date</th>
              <th className="px-4 py-3 text-left text-gray-400">Symbol</th>
              <th className="px-4 py-3 text-left text-gray-400">Side</th>
              <th className="px-4 py-3 text-right text-gray-400">Entry</th>
              <th className="px-4 py-3 text-right text-gray-400">Exit</th>
              <th className="px-4 py-3 text-right text-gray-400">PnL</th>
              <th className="px-4 py-3 text-left text-gray-400">Strategy</th>
              <th className="px-4 py-3 text-center text-gray-400">Rating</th>
            </tr>
          </thead>
          <tbody>
            {entries.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-500">No journal entries yet</td></tr>
            ) : entries.map((e) => (
              <>
                <tr key={e.id}
                  onClick={() => setExpandedId(expandedId === e.id ? null : e.id)}
                  className="border-b border-gray-700/50 hover:bg-gray-750 cursor-pointer transition">
                  <td className="px-4 py-3 text-gray-300">{new Date(e.created_at).toLocaleDateString()}</td>
                  <td className="px-4 py-3 font-medium text-white">{e.trade_symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${e.trade_side === 'BUY' ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}`}>
                      {e.trade_side || '-'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-gray-300">{e.entry_price?.toFixed(4) ?? '-'}</td>
                  <td className="px-4 py-3 text-right text-gray-300">{e.exit_price?.toFixed(4) ?? '-'}</td>
                  <td className={`px-4 py-3 text-right font-medium ${(e.pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {e.pnl != null ? `$${e.pnl.toFixed(2)}` : '-'}
                  </td>
                  <td className="px-4 py-3 text-gray-400">{e.strategy_tag || '-'}</td>
                  <td className="px-4 py-3 text-center text-yellow-400">{'★'.repeat(e.rating || 0)}{'☆'.repeat(5 - (e.rating || 0))}</td>
                </tr>
                {expandedId === e.id && (
                  <tr key={`${e.id}-detail`} className="bg-gray-800/50">
                    <td colSpan={8} className="px-6 py-4 space-y-2">
                      {e.notes && <div><span className="text-gray-500 text-xs">Notes:</span><p className="text-gray-300 text-sm">{e.notes}</p></div>}
                      {e.mistakes && <div><span className="text-red-500 text-xs">Mistakes:</span><p className="text-red-300 text-sm">{e.mistakes}</p></div>}
                      {e.lessons && <div><span className="text-blue-500 text-xs">Lessons:</span><p className="text-blue-300 text-sm">{e.lessons}</p></div>}
                      <div className="text-gray-500 text-xs">Created by {e.created_by} on {new Date(e.created_at).toLocaleString()}</div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
