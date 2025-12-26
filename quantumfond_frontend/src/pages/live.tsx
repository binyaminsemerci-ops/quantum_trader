import { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import InsightCard from '../components/InsightCard';
import { safeNum } from '../utils/formatters';

export default function Live() {
  const [trades, setTrades] = useState<any[]>([]);
  const signals = useWebSocket('signals');
  const positions = useWebSocket('positions');

  useEffect(() => {
    const apiUrl = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8026';
    fetch(`${apiUrl}/trades/active`)
      .then(res => res.json())
      .then(data => setTrades(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Live AI Trading</h1>

      {/* AI Signal & Position Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <InsightCard 
          title={`Last AI Signal - ${signals?.model || '—'}`}
          value={`${signals?.signal || '—'} ${signals?.symbol || ''}`}
          change={signals?.confidence ? parseFloat(safeNum(signals.confidence * 100, 1)) : undefined}
          trend="up"
        />
        <InsightCard 
          title="Last Position" 
          value={`${positions?.symbol || '—'} ${positions?.direction || ''}`}
          icon="📊"
        />
      </div>

      {/* Active Trades Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-950">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Symbol</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Side</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Quantity</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Price</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">P&L</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {trades.map((trade) => (
              <tr key={trade.id} className="hover:bg-gray-800/50">
                <td className="px-6 py-4 text-sm font-medium text-white">{trade.symbol}</td>
                <td className="px-6 py-4 text-sm">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    trade.side === 'LONG' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {trade.side}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-300">{trade.quantity}</td>
                <td className="px-6 py-4 text-sm text-gray-300">${safeNum(trade.price)}</td>
                <td className="px-6 py-4 text-sm">
                  <span className={(trade.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}>
                    ${safeNum(trade.pnl)}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-400">{trade.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
