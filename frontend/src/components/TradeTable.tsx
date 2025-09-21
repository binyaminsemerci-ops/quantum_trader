import React, { useEffect, useState } from "react";
import type { Trade as SharedTrade } from '../types';

export type TradeTableProps = {
  initialTrades?: SharedTrade[] | null;
};

const TradesTable: React.FC<TradeTableProps> = ({ initialTrades = null }): JSX.Element => {
  const [trades, setTrades] = useState<SharedTrade[]>(Array.isArray(initialTrades) ? initialTrades : []);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    async function fetchTrades() {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL}/trades`);
        if (!response.ok) {
          throw new Error(`Failed to fetch trades: ${response.status}`);
        }
        const data = await response.json();
        if (!mounted) return;
        setTrades(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error(error);
        if (mounted) setTrades([]);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    fetchTrades();
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return <p>Loading trades...</p>;
  }

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">📊 Trades</h2>
      <table className="min-w-full bg-white shadow rounded-xl overflow-hidden">
        <thead>
          <tr className="bg-gray-200 text-left">
            <th className="px-4 py-2">ID</th>
            <th className="px-4 py-2">Pair</th>
            <th className="px-4 py-2">Side</th>
            <th className="px-4 py-2">Amount</th>
            <th className="px-4 py-2">Price</th>
            <th className="px-4 py-2">Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => {
            const id = trade && (trade.id ?? '') as string | number;
            const pair = (trade && (trade.symbol ?? (trade.pair as string)) as string) ?? '-';
            const side = trade && trade.side ? String(trade.side).toUpperCase() : '-';
            const qty = trade ? (trade.quantity ?? trade.amount ?? '-') : '-';
            const price = trade ? (trade.price ?? '-') : '-';
            const ts = trade && trade.timestamp ? new Date(String((trade as any).timestamp)).toLocaleString() : '-';

            return (
              <tr key={String(id)} className="border-t">
                <td className="px-4 py-2">{String(id)}</td>
                <td className="px-4 py-2">{pair}</td>
                <td className={`px-4 py-2 font-bold ${side === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>
                  {side}
                </td>
                <td className="px-4 py-2">{String(qty)}</td>
                <td className="px-4 py-2">{String(price)}</td>
                <td className="px-4 py-2">{ts}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
export default TradesTable;
