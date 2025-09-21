import { useEffect, useState } from 'react';

type Props = { symbol?: string; limit?: number };

export default function CandlesChart({ symbol = 'BTCUSDT', limit = 50 }: Props): JSX.Element {
  const [loading, setLoading] = useState(true);
  const [candles, setCandles] = useState<any[]>([]);

  useEffect(() => {
    let canceled = false;
    async function fetchCandles() {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/candles?symbol=${symbol}&limit=${limit}`);
        const data = await res.json();
        if (!canceled) setCandles(data.candles || []);
      } catch (err) {
        console.error('Failed to fetch candles', err);
      } finally {
        if (!canceled) setLoading(false);
      }
    }
    fetchCandles();
    return () => { canceled = true; };
  }, [symbol, limit]);

  if (loading) return <div className="p-4 bg-white rounded shadow animate-pulse">Loading candles…</div>;
  if (!candles.length) return <div className="p-4 bg-white rounded shadow">No candles</div>;

  return (
    <div className="p-4 bg-white rounded shadow">
      <h2 className="text-xl font-bold mb-2">📊 {symbol} Candles</h2>
      <div className="text-sm text-gray-600">Showing last {candles.length} candles.</div>
    </div>
  );
}
