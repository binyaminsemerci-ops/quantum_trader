import { useEffect, useState } from 'react';
import api from '../utils/api';
import type { OHLCV } from '../types';

type Props = { symbol?: string; limit?: number };

export default function CandlesChart({ symbol = 'BTCUSDT', limit = 50 }: Props): JSX.Element {
  const [loading, setLoading] = useState(true);
  const [candles, setCandles] = useState<OHLCV[]>([]);

  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        // backend exposes /api/candles which may return { candles: OHLCV[] }
        const resp = await api.get(`/candles?symbol=${encodeURIComponent(symbol)}&limit=${encodeURIComponent(String(limit))}`);
        if (!canceled && resp && 'data' in resp && Array.isArray(resp.data)) {
          setCandles(resp.data as OHLCV[]);
        }
      } catch (err) {
        console.error('Failed to fetch candles', err);
      } finally {
        if (!canceled) setLoading(false);
      }
    })();
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
