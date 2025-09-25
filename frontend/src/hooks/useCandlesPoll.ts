import { useEffect, useState, useRef } from 'react';
import api from '../utils/api';
import { fetchRecentPrices } from '../api/prices';
import type { OHLCV } from '../types';

type Result = { data: OHLCV[]; loading: boolean; error?: string };

export default function useCandlesPoll(symbol = 'BTCUSDC', limit = 200, intervalMs = 5000) {
  const [data, setData] = useState<OHLCV[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    let timer: NodeJS.Timeout;

    async function fetchOnce() {
      try {
        const resp = await api.get(`/candles?symbol=${encodeURIComponent(symbol)}&limit=${encodeURIComponent(String(limit))}`);
        if (resp && 'data' in resp && Array.isArray(resp.data) && resp.data.length) {
          setData(resp.data as OHLCV[]);
        } else {
          // Fallback to helper which calls /prices/recent or returns deterministic mock candles
          const candles = await fetchRecentPrices(symbol, limit);
          // map Candle -> OHLCV shape
          setData(candles.map((c) => ({ timestamp: c.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume })) as OHLCV[]);
        }
      } catch (err: unknown) {
        setError(String(err));
        try {
          const candles = await fetchRecentPrices(symbol, limit);
          setData(candles.map((c) => ({ timestamp: c.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume })) as OHLCV[]);
        } catch {}
      } finally {
        if (mounted.current) setLoading(false);
      }
    }

    fetchOnce();
    timer = setInterval(fetchOnce, intervalMs);

    return () => {
      mounted.current = false;
      if (timer) clearInterval(timer);
    };
  }, [symbol, limit, intervalMs]);

  return { data, loading, error } as Result;
}
