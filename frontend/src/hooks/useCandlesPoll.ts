import { useEffect, useState, useRef } from 'react';
import api from '../utils/api';
import type { OHLCV } from '../types';

type Result = { data: OHLCV[]; loading: boolean; error?: string };

export default function useCandlesPoll(symbol = 'BTCUSDT', limit = 200, intervalMs = 5000) {
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
        if (resp && 'data' in resp && Array.isArray(resp.data)) {
          setData(resp.data as OHLCV[]);
        }
      } catch (err: unknown) {
        setError(String(err));
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
