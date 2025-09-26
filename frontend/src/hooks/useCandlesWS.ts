import { useEffect, useRef, useState } from 'react';
import type { OHLCV } from '../types';

type Result = { data: OHLCV[]; connected: boolean; error?: string };

export default function useCandlesWS(path = '/ws/dashboard') {
  const [data, setData] = useState<OHLCV[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.host;
    const url = `${scheme}://${host}${path}`;
    let cancelled = false;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          if (payload && Array.isArray(payload.chart)) {
            setData(payload.chart as OHLCV[]);
          }
        } catch (err) {
          console.warn('Failed to parse ws message', err);
        }
      };

      ws.onerror = () => {
        setError('WebSocket error');
      };

      ws.onclose = () => {
        if (!cancelled) setConnected(false);
      };
    } catch (err) {
      setError(String(err));
    }

    return () => {
      cancelled = true;
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {}
      }
    };
  }, [path]);

  return { data, connected, error } as Result;
}
