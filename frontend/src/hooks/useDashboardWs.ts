import { useEffect, useRef, useState } from 'react';

export interface DashboardWsPayload {
  stats?: any;
  trades?: any[];
  logs?: any[];
  chart?: any[];
}

interface Options {
  enabled?: boolean;
  url?: string;
  reconnectMs?: number;
}

export function useDashboardWs(options: Options = {}) {
  // In development, use relative path for Vite proxy; in production use absolute URL
  const isDev = (import.meta as any).env?.DEV;
  const defaultUrl = isDev ? `ws://${window.location.host}/ws/dashboard` : `ws://127.0.0.1:8000/ws/dashboard`;
  const { enabled = true, url = defaultUrl, reconnectMs = 4000 } = options;
  const [data, setData] = useState<DashboardWsPayload | null>(null);
  const [status, setStatus] = useState<'idle'|'connecting'|'open'|'closed'|'error'>('idle');
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) return; // keep disabled to avoid noise
    let cancelled = false;
    function connect() {
      if (cancelled) return;
      setStatus('connecting');
      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;
        ws.onopen = () => { if (!cancelled) setStatus('open'); };
        ws.onmessage = (ev) => {
          try {
            const parsed = JSON.parse(ev.data);
            setData(parsed);
          } catch { /* ignore */ }
        };
        ws.onerror = () => { if (!cancelled) setStatus('error'); };
        ws.onclose = () => {
          if (cancelled) return;
            setStatus('closed');
            retryRef.current = window.setTimeout(connect, reconnectMs);
        };
      } catch {
        setStatus('error');
        retryRef.current = window.setTimeout(connect, reconnectMs);
      }
    }
    connect();
    return () => {
      cancelled = true;
      if (retryRef.current) window.clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, [enabled, url, reconnectMs]);

  return { data, status };
}
