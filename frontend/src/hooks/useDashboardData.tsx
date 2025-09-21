import React, { createContext, useContext, useEffect, useState } from 'react';

type DashboardData = any;
type ToastShape = { message?: string; type?: string } | null;

type DashboardContextType = {
  data: DashboardData;
  connected: boolean;
  paused: boolean;
  setPaused: (v: boolean) => void;
  fallback: boolean;
  lastUpdated: string | null;
  toast?: ToastShape;
  setToast?: (t: ToastShape) => void;
};

const DashboardContext = createContext<DashboardContextType | null>(null);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<DashboardData>({ stats: null, trades: [], logs: [], chart: [] });
  const [connected, setConnected] = useState<boolean>(false);
  const [paused, setPaused] = useState<boolean>(false);
  const [fallback, setFallback] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastShape>(null);

  async function fetchFallback() {
    try {
      const [statsRes, tradesRes, logsRes, chartRes] = await Promise.all([
        fetch('http://127.0.0.1:8000/api/stats'),
        fetch('http://127.0.0.1:8000/api/trades'),
        fetch('http://127.0.0.1:8000/api/trade_logs?limit=50'),
        fetch('http://127.0.0.1:8000/api/chart'),
      ]);

      const stats = await statsRes.json();
      const trades = await tradesRes.json();
      const logs = await logsRes.json();
      const chart = await chartRes.json();

      setData({ stats, trades: trades.trades || [], logs: logs.logs || [], chart: chart || [] });
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('Fallback fetch error:', err);
    }
  }

  useEffect(() => {
    if (paused) return;

    if (!fallback) {
      const ws = new WebSocket('ws://127.0.0.1:8000/ws/dashboard');

      ws.onopen = () => {
        setConnected(true);
        setFallback(false);
      };

      ws.onclose = () => {
        setConnected(false);
        setFallback(true);
      };

      ws.onerror = () => {
        setConnected(false);
        setFallback(true);
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const payload = JSON.parse(event.data);
          setData(payload);
          setLastUpdated(new Date().toLocaleTimeString());

          if (payload.logs && payload.logs.length > 0) {
            const latest = payload.logs[0];
            setToast({
              message: `Trade ${String(latest.status).toUpperCase()}: ${latest.symbol} ${latest.side} ${latest.qty}@${latest.price}`,
              type: latest.status === 'accepted' ? 'success' : 'error',
            });
          }
        } catch (err) {
          console.error('WS payload parse error', err);
        }
      };

      return () => ws.close();
    }
  }, [paused, fallback]);

  useEffect(() => {
    if (paused || !fallback) return;
    fetchFallback();
    const id = setInterval(fetchFallback, 3000);
    return () => clearInterval(id);
  }, [paused, fallback]);

  return (
    <DashboardContext.Provider value={{ data, connected, paused, setPaused, fallback, lastUpdated, toast, setToast }}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboardData(): DashboardContextType {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboardData must be used within DashboardProvider');
  return ctx;
}
