<<<<<<< Updated upstream
import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';

type Trade = {
  id?: string | number;
  symbol?: string;
  side?: string;
  qty?: number;
  price?: number;
  status?: string;
  timestamp?: string;
};

type LogItem = {
  timestamp?: string;
  symbol?: string;
  side?: string;
  qty?: number;
  price?: number;
  status?: string;
};

type ChartPoint = { timestamp?: string; equity?: number };

type Stats = {
  analytics?: { win_rate?: number; sharpe_ratio?: number; trades_count?: number };
  risk?: { max_trade_exposure?: number; daily_loss_limit?: number; exposure_per_symbol?: Record<string, number> };
  pnl_per_symbol?: Record<string, number>;
};

type DashboardData = {
  stats?: Stats | null;
  trades?: Trade[];
  logs?: LogItem[];
  chart?: ChartPoint[];
  candles?: any[];
} | null;
export type ToastShape = { message?: string; type?: string } | null;

type DashboardContextType = {
  data: DashboardData;
  connected: boolean;
  paused: boolean;
  setPaused: (v: boolean) => void;
  fallback: boolean;
  lastUpdated: string | null;
  toast: ToastShape;
  setToast: (t: ToastShape) => void;
};

const DashboardContext = createContext<DashboardContextType | null>(null);

export function DashboardProvider({ children }: { children: ReactNode }) {
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

=======
<<<<<<<< Updated upstream:frontend/src/hooks/useDashboardData.jsx
// Auto-generated re-export stub
export { default } from './useDashboardData.tsx';
========
// frontend/src/hooks/useDashboardData.tsx
import { createContext, useContext, useEffect, useState } from "react";

const DashboardContext = createContext(null);

export function DashboardProvider({ children }) {
  const [data, setData] = useState({
    stats: null,
    trades: [],
    logs: [],
    chart: [],
  });
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [toast, setToast] = useState(null);

  // --- Fallback fetch via REST ---
  async function fetchFallback() {
    try {
      const [statsRes, tradesRes, logsRes, chartRes] = await Promise.all([
        fetch("http://127.0.0.1:8000/api/stats"),
        fetch("http://127.0.0.1:8000/api/trades"),
        fetch("http://127.0.0.1:8000/api/trade_logs?limit=50"),
        fetch("http://127.0.0.1:8000/api/chart"),
      ]);

  const stats = await statsRes.json();
  const trades = await tradesRes.json();
  const logs = await logsRes.json();
  const chart = await chartRes.json();

      setData({
        stats: stats,
        trades: trades.trades || [],
        logs: logs.logs || [],
        chart: chart || [],
      });
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Fallback fetch error:", err);
    }
  }

  // --- WebSocket connection ---
>>>>>>> Stashed changes
  useEffect(() => {
    if (paused) return;

    if (!fallback) {
<<<<<<< Updated upstream
      const ws = new WebSocket('ws://127.0.0.1:8000/ws/dashboard');
=======
      const ws = new WebSocket("ws://127.0.0.1:8000/ws/dashboard");
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        setData(payload);
        setLastUpdated(new Date().toLocaleTimeString());

        // 🚀 Toast når en trade kommer inn
        if (payload.logs && payload.logs.length > 0) {
          const latest = payload.logs[0];
          setToast({
            message: `Trade ${latest.status.toUpperCase()}: ${latest.symbol} ${latest.side} ${latest.qty}@${latest.price}`,
            type: latest.status === "accepted" ? "success" : "error",
          });
>>>>>>> Stashed changes
        }
      };

      return () => ws.close();
    }
  }, [paused, fallback]);

<<<<<<< Updated upstream
  useEffect(() => {
    if (paused || !fallback) return;
    fetchFallback();
=======
  // --- Polling if fallback active ---
  useEffect(() => {
    if (paused || !fallback) return;

    fetchFallback(); // initial
>>>>>>> Stashed changes
    const id = setInterval(fetchFallback, 3000);
    return () => clearInterval(id);
  }, [paused, fallback]);

  return (
<<<<<<< Updated upstream
    <DashboardContext.Provider value={{ data, connected, paused, setPaused, fallback, lastUpdated, toast, setToast }}>
=======
    <DashboardContext.Provider
      value={{
        data,
        connected,
        paused,
        setPaused,
        fallback,
        lastUpdated,
        toast,
        setToast,
      }}
    >
>>>>>>> Stashed changes
      {children}
    </DashboardContext.Provider>
  );
}

<<<<<<< Updated upstream
export function useDashboardData(): DashboardContextType {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboardData must be used within DashboardProvider');
  return ctx;
}
=======
export function useDashboardData() {
  return useContext(DashboardContext);
}
>>>>>>>> Stashed changes:frontend/src/hooks/useDashboardData.tsx
>>>>>>> Stashed changes
