import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { Trade, OHLCV, StatSummary } from '../types';

type LogItem = {
  timestamp?: string;
  symbol?: string;
  side?: string;
  qty?: number;
  price?: number;
  status?: string;
};

type ChartPoint = { timestamp?: string; equity?: number };

type Stats = StatSummary & {
  analytics?: { win_rate?: number; sharpe_ratio?: number; trades_count?: number };
  risk?: { max_trade_exposure?: number; daily_loss_limit?: number; exposure_per_symbol?: Record<string, number> };
  pnl_per_symbol?: Record<string, number>;
};

type DashboardData = {
  stats?: Stats | null;
  trades?: Trade[];
  logs?: LogItem[];
  chart?: ChartPoint[];
  candles?: OHLCV[];
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
  const { safeJson } = await import('../utils/api');
  const statsRaw = await safeJson(statsRes);
  const tradesRaw = await safeJson(tradesRes);
  const logsRaw = await safeJson(logsRes);
  const chartRaw = await safeJson(chartRes);

  const statsVal = (statsRaw && typeof statsRaw === 'object') ? (statsRaw as Stats) : null;
  const tradesVal = Array.isArray(tradesRaw) ? (tradesRaw as Trade[]) : (tradesRaw && typeof tradesRaw === 'object' && Array.isArray((tradesRaw as any).trades) ? (tradesRaw as any).trades : []);
  const logsVal = Array.isArray(logsRaw) ? (logsRaw as LogItem[]) : (logsRaw && typeof logsRaw === 'object' && Array.isArray((logsRaw as any).logs) ? (logsRaw as any).logs : []);
  const chartVal = Array.isArray(chartRaw) ? (chartRaw as ChartPoint[]) : [];

  setData({ stats: statsVal, trades: tradesVal, logs: logsVal, chart: chartVal });
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
          const { safeParse, extractToastFromPayload } = require('../utils/ws');
          const payload = safeParse(event.data);
          if (payload && typeof payload === 'object') {
            setData(payload);
            setLastUpdated(new Date().toLocaleTimeString());
            const t = extractToastFromPayload(payload);
            if (t) setToast(t);
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
