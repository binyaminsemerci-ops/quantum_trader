// frontend/src/hooks/useAutoRefresh.tsx
import React, { createContext, useContext, useEffect, useState } from "react";
import type { Trade, Signal, OHLCV } from '../types';
import { parseTradesPayload, parseLogsPayload, parseChartPayload } from '../lib/parseFallback';

export type DashboardData = {
  stats: Record<string, unknown> | null;
  trades: Trade[];
  logs: Record<string, unknown>[];
  chart: OHLCV[];
};

export type DashboardContextType = {
  data: DashboardData;
  connected: boolean;
  paused: boolean;
  setPaused: React.Dispatch<React.SetStateAction<boolean>>;
  fallback: boolean;
  lastUpdated: string | null;
};

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<DashboardData>({
    stats: null,
    trades: [],
    logs: [],
    chart: [],
  });
  const [connected, setConnected] = useState<boolean>(false);
  const [paused, setPaused] = useState<boolean>(false);
  const [fallback, setFallback] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

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

      // runtime guards for payload shapes
      const parsedTrades = parseTradesPayload(trades);
      const parsedLogs = parseLogsPayload(logs);
      const parsedChart = parseChartPayload(chart);

      setData({
        stats: stats,
        trades: parsedTrades,
        logs: parsedLogs,
        chart: parsedChart,
      });
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Fallback fetch error:", err);
    }
  }

  // --- WebSocket connection ---
  useEffect(() => {
    if (paused) return;

    if (!fallback) {
  const ws = new WebSocket("ws://127.0.0.1:8000/ws/dashboard");

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
        } catch (e) {
          console.error('Invalid websocket payload', e);
        }
      };

      return () => ws.close();
    }
  }, [paused, fallback]);

  // --- Polling if fallback active ---
  useEffect(() => {
    if (paused || !fallback) return;

    fetchFallback(); // initial
    const id = setInterval(fetchFallback, 3000);
    return () => clearInterval(id);
  }, [paused, fallback]);

  return (
    <DashboardContext.Provider
      value={{ data, connected, paused, setPaused, fallback, lastUpdated }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboardData(): DashboardContextType {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboardData must be used within a DashboardProvider');
  return ctx;
}
