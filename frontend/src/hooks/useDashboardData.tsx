import { useState, useEffect } from "react";
import type { StatSummary } from "../types";

export type DashboardData = {
  stats: StatSummary;
  chart: Array<{ timestamp: string; equity: number }>;

  logs: any[];
  watchlist: Array<{ symbol: string; price: number }>;

  //   TODO: Legg til flere felt etter behov
};

export function useDashboardData(): {
  connected: boolean;
  paused: boolean;
  setPaused: (p: boolean) => void;
  fallback: boolean;
  lastUpdated: string | null;
  toast: any;
  setToast: (t: any) => void;
  stats: StatSummary;
  chart: DashboardData["chart"];
  logs: any[];
  watchlist: DashboardData["watchlist"];
} {
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [toast, setToast] = useState<any>(null);

  // ✅ Legg til sikre defaults for dashboard-data
  const [stats, setStats] = useState<StatSummary>({
    profit: 0,
    winRate: 0,
    sharpe: 0,
    maxDrawdown: 0,
  });

  const [chart, setChart] = useState<DashboardData["chart"]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [watchlist, setWatchlist] = useState<DashboardData["watchlist"]>([]);

  useEffect(() => {
    // TODO: koble til WebSocket eller fetch fra backend
    // Nå setter vi dummy-data så frontend ikke krasjer
    setTimeout(() => {
      setStats({
        profit: 12.5,
        winRate: 65,
        sharpe: 1.8,
        maxDrawdown: -5.2,
      });
      setChart([
        { timestamp: "2024-01-01", equity: 10000 },
        { timestamp: "2024-02-01", equity: 10500 },
        { timestamp: "2024-03-01", equity: 11200 },
      ]);
      setLogs([{ id: 1, symbol: "BTCUSDT", side: "BUY", qty: 0.1, price: 42000 }]);
      setWatchlist([{ symbol: "BTCUSDT", price: 42000 }, { symbol: "ETHUSDT", price: 3000 }]);
      setConnected(true);
      setLastUpdated(new Date().toLocaleTimeString());
    }, 1000);
  }, []);

  return {
    connected,
    paused,
    setPaused,
    fallback,
    lastUpdated,
    toast,
    setToast,
    stats,
    chart,
    logs,
    watchlist,
  };
}
