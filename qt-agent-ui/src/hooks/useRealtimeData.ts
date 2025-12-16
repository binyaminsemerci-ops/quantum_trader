import { useState } from "react";
import { useWebSocket } from "./useWebSocket";

interface DashboardData {
  stats: {
    total_trades: number;
    avg_price: number;
    active_symbols: number;
    pnl: number;
    pnl_per_symbol: Record<string, number>;
    risk: any;
    analytics: any;
  };
  trades: any[];
  logs: any[];
  chart: any[];
}

export function useDashboardStream() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  
  const { isConnected, error } = useWebSocket({
    url: "ws://localhost:8000/ws/dashboard",
    onMessage: (data) => {
      setDashboardData(data);
    },
  });

  return {
    data: dashboardData,
    isConnected,
    error,
  };
}

export function useRealtimeMetrics() {
  const { data } = useDashboardStream();
  
  return {
    data: data?.stats || null,
    loading: false,
    error: null,
  };
}

export function useRealtimeTrades() {
  const { data } = useDashboardStream();
  
  return {
    data: data?.trades || [],
    loading: false,
    error: null,
  };
}

export function useRealtimeLogs() {
  const { data } = useDashboardStream();
  
  return {
    data: data?.logs || [],
    loading: false,
    error: null,
  };
}

export function useRealtimeChart() {
  const { data } = useDashboardStream();
  
  return {
    data: data?.chart || [],
    loading: false,
    error: null,
  };
}
