// frontend/src/hooks/useDashboardData.jsx
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

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        setData(payload);
        setLastUpdated(new Date().toLocaleTimeString());
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

export function useDashboardData() {
  return useContext(DashboardContext);
}
