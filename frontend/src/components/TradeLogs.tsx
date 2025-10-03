import { useEffect, useState } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';

type LogItem = { id?: number; timestamp?: string; symbol?: string; side?: string; qty?: number; price?: number; status?: string; reason?: string };

export default function TradeLogs(): JSX.Element {
  const { data } = useDashboardData();
  const initial: LogItem[] = data?.logs || [];
  const [logs, setLogs] = useState<LogItem[]>(initial);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    let mounted = true;
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/trade_logs`);
    ws.onmessage = (ev) => {
      if (!mounted) return;
      try {
        const arr = JSON.parse(ev.data) as LogItem[];
        // append new logs to the front
        setLogs((prev) => [...arr.reverse(), ...prev].slice(0, 200));
      } catch (e) {
        // ignore
      }
    };
    ws.onopen = () => {
      // on open we could fetch recent logs if initial is empty
      if (mounted && logs.length === 0) {
        fetch('/api/trade_logs?limit=100')
          .then((r) => r.json())
          .then((j) => {
            if (!mounted) return;
            setLogs(j.logs || []);
          })
          .catch(() => {});
      }
    };
    ws.onclose = () => {
      // keep existing logs; client can refresh
    };
    return () => {
      mounted = false;
      try { ws.close(); } catch (e) {}
    };
  }, []);

  const filtered = filter === 'all' ? logs : logs.filter((l) => l.status === filter);

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">📜 Trade Logs</h2>
        <div className="space-x-2">
          <select aria-label="Filter trade logs" value={filter} onChange={(e) => setFilter(e.target.value)} className="border rounded px-2 py-1">
            <option value="all">All</option>
            <option value="simulated">Simulated</option>
            <option value="simulated">Simulated</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-200 rounded-lg">
          <thead className="bg-gray-100 text-left">
            <tr>
              <th className="px-4 py-2">Timestamp</th>
              <th className="px-4 py-2">Symbol</th>
              <th className="px-4 py-2">Side</th>
              <th className="px-4 py-2">Qty</th>
              <th className="px-4 py-2">Price</th>
              <th className="px-4 py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((log, i) => (
              <tr key={log.id ?? i} className="border-b hover:bg-gray-50">
                <td className="px-4 py-2">{log.timestamp}</td>
                <td className="px-4 py-2">{log.symbol}</td>
                <td className="px-4 py-2">{log.side}</td>
                <td className="px-4 py-2">{log.qty}</td>
                <td className="px-4 py-2">{log.price}</td>
                <td className="px-4 py-2">{log.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
