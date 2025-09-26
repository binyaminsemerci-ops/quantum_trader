import { useState } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';

type LogItem = { timestamp?: string; symbol?: string; side?: string; qty?: number; price?: number; status?: string };

export default function TradeLogs(): JSX.Element {
  const { data } = useDashboardData();
  const logs: LogItem[] = data?.logs || [];
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'all' ? logs : logs.filter((l) => l.status === filter);

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">📜 Trade Logs</h2>
        <div className="space-x-2">
          <select value={filter} onChange={(e) => setFilter(e.target.value)} className="border rounded px-2 py-1">
            <option value="all">All</option>
            <option value="accepted">Accepted</option>
            <option value="rejected">Rejected</option>
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
              <tr key={i} className="border-b hover:bg-gray-50">
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
