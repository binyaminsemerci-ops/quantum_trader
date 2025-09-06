import { useState } from "react";
import { useDashboardData } from "../hooks/useDashboardData.jsx";

export default function TradeLogs() {
  const { data } = useDashboardData();
  const logs = data.logs || [];

  const [filter, setFilter] = useState("all");

  const filtered =
    filter === "all" ? logs : logs.filter((l) => l.status === filter);

  const exportCSV = () => {
    const rows = [
      ["timestamp", "symbol", "side", "qty", "price", "status", "reason"],
      ...logs.map((l) => [
        l.timestamp,
        l.symbol,
        l.side,
        l.qty,
        l.price,
        l.status,
        l.reason || "",
      ]),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "trade_logs.csv";
    a.click();
  };

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">📜 Trade Logs</h2>
        <div className="space-x-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border rounded px-2 py-1"
          >
            <option value="all">All</option>
            <option value="accepted">Accepted</option>
            <option value="rejected">Rejected</option>
          </select>
          <button
            onClick={exportCSV}
            className="px-3 py-1 bg-gray-700 text-white rounded"
          >
            Export CSV
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-200 dark:border-gray-700 rounded-lg">
          <thead className="bg-gray-100 dark:bg-gray-700 dark:text-gray-200">
            <tr>
              <th className="px-4 py-2 text-left">Timestamp</th>
              <th className="px-4 py-2 text-left">Symbol</th>
              <th className="px-4 py-2 text-left">Side</th>
              <th className="px-4 py-2 text-left">Qty</th>
              <th className="px-4 py-2 text-left">Price</th>
              <th className="px-4 py-2 text-left">Status</th>
              <th className="px-4 py-2 text-left">Reason</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((log, i) => (
              <tr
                key={i}
                className="border-b dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800"
              >
                <td className="px-4 py-2">{log.timestamp}</td>
                <td className="px-4 py-2">{log.symbol}</td>
                <td className="px-4 py-2">{log.side}</td>
                <td className="px-4 py-2">{log.qty}</td>
                <td className="px-4 py-2">{log.price}</td>
                <td className="px-4 py-2">
                  <span
                    className={`px-2 py-1 rounded text-white text-xs font-semibold ${
                      log.status === "accepted"
                        ? "bg-green-600"
                        : "bg-red-600"
                    }`}
                  >
                    {log.status}
                  </span>
                </td>
                <td className="px-4 py-2">{log.reason || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
