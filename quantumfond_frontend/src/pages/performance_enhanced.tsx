import { useEffect, useState } from "react";
import InsightCard from "../components/InsightCard";
import { safeNum, safePct } from "../utils/formatters";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function PerformanceEnhanced() {
  const [metrics, setMetrics] = useState<any>({});
  const [curve, setCurve] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  const fetchPerformanceData = async () => {
    try {
      const response = await fetch(`${API_URL}/performance/metrics`);
      const data = await response.json();
      setMetrics(data.metrics || {});
      setCurve(data.curve || []);
      setLoading(false);
    } catch (err) {
      console.error("Failed to fetch performance data:", err);
      setLoading(false);
    }
  };

  const handleExport = (format: string) => {
    window.open(`${API_URL}/reports/export/${format}`, "_blank");
  };

  return (
    <div className="space-y-6 p-4">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Performance Analytics</h1>
        <div className="flex gap-2">
          <button
            onClick={() => handleExport("json")}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition"
          >
            📄 JSON
          </button>
          <button
            onClick={() => handleExport("csv")}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition"
          >
            📊 CSV
          </button>
          <button
            onClick={() => handleExport("pdf")}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition"
          >
            📑 PDF
          </button>
        </div>
      </div>

      {loading ? (
        <div className="text-center text-gray-400 py-12">Loading metrics...</div>
      ) : (
        <>
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <InsightCard
              title="Total Return"
              value={`$${safeNum(metrics.total_return)}`}
              trend={metrics.total_return > 0 ? "up" : "down"}
              icon="💰"
            />
            <InsightCard
              title="Win Rate"
              value={safePct(metrics.winrate, 1)}
              trend={metrics.winrate > 0.5 ? "up" : "down"}
              icon="🎯"
            />
            <InsightCard
              title="Profit Factor"
              value={safeNum(metrics.profit_factor)}
              trend={metrics.profit_factor > 1 ? "up" : "down"}
              icon="⚖️"
            />
            <InsightCard
              title="Sharpe Ratio"
              value={safeNum(metrics.sharpe)}
              trend={metrics.sharpe > 1 ? "up" : "down"}
              icon="📈"
            />
            <InsightCard
              title="Sortino Ratio"
              value={safeNum(metrics.sortino)}
              trend={metrics.sortino > 1 ? "up" : "down"}
              icon="📊"
            />
            <InsightCard
              title="Max Drawdown"
              value={`$${safeNum(metrics.max_drawdown)}`}
              trend="down"
              icon="⚠️"
            />
          </div>

          {/* Trade Statistics */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-white">Trade Statistics</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              <div>
                <div className="text-sm text-gray-400 mb-1">Total Trades</div>
                <div className="text-2xl font-bold text-white">{metrics.total_trades || 0}</div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Winning Trades</div>
                <div className="text-2xl font-bold text-green-400">{metrics.winning_trades || 0}</div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Losing Trades</div>
                <div className="text-2xl font-bold text-red-400">{metrics.losing_trades || 0}</div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Avg Win</div>
                <div className="text-2xl font-bold text-green-400">${safeNum(metrics.average_win)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Avg Loss</div>
                <div className="text-2xl font-bold text-red-400">${safeNum(metrics.average_loss)}</div>
              </div>
            </div>
          </div>

          {/* Equity Curve */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-white">Equity Curve</h2>
            {curve.length > 0 ? (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curve} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#9ca3af"
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                    />
                    <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1f2937",
                        border: "1px solid #374151",
                        borderRadius: "8px",
                        color: "#fff"
                      }}
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                      formatter={(value: any) => [`$${safeNum(value)}`, "Equity"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={false}
                      name="Equity"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-80 flex items-center justify-center text-gray-500">
                No equity curve data available
              </div>
            )}
          </div>

          {/* Risk Metrics */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-white">Risk-Adjusted Performance</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Sharpe Ratio</span>
                <div className="text-right">
                  <span className="text-white font-mono text-lg">{safeNum(metrics.sharpe)}</span>
                  <span className="text-xs text-gray-500 ml-2">
                    {metrics.sharpe > 2 ? "Excellent" : metrics.sharpe > 1 ? "Good" : "Poor"}
                  </span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Sortino Ratio</span>
                <div className="text-right">
                  <span className="text-white font-mono text-lg">{safeNum(metrics.sortino)}</span>
                  <span className="text-xs text-gray-500 ml-2">
                    {metrics.sortino > 2 ? "Excellent" : metrics.sortino > 1 ? "Good" : "Poor"}
                  </span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Max Drawdown</span>
                <div className="text-right">
                  <span className="text-red-400 font-mono text-lg">${safeNum(Math.abs(metrics.max_drawdown))}</span>
                  <span className="text-xs text-gray-500 ml-2">
                    {Math.abs(metrics.max_drawdown) < 1000 ? "Low Risk" : Math.abs(metrics.max_drawdown) < 5000 ? "Medium Risk" : "High Risk"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
