import { useEffect, useState } from "react";
import InsightCard from "../components/InsightCard";
import { Chart as ChartJS, registerables } from "chart.js";

ChartJS.register(...registerables);

const RL_DASHBOARD_URL = "/api/rl-dashboard/";

interface RLDashboardResponse {
  timestamp: string;
  symbols: string[];
  rewards: { [key: string]: number };
  metadata?: any;
}

export default function RLIntelligence() {
  const initialSymbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"];
  const [symbols, setSymbols] = useState<string[]>(initialSymbols);
  const [perf, setPerf] = useState<{ [key: string]: number }>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const chartInstances: { [key: string]: ChartJS } = {};

    symbols.forEach((sym) => {
      const canvas = document.getElementById(`chart-${sym}`) as HTMLCanvasElement;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          chartInstances[sym] = new ChartJS(ctx, {
            type: "line",
            data: {
              labels: [],
              datasets: [
                {
                  label: "Reward",
                  data: [],
                  borderColor: "#00ffcc",
                  backgroundColor: "rgba(0, 255, 204, 0.1)",
                  tension: 0.4,
                },
                {
                  label: "Policy Δ",
                  data: [],
                  borderColor: "#ff00aa",
                  backgroundColor: "rgba(255, 0, 170, 0.1)",
                  tension: 0.4,
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: { display: false },
                y: {
                  ticks: { color: "#aaa" },
                  grid: { color: "#333" },
                },
              },
              plugins: {
                legend: {
                  labels: { color: "#aaa" },
                },
              },
            },
          });
        }
      }
    });

    const fetchData = async () => {
      try {
        const response = await fetch(RL_DASHBOARD_URL);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data: RLDashboardResponse = await response.json();
        const rewardsData = data.rewards || {};

        // Update symbols from backend response
        if (data.symbols && data.symbols.length > 0) {
          setSymbols(data.symbols);
        }

        const newPerf: { [key: string]: number } = {};

        symbols.forEach((sym) => {
          const rewardHistory = Array.isArray(rewardsData[sym]) ? rewardsData[sym] : [rewardsData[sym] || 0];
          const latestReward = rewardHistory[rewardHistory.length - 1] || 0;

          const chart = chartInstances[sym];
          if (chart && chart.data.labels) {
            chart.data.labels.push("");
            (chart.data.datasets[0].data as number[]).push(latestReward);
            (chart.data.datasets[1].data as number[]).push(latestReward * 0.5);

            if (chart.data.labels.length > 80) {
              chart.data.labels.shift();
              (chart.data.datasets[0].data as number[]).shift();
              (chart.data.datasets[1].data as number[]).shift();
            }

            chart.update("none");
          }

          newPerf[sym] = latestReward;
        });

        setPerf(newPerf);
        setLoading(false);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch RL data:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000);

    return () => {
      clearInterval(interval);
      Object.values(chartInstances).forEach((chart) => {
        if (chart) chart.destroy();
      });
    };
  }, [symbols]);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6 text-green-400">
        🧠 RL Intelligence
      </h1>

      {/* Insight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <InsightCard
          title="Symbols Tracked"
          value={symbols.length}
          icon="📊"
        />
        <InsightCard
          title="Best Performer"
          value={(() => {
            const entries = Object.entries(perf);
            if (entries.length === 0) return "N/A";
            const best = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
            return best[0];
          })()}
          change={Math.max(...Object.values(perf), 0) * 100}
          trend="up"
          icon="🏆"
        />
        <InsightCard
          title="Avg Reward"
          value={(() => {
            const vals = Object.values(perf);
            if (vals.length === 0) return "0.00%";
            const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
            return `${(avg * 100).toFixed(2)}%`;
          })()}
          trend={
            Object.values(perf).reduce((a, b) => a + b, 0) > 0 ? "up" : "down"
          }
          icon="📈"
        />
        <InsightCard
          title="Status"
          value={error ? "Offline" : "Live"}
          trend={error ? "down" : "up"}
          icon={error ? "⚠️" : "✅"}
        />
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg">
          <p className="text-red-400">
            ⚠️ Unable to connect to RL Dashboard: {error}
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Ensure RL Dashboard is running at {RL_DASHBOARD_URL}
          </p>
        </div>
      )}

      {loading && (
        <div className="mb-6 p-4 bg-blue-900/30 border border-blue-500 rounded-lg">
          <p className="text-blue-400">🔄 Loading RL data...</p>
        </div>
      )}

      {/* Performance Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        {symbols.map((sym) => (
          <div
            key={sym}
            className="bg-gray-800 border border-gray-700 rounded-lg p-4"
          >
            <h2 className="text-lg font-semibold text-green-400 mb-4">{sym}</h2>
            <div className="h-64">
              <canvas id={`chart-${sym}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Performance Heatmap */}
      <div>
        <h2 className="text-2xl font-bold text-green-400 mb-4">
          🔥 RL Performance Heatmap
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {symbols.map((sym) => {
            const reward = perf[sym] || 0;
            const bgColor =
              reward > 0.02
                ? "#00ff66"
                : reward > 0
                ? "#33ff99"
                : reward > -0.02
                ? "#ffcc00"
                : "#ff0066";
            return (
              <div
                key={sym}
                className="text-center p-6 rounded-lg border border-gray-700"
                style={{ backgroundColor: bgColor }}
              >
                <p className="font-bold text-black text-lg">{sym}</p>
                <p className="text-black text-2xl font-mono">
                  {reward.toFixed(4)}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* System Info */}
      <div className="mt-10 p-4 bg-gray-800 border border-gray-700 rounded-lg">
        <h3 className="text-lg font-semibold text-green-400 mb-2">
          📡 System Info
        </h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Dashboard URL:</span>
            <br />
            <span className="text-green-400 font-mono">{RL_DASHBOARD_URL}</span>
          </div>
          <div>
            <span className="text-gray-400">Update Interval:</span>
            <br />
            <span className="text-green-400">3 seconds</span>
          </div>
          <div>
            <span className="text-gray-400">Symbols Tracked:</span>
            <br />
            <span className="text-green-400">{symbols.length}</span>
          </div>
          <div>
            <span className="text-gray-400">Status:</span>
            <br />
            <span className={error ? "text-red-400" : "text-green-400"}>
              {error ? "⚠️ Disconnected" : "✅ Connected"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
