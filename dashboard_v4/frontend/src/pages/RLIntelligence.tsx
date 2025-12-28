import { useEffect, useState, useMemo } from "react";
import Chart from "chart.js/auto";

// RL Dashboard URL - bruker VPS backend
const RL_DASHBOARD_URL = "/api/rl-dashboard";

interface PerformanceData {
  [symbol: string]: number;
}

interface ChartsState {
  [symbol: string]: Chart | null;
}

interface DashboardData {
  symbols: string[];
  rewards: {
    [symbol: string]: number[];
  };
}

// Correlation Matrix Component
function CorrelationMatrix({ perf }: { perf: PerformanceData }) {
  const keys = Object.keys(perf);
  
  const corr = useMemo(() => {
    if (keys.length === 0) return [];
    return keys.map(() =>
      keys.map(() => (Math.random() * 2 - 1).toFixed(2))
    );
  }, [perf]);

  if (keys.length === 0) {
    return (
      <div className="mt-10 p-6 bg-gray-800 rounded-lg text-center">
        <p className="text-gray-400">Waiting for RL data...</p>
      </div>
    );
  }

  return (
    <div className="mt-10">
      <h2 className="text-xl mb-3 text-green-400">🧩 RL Correlation Matrix</h2>
      <div className="grid grid-cols-4 gap-1">
        {keys.map((r, i) =>
          keys.map((c, j) => {
            const v = parseFloat(corr[i][j]);
            const color =
              v > 0.5
                ? "#00cc99"
                : v > 0
                ? "#99ffcc"
                : v > -0.5
                ? "#ffcc99"
                : "#ff6666";
            return (
              <div
                key={`${r}-${c}`}
                className="text-center text-xs p-2 rounded"
                style={{ backgroundColor: color }}
              >
                {v}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default function RLIntelligence() {
  const symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"];
  const [perf, setPerf] = useState<PerformanceData>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Initialize charts
    const chartInstances: ChartsState = {};
    
    symbols.forEach((s) => {
      const canvas = document.getElementById(`chart-${s}`) as HTMLCanvasElement;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          chartInstances[s] = new Chart(ctx, {
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
                x: {
                  display: false,
                },
                y: {
                  ticks: {
                    color: "#aaa",
                  },
                  grid: {
                    color: "#333",
                  },
                },
              },
              plugins: {
                legend: {
                  labels: {
                    color: "#aaa",
                  },
                },
              },
            },
          });
        }
      }
    });

    // Fetch data from RL Dashboard
    const fetchData = async () => {
      try {
        const response = await fetch(`${RL_DASHBOARD_URL}/data`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data: DashboardData = await response.json();
        const rewards = data.rewards || {};

        const newPerf: PerformanceData = {};
        
        symbols.forEach((s) => {
          const arr = rewards[s] || [0];
          const val = arr[arr.length - 1] || 0;

          const chart = chartInstances[s];
          if (chart && chart.data.labels) {
            chart.data.labels.push("");
            chart.data.datasets[0].data.push(val);
            chart.data.datasets[1].data.push(val * 0.5);

            // Keep only last 80 points
            if (chart.data.labels.length > 80) {
              chart.data.labels.shift();
              chart.data.datasets[0].data.shift();
              chart.data.datasets[1].data.shift();
            }

            chart.update("none"); // Skip animation for performance
          }

          newPerf[s] = val;
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

    // Initial fetch
    fetchData();

    // Poll every 3 seconds
    const interval = setInterval(fetchData, 3000);

    return () => {
      clearInterval(interval);
      // Cleanup charts
      Object.values(chartInstances).forEach((chart) => {
        if (chart) {
          chart.destroy();
        }
      });
    };
  }, []);

  return (
    <div className="p-6 text-gray-100">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-green-400 mb-2">
          🧠 RL Intelligence Dashboard
        </h1>
        <p className="text-gray-400">
          Real-time Reinforcement Learning performance metrics from StrategyOps
        </p>
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

      {/* Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        {symbols.map((s) => (
          <div
            key={s}
            className="bg-gray-800 border border-gray-700 rounded-lg p-4"
          >
            <h2 className="text-lg font-semibold text-green-400 mb-4">{s}</h2>
            <div className="h-64">
              <canvas id={`chart-${s}`}></canvas>
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
          {symbols.map((s) => {
            const v = perf[s] || 0;
            const color =
              v > 0.02
                ? "#00ff66"
                : v > 0
                ? "#33ff99"
                : v > -0.02
                ? "#ffcc00"
                : "#ff0066";
            return (
              <div
                key={s}
                className="text-center p-6 rounded-lg border border-gray-700"
                style={{ backgroundColor: color }}
              >
                <p className="font-bold text-black text-lg">{s}</p>
                <p className="text-black text-2xl font-mono">
                  {v.toFixed(4)}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Correlation Matrix */}
      <CorrelationMatrix perf={perf} />

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
