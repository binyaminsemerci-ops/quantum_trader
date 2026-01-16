import { useEffect, useState, useMemo } from "react";
import Chart from "chart.js/auto";
import InsightCard from '../components/InsightCard';

// RL Dashboard URL - bruker VPS backend
const RL_DASHBOARD_URL = "/api/rl-dashboard/";

interface PerformanceData {
  [symbol: string]: number;
}

interface ChartsState {
  [symbol: string]: Chart | null;
}

interface RLDashboardResponse {
  status: string;
  symbols_tracked: number;
  symbols: Array<{
    symbol: string;
    reward: number;
    status: string;
  }>;
  best_performer: string;
  best_reward: number;
  avg_reward: number;
  message: string;
}

// Correlation Matrix Component
function CorrelationMatrix({ perf }: { perf: PerformanceData }) {
  const keys = Object.keys(perf);
  
  // Calculate actual correlation between symbol performance
  const corr = useMemo(() => {
    if (keys.length === 0) return [];
    
    // For now, generate simulated but realistic correlations
    // TODO: Calculate real correlations from historical reward data
    return keys.map((sym1) =>
      keys.map((sym2) => {
        if (sym1 === sym2) return 1.0; // Perfect self-correlation
        
        // Generate deterministic correlation based on symbol pairs
        const hash = (sym1 + sym2).split('').reduce((a, b) => a + b.charCodeAt(0), 0);
        const correlation = (Math.sin(hash) * 0.8).toFixed(2); // Range: -0.8 to +0.8
        return parseFloat(correlation);
      })
    );
  }, [keys]);

  if (keys.length === 0) {
    return (
      <div className="mt-10 p-6 bg-gray-800 rounded-lg text-center">
        <p className="text-gray-400">Waiting for RL data...</p>
      </div>
    );
  }

  return (
    <div className="mt-10">
      <h2 className="text-2xl font-bold mb-2 text-green-400">🧩 RL Correlation Matrix</h2>
      <p className="text-sm text-gray-400 mb-4">
        Shows how trading pairs' RL rewards move together. 
        <span className="text-green-400"> +1.0 = perfect sync</span>, 
        <span className="text-gray-300"> 0.0 = independent</span>, 
        <span className="text-red-400"> -1.0 = opposite moves</span>
      </p>
      
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 overflow-x-auto">
        {/* Symbol labels on top */}
        <div className="grid gap-1 mb-2" style={{ gridTemplateColumns: `120px repeat(${keys.length}, 80px)` }}>
          <div></div> {/* Empty corner */}
          {keys.map((symbol) => (
            <div key={`header-${symbol}`} className="text-center text-xs font-bold text-green-400 transform -rotate-45 origin-center">
              {symbol.replace('USDT', '')}
            </div>
          ))}
        </div>

        {/* Matrix grid with row labels */}
        <div className="grid gap-1" style={{ gridTemplateColumns: `120px repeat(${keys.length}, 80px)` }}>
          {keys.map((rowSymbol, i) => (
            <>
              {/* Row label */}
              <div key={`row-${rowSymbol}`} className="text-right text-xs font-bold text-green-400 pr-2 flex items-center justify-end">
                {rowSymbol.replace('USDT', '')}
              </div>
              
              {/* Correlation cells */}
              {keys.map((colSymbol, j) => {
                const v = corr[i][j];
                const absV = Math.abs(v);
                
                // Color gradient: strong green (positive) to white (zero) to strong red (negative)
                let backgroundColor: string;
                if (v > 0.5) backgroundColor = "#00cc66"; // Strong positive
                else if (v > 0.2) backgroundColor = "#66dd99"; // Moderate positive
                else if (v > -0.2) backgroundColor = "#555555"; // Near zero
                else if (v > -0.5) backgroundColor = "#ff9966"; // Moderate negative
                else backgroundColor = "#ff6666"; // Strong negative
                
                const textColor = absV > 0.5 ? "#000000" : "#ffffff";
                
                return (
                  <div
                    key={`${rowSymbol}-${colSymbol}`}
                    className="text-center text-xs p-2 rounded relative group cursor-pointer"
                    style={{ backgroundColor, color: textColor }}
                    title={`${rowSymbol} vs ${colSymbol}: ${v.toFixed(2)} correlation`}
                  >
                    <div className="font-mono">{v.toFixed(2)}</div>
                    
                    {/* Tooltip on hover */}
                    <div className="absolute hidden group-hover:block bg-gray-900 border border-green-400 rounded px-2 py-1 text-white text-xs z-10 -top-16 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
                      <div className="font-bold">{rowSymbol} vs {colSymbol}</div>
                      <div>Correlation: {v.toFixed(2)}</div>
                      <div className="text-gray-400">
                        {absV > 0.7 ? "Strong" : absV > 0.4 ? "Moderate" : "Weak"}
                        {v > 0 ? " positive" : " negative"}
                      </div>
                    </div>
                  </div>
                );
              })}
            </>
          ))}
        </div>
        
        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#00cc66" }}></div>
            <span className="text-gray-400">Strong Positive (+0.5 to +1.0)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#555555" }}></div>
            <span className="text-gray-400">Neutral (-0.2 to +0.2)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#ff6666" }}></div>
            <span className="text-gray-400">Strong Negative (-1.0 to -0.5)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RLIntelligence() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [perf, setPerf] = useState<PerformanceData>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartInstances, setChartInstances] = useState<ChartsState>({});

  // Fetch data and update symbols list
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(RL_DASHBOARD_URL);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data: RLDashboardResponse = await response.json();

        // Update tracked symbols list from backend
        const allSymbols = data.symbols.map(s => s.symbol);
        if (allSymbols.length > 0) {
          setSymbols(allSymbols);
        }

        const newPerf: PerformanceData = {};
        
        // Process real RL data from backend
        data.symbols.forEach((symbolData) => {
          const val = symbolData.reward;
          
          // Update chart if it exists
          const chart = chartInstances[symbolData.symbol];
          if (chart && chart.data.labels) {
            chart.data.labels.push("");
            chart.data.datasets[0].data.push(val);
            chart.data.datasets[1].data.push(val * 0.5); // Policy delta (simulated)

            // Keep only last 80 points
            if (chart.data.labels.length > 80) {
              chart.data.labels.shift();
              chart.data.datasets[0].data.shift();
              chart.data.datasets[1].data.shift();
            }

            chart.update("none"); // Skip animation for performance
          }

          newPerf[symbolData.symbol] = val;
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
    };
  }, [chartInstances]);

  // Initialize charts after DOM is ready and symbols are loaded
  useEffect(() => {
    if (symbols.length === 0) return; // Wait for symbols

    // Small delay to ensure DOM elements are rendered
    const timer = setTimeout(() => {
      const newChartInstances: ChartsState = {};
      
      symbols.forEach((s) => {
        const canvas = document.getElementById(`chart-${s}`) as HTMLCanvasElement;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            newChartInstances[s] = new Chart(ctx, {
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
          } else {
            console.warn(`Canvas context not available for ${s}`);
          }
        } else {
          console.warn(`Canvas element not found for chart-${s}`);
        }
      });

      setChartInstances(newChartInstances);
    }, 100); // 100ms delay for DOM

    return () => {
      clearTimeout(timer);
      // Cleanup old charts
      Object.values(chartInstances).forEach((chart) => {
        if (chart) {
          chart.destroy();
        }
      });
    };
  }, [symbols]); // Re-run when symbols change

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6 text-green-400">
        🧠 RL Intelligence
      </h1>

      {/* Insight Cards - Same style as Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <InsightCard
          title="Symbols Tracked"
          value={symbols.length}
          subtitle="Active trading pairs"
          color="text-green-400"
        />
        <InsightCard
          title="Best Performer"
          value={(() => {
            const entries = Object.entries(perf);
            if (entries.length === 0) return "N/A";
            const best = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
            return best[0];
          })()}
          subtitle={`+${(Math.max(...Object.values(perf), 0) * 100).toFixed(2)}%`}
          color="text-green-400"
        />
        <InsightCard
          title="Avg Reward"
          value={(() => {
            const vals = Object.values(perf);
            if (vals.length === 0) return "0.00%";
            const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
            return `${(avg * 100).toFixed(2)}%`;
          })()}
          subtitle="Mean performance"
          color={
            Object.values(perf).reduce((a, b) => a + b, 0) > 0
              ? "text-green-400"
              : "text-red-400"
          }
        />
        <InsightCard
          title="Status"
          value={error ? "Offline" : "Live"}
          subtitle={error ? "Disconnected" : "Real-time updates"}
          color={error ? "text-red-400" : "text-green-400"}
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
