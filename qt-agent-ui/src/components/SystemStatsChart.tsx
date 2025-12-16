import { useEffect, useState } from "react";
import { Activity, TrendingUp, Target, Zap } from "lucide-react";

interface SystemMetrics {
  total_trades: number;
  win_rate: number;
  active_positions: number;
  signals_today: number;
  avg_latency_ms?: number;
  uptime_hours?: number;
}

export default function SystemStatsChart() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    total_trades: 0,
    win_rate: 0,
    active_positions: 0,
    signals_today: 0,
    avg_latency_ms: 0,
    uptime_hours: 0,
  });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [metricsRes, signalsRes, positionsRes] = await Promise.all([
          fetch("http://localhost:8000/api/metrics/system"),
          fetch("http://localhost:8000/api/ai/signals/latest"),
          fetch("http://localhost:8000/positions"),
        ]);

        const metricsData = await metricsRes.json();
        const signalsData = await signalsRes.json();
        const positionsData = await positionsRes.json();

        // Count actual positions (handle both array and object responses)
        let positionsCount = 0;
        if (Array.isArray(positionsData)) {
          positionsCount = positionsData.length;
        } else if (positionsData && typeof positionsData === 'object') {
          // If it's an object with positions array
          positionsCount = positionsData.positions?.length || 0;
        }

        // Filter signals from today
        const today = new Date().toDateString();
        const signalsToday = Array.isArray(signalsData) ? signalsData.filter((sig: any) => 
          new Date(sig.timestamp).toDateString() === today
        ).length : 0;

        setMetrics({
          total_trades: metricsData.total_trades || 0,
          win_rate: metricsData.win_rate || 0,
          active_positions: positionsCount,
          signals_today: signalsToday,
          avg_latency_ms: metricsData.avg_latency_ms || 12,
          uptime_hours: metricsData.uptime_hours || 24,
        });
      } catch (error) {
        console.error("Error fetching system metrics:", error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 15000); // Update every 15s

    return () => clearInterval(interval);
  }, []);

  const stats = [
    {
      icon: TrendingUp,
      label: "Total Trades",
      value: metrics.total_trades,
      color: "text-blue-500",
    },
    {
      icon: Target,
      label: "Win Rate",
      value: `${(metrics.win_rate * 100).toFixed(1)}%`,
      color: "text-emerald-500",
    },
    {
      icon: Activity,
      label: "Active Positions",
      value: metrics.active_positions,
      color: "text-purple-500",
    },
    {
      icon: Zap,
      label: "Signals Today",
      value: metrics.signals_today,
      color: "text-amber-500",
    },
  ];

  return (
    <div className="h-full flex flex-col justify-between">
      <div className="grid grid-cols-2 gap-3">
        {stats.map((stat, i) => (
          <div key={i} className="p-3 rounded-lg bg-black/5 dark:bg-white/5 flex items-center gap-2">
            <stat.icon className={`w-5 h-5 ${stat.color}`} />
            <div>
              <div className="text-xs text-slate-500">{stat.label}</div>
              <div className="text-lg font-bold">{stat.value}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
        <div className="flex justify-between text-sm">
          <span className="text-slate-500">Avg Latency:</span>
          <span className="font-semibold">{metrics.avg_latency_ms}ms</span>
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span className="text-slate-500">Uptime:</span>
          <span className="font-semibold">{metrics.uptime_hours}h</span>
        </div>
      </div>
    </div>
  );
}
