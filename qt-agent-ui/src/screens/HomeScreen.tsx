import { useState, useEffect } from "react";
import { WidgetShell, ScreenGrid, span } from "../components/Widget";
import { useMetrics, usePositions } from "../hooks/useData";
import AiOsStatusWidget from "../components/AiOsStatusWidget";

export default function HomeScreen() {
  const [time, setTime] = useState(new Date());
  const { data: metrics } = useMetrics();
  const { data: positions } = usePositions();

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const timeStr = time.toLocaleTimeString("no-NO", { hour: "2-digit", minute: "2-digit" });
  const dateStr = time.toLocaleDateString("no-NO", { weekday: "long", day: "2-digit", month: "long" });

  return (
    <div className="mx-auto max-w-[1280px] px-4 pb-24 pt-4">
      <ScreenGrid>
        {/* Stor klokke/infovindu (venstre) */}
        <WidgetShell className={`${span.twoThirds} h-44 flex flex-col justify-center`}>
          <div className="text-5xl font-semibold tracking-tight">{timeStr}</div>
          <div className="text-slate-500 mt-2">{dateStr}</div>
        </WidgetShell>

        {/* Status widgets (høyre topp) */}
        <WidgetShell title="AI Status" className={`${span.third} h-44`}>
          <div className="text-2xl font-bold text-emerald-600">
            {metrics?.ai_status || "READY"}
          </div>
          <div className="text-sm text-slate-500 mt-2">
            Mode: {metrics?.autonomous_mode ? "AUTO" : "MANUAL"}
          </div>
        </WidgetShell>

        <WidgetShell title="Active Positions" className={`${span.third} h-44`}>
          <div className="text-3xl font-bold">{positions.length}</div>
          <div className="text-sm text-slate-500 mt-2">
            Total PnL: ${positions.reduce((sum, p) => sum + (p.pnl || 0), 0).toFixed(2)}
          </div>
        </WidgetShell>

        <WidgetShell title="Total Trades" className={`${span.third} h-44`}>
          <div className="text-3xl font-bold">{metrics?.total_trades || 0}</div>
          <div className="text-sm text-slate-500 mt-2">
            Win Rate: {(metrics?.win_rate || 0).toFixed(1)}%
          </div>
        </WidgetShell>

        {/* AI-OS System Health Widget */}
        <div className={`${span.twoThirds}`}>
          <AiOsStatusWidget />
        </div>

        {/* System Overview - Better layout */}
        <WidgetShell title="System Overview" className={`${span.third}`}>
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-black/5 dark:bg-white/5">
              <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Recent Activity</h4>
              <div className="text-xs text-slate-700 dark:text-slate-300">
                Signals: {metrics?.signals_count || 0} • Positions: {positions.length}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-black/5 dark:bg-white/5">
              <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Performance</h4>
              <div className="text-xs text-slate-700 dark:text-slate-300">
                PnL: ${metrics?.pnl_usd?.toFixed(2) || "0.00"}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-black/5 dark:bg-white/5">
              <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Health</h4>
              <div className="text-xs text-emerald-600 dark:text-emerald-400">All Systems Operational</div>
            </div>
          </div>
        </WidgetShell>
      </ScreenGrid>
    </div>
  );
}
