import { WidgetShell, ScreenGrid, span } from "../components/Widget";
import { useSignals, useModelInfo, useMetrics } from "../hooks/useData";

export default function SignalsScreen() {
  const { data: signals } = useSignals();
  const { data: modelInfo } = useModelInfo();
  const { data: metrics } = useMetrics();

  // Calculate signal distribution
  const buyCount = signals.filter(s => s.action === "BUY").length;
  const sellCount = signals.filter(s => s.action === "SELL").length;
  const holdCount = signals.filter(s => s.action === "HOLD").length;

  return (
    <div className="mx-auto max-w-[1280px] px-4 pb-24 pt-4">
      <ScreenGrid>
        <WidgetShell title="Distribution (24h)" className={`${span.third} h-[260px]`}>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">BUY signals</span>
              <span className="text-lg font-bold text-emerald-600">{buyCount}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">SELL signals</span>
              <span className="text-lg font-bold text-rose-600">{sellCount}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">HOLD signals</span>
              <span className="text-lg font-bold text-slate-400">{holdCount}</span>
            </div>
            <div className="pt-3 border-t mt-3">
              <div className="text-xs text-slate-500">Total: {signals.length} signals</div>
            </div>
          </div>
        </WidgetShell>

        <WidgetShell title="Model Info" className={`${span.third} h-[260px]`}>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Type:</span>
              <span className="font-semibold">{modelInfo?.model_type || "N/A"}</span>
            </div>
            <div className="flex justify-between">
              <span>Status:</span>
              <span className="font-semibold text-emerald-600">{modelInfo?.status || "READY"}</span>
            </div>
            <div className="flex justify-between">
              <span>Accuracy:</span>
              <span className="font-semibold">{((modelInfo?.accuracy || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Training Samples:</span>
              <span className="font-semibold">{modelInfo?.samples || "N/A"}</span>
            </div>
            {modelInfo?.training_date && (
              <div className="text-xs text-slate-500 pt-2 border-t">
                Last trained: {new Date(modelInfo.training_date).toLocaleDateString("no-NO")}
              </div>
            )}
          </div>
        </WidgetShell>

        <WidgetShell title="Health" className={`${span.third} h-[260px]`}>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
              <span className="text-sm">AI System Online</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
              <span className="text-sm">Backend Connected</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-sky-500"></div>
              <span className="text-sm">Mode: {metrics?.autonomous_mode ? "AUTO" : "MANUAL"}</span>
            </div>
            <div className="pt-3 border-t text-xs text-slate-500">
              All systems operational
            </div>
          </div>
        </WidgetShell>

        <WidgetShell title="Live Signal Feed" className={`${span.full} h-[360px]`}>
          <div className="h-full overflow-y-auto overflow-x-hidden">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-[var(--panel)] z-10 border-b border-slate-200 dark:border-slate-700">
                <tr className="text-left">
                  <th className="py-2 px-2">Time</th>
                  <th className="px-2">Symbol</th>
                  <th className="px-2">Action</th>
                  <th className="px-2">Confidence</th>
                  <th className="px-2">Price</th>
                  <th className="px-2">Reason</th>
                </tr>
              </thead>
              <tbody>
                {signals.length === 0 ? (
                  <tr><td colSpan={6} className="py-8 text-center text-slate-400">No signals available</td></tr>
                ) : (
                  signals.map((sig, i) => (
                    <tr key={i} className="border-b border-slate-100 dark:border-slate-700">
                      <td className="py-2 px-2 text-xs text-slate-500">
                        {new Date(sig.timestamp).toLocaleString("no-NO", { 
                          hour: "2-digit", 
                          minute: "2-digit",
                          day: "2-digit",
                          month: "short"
                        })}
                      </td>
                      <td className="px-2 font-medium">{sig.symbol}</td>
                      <td className="px-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          sig.action === "BUY" ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300" :
                          sig.action === "SELL" ? "bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-300" :
                          "bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300"
                        }`}>
                          {sig.action}
                        </span>
                      </td>
                      <td className="px-2">{(sig.confidence * 100).toFixed(0)}%</td>
                      <td className="px-2">${sig.price?.toFixed(2)}</td>
                      <td className="px-2 text-xs text-slate-500 max-w-xs truncate" title={sig.reason || "—"}>{sig.reason || "—"}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </WidgetShell>
      </ScreenGrid>
    </div>
  );
}
