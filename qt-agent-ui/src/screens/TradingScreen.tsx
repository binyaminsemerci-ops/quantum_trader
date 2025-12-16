import { WidgetShell, ScreenGrid, span } from "../components/Widget";
import { useMetrics, usePositions, useSignals, useModelInfo } from "../hooks/useData";
import PriceChart from "../components/PriceChart";
import PnLChart from "../components/PnLChart";
import SystemStatsChart from "../components/SystemStatsChart";

export default function TradingScreen() {
  const { data: metrics } = useMetrics();
  const { data: positions } = usePositions();
  const { data: signals } = useSignals();
  const { data: modelInfo } = useModelInfo();

  const latestSignal = signals[0];

  return (
    <div className="mx-auto max-w-[1280px] px-4 pb-24 pt-4">
      <ScreenGrid>
        {/* A: Price/Equity – 8x2 */}
        <WidgetShell title="Price Chart (BTCUSDT)" className={`${span.twoThirds} h-[300px]`}>
          <PriceChart symbol="BTCUSDT" limit={100} />
        </WidgetShell>

        {/* B: AI-dock – 4x2 */}
        <WidgetShell title="AI Status" className={`${span.third} h-[300px]`}>
          <div className="space-y-3 text-sm">
            <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20">
              <div className="text-xs text-emerald-700 dark:text-emerald-300 font-medium mb-1">Latest Signal</div>
              {latestSignal ? (
                <>
                  <div className="font-bold">{latestSignal.symbol}</div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      latestSignal.action === "BUY" ? "bg-emerald-600 text-white" : 
                      latestSignal.action === "SELL" ? "bg-rose-600 text-white" : 
                      "bg-slate-400 text-white"
                    }`}>
                      {latestSignal.action}
                    </span>
                    <span className="text-xs">{(latestSignal.confidence * 100).toFixed(0)}% conf</span>
                  </div>
                </>
              ) : (
                <div className="text-slate-500">No signals yet</div>
              )}
            </div>

            <div>
              <div className="text-xs text-slate-500 mb-2">System Status</div>
              <div className="flex items-center gap-2 mb-1">
                <div className={`w-2 h-2 rounded-full ${metrics?.autonomous_mode ? "bg-emerald-500" : "bg-slate-400"}`} />
                <span>Mode: {metrics?.autonomous_mode ? "AUTO" : "MANUAL"}</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-sky-500" />
                <span>Status: {metrics?.ai_status || "READY"}</span>
              </div>
            </div>

            <div>
              <div className="text-xs text-slate-500 mb-2">AI Model</div>
              <div className="text-xs space-y-1">
                <div>Type: {modelInfo?.model_type || "N/A"}</div>
                <div>Accuracy: {((modelInfo?.accuracy || 0) * 100).toFixed(1)}%</div>
                <div>Features: {modelInfo?.features_count || 0}</div>
              </div>
            </div>

            <div>
              <div className="text-xs text-slate-500 mb-2">Performance</div>
              <div className="text-xs space-y-1">
                <div>Trades: {metrics?.total_trades || 0}</div>
                <div>Win Rate: {(metrics?.win_rate || 0).toFixed(1)}%</div>
                <div>PnL: ${(metrics?.pnl_usd || 0).toFixed(2)}</div>
              </div>
            </div>
          </div>
        </WidgetShell>

        {/* C: PnL by Day – 6x2 */}
        <WidgetShell title="PnL by Day" className={`${span.half} h-[260px]`}>
          <PnLChart />
        </WidgetShell>

        {/* D: Latency/Runtime – 6x2 */}
        <WidgetShell title="System Stats" className={`${span.half} h-[260px]`}>
          <SystemStatsChart />
        </WidgetShell>

        {/* E: Positions – 6x2 */}
        <WidgetShell title="Active Positions" className={`${span.half} h-[260px]`}>
          <div className="h-[220px] overflow-y-auto overflow-x-hidden">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[var(--panel)] z-10">
                <tr className="text-left border-b border-slate-200 dark:border-slate-700">
                  <th className="py-2 px-1">Symbol</th>
                  <th className="px-1">Side</th>
                  <th className="px-1">Size</th>
                  <th className="px-1">Entry</th>
                  <th className="px-1">PnL</th>
                </tr>
              </thead>
              <tbody>
                {positions.length === 0 ? (
                  <tr><td colSpan={5} className="py-4 text-center text-slate-400">No active positions</td></tr>
                ) : (
                  positions.map((pos, i) => (
                    <tr key={i} className="border-b border-slate-100 dark:border-slate-700">
                      <td className="py-2 px-1 font-medium">{pos.symbol}</td>
                      <td className={`px-1 ${pos.side === "LONG" ? "text-emerald-600" : "text-rose-600"}`}>
                        {pos.side}
                      </td>
                      <td className="px-1">{pos.size}</td>
                      <td className="px-1">${pos.entry_price?.toFixed(2)}</td>
                      <td className={`px-1 ${pos.pnl >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                        ${pos.pnl?.toFixed(2)} ({pos.pnl_pct?.toFixed(2)}%)
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </WidgetShell>

        {/* F: Signals – 6x2 */}
        <WidgetShell title="Signal Feed" className={`${span.half} h-[260px]`}>
          <div className="h-[220px] overflow-y-auto overflow-x-hidden">
            <div className="space-y-2 pr-2">
              {signals.length === 0 ? (
                <div className="text-center text-slate-400 py-4">No signals yet</div>
              ) : (
                signals.slice(0, 10).map((sig, i) => (
                  <div key={i} className="text-xs p-2 rounded-lg bg-black/5 dark:bg-white/5 flex-shrink-0">
                    <div className="flex justify-between items-start">
                      <div>
                        <span className="font-medium">{sig.symbol}</span>
                        <span className={`ml-2 ${sig.action === "BUY" ? "text-emerald-600" : sig.action === "SELL" ? "text-rose-600" : "text-slate-400"}`}>
                          {sig.action}
                        </span>
                      </div>
                      <span className="text-slate-500 text-[10px]">
                        {new Date(sig.timestamp).toLocaleTimeString("no-NO", { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>
                    <div className="text-slate-500 mt-1">
                      Conf: {(sig.confidence * 100).toFixed(0)}% @ ${sig.price?.toFixed(2)}
                    </div>
                    {sig.reason && <div className="text-slate-400 text-[10px] mt-1 truncate" title={sig.reason}>{sig.reason}</div>}
                  </div>
                ))
              )}
            </div>
          </div>
        </WidgetShell>
      </ScreenGrid>
    </div>
  );
}
