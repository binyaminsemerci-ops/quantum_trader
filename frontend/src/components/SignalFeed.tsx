import { useEffect, useState } from "react";
import SignalDetail from "./SignalDetail";

export type Signal = {
  id: string;
  symbol: string;
  score: number;
  direction?: "LONG" | "SHORT";
  confidence?: number;
  timestamp: string;
  details?: Record<string, any>;
};

function toLocalString(iso: string) {
  try {
    const d = new Date(iso);
    // Example: "23:06:20 UTC" or local tz name when available
    const time = d.toLocaleTimeString();
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    return `${time} ${tz}`;
  } catch (err) {
    return iso;
  }
}

export default function SignalFeed() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [selected, setSelected] = useState<Signal | null>(null);

  const [pageSize, setPageSize] = useState<number>(10);

  useEffect(() => {
    let mounted = true;
    // Poll the paginated endpoint every 5s
    async function fetchSignals() {
      try {
        const base = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
        const res = await fetch(`${base}/signals/?page=1&page_size=${pageSize}`);
        if (!res.ok) return;
        const data = await res.json();
        // data.items contains typed signals; map backend 'side' to direction
        const mapped: Signal[] = (data.items || []).map((s: any) => ({
          id: s.id,
          symbol: s.symbol,
          score: s.score,
          direction: s.side === "buy" ? "LONG" : "SHORT",
          confidence: s.confidence,
          timestamp: s.timestamp,
          details: s.details,
        }));
        if (mounted) setSignals(mapped);
      } catch (err) {
        // swallow network errors for the stub
      }
    }

    fetchSignals();
    const id = setInterval(fetchSignals, 5000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [pageSize]);

  return (
    <div className="p-2 border rounded">
      <h3 className="font-semibold">Signal Feed (mock)</h3>
      <div className="flex items-center gap-2 mb-2">
        <label htmlFor="signal-page-size" className="text-sm">Page size:</label>
        <select id="signal-page-size" value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))} className="text-sm p-1 border rounded">
          <option value={5}>5</option>
          <option value={10}>10</option>
          <option value={20}>20</option>
        </select>
      </div>

      <ul className="space-y-2">
        {signals.map((s) => (
          <li key={s.id} className="p-2 border rounded-md hover:bg-slate-50 cursor-pointer" onClick={() => setSelected(s)}>
            <div className="flex justify-between items-start">
              <div>
                <div className="text-sm text-slate-600">{toLocalString(s.timestamp)}</div>
                <div className="flex items-baseline gap-3 mt-1">
                  <div className="font-mono text-sm">{s.symbol}</div>
                  <div>
                    {s.direction ? (
                      <span
                        className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                          s.direction === "LONG" ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
                        }`}
                      >
                        {s.direction}
                      </span>
                    ) : (
                      <span className="text-sm">—</span>
                    )}
                  </div>
                  <div className="text-sm">score: {s.score}</div>
                  {s.confidence !== undefined && (
                    <div className="text-xs text-slate-500">{Math.round(s.confidence * 100)}%</div>
                  )}
                </div>
                {s.details?.note && <div className="text-xs text-slate-400 mt-1">{s.details.note}</div>}
              </div>
            </div>
          </li>
        ))}
      </ul>
      <SignalDetail signal={selected} onClose={() => setSelected(null)} />
      {!signals.length && <div className="text-sm text-muted">No signals yet</div>}
    </div>
  );
}
