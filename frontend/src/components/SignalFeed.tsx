import { useEffect, useState } from "react";

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

  useEffect(() => {
    let mounted = true;
    // Minimal polling stub: fetch the latest signals every 5s
    async function fetchSignals() {
      try {
        const base = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
        const res = await fetch(`${base}/signals/recent`);
        if (!res.ok) return;
        const data = (await res.json()) as Signal[];
        if (mounted) setSignals(data.slice(0, 20));
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
  }, []);

  return (
    <div className="p-2 border rounded">
      <h3 className="font-semibold">Signal Feed (mock)</h3>
      <ul>
        {signals.map((s) => (
          <li key={s.id} className="py-1">
            <div className="text-sm text-slate-600">{toLocalString(s.timestamp)}</div>
            <div className="flex items-baseline gap-3">
              <div className="font-mono text-sm">{s.symbol}</div>
              <div className="text-sm">{s.direction ?? "—"}</div>
              <div className="text-sm">score: {s.score}</div>
              {s.confidence !== undefined && <div className="text-xs text-slate-500">conf: {s.confidence}</div>}
            </div>
            {s.details?.note && <div className="text-xs text-slate-400">{s.details.note}</div>}
          </li>
        ))}
      </ul>
      {!signals.length && <div className="text-sm text-muted">No signals yet</div>}
    </div>
  );
}
