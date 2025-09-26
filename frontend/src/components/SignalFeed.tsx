import React, { useEffect, useState } from "react";

export type Signal = {
  id: string;
  symbol: string;
  score: number;
  timestamp: string;
};

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
          <li key={s.id}>{s.timestamp} — {s.symbol} ({s.score})</li>
        ))}
      </ul>
      {!signals.length && <div className="text-sm text-muted">No signals yet</div>}
    </div>
  );
}
