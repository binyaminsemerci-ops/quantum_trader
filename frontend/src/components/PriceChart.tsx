import React from "react";

// Minimal chart stub — later replace with Recharts or TradingView
type PricePoint = { time: string; open: number; high: number; low: number; close: number };

export default function PriceChart({ data }: { data: PricePoint[] }) {
  return (
    <div className="p-2 border rounded">
      <h3 className="font-semibold">Price chart (mock)</h3>
      <ul>
        {data.slice(-10).map((p) => (
          <li key={p.time}>{p.time}: {p.close}</li>
        ))}
      </ul>
    </div>
  );
}
