import { useEffect, useMemo, useState } from "react";
import { fetchRecentPrices } from "../api/prices";
import type { Candle } from "../api/prices";

type PricePoint = Candle;

function formatNumber(n: number) {
  return Number.isFinite(n) ? n.toFixed(2) : "-";
}

// A very small, dependency-free SVG price chart.
export default function PriceChart({ data }: { data?: PricePoint[] }) {
  const [internal, setInternal] = useState<PricePoint[] | null>(null);

  useEffect(() => {
    if (!data) {
      let mounted = true;
      // try fetch recent prices from backend; fallback to generated demo series
      fetchRecentPrices()
        .then((d) => mounted && setInternal(d))
        .catch(() => {
          if (!mounted) return;
          // generate a small demo series (random walk) for visual purposes
          const now = Date.now();
          const seed = 100 + Math.random() * 10;
          const demo: PricePoint[] = Array.from({ length: 40 }).map((_, i) => {
            const t = new Date(now - (40 - i) * 60000).toISOString();
            const open = seed + Math.sin(i / 5) * 2 + (i * 0.1);
            const close = open + (Math.random() - 0.5) * 1.5;
            const high = Math.max(open, close) + Math.random() * 0.8;
            const low = Math.min(open, close) - Math.random() * 0.8;
            const volume = Math.round(10 + Math.random() * 5);
            return { time: t, open, high, low, close, volume };
          });
          setInternal(demo);
        });
      return () => {
        mounted = false;
      };
    }
    return;
  }, [data]);

  const points = data ?? internal ?? [];

  const latest = points[points.length - 1];

  const svg = useMemo(() => {
    if (!points.length) return null;
    const w = 600;
    const h = 200;
    const padding = 20;
    const prices = points.map((p) => p.close);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const x = (i: number) => padding + (i / Math.max(1, points.length - 1)) * (w - padding * 2);
    const y = (v: number) => padding + ((max - v) / range) * (h - padding * 2);

    const path = points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(p.close)}`)
      .join(" ");

    const candles = points.map((p, i) => {
      const cx = x(i);
      const cy1 = y(p.open);
      const cy2 = y(p.close);
      const chigh = y(p.high);
      const clow = y(p.low);
      const color = p.close >= p.open ? "green" : "red";
      return { cx, cy1, cy2, chigh, clow, color };
    });

    return (
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        <rect x={0} y={0} width={w} height={h} fill="transparent" />
        <path d={path} fill="none" stroke="#2563eb" strokeWidth={2} strokeOpacity={0.9} />
        {candles.map((c, i) => (
          <g key={i}>
            {/* wick */}
            <line x1={c.cx} x2={c.cx} y1={c.chigh} y2={c.clow} stroke={c.color} strokeWidth={1} />
            {/* body */}
            <rect x={c.cx - 4} y={Math.min(c.cy1, c.cy2)} width={8} height={Math.max(1, Math.abs(c.cy2 - c.cy1))} fill={c.color} />
          </g>
        ))}
      </svg>
    );
  }, [points]);

  return (
    <div className="p-2 border rounded">
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-semibold">Price chart</h3>
        {latest && <div className="text-sm">Latest: {formatNumber(latest.close)}</div>}
      </div>
      <div>{svg ?? <div className="text-sm text-muted">Loading chart...</div>}</div>
    </div>
  );
}
