import { useEffect, useMemo, useState } from "react";
import { fetchRecentPrices } from "../api/prices";
import type { Candle } from "../api/prices";
import type { Signal } from "./SignalFeed";

type PricePoint = Candle;

function formatNumber(n: number) {
  return Number.isFinite(n) ? n.toFixed(2) : "-";
}

// A very small, dependency-free SVG price chart.
export default function PriceChart({ data, signals }: { data?: PricePoint[]; signals?: Signal[] }) {
  const [internal, setInternal] = useState<PricePoint[] | null>(null);
  const [highlightIdx, setHighlightIdx] = useState<number | null>(null);

  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (data) return;
    let mounted = true;
    fetchRecentPrices()
      .then(d => { if (mounted) { setInternal(d); setError(null); } })
      .catch(() => { if (mounted) setError('Failed to load price data'); });
    return () => { mounted = false; };
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

  // Simple overlay: render small markers above the chart for signals
  const overlay = useMemo(() => {
    if (!points.length || !signals || !signals.length) return null;
    const w = 600;
    const padding = 20;
    const prices = points.map((p) => p.close);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const x = (i: number) => padding + (i / Math.max(1, points.length - 1)) * (w - padding * 2);
    const y = (v: number) => padding + ((max - v) / range) * (200 - padding * 2);

    // map signal timestamps to nearest index in points
    const markers = signals
      .map((s) => {
        const sigTime = new Date(s.timestamp).getTime();
        let bestIdx = 0;
        let bestDiff = Infinity;
        points.forEach((p, i) => {
          const dt = Math.abs(new Date(p.time).getTime() - sigTime);
          if (dt < bestDiff) {
            bestDiff = dt;
            bestIdx = i;
          }
        });
        return { idx: bestIdx, score: s.score, side: s.direction };
      })
      .slice(0, 20);

    return (
      <svg width={600} height={200} viewBox={`0 0 ${600} ${200}`} className="absolute top-0 left-0 pointer-events-none">
        {markers.map((m, i) => (
          <g key={i} transform={`translate(${x(m.idx)}, ${y(points[m.idx].close) - 10})`}>
            <circle r={6} fill={m.side === "LONG" ? "#10b981" : "#ef4444"} opacity={0.9} />
            <text x={10} y={4} fontSize={10} fill="#111">{Math.round(m.score * 100)}%</text>
          </g>
        ))}
        {highlightIdx !== null && points[highlightIdx] && (
          <g transform={`translate(${x(highlightIdx)}, ${y(points[highlightIdx].close) - 10})`}>
            <circle r={12} fill="none" stroke="#f59e0b" strokeWidth={2} />
            <circle r={6} fill="#f59e0b" opacity={0.95} />
          </g>
        )}
      </svg>
    );
  }, [points, signals]);

  useEffect(() => {
    function onFocus(e: any) {
      const detail = e?.detail;
      if (!detail || !detail.timestamp) return;
      const sigTime = new Date(detail.timestamp).getTime();
      let bestIdx = 0;
      let bestDiff = Infinity;
      points.forEach((p, i) => {
        const dt = Math.abs(new Date(p.time).getTime() - sigTime);
        if (dt < bestDiff) {
          bestDiff = dt;
          bestIdx = i;
        }
      });
      setHighlightIdx(bestIdx);
      setTimeout(() => setHighlightIdx(null), 4000);
    }

    window.addEventListener('focus-signal', onFocus as any);
    return () => window.removeEventListener('focus-signal', onFocus as any);
  }, [points]);

  return (
    <div className="p-2 border rounded">
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-semibold">Price chart</h3>
        {latest && <div className="text-sm">Latest: {formatNumber(latest.close)}</div>}
      </div>
      <div className="relative">{svg ?? <div className="text-sm text-muted">{error || 'Loading chart...'}</div>}
        {overlay}
      </div>
    </div>
  );
}
