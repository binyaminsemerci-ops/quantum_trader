import { useEffect, useMemo, useState } from "react";
import { fetchRecentPrices } from "../api/prices";
import { fetchSignals } from "../api/signals";
import type { Candle } from "../api/prices";
import type { Signal } from "../api/signals";

type PricePoint = Candle;

type Props = {
  data?: PricePoint[];
  signals?: Signal[];
  source?: 'live' | 'demo';
};

function formatNumber(n: number) {
  return Number.isFinite(n) ? n.toFixed(2) : '-';
}

export default function PriceChart({ data, signals, source }: Props) {
  const [internalPrices, setInternalPrices] = useState<PricePoint[] | null>(data ?? null);
  const [priceSource, setPriceSource] = useState<'live' | 'demo' | null>(source ?? null);
  const [loading, setLoading] = useState<boolean>(!data);
  const [error, setError] = useState<string | null>(source === 'demo' ? 'Using demo price data (backend fetch failed)' : null);
  const [internalSignals, setInternalSignals] = useState<Signal[] | null>(null);
  const [highlightIdx, setHighlightIdx] = useState<number | null>(null);

  useEffect(() => {
    if (data) {
      setInternalPrices(null);
      setPriceSource(source ?? null);
      setLoading(false);
      setError(source === 'demo' ? 'Using demo price data (backend fetch failed)' : null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchRecentPrices()
      .then((result) => {
        if (cancelled) return;
        setInternalPrices(result.candles);
        setPriceSource(result.source);
        setError(result.source === 'demo' ? 'Using demo price data (backend fetch failed)' : null);
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn('price fetch failed', err);
        setError('Unable to load price data');
        setInternalPrices([]);
        setPriceSource('demo');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [data, source]);

  useEffect(() => {
    if (signals) {
      setInternalSignals(null);
      return;
    }
    let cancelled = false;
    fetchSignals({ limit: 40 })
      .then(({ items }) => {
        if (!cancelled) setInternalSignals(items);
      })
      .catch((err) => {
        if (!cancelled) {
          console.warn('signal overlay fetch failed', err);
          setInternalSignals([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [signals]);

  const points = data ?? internalPrices ?? [];
  const overlaySignals = signals ?? internalSignals ?? [];
  const latest = points[points.length - 1];
  const resolvedSource = source ?? priceSource ?? (overlaySignals.some((s) => s.source === 'demo') ? 'demo' : null);

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

    const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(p.close)}`).join(' ');

    const candles = points.map((p, i) => {
      const cx = x(i);
      const cy1 = y(p.open);
      const cy2 = y(p.close);
      const chigh = y(p.high);
      const clow = y(p.low);
      const color = p.close >= p.open ? 'green' : 'red';
      return { cx, cy1, cy2, chigh, clow, color };
    });

    return (
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        <rect x={0} y={0} width={w} height={h} fill="transparent" />
        <path d={path} fill="none" stroke="#2563eb" strokeWidth={2} strokeOpacity={0.9} />
        {candles.map((c, i) => (
          <g key={i}>
            <line x1={c.cx} x2={c.cx} y1={c.chigh} y2={c.clow} stroke={c.color} strokeWidth={1} />
            <rect x={c.cx - 4} y={Math.min(c.cy1, c.cy2)} width={8} height={Math.max(1, Math.abs(c.cy2 - c.cy1))} fill={c.color} />
          </g>
        ))}
      </svg>
    );
  }, [points]);

  const overlay = useMemo(() => {
    if (!points.length || !overlaySignals.length) return null;
    const w = 600;
    const padding = 20;
    const prices = points.map((p) => p.close);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const x = (i: number) => padding + (i / Math.max(1, points.length - 1)) * (w - padding * 2);
    const y = (v: number) => padding + ((max - v) / range) * (200 - padding * 2);

    const markers = overlaySignals
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
        return { idx: bestIdx, score: s.score, direction: s.direction };
      })
      .slice(0, 20);

    return (
      <svg width={600} height={200} viewBox={`0 0 ${600} ${200}`} className="absolute top-0 left-0 pointer-events-none">
        {markers.map((m, i) => (
          <g key={i} transform={`translate(${x(m.idx)}, ${y(points[m.idx].close) - 10})`}>
            <circle r={6} fill={m.direction === 'LONG' ? '#10b981' : m.direction === 'SHORT' ? '#ef4444' : '#64748b'} opacity={0.9} />
            <text x={10} y={4} fontSize={10} fill="#111">
              {Math.round(m.score * 100)}%
            </text>
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
  }, [points, overlaySignals, highlightIdx]);

  useEffect(() => {
    function onFocus(e: Event) {
      const detail = (e as CustomEvent).detail as { timestamp?: string } | undefined;
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
      window.setTimeout(() => setHighlightIdx(null), 4000);
    }

    window.addEventListener('focus-signal', onFocus as EventListener);
    return () => window.removeEventListener('focus-signal', onFocus as EventListener);
  }, [points]);

  return (
    <div className="p-2 border rounded">
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-semibold">Price chart</h3>
        {latest && <div className="text-sm">Latest: {formatNumber(latest.close)}</div>}
      </div>
      {resolvedSource === 'demo' && (
        <div className="mb-2 text-xs text-amber-600" role="status">
          Using demo price data (backend fetch failed or live data disabled)
        </div>
      )}
      {error && resolvedSource !== 'demo' && (
        <div className="mb-2 text-xs text-amber-600" role="alert">
          {error}
        </div>
      )}
      <div className="relative">
        {svg ?? (
          <div className="text-sm text-slate-500">
            {loading ? 'Loading chart...' : 'No price data available'}
          </div>
        )}
        {overlay}
      </div>
    </div>
  );
}
