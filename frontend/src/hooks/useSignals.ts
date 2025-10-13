import { useEffect, useState } from 'react';

export interface LiveSignal {
  id?: string;
  timestamp: string; // ISO string
  symbol: string;
  side: 'buy' | 'sell';
  score?: number;       // 0-1 strength
  confidence?: number;  // 0-1 confidence
  // Normalised field for the price chart overlay (LONG/SHORT)
  direction?: 'LONG' | 'SHORT';
}

/**
 * Poll the backend AI signals endpoint and normalise for UI.
 * Tries canonical path first ("/signals/recent") then legacy "/api/signals/recent".
 * Falls back to empty array (no demo data) when unavailable.
 */
export function useSignals(options: { intervalMs?: number; limit?: number; symbol?: string } = {}) {
  const { intervalMs = 8000, limit = 20, symbol } = options;
  const [signals, setSignals] = useState<LiveSignal[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchOnce() {
      setLoading(true);
      setError(null);
      const qs = new URLSearchParams({ limit: String(limit), profile: 'mixed' });
      if (symbol) qs.set('symbol', symbol);
      const paths = [
        `/signals/recent?${qs.toString()}`,
        `/api/signals/recent?${qs.toString()}` // fallback path variant
      ];
      for (const p of paths) {
        try {
          const res = await fetch(p);
          if (!res.ok) continue;
            const raw = await res.json();
            if (cancelled) return;
            if (Array.isArray(raw)) {
              const mapped: LiveSignal[] = raw.map((r: any, idx) => ({
                id: r.id ?? `${Date.now()}-${idx}`,
                timestamp: typeof r.timestamp === 'string' ? r.timestamp : new Date(r.timestamp).toISOString(),
                symbol: r.symbol ?? 'UNKNOWN',
                side: r.side === 'sell' ? 'sell' : 'buy',
                score: typeof r.score === 'number' ? r.score : (typeof r.strength === 'number' ? r.strength : undefined),
                confidence: typeof r.confidence === 'number' ? r.confidence : undefined,
                direction: r.side === 'sell' ? 'SHORT' : 'LONG'
              }));
              setSignals(mapped);
              setLoading(false);
              return;
            }
        } catch (e: any) {
          // continue loop to try next path
        }
      }
      if (!cancelled) {
        setSignals([]); // no demo fallback – show nothing if backend not ready
        setLoading(false);
        setError(prev => prev ?? 'No signals available');
      }
    }

    fetchOnce();
    const id = setInterval(fetchOnce, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs, limit, symbol]);

  return { signals, loading, error };
}
