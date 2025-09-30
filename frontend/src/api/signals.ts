export type Signal = {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT' | 'FLAT';
  score: number;
  confidence?: number;
  timestamp: string;
  side?: string;
  details?: Record<string, unknown>;
  source?: string;
};

export type FetchSignalsParams = {
  limit?: number;
  symbol?: string;
  profile?: 'mixed' | 'left' | 'right';
};

export type SignalsResult = {
  items: Signal[];
  source?: string;
};

function toDirection(side: unknown): Signal['direction'] {
  const normalized = typeof side === 'string' ? side.toLowerCase() : undefined;
  if (normalized === 'buy') return 'LONG';
  if (normalized === 'sell') return 'SHORT';
  return 'FLAT';
}

export async function fetchSignals({
  limit = 20,
  symbol = 'BTCUSDT',
  profile = 'mixed',
}: FetchSignalsParams = {}): Promise<SignalsResult> {
  const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
  const params = new URLSearchParams({
    limit: String(limit),
    symbol,
    profile,
  });

  try {
    const res = await fetch(`${base}/signals/recent?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    const rawItems: any[] = Array.isArray(payload)
      ? payload
      : payload && Array.isArray((payload as any).items)
      ? (payload as any).items
      : [];

    const items: Signal[] = rawItems.map((item, idx) => {
      const timestamp = item.timestamp ?? new Date().toISOString();
      const side = typeof item.side === 'string' ? item.side : undefined;
      const direction = toDirection(side);
      const details = (item.details && typeof item.details === 'object') ? (item.details as Record<string, unknown>) : undefined;
      const confidence = item.confidence != null ? Number(item.confidence) : undefined;
      return {
        id: String(item.id ?? `${symbol}-${idx}`),
        symbol: item.symbol ?? symbol,
        direction,
        side,
        score: Number(item.score ?? 0),
        confidence,
        timestamp: String(timestamp),
        details,
        source: (details?.source as string | undefined) ?? (direction === 'FLAT' ? 'demo' : undefined),
      };
    });

    const firstSource = items.find((item) => item.source)?.source;
    return { items, source: firstSource };
  } catch (err) {
    const now = Date.now();
    const safeLimit = Math.max(1, limit);
    const fallback: Signal[] = Array.from({ length: safeLimit }).map((_, i) => {
      const timestamp = new Date(now - i * 5 * 60_000).toISOString();
      const direction = i % 2 === 0 ? 'LONG' : 'SHORT';
      return {
        id: `demo-${symbol}-${i}`,
        symbol,
        direction,
        score: 0.5 + (i % 5) * 0.05,
        confidence: 0.5,
        timestamp,
        source: 'demo',
        details: { source: 'demo', note: `fallback signal ${i}` },
      };
    });
    return { items: fallback, source: 'demo' };
  }
}
