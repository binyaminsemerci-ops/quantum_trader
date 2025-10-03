export type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type PriceResult = {
  candles: Candle[];
  source: 'live' | 'demo';
};

function toNumber(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function fallbackCandles(limit: number): Candle[] {
  const now = Date.now();
  const safeLimit = Math.max(1, limit);
  return Array.from({ length: safeLimit }).map((_, i) => {
    const base = 100 + i * 0.25;
    return {
      time: new Date(now - (safeLimit - i) * 60_000).toISOString(),
      open: base,
      high: base + 1,
      low: base - 1,
      close: base + Math.sin(i / 4) * 1.5,
      volume: 10 + i,
    };
  });
}

export async function fetchRecentPrices(symbol = 'BTCUSDT', limit = 50): Promise<PriceResult> {
  const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
  try {
    const safeLimit = Math.max(1, limit);
    const q = new URLSearchParams({ symbol, limit: String(safeLimit) });
    const res = await fetch(`${base}/prices/recent?${q.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    const raw = Array.isArray(payload)
      ? payload
      : payload && Array.isArray((payload as any).data)
      ? (payload as any).data
      : [];
    const candles = raw.slice(0, safeLimit).map((d: any) => ({
      time: String(d.time ?? d.timestamp ?? new Date().toISOString()),
      open: toNumber(d.open),
      high: toNumber(d.high),
      low: toNumber(d.low),
      close: toNumber(d.close),
      volume: toNumber(d.volume ?? 0),
    }));
    return { candles, source: 'live' };
  } catch (err) {
    const safeLimit = Math.max(1, limit);
    return { candles: fallbackCandles(safeLimit), source: 'demo' };
  }
}
