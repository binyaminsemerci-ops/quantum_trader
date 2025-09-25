// NOTE: we intentionally avoid importing OHLCV here to keep this helper minimal

const API_BASE = 'http://localhost:8000';

export type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export async function fetchRecentPrices(symbol = 'BTCUSDC', limit = 50): Promise<Candle[]> {
  try {
    const q = new URLSearchParams({ symbol, limit: String(limit) });
    const res = await fetch(`${API_BASE}/prices/recent?${q.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = (await res.json()) as unknown;
    const data = Array.isArray(payload) ? payload : (payload && (payload as any).data) || [];
    return data.map((d: any) => ({
      time: d.time || d.timestamp || new Date().toISOString(),
      open: Number(d.open),
      high: Number(d.high),
      low: Number(d.low),
      close: Number(d.close),
      volume: Number(d.volume) || 0,
    }));
  } catch (err) {
    const now = Date.now();
    const candles: Candle[] = Array.from({ length: limit }).map((_, i) => ({
      time: new Date(now - (limit - i) * 60000).toISOString(),
      open: 100 + i,
      high: 101 + i,
      low: 99 + i,
      close: 100 + i,
      volume: 10 + i,
    }));
    return candles;
  }
}
