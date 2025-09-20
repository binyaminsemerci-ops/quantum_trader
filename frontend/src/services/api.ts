const API_BASE = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000';

type PricePoint = { timestamp: string; open: number; high: number; low: number; close: number; volume?: number };

type Signal = {
  id?: string | number;
  symbol?: string;
  side?: 'buy' | 'sell' | string;
  score?: number;
  created_at?: string;
};

type Sentiment = {
  symbol?: string;
  positive?: number;
  neutral?: number;
  negative?: number;
  updated_at?: string;
};

// Minimal typed adapters for services used by components. Keep names stable to avoid
// changing callers during the conservative migration.
export async function fetchPriceData(symbol: string, interval: string): Promise<PricePoint[]> {
  const res = await fetch(`${API_BASE}/prices/${encodeURIComponent(symbol)}?interval=${encodeURIComponent(interval)}`);
  if (!res.ok) throw new Error(`Failed to fetch price data: ${await res.text()}`);
  const data = await res.json();
  return data ?? [];
}

export async function fetchTradingSignals(symbol?: string): Promise<Signal[]> {
  const url = symbol ? `${API_BASE}/signals?symbol=${encodeURIComponent(symbol)}` : `${API_BASE}/signals`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch trading signals: ${await res.text()}`);
  const data = await res.json();
  return Array.isArray(data) ? (data as Signal[]) : [];
}

export async function fetchSentimentData(symbol?: string): Promise<Sentiment | null> {
  const url = symbol ? `${API_BASE}/sentiment?symbol=${encodeURIComponent(symbol)}` : `${API_BASE}/sentiment`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch sentiment data: ${await res.text()}`);
  const data = await res.json();
  return data && typeof data === 'object' ? (data as Sentiment) : null;
}

export default { fetchPriceData, fetchTradingSignals, fetchSentimentData };
