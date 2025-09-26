export type Signal = {
  id: string;
  symbol: string;
  score: number;
  timestamp: string;
};

export async function fetchRecentSignals(): Promise<Signal[]> {
  const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
  try {
    const res = await fetch(`${base}/signals/recent`);
    if (!res.ok) return [];
    const payload = await res.json();
    return Array.isArray(payload) ? payload as Signal[] : (payload.data || []);
  } catch (err) {
    return [];
  }
}
