import { safeJson } from '../utils/api';

const X_API_KEY = import.meta.env.VITE_X_API_KEY as string | undefined;

type TwitterSentiment = { symbol: string; score?: number; details?: Record<string, unknown> } | { error: string };

export async function fetchTwitterSentiment(symbol?: string): Promise<TwitterSentiment | null> {
  try {
    const url = symbol ? `/api/twitter/sentiment?symbol=${encodeURIComponent(symbol)}` : '/api/twitter/sentiment';
    const response = await fetch(url, { headers: X_API_KEY ? { 'X-API-Key': X_API_KEY } : {} });
    if (!response.ok) throw new Error(`Twitter API error ${response.status}: ${await response.text()}`);
    const data = await safeJson(response);
    return (data && typeof data === 'object') ? (data as TwitterSentiment) : null;
  } catch (error: unknown) {
    console.error('Error fetching Twitter sentiment:', error);
    let message = String(error);
    if (typeof error === 'object' && error !== null && 'message' in error) {
      const maybe = (error as { message?: unknown }).message;
      if (typeof maybe === 'string') message = maybe;
    }
    return { error: message };
  }
}

export default { fetchTwitterSentiment };
