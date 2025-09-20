const X_API_KEY = import.meta.env.VITE_X_API_KEY as string | undefined;

type TwitterSentiment = { symbol: string; score?: number; details?: any } | { error: string };

export async function fetchTwitterSentiment(symbol: string): Promise<TwitterSentiment> {
  try {
    const response = await fetch(`http://localhost:8000/twitter/sentiment/${encodeURIComponent(symbol)}`, {
      headers: X_API_KEY ? { 'X-API-Key': X_API_KEY } : {}
    });
    if (!response.ok) throw new Error(`Twitter API error ${response.status}: ${await response.text()}`);
    const data = await response.json();
    return data;
  } catch (error: unknown) {
    console.error('Error fetching Twitter sentiment:', error);
    const message = (error as any)?.message ?? String(error);
    return { error: message };
  }
}

export default { fetchTwitterSentiment };
