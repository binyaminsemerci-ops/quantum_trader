// Minimal TypeScript migration of the trading helper
export type TradePayload = {
  symbol: string;
  side: string;
  quantity: number;
};

export async function submitTrade(payload: TradePayload = { symbol: 'BTC/USDT', side: 'BUY', quantity: 0.01 }) {
  try {
    const response = await fetch('/api/trade', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      return { ok: false, error: data?.detail?.message || 'Failed to submit trade', data };
    }
    return { ok: true, data };
  } catch (error: any) {
    return { ok: false, error: error?.message || String(error) };
  }
}

export const exampleTestCases: Array<{ name: string; data: TradePayload }> = [
  { name: 'Valid Trade', data: { symbol: 'BTC/USDT', side: 'BUY', quantity: 0.01 } },
  { name: 'Invalid Side', data: { symbol: 'BTC/USDT', side: 'INVALID', quantity: 0.01 } },
  { name: 'Small Quantity', data: { symbol: 'BTC/USDT', side: 'BUY', quantity: 0.0001 } },
  { name: 'ETH Balance Error', data: { symbol: 'ETH/USDT', side: 'BUY', quantity: 0.1 } },
  { name: 'Connection Error', data: { symbol: 'XRP/USDT', side: 'SELL', quantity: 10 } }
];
