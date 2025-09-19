// Lightweight typings to help the migration
import axios from 'axios';

export type ApiResponse<T = unknown> = { data?: T; error?: string | null };

export const trainModel = (symbol: string) => {
  return axios.post(`/api/ai/train/${symbol}`);
};

export const getPrediction = (symbol: string) => {
  return axios.post(`/api/ai/predict/${symbol}`);
};
const API_BASE = "/api"; // proxes til FastAPI

async function request<T = unknown>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const error = await res.text();
    return { error: `API error ${res.status}: ${error}` };
  }
  try {
    const payload = await res.json();
    return { data: payload };
  } catch (e) {
    return { data: undefined };
  }
}

export const api = {
  // Spot
  getSpotBalance: () => request('/binance/spot/balance'),
  getSpotPrice: (symbol: string) => request(`/binance/spot/price/${symbol}`),
  placeSpotOrder: (symbol: string, side: string, quantity: number) =>
    request('/binance/spot/order', {
      method: 'POST',
      body: JSON.stringify({ symbol, side, quantity }),
    }),

  // Futures
  getFuturesBalance: () => request('/binance/futures/balance'),
  getFuturesPrice: (symbol: string) => request(`/binance/futures/price/${symbol}`),
  placeFuturesOrder: (symbol: string, side: string, quantity: number) =>
    request('/binance/futures/order', {
      method: 'POST',
      body: JSON.stringify({ symbol, side, quantity }),
    }),
  getOpenFuturesOrders: (symbol?: string) =>
    request(`/binance/futures/orders${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ''}`),
  cancelFuturesOrder: (symbol: string, orderId: string | number) =>
    request(`/binance/futures/order/${encodeURIComponent(String(symbol))}/${encodeURIComponent(String(orderId))}`, { method: 'DELETE' }),

  // Andre API-er
  getStats: () => request('/stats'),
  getTrades: () => request('/trades'),
  getChart: () => request('/chart'),
  getSettings: () => request('/settings'),
  saveSettings: (settings: unknown) =>
    request('/settings', { method: 'POST', body: JSON.stringify(settings) }),
  // compatibility wrappers expected by some components
  get: <T = unknown>(endpoint: string): Promise<ApiResponse<T>> => request<T>(endpoint),
  post: <T = unknown>(endpoint: string, body: unknown = null, opts: Record<string, any> = {}): Promise<ApiResponse<T>> => {
    // support axios-like params via opts.params
    let url = endpoint;
    if (opts && opts.params) {
      const qs = new URLSearchParams(opts.params).toString();
      url = `${endpoint}${endpoint.includes('?') ? '&' : '?'}${qs}`;
    }
    return request<T>(url, body ? { method: 'POST', body: JSON.stringify(body) } : { method: 'POST' });
  },
  delete: <T = unknown>(endpoint: string): Promise<ApiResponse<T>> => request<T>(endpoint, { method: 'DELETE' }),
};

export default api;
