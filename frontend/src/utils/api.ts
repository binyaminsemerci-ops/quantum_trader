// Lightweight typings to help the migration
import axios from 'axios';
import type { Trade, StatSummary, OHLCV, ApiResponse as ApiResponseType } from '../types';

export type ApiResponse<T = unknown> = ApiResponseType<T>;

// axios-based AI endpoints (keep original runtime behavior; typed loosely)
export const trainModel = (symbol: string): Promise<any> => {
  return axios.post(`/api/ai/train/${symbol}`) as Promise<any>;
};

export const getPrediction = (symbol: string): Promise<any> => {
  return axios.post(`/api/ai/predict/${symbol}`) as Promise<any>;
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
  // these are loosely-typed because balance payloads can vary; consumers should narrow
  getSpotBalance: (): Promise<ApiResponse<Record<string, unknown>>> => request('/binance/spot/balance'),
  getSpotPrice: (symbol: string): Promise<ApiResponse<Record<string, unknown>>> => request(`/binance/spot/price/${symbol}`),
  placeSpotOrder: (symbol: string, side: string, quantity: number) =>
    request('/binance/spot/order', {
      method: 'POST',
      body: JSON.stringify({ symbol, side, quantity }),
    }),

  // Futures
  getFuturesBalance: (): Promise<ApiResponse<Record<string, unknown>>> => request('/binance/futures/balance'),
  getFuturesPrice: (symbol: string): Promise<ApiResponse<Record<string, unknown>>> => request(`/binance/futures/price/${symbol}`),
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
  // Domain-typed helpers (conservative): map to existing types in frontend/src/types
  getStats: (): Promise<ApiResponse<StatSummary>> => request<StatSummary>('/stats'),
  getTrades: (): Promise<ApiResponse<Trade[]>> => request<Trade[]>('/trades'),
  getChart: (): Promise<ApiResponse<OHLCV[]>> => request<OHLCV[]>('/chart'),
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
