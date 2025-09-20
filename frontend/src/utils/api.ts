// Canonical typed API adapter for the frontend.
// Keep this module single-responsibility: expose a conservative domain-typed `api` object
// plus AI helpers. Other service wrappers should re-export from here.

import axios from 'axios';
import type { Trade, StatSummary, OHLCV, ApiResponse as ApiResponseType } from '../types';

export type ApiResponse<T = unknown> = ApiResponseType<T>;

const API_BASE = "/api"; // proxied to backend

async function request<T = unknown>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const error = await res.text();
    return { error: `API error ${res.status}: ${error}` } as ApiResponse<T>;
  }
  try {
    const payload = await res.json();
    return { data: payload } as ApiResponse<T>;
  } catch (e) {
    return { data: undefined } as ApiResponse<T>;
  }
}

export const api = {
  // Spot (loosely typed)
  getSpotBalance: (): Promise<ApiResponse<Record<string, unknown>>> => request('/binance/spot/balance'),
  getSpotPrice: (symbol: string): Promise<ApiResponse<Record<string, unknown>>> => request(`/binance/spot/price/${symbol}`),
  placeSpotOrder: (symbol: string, side: string, quantity: number) =>
    request('/binance/spot/order', {
      method: 'POST',
      body: JSON.stringify({ symbol, side, quantity }),
    }),

  // Futures (loosely typed)
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

  // Domain-typed helpers
  getStats: (): Promise<ApiResponse<StatSummary>> => request<StatSummary>('/stats'),
  getTrades: (): Promise<ApiResponse<Trade[]>> => request<Trade[]>('/trades'),
  getChart: (): Promise<ApiResponse<OHLCV[]>> => request<OHLCV[]>('/chart'),

  getSettings: () => request('/settings'),
  saveSettings: (settings: unknown) =>
    request('/settings', { method: 'POST', body: JSON.stringify(settings) }),

  // compatibility wrappers
  get: <T = unknown>(endpoint: string): Promise<ApiResponse<T>> => request<T>(endpoint),
  post: <T = unknown>(endpoint: string, body: unknown = null, opts: Record<string, any> = {}): Promise<ApiResponse<T>> => {
    let url = endpoint;
    if (opts && opts.params) {
      const qs = new URLSearchParams(opts.params).toString();
      url = `${endpoint}${endpoint.includes('?') ? '&' : '?'}${qs}`;
    }
    return request<T>(url, body ? { method: 'POST', body: JSON.stringify(body) } : { method: 'POST' });
  },
  delete: <T = unknown>(endpoint: string): Promise<ApiResponse<T>> => request<T>(endpoint, { method: 'DELETE' }),
};

// AI helpers (axios-based; typed loosely to avoid coupling to AI server types)
export const trainModel = (symbol: string) => axios.post(`/api/ai/train/${symbol}`);
export const getPrediction = (symbol: string) => axios.post(`/api/ai/predict/${symbol}`);

export default api;
