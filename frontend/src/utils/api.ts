import axios from 'axios';
import type { Trade, StatSummary, OHLCV, ApiResponse as ApiResponseType, SpotBalance, FuturesBalance } from '../types';

export type Price = { symbol?: string; price?: number };

export type ApiResponse<T = unknown> = ApiResponseType<T> | { error?: string; data?: T };

const API_BASE = '/api';

// Runtime-configurable default exchange used by frontend API helpers.
// The Settings page will set this value when the app loads saved settings.
export let DEFAULT_EXCHANGE = 'binance';

export function setDefaultExchange(name: string) {
  DEFAULT_EXCHANGE = name || 'binance';
}

export function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null;
}

export async function safeJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return undefined;
  }
}

export function ensureArray<T>(x: unknown): T[] {
  return Array.isArray(x) ? (x as T[]) : [];
}

/**
 * Extract an array from a payload which may be the array itself or an object wrapper
 * e.g. payload = [{...}] or payload = { trades: [{...}] }
 */
export function extractWrapperArray<T = unknown>(payload: unknown, key: string): T[] {
  if (Array.isArray(payload)) return payload as T[];
  if (isRecord(payload) && Array.isArray(payload[key])) return payload[key] as T[];
  return [];
}

async function request<T = unknown>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (!res.ok) {
    const error = await res.text();
    return { error: `API error ${res.status}: ${error}` } as ApiResponse<T>;
  }

  const payload = await safeJson(res);
  // If caller expects an array-like result, return empty array instead of raw undefined
  return { data: payload as T } as ApiResponse<T>;
}

export const api = {
  // Spot
  getSpotBalance: (): Promise<ApiResponse<SpotBalance>> => request(`/binance/spot/balance?exchange=${DEFAULT_EXCHANGE}`),
  getSpotPrice: (symbol: string): Promise<ApiResponse<Price>> => request(`/binance/spot/price/${symbol}?exchange=${DEFAULT_EXCHANGE}`),
  placeSpotOrder: (symbol: string, side: string, quantity: number) =>
    request(`/binance/spot/order?exchange=${DEFAULT_EXCHANGE}`, { method: 'POST', body: JSON.stringify({ symbol, side, quantity }) }),

  // Futures
  getFuturesBalance: (): Promise<ApiResponse<FuturesBalance>> => request(`/binance/futures/balance?exchange=${DEFAULT_EXCHANGE}`),
  getFuturesPrice: (symbol: string): Promise<ApiResponse<Price>> => request(`/binance/futures/price/${symbol}?exchange=${DEFAULT_EXCHANGE}`),
  placeFuturesOrder: (symbol: string, side: string, quantity: number) =>
    request(`/binance/futures/order?exchange=${DEFAULT_EXCHANGE}`, { method: 'POST', body: JSON.stringify({ symbol, side, quantity }) }),
  getOpenFuturesOrders: (symbol?: string) => request(`/binance/futures/orders${symbol ? `?symbol=${encodeURIComponent(symbol)}&exchange=${DEFAULT_EXCHANGE}` : `?exchange=${DEFAULT_EXCHANGE}`}`),
  cancelFuturesOrder: (symbol: string, orderId: string | number) =>
    request(`/binance/futures/order/${encodeURIComponent(String(symbol))}/${encodeURIComponent(String(orderId))}?exchange=${DEFAULT_EXCHANGE}`, { method: 'DELETE' }),

  // Domain helpers
  getStats: (): Promise<ApiResponse<StatSummary>> => request<StatSummary>('/stats'),
  getTrades: (): Promise<ApiResponse<Trade[]>> => request<Trade[]>('/trades'),
  getChart: (): Promise<ApiResponse<OHLCV[]>> => request<OHLCV[]>('/chart'),

  getSettings: () => request('/settings'),
  saveSettings: (settings: unknown) => request('/settings', { method: 'POST', body: JSON.stringify(settings) }),

  // compatibility wrappers
  get: <T = unknown>(endpoint: string): Promise<ApiResponse<T>> => request<T>(endpoint),
  post: <T = unknown>(endpoint: string, body: unknown = null, opts: Record<string, unknown> = {}): Promise<ApiResponse<T>> => {
    let url = endpoint;
    if (opts && (opts as Record<string, unknown>).params) {
      const params = (opts as Record<string, unknown>).params;
      if (params && typeof params === 'object') {
        const flat: Record<string, string> = {};
        for (const k of Object.keys(params)) {
          const v = (params as Record<string, unknown>)[k];
          flat[k] = v == null ? '' : String(v);
        }
        const qs = new URLSearchParams(flat).toString();
        url = `${endpoint}${endpoint.includes('?') ? '&' : '?'}${qs}`;
      }
    }
    return request<T>(url, body ? { method: 'POST', body: JSON.stringify(body) } : { method: 'POST' });
  },
  delete: <T = unknown>(endpoint: string): Promise<ApiResponse<T>> => request<T>(endpoint, { method: 'DELETE' }),
};

export const trainModel = (symbol: string) => axios.post(`/api/ai/train/${symbol}`);
export const getPrediction = (symbol: string) => axios.post(`/api/ai/predict/${symbol}`);

export default api;
