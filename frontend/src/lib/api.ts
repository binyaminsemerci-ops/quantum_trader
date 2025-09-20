// Ny fil: axios-instans + eksportert base-URL
import axios from 'axios';

export const API_BASE = (import.meta.env.VITE_API_URL as string) ?? '/api';

// små helper: om API_BASE er relativ (/api) så la axios bruke relative path (vite proxy)
const axiosBase = API_BASE.startsWith('/') ? API_BASE : API_BASE;

export const api = axios.create({
  baseURL: axiosBase,
  timeout: 10000,
});

// Health check
export const checkHealth = async (): Promise<{ ok: true; data: unknown } | { ok: false; error: unknown }> => {
  try {
    const res = await api.get('/health');
    return { ok: true, data: res.data };
  } catch (err: unknown) {
    // avoid assuming axios error shape; return unknown error
    console.error('Health check failed:', (err as any)?.response?.status ?? 'no-status', (err as any)?.message ?? err);
    return { ok: false, error: err };
  }
};

// Trades
export const fetchTrades = async (): Promise<unknown[]> => {
  try {
    const res = await api.get('/trades');
    return Array.isArray(res.data) ? res.data : [];
  } catch (err: unknown) {
    console.error('fetchTrades error:', (err as any)?.response?.status ?? 'no-status', (err as any)?.message ?? err);
    throw err;
  }
};

export type CreateTradeInput = {
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  reason: string;
};

export const createTrade = async (trade: CreateTradeInput): Promise<unknown> => {
  try {
    const res = await api.post('/trades', trade);
    return res.data;
  } catch (err: unknown) {
    console.error('createTrade error:', (err as any)?.response?.status ?? 'no-status', (err as any)?.message ?? err);
    throw err;
  }
};

// Stats
export const fetchStats = async (): Promise<unknown> => {
  try {
    const res = await api.get('/stats');
    return res.data;
  } catch (err: unknown) {
    console.error('fetchStats error:', (err as any)?.response?.status ?? 'no-status', (err as any)?.message ?? err);
    throw err;
  }
};

// Settings
export const fetchSettings = async (): Promise<unknown> => {
  try {
    const res = await api.get('/settings');
    return res.data;
  } catch (err: unknown) {
    console.error('fetchSettings error:', (err as any)?.response?.status ?? 'no-status', (err as any)?.message ?? err);
    throw err;
  }
};

export type UpdateSettings = { api_key: string; api_secret: string };
export const updateSettings = async (settings: UpdateSettings): Promise<unknown> => {
  try {
    const res = await api.post('/settings', settings);
    return res.data;
  } catch (err: unknown) {
    console.error('updateSettings error:', (err as any)?.response?.status ?? 'no-status', (err as any)?.message ?? err);
    throw err;
  }
};
