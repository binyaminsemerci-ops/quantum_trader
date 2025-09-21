import type { Trade, StatSummary, OHLCV, ApiResponse } from '../types';

const API_BASE = 'http://localhost:8000';

async function safeJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return undefined;
  }
}

export async function fetchTrades(): Promise<ApiResponse<Trade[]>> {
  const res = await fetch(`${API_BASE}/trades`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const payload = await safeJson(res);
  const data = Array.isArray(payload) ? (payload as Trade[]) : ((payload as any)?.trades ?? payload);
  return { data: data as Trade[] };
}

export async function fetchStats(): Promise<ApiResponse<StatSummary>> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const payload = await safeJson(res);
  return { data: (payload as StatSummary) };
}

export async function fetchChart(): Promise<ApiResponse<OHLCV[]>> {
  const res = await fetch(`${API_BASE}/chart`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const payload = await safeJson(res);
  const data = (payload as any)?.data ?? payload;
  return { data: data as OHLCV[] };
}

export async function fetchSettings(): Promise<ApiResponse<Record<string, unknown>>> {
  const res = await fetch(`${API_BASE}/settings`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  return { data: (await safeJson(res)) as Record<string, unknown> };
}

export async function fetchBinance(): Promise<ApiResponse<Record<string, unknown>>> {
  const res = await fetch(`${API_BASE}/binance`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  return { data: (await safeJson(res)) as Record<string, unknown> };
}
