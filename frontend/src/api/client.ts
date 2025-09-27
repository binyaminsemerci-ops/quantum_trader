import type { Trade, StatSummary, OHLCV, ApiResponse } from '../types';

const API_BASE = 'http://localhost:8000';

async function safeJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return undefined;
  }
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null;
}

export async function fetchTrades(): Promise<ApiResponse<Trade[]>> {
  const res = await fetch(`${API_BASE}/trades`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const payload = await safeJson(res);
  let data: unknown = payload;
  if (Array.isArray(payload)) data = payload;
  else if (isRecord(payload) && Array.isArray((payload as Record<string, unknown>)['trades'])) {
    data = (payload as Record<string, unknown>)['trades'];
  }
  return { data: (Array.isArray(data) ? (data as Trade[]) : []) };
}

export async function fetchStats(): Promise<ApiResponse<StatSummary>> {
  const res = await fetch(`${API_BASE}/stats/overview`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const payload = await safeJson(res);
  return { data: (payload as StatSummary) };
}

export async function fetchChart(): Promise<ApiResponse<OHLCV[]>> {
  const res = await fetch(`${API_BASE}/chart`);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const payload = await safeJson(res);
  let data: unknown = payload;
  if (isRecord(payload) && Array.isArray((payload as Record<string, unknown>)['data'])) {
    data = (payload as Record<string, unknown>)['data'];
  }
  return { data: (Array.isArray(data) ? (data as OHLCV[]) : []) };
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
