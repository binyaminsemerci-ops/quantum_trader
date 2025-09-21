import type { Trade, StatSummary, OHLCV } from '../types';

const API_BASE = 'http://localhost:8000';

export async function fetchTrades(): Promise<Trade[]> {
  const res = await fetch(`${API_BASE}/trades`);
  return (await res.json()) as Trade[];
}

export async function fetchStats(): Promise<StatSummary> {
  const res = await fetch(`${API_BASE}/stats`);
  return (await res.json()) as StatSummary;
}

export async function fetchChart(): Promise<OHLCV[]> {
  const res = await fetch(`${API_BASE}/chart`);
  return (await res.json()) as OHLCV[];
}

export async function fetchSettings(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/settings`);
  return (await res.json()) as Record<string, unknown>;
}

export async function fetchBinance(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/binance`);
  return (await res.json()) as Record<string, unknown>;
}
