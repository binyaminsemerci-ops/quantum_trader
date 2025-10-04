import type { Trade, StatSummary, OHLCV, ApiResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Enhanced error handling
class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: Response
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Toast notification support
type ToastNotification = {
  showSuccess: (message: string) => void;
  showError: (message: string) => void;
};

let toastNotifier: ToastNotification | null = null;

export function setToastNotifier(notifier: ToastNotification): void {
  toastNotifier = notifier;
}

async function safeJson(res: Response): Promise<unknown> {
  try {
    const text = await res.text();
    if (!text.trim()) return null;
    return JSON.parse(text);
  } catch (error) {
    console.warn('Failed to parse JSON response:', error);
    return null;
  }
}

async function handleResponse<T>(res: Response, defaultValue: T): Promise<ApiResponse<T>> {
  if (!res.ok) {
    const errorData = await safeJson(res);
    const errorMessage = isRecord(errorData) && typeof errorData.message === 'string'
      ? errorData.message
      : `HTTP ${res.status}: ${res.statusText}`;

    console.error('API Error:', errorMessage, errorData);
    return { error: errorMessage };
  }

  const payload = await safeJson(res);
  return { data: payload as T };
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null;
}

export async function fetchTrades(): Promise<ApiResponse<Trade[]>> {
  try {
    const res = await fetch(`${API_BASE}/trades`);

    if (!res.ok) {
      const errorData = await safeJson(res);
      const errorMessage = isRecord(errorData) && typeof errorData.message === 'string'
        ? errorData.message
        : `HTTP ${res.status}: ${res.statusText}`;
      return { error: errorMessage };
    }

    const payload = await safeJson(res);
    let data: unknown = payload;

    if (Array.isArray(payload)) {
      data = payload;
    } else if (isRecord(payload) && Array.isArray((payload as Record<string, unknown>)['trades'])) {
      data = (payload as Record<string, unknown>)['trades'];
    }

    return { data: (Array.isArray(data) ? (data as Trade[]) : []) };
  } catch (error) {
    console.error('Error fetching trades:', error);
    return { error: error instanceof Error ? error.message : 'Network error' };
  }
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
