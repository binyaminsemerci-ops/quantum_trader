/**
 * Enhanced API client for Quantum Trader frontend
 * Provides type-safe methods for all backend endpoints with proper error handling
 */
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

interface CreateTradeRequest {
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  price: number;
}

interface CreateTradeResponse {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  status: string;
  timestamp: string;
  message: string;
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

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null;
}

async function apiRequest<T>(
  endpoint: string, 
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const url = `${API_BASE}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await safeJson(response);
      const errorMessage = isRecord(errorData) && typeof errorData.message === 'string' 
        ? errorData.message 
        : `HTTP ${response.status}: ${response.statusText}`;
      
      console.error(`API Error [${endpoint}]:`, errorMessage, errorData);
      return { error: errorMessage };
    }

    const data = await safeJson(response);
    return { data: data as T };
    
  } catch (error) {
    console.error(`Network error [${endpoint}]:`, error);
    return { 
      error: error instanceof Error ? error.message : 'Network error occurred' 
    };
  }
}

// Trades API
export async function fetchTrades(): Promise<ApiResponse<Trade[]>> {
  return apiRequest<Trade[]>('/trades');
}

export async function createTrade(trade: CreateTradeRequest): Promise<ApiResponse<CreateTradeResponse>> {
  return apiRequest<CreateTradeResponse>('/trades', {
    method: 'POST',
    body: JSON.stringify(trade),
  });
}

// Stats API  
export async function fetchStats(): Promise<ApiResponse<StatSummary>> {
  return apiRequest<StatSummary>('/stats/overview');
}

// Prices API
export async function fetchPrices(): Promise<ApiResponse<Record<string, number>>> {
  return apiRequest<Record<string, number>>('/prices');
}

// Signals API
export async function fetchSignals(): Promise<ApiResponse<any[]>> {
  return apiRequest<any[]>('/signals');
}

// Candles API
export async function fetchCandles(
  symbol: string = 'BTCUSDT', 
  interval: string = '1h'
): Promise<ApiResponse<OHLCV[]>> {
  return apiRequest<OHLCV[]>(`/candles?symbol=${symbol}&interval=${interval}`);
}

// Settings API
export async function fetchSettings(): Promise<ApiResponse<Record<string, any>>> {
  return apiRequest<Record<string, any>>('/settings');
}

export async function updateSettings(settings: Record<string, any>): Promise<ApiResponse<Record<string, any>>> {
  return apiRequest<Record<string, any>>('/settings', {
    method: 'POST', 
    body: JSON.stringify(settings),
  });
}

// Health check
export async function healthCheck(): Promise<ApiResponse<{ message: string }>> {
  return apiRequest<{ message: string }>('/');
}

// Export enhanced client as default
export default {
  trades: {
    fetch: fetchTrades,
    create: createTrade,
  },
  stats: {
    fetch: fetchStats,
  },
  prices: {
    fetch: fetchPrices,
  },
  signals: {
    fetch: fetchSignals,
  },
  candles: {
    fetch: fetchCandles,
  },
  settings: {
    fetch: fetchSettings,
    update: updateSettings,
  },
  health: healthCheck,
};