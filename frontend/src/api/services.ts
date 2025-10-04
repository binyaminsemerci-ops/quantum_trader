import type { ApiResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export type Settings = {
  api_key?: string;
  api_secret?: string;
  risk_percentage?: number;
  max_position_size?: number;
  trading_enabled?: boolean;
  notifications_enabled?: boolean;
  [key: string]: any; // For backward compatibility
};

export type SettingsUpdateResponse = {
  status: string;
  message: string;
  updated_fields: string[];
};

async function handleApiResponse<T>(response: Response): Promise<ApiResponse<T>> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      } else if (errorData.message) {
        errorMessage = errorData.message;
      }
    } catch (e) {
      // If we can't parse the error response, use the default message
    }

    return { error: errorMessage };
  }

  try {
    const data = await response.json();
    return { data };
  } catch (error) {
    return { error: 'Failed to parse response' };
  }
}

export async function fetchSettings(): Promise<ApiResponse<Settings>> {
  try {
    const response = await fetch(`${API_BASE}/settings`);
    return handleApiResponse<Settings>(response);
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'Network error occurred'
    };
  }
}

export async function updateSettings(settings: Partial<Settings>): Promise<ApiResponse<SettingsUpdateResponse>> {
  try {
    const response = await fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    });

    return handleApiResponse<SettingsUpdateResponse>(response);
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'Network error occurred'
    };
  }
}

export async function createTrade(trade: {
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  price: number;
}): Promise<ApiResponse<any>> {
  try {
    const response = await fetch(`${API_BASE}/trades`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(trade),
    });

    return handleApiResponse<any>(response);
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'Network error occurred'
    };
  }
}
