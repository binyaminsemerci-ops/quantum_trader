const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public response?: any
  ) {
    super(`API Error: ${status} ${statusText}`);
    this.name = 'ApiError';
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${url}`);
    
    if (!response.ok) {
      let errorResponse;
      try {
        errorResponse = await response.json();
      } catch {
        errorResponse = await response.text();
      }
      throw new ApiError(response.status, response.statusText, errorResponse);
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new Error(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

interface GlobalSummary {
  period: {
    start_date: string;
    end_date: string;
    days: number;
  };
  balance: {
    initial: number;
    current: number;
    pnl_total: number;
    pnl_pct: number;
  };
  trades: {
    total: number;
    winning: number;
    losing: number;
    win_rate: number;
  };
  risk: {
    max_drawdown: number;
    sharpe_ratio: number;
    profit_factor: number;
    avg_r_multiple: number;
  };
  best_worst: {
    best_trade_pnl: number;
    worst_trade_pnl: number;
    best_day_pnl: number;
    worst_day_pnl: number;
  };
  streaks: {
    longest_win_streak: number;
    longest_loss_streak: number;
    current_streak: number;
    current_streak_type: string;
  };
  costs: {
    total_commission: number;
    total_slippage: number;
  };
}

interface EquityPoint {
  timestamp: string;
  equity: number;
  balance: number;
}

interface StrategyStats {
  strategy_id: string;
  total_trades: number;
  total_pnl: number;
  win_rate: number;
}

interface SymbolStats {
  symbol: string;
  total_trades: number;
  total_pnl: number;
  win_rate: number;
}

export async function fetchAnalytics() {
  try {
    const [summary, equityCurve, topStrategies, topSymbols] = await Promise.all([
      fetchJson<GlobalSummary>('/api/pal/global/summary').catch(() => ({
        period: { start_date: '', end_date: '', days: 0 },
        balance: { initial: 10000, current: 10000, pnl_total: 0, pnl_pct: 0 },
        trades: { total: 0, winning: 0, losing: 0, win_rate: 0 },
        risk: { max_drawdown: 0, sharpe_ratio: 0, profit_factor: 0, avg_r_multiple: 0 },
        best_worst: { best_trade_pnl: 0, worst_trade_pnl: 0, best_day_pnl: 0, worst_day_pnl: 0 },
        streaks: { longest_win_streak: 0, longest_loss_streak: 0, current_streak: 0, current_streak_type: 'none' },
        costs: { total_commission: 0, total_slippage: 0 }
      })),
      fetchJson<EquityPoint[]>('/api/pal/global/equity-curve?days=30').catch(() => []),
      fetchJson<StrategyStats[]>('/api/pal/strategies/top?limit=5').catch(() => []),
      fetchJson<SymbolStats[]>('/api/pal/symbols/top?limit=5').catch(() => [])
    ]);

    return {
      summary,
      equityCurve,
      topStrategies: topStrategies || [],
      topSymbols: topSymbols || []
    };
  } catch (error) {
    console.error('Analytics fetch error:', error);
    throw error;
  }
}
