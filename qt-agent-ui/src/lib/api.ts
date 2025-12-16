// API client for Quantum Trader backend
const API_BASE = "http://localhost:8000";

export interface MetricsData {
  total_trades?: number;
  win_rate?: number;
  pnl_usd?: number;
  ai_status?: string;
  autonomous_mode?: boolean;
  positions_count?: number;
  signals_count?: number;
}

export interface Position {
  symbol: string;
  side: "LONG" | "SHORT";
  size: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
}

export interface Signal {
  timestamp: string;
  symbol: string;
  action: "BUY" | "SELL" | "HOLD";
  confidence: number;
  price: number;
  reason?: string;
}

export interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: "BUY" | "SELL";
  price: number;
  quantity: number;
  pnl?: number;
  status: string;
}

export interface ModelInfo {
  status: string;
  last_trained?: string;
  accuracy?: number;
  features_count?: number;
  model_type?: string;
}

export interface OHLCVData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface AiOsModuleHealth {
  name: string;
  health: string;
  last_activity: string;
  note: string;
}

export interface AiOsStatus {
  overall_health: string;
  risk_mode: string;
  emergency_brake: boolean;
  new_trades_allowed: boolean;
  modules: AiOsModuleHealth[];
}

export const api = {
  // Metrics endpoint - use /api/metrics/system
  async getMetrics(): Promise<MetricsData> {
    try {
      const res = await fetch(`${API_BASE}/api/metrics/system`);
      if (!res.ok) {
        return {
          total_trades: 0,
          win_rate: 0,
          pnl_usd: 0,
          ai_status: "READY",
          autonomous_mode: true,
          positions_count: 0,
          signals_count: 0
        };
      }
      const data = await res.json();
      
      // Get real positions count from /positions endpoint
      let actualPositionsCount = 0;
      try {
        const posRes = await fetch(`${API_BASE}/positions`);
        if (posRes.ok) {
          const positions = await posRes.json();
          actualPositionsCount = Array.isArray(positions) ? positions.length : 0;
        }
      } catch {
        // Use backend count if positions fetch fails
        actualPositionsCount = data.positions_count || data.open_positions || 0;
      }
      
      return {
        total_trades: data.total_trades || 0,
        win_rate: data.win_rate || 0,  // Backend returns percentage already
        pnl_usd: data.total_pnl || data.pnl_usd || 0,
        ai_status: data.ai_status || "READY",
        autonomous_mode: data.autonomous_mode !== false,
        positions_count: actualPositionsCount,
        signals_count: data.signals_count || 0
      };
    } catch {
      return {
        total_trades: 0,
        win_rate: 0,
        pnl_usd: 0,
        ai_status: "READY",
        autonomous_mode: true,
        positions_count: 0,
        signals_count: 0
      };
    }
  },

  // Positions endpoint - use /positions (simpler endpoint)
  async getPositions(): Promise<Position[]> {
    try {
      // Try /positions first (simpler endpoint)
      const res = await fetch(`${API_BASE}/positions`);
      if (!res.ok) return [];
      const data = await res.json();
      
      // Data might be array of positions or object with positions array
      const positionsArray = Array.isArray(data) ? data : (data.positions || []);
      
      return positionsArray.map((p: any) => ({
        symbol: p.symbol || p.pair,
        side: ((p.side || "LONG").toUpperCase()) as "LONG" | "SHORT",
        size: Math.abs(p.size || p.quantity || p.qty || 0),
        entry_price: p.entry_price || p.avg_entry || p.avgPrice || 0,
        current_price: p.current_price || p.mark_price || p.markPrice || 0,
        pnl: p.pnl || p.unrealized_pnl || p.unrealizedPnl || 0,
        pnl_pct: p.pnl_pct || p.pnlPct || 0
      }));
    } catch {
      return [];
    }
  },

  // Signals endpoint - use /api/ai/signals/latest
  async getSignals(limit = 100): Promise<Signal[]> {
    try {
      const res = await fetch(`${API_BASE}/api/ai/signals/latest?limit=${limit}`);
      if (!res.ok) return [];
      const data = await res.json();
      return Array.isArray(data) ? data : [];
    } catch {
      return [];
    }
  },

  // Trades endpoint - use /api/trades
  async getTrades(limit = 100): Promise<Trade[]> {
    try {
      const res = await fetch(`${API_BASE}/api/trades?limit=${limit}`);
      if (!res.ok) return [];
      const data = await res.json();
      return Array.isArray(data) ? data : [];
    } catch {
      return [];
    }
  },

  // Model info endpoint
  async getModelInfo(): Promise<ModelInfo> {
    try {
      const res = await fetch(`${API_BASE}/api/ai/model/status`);
      if (!res.ok) throw new Error("Failed to fetch model info");
      return res.json();
    } catch {
      return {
        status: "UNKNOWN",
        model_type: "N/A"
      };
    }
  },

  // OHLCV data endpoint - use /candles
  async getOHLCV(symbol = "BTCUSDT", limit = 100): Promise<OHLCVData[]> {
    try {
      const res = await fetch(`${API_BASE}/candles?symbol=${symbol}&limit=${limit}`);
      if (!res.ok) return [];
      const data = await res.json();
      // Backend returns {symbol: string, candles: array}
      return data?.candles || [];
    } catch {
      return [];
    }
  },

  // Health check
  async getHealth(): Promise<any> {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("Failed to fetch health");
      return res.json();
    } catch {
      return { status: "unknown" };
    }
  },

  // AI-OS Status endpoint
  async getAiOsStatus(): Promise<AiOsStatus> {
    try {
      const res = await fetch(`${API_BASE}/api/aios_status`);
      if (!res.ok) {
        return {
          overall_health: "UNKNOWN",
          risk_mode: "NORMAL",
          emergency_brake: false,
          new_trades_allowed: false,
          modules: []
        };
      }
      return await res.json();
    } catch {
      return {
        overall_health: "UNKNOWN",
        risk_mode: "NORMAL",
        emergency_brake: false,
        new_trades_allowed: false,
        modules: []
      };
    }
  },
};
