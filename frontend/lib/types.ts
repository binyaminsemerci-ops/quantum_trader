// TypeScript types for Dashboard API
// Matches backend/api/dashboard/models.py

export type MarketRegime = 
  | 'HIGH_VOL_TRENDING'
  | 'LOW_VOL_TRENDING'
  | 'HIGH_VOL_RANGING'
  | 'LOW_VOL_RANGING'
  | 'CHOPPY'
  | 'UNKNOWN';

export type ESSState = 'ARMED' | 'TRIPPED' | 'COOLING' | 'UNKNOWN';
export type ServiceStatus = 'OK' | 'DEGRADED' | 'DOWN' | 'UNKNOWN';
export type SignalDirection = 'BUY' | 'SELL' | 'HOLD';
export type PositionSide = 'LONG' | 'SHORT';
export type EventType = 
  | 'position_updated'
  | 'pnl_updated'
  | 'signal_generated'
  | 'ess_state_changed'
  | 'health_alert'
  | 'trade_executed'
  | 'order_placed'
  | 'strategy_updated' // Sprint 4 Del 2
  | 'rl_sizing_updated' // Sprint 4 Del 2
  | 'regime_changed' // Sprint 4 Del 2
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'snapshot'
  | 'update'
  | 'heartbeat'
  | 'pong';

export interface DashboardPosition {
  symbol: string;
  side: PositionSide;
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  value: number;
}

export interface DashboardSignal {
  timestamp: string;
  symbol: string;
  direction: SignalDirection;
  confidence: number;
  strategy: string;
  target_size?: number;
}

// Sprint 4 Del 2: Strategy & RL Inspector
export interface DashboardRLSizing {
  symbol: string;
  proposed_risk_pct: number;
  capped_risk_pct: number;
  proposed_leverage: number;
  capped_leverage: number;
  volatility_bucket: string; // "LOW", "MEDIUM", "HIGH", "EXTREME"
}

export interface DashboardStrategy {
  active_strategy: string;
  regime: MarketRegime;
  ensemble_scores: Record<string, number>; // {"xgb": 0.73, "lgbm": 0.69, ...}
  rl_sizing?: DashboardRLSizing;
}

export interface DashboardPortfolio {
  equity: number;
  cash: number;
  margin_used: number;
  margin_available: number;
  total_pnl: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  weekly_pnl: number;
  monthly_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  position_count: number;
}

export interface DashboardRisk {
  ess_state: ESSState;
  ess_reason?: string;
  ess_tripped_at?: string;
  daily_pnl_pct: number; // Sprint 4 Del 2: current daily PnL%
  daily_drawdown_pct: number;
  weekly_drawdown_pct: number;
  max_drawdown_pct: number;
  max_allowed_dd_pct: number; // Sprint 4 Del 2: policy limit
  exposure_total: number;
  exposure_long: number;
  exposure_short: number;
  exposure_net: number;
  open_risk_pct: number; // Sprint 4 Del 2: total risk from open positions
  max_risk_per_trade_pct: number; // Sprint 4 Del 2: policy limit
  risk_limit_used_pct: number;
}

export interface ServiceHealthInfo {
  name: string;
  status: ServiceStatus;
  latency_ms?: number;
  last_check?: string;
}

export interface DashboardSystemHealth {
  overall_status: ServiceStatus;
  alerts_count: number;
  last_alert?: string;
  services: ServiceHealthInfo[];
}

export interface DashboardSnapshot {
  schema_version: number; // Sprint 4 Del 3: API versioning
  timestamp: string;
  partial_data: boolean; // Sprint 4 Del 3: Indicates if some services failed
  errors: string[]; // Sprint 4 Del 3: List of service errors
  portfolio: DashboardPortfolio;
  positions: DashboardPosition[];
  signals: DashboardSignal[];
  risk: DashboardRisk;
  system: DashboardSystemHealth;
  strategy?: DashboardStrategy; // Sprint 4 Del 2
}

export interface DashboardEvent {
  type: EventType;
  timestamp: string;
  payload: any;
}
