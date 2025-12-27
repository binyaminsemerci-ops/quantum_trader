// TypeScript types for TP Dashboard API responses

export interface TPDashboardKey {
  strategy_id: string;
  symbol: string;
}

export interface TPDashboardProfileLeg {
  label: string;
  r_multiple: number;
  size_fraction: number;
  kind: string;
}

export interface TPDashboardProfile {
  profile_id: string;
  legs: TPDashboardProfileLeg[];
  trailing_profile_id?: string | null;
}

export interface TPDashboardMetrics {
  tp_hit_rate: number;
  tp_attempts: number;
  tp_hits: number;
  tp_misses: number;
  avg_slippage_pct?: number;
  max_slippage_pct?: number;
  avg_time_to_tp_minutes?: number;
  total_tp_profit_usd?: number;
  avg_tp_profit_usd?: number;
  premature_exits?: number;
  missed_opportunities_usd?: number;
}

export interface TPDashboardRecommendation {
  has_recommendation: boolean;
  profile_id?: string | null;
  suggested_scale_factor?: number | null;
  reason?: string | null;
}

export interface TPDashboardEntry {
  key: TPDashboardKey;
  profile: TPDashboardProfile;
  metrics: TPDashboardMetrics;
  recommendation: TPDashboardRecommendation;
}

export interface TPDashboardSummary {
  best: TPDashboardEntry[];
  worst: TPDashboardEntry[];
}

export interface FilterState {
  strategyId: string | null;
  symbol: string | null;
  search: string;
  onlyWithRecommendation: boolean;
}
