// API client for Dashboard
import type { DashboardSnapshot } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class DashboardAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Fetch complete dashboard snapshot
   * 
   * Calls multiple endpoints in parallel:
   * - /api/dashboard/overview (portfolio, PnL, ESS)
   * - /api/dashboard/trading (positions, orders, signals)
   * - /api/dashboard/system (microservices health)
   */
  async fetchSnapshot(): Promise<DashboardSnapshot> {
    try {
      // Fetch all endpoints in parallel
      const [overviewRes, tradingRes, systemRes] = await Promise.all([
        fetch(`${this.baseUrl}/api/dashboard/overview`),
        fetch(`${this.baseUrl}/api/dashboard/trading`),
        fetch(`${this.baseUrl}/api/dashboard/system`),
      ]);

      if (!overviewRes.ok) {
        throw new Error(`Overview endpoint: HTTP ${overviewRes.status}`);
      }

      const overview = await overviewRes.json();
      const trading = tradingRes.ok ? await tradingRes.json() : { open_positions: [], recent_orders: [], recent_signals: [] };
      const system = systemRes.ok ? await systemRes.json() : { services_health: [], exchanges_health: [] };

      // Map positions from trading data
      const positions = (trading.open_positions || []).map((pos: any) => ({
        symbol: pos.symbol || '',
        side: pos.side || 'LONG',
        size: pos.size || 0,
        entry_price: pos.entry_price || 0,
        current_price: pos.current_price || 0,
        unrealized_pnl: pos.unrealized_pnl || 0,
        unrealized_pnl_pct: pos.unrealized_pnl_pct || 0,
        value: pos.value || 0,
      }));

      // Map signals from trading data
      const signals = (trading.recent_signals || []).map((sig: any) => ({
        timestamp: sig.timestamp || new Date().toISOString(),
        symbol: sig.symbol || '',
        direction: sig.direction || 'HOLD',
        confidence: sig.confidence || 0,
        strategy: sig.strategy || '',
        target_size: sig.target_size,
      }));

      // Map system services
      const services = [
        ...(system.services_health || []).map((svc: any) => ({
          name: svc.name || '',
          status: svc.status === 'healthy' ? 'OK' : svc.status === 'connected' ? 'OK' : 'UNKNOWN',
          latency_ms: undefined,
          last_check: undefined,
        })),
        ...(system.exchanges_health || []).map((ex: any) => ({
          name: ex.exchange || '',
          status: ex.status === 'connected' ? 'OK' : 'DOWN',
          latency_ms: ex.latency_ms,
          last_check: undefined,
        })),
      ];

      const snapshot: DashboardSnapshot = {
        schema_version: 3,
        timestamp: overview.timestamp || new Date().toISOString(),
        partial_data: !tradingRes.ok || !systemRes.ok,
        errors: [
          ...(!tradingRes.ok ? ['trading endpoint unavailable'] : []),
          ...(!systemRes.ok ? ['system endpoint unavailable'] : []),
        ],
        portfolio: {
          equity: overview.global_pnl?.equity || 0,
          cash: 0,
          margin_used: 0,
          margin_available: 0,
          total_pnl: overview.global_pnl?.total_pnl || 0,
          daily_pnl: overview.global_pnl?.daily_pnl || 0,
          daily_pnl_pct: overview.global_pnl?.daily_pnl_pct || 0,
          weekly_pnl: overview.global_pnl?.weekly_pnl || 0,
          monthly_pnl: overview.global_pnl?.monthly_pnl || 0,
          realized_pnl: 0,
          unrealized_pnl: 0,
          position_count: positions.length,
        },
        positions,
        signals,
        risk: {
          ess_state: (overview.ess_status?.status || 'UNKNOWN') as any,
          ess_reason: overview.ess_status?.reason || undefined,
          ess_tripped_at: undefined,
          daily_pnl_pct: overview.global_pnl?.daily_pnl_pct || 0,
          daily_drawdown_pct: 0,
          weekly_drawdown_pct: 0,
          max_drawdown_pct: 0,
          max_allowed_dd_pct: 5.0,
          exposure_total: overview.exposure_per_exchange?.reduce(
            (sum: number, ex: any) => sum + Math.abs(ex.net_exposure || 0),
            0
          ) || 0,
          exposure_long: 0,
          exposure_short: 0,
          exposure_net: overview.exposure_per_exchange?.reduce(
            (sum: number, ex: any) => sum + (ex.net_exposure || 0),
            0
          ) || 0,
          open_risk_pct: 0,
          max_risk_per_trade_pct: 2.0,
          risk_limit_used_pct: 0,
        },
        system: {
          overall_status: services.some(s => s.status === 'DOWN') ? 'DOWN' : 'OK',
          alerts_count: 0,
          last_alert: undefined,
          services,
        },
      };
      
      return snapshot;
    } catch (error) {
      console.error('[DashboardAPI] Fetch snapshot failed:', error);
      throw error;
    }
  }

  /**
   * Check dashboard API health
   */
  async checkHealth(): Promise<{ status: string; service: string; timestamp: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/dashboard/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('[DashboardAPI] Health check failed:', error);
      throw error;
    }
  }
}

// Singleton instance
export const dashboardAPI = new DashboardAPI();

// Helper functions for easy import
export const fetchDashboardSnapshot = () => dashboardAPI.fetchSnapshot();
export const checkDashboardHealth = () => dashboardAPI.checkHealth();
