/**
 * Overview Tab Component
 * DASHBOARD-V3-001: Full Visual UI
 * 
 * Displays:
 * - GO-LIVE status
 * - Global Risk State
 * - ESS Status
 * - Global PnL (today)
 * - Exposure per exchange
 */

import { useEffect, useState } from 'react';
import DashboardCard from '../DashboardCard';
import { useDashboardStream } from '@/hooks/useDashboardStream';

interface OverviewData {
  timestamp: string;
  environment: string;
  go_live_active: boolean;
  global_pnl: {
    equity: number;
    daily_pnl: number;
    daily_pnl_pct: number;
    weekly_pnl: number;
    monthly_pnl: number;
    total_pnl: number;
  };
  exposure_per_exchange: Array<{ exchange: string; exposure: number }>;
  global_risk_state: 'OK' | 'WARNING' | 'CRITICAL';
  ess_status: {
    status: string;
    triggers_today: number;
    daily_loss: number;
    threshold: number;
  };
  capital_profiles_summary: any[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function OverviewTab() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // [PHASE 9] Real-time WebSocket stream
  const { data: streamData, connected: wsConnected } = useDashboardStream();

  const fetchOverviewData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard/overview`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch overview');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviewData();
    // Poll every 5 seconds
    const interval = setInterval(fetchOverviewData, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="dashboard-card h-32 animate-pulse bg-gray-200" />
        ))}
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <p className="text-danger text-lg">⚠️ {error || 'No data'}</p>
        <button onClick={fetchOverviewData} className="mt-4 px-4 py-2 bg-primary text-white rounded-lg">
          Retry
        </button>
      </div>
    );
  }

  // const riskStateColor = 
  //   data.global_risk_state === 'OK' ? 'bg-success text-white' :
  //   data.global_risk_state === 'WARNING' ? 'bg-warning text-white' :
  //   'bg-danger text-white';

  // const essStatusColor =
  //   data.ess_status.status === 'INACTIVE' ? 'bg-success text-white' :
  //   data.ess_status.status === 'ACTIVE' ? 'bg-danger text-white' :
  //   'bg-gray-500 text-white';
  
  // [PHASE 9] Use real-time data if available
  const isGoLiveActive = streamData?.go_live_active ?? data.go_live_active;
  const currentRiskState = streamData?.risk_state ?? data.global_risk_state;
  const isESSActive = streamData?.ess_active ?? (data.ess_status.status === 'ACTIVE');
  const dailyPnL = streamData?.daily_pnl ?? data.global_pnl.daily_pnl;
  const dailyPnLPct = streamData?.daily_pnl_pct ?? data.global_pnl.daily_pnl_pct;

  return (
    <div className="space-y-6">
      {/* [PHASE 9] WebSocket Status Indicator */}
      {wsConnected && (
        <div className="flex items-center justify-end text-xs text-success">
          <span className="w-2 h-2 rounded-full bg-success mr-2 animate-pulse" />
          Live updates active
        </div>
      )}
      
      {/* Top Row: Key Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* GO-LIVE Status */}
        <DashboardCard title="GO-LIVE Status">
          <div className="text-center py-4">
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-bold ${
              isGoLiveActive ? 'bg-success text-white' : 'bg-gray-500 text-white'
            }`}>
              {isGoLiveActive ? '✅ ACTIVE' : '⏸️ INACTIVE'}
            </div>
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              Env: {data.environment.toUpperCase()}
            </p>
            {streamData && (
              <p className="text-xs text-gray-500 mt-1">
                {streamData.open_positions_count} positions
              </p>
            )}
          </div>
        </DashboardCard>

        {/* Global Risk State */}
        <DashboardCard title="Global Risk State">
          <div className="text-center py-4">
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-bold ${
              currentRiskState === 'OK' ? 'bg-success text-white' :
              currentRiskState === 'WARNING' ? 'bg-warning text-white' :
              'bg-danger text-white'
            }`}>
              {currentRiskState}
            </div>
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              System risk assessment
            </p>
            {streamData && streamData.blocked_trades_last_5m > 0 && (
              <p className="text-xs text-danger mt-1">
                🚫 {streamData.blocked_trades_last_5m} blocked (5m)
              </p>
            )}
          </div>
        </DashboardCard>

        {/* ESS Status */}
        <DashboardCard title="Emergency Stop System">
          <div className="text-center py-4">
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-bold ${
              isESSActive ? 'bg-danger text-white' : 'bg-success text-white'
            }`}>
              {isESSActive ? 'ACTIVE' : 'INACTIVE'}
            </div>
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              Triggers today: {data.ess_status.triggers_today}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Threshold: {data.ess_status.threshold}%
            </p>
            {streamData?.ess_reason && (
              <p className="text-xs text-danger mt-1">
                {streamData.ess_reason}
              </p>
            )}
          </div>
        </DashboardCard>

        {/* Daily PnL */}
        <DashboardCard title="Daily PnL">
          <div className="text-center py-4">
            <div className={`text-3xl font-bold ${
              dailyPnL >= 0 ? 'text-success' : 'text-danger'
            }`}>
              ${dailyPnL.toFixed(2)}
            </div>
            <p className={`text-sm ${
              dailyPnLPct >= 0 ? 'text-success' : 'text-danger'
            }`}>
              {dailyPnLPct >= 0 ? '+' : ''}{dailyPnLPct.toFixed(2)}%
            </p>
            {streamData && streamData.failovers_last_5m > 0 && (
              <p className="text-xs text-warning mt-1">
                ⚠️ {streamData.failovers_last_5m} failovers (5m)
              </p>
            )}
          </div>
        </DashboardCard>
      </div>

      {/* Second Row: PnL Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Equity */}
        <DashboardCard title="Total Equity">
          <div className="text-center py-4">
            <div className="text-3xl font-bold text-gray-900 dark:text-white">
              ${data.global_pnl.equity.toFixed(2)}
            </div>
            <p className="mt-1 text-xs text-gray-600 dark:text-gray-400">
              Current portfolio value
            </p>
          </div>
        </DashboardCard>

        {/* Weekly PnL */}
        <DashboardCard title="Weekly PnL">
          <div className="text-center py-4">
            <div className={`text-3xl font-bold ${
              data.global_pnl.weekly_pnl >= 0 ? 'text-success' : 'text-danger'
            }`}>
              ${data.global_pnl.weekly_pnl.toFixed(2)}
            </div>
          </div>
        </DashboardCard>

        {/* Monthly PnL */}
        <DashboardCard title="Monthly PnL">
          <div className="text-center py-4">
            <div className={`text-3xl font-bold ${
              data.global_pnl.monthly_pnl >= 0 ? 'text-success' : 'text-danger'
            }`}>
              ${data.global_pnl.monthly_pnl.toFixed(2)}
            </div>
          </div>
        </DashboardCard>
      </div>

      {/* Third Row: Exposure per Exchange */}
      <DashboardCard title="Exposure per Exchange">
        <div className="p-4">
          {data.exposure_per_exchange && data.exposure_per_exchange.length > 0 ? (
            <div className="space-y-3">
              {data.exposure_per_exchange.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <span className="font-medium text-gray-700 dark:text-gray-300">
                    {item.exchange}
                  </span>
                  <span className="text-lg font-bold text-gray-900 dark:text-white">
                    ${item.exposure.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-gray-600 dark:text-gray-400 py-8">
              No exchange exposure data
            </p>
          )}
        </div>
      </DashboardCard>

      {/* Update Timestamp */}
      <div className="text-center text-xs text-gray-500 dark:text-gray-400">
        Last updated: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
