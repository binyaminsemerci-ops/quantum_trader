/**
 * useDashboardStream Hook
 * DASHBOARD-V3-001: Phase 9 - Real-time WebSocket Integration
 * 
 * Provides real-time dashboard metrics via WebSocket with polling fallback.
 * 
 * Usage:
 *   const { data, connected, error } = useDashboardStream();
 * 
 * Returns:
 *   - data: Real-time metrics (positions count, blocked trades, risk state, etc.)
 *   - connected: WebSocket connection status
 *   - error: Error message if connection fails
 */

import { useEffect, useState } from 'react';
import { dashboardWebSocket } from '@/lib/websocket';
import type { DashboardEvent } from '@/lib/types';

export interface DashboardStreamData {
  timestamp: string;
  
  // GO-LIVE status
  go_live_active: boolean;
  
  // Risk state
  risk_state: 'OK' | 'WARNING' | 'CRITICAL';
  
  // ESS status
  ess_active: boolean;
  ess_reason?: string;
  
  // Live counters
  open_positions_count: number;
  pending_orders_count: number;
  
  // Real-time risk metrics
  blocked_trades_last_5m: number;
  scaled_trades_last_5m: number;
  
  // Failover events
  failovers_last_5m: number;
  
  // PnL
  daily_pnl: number;
  daily_pnl_pct: number;
  
  // Exposure
  total_exposure: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const FALLBACK_POLL_INTERVAL = 5000; // 5 seconds

export function useDashboardStream() {
  const [data, setData] = useState<DashboardStreamData | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number>(0);

  // Fallback: Poll overview endpoint if WebSocket disconnected
  useEffect(() => {
    if (connected) return; // WebSocket active, no polling needed

    const pollOverview = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard/overview`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const overview = await response.json();
        
        // Map overview data to stream format
        setData({
          timestamp: overview.timestamp,
          go_live_active: overview.go_live_active || false,
          risk_state: overview.global_risk_state || 'OK',
          ess_active: overview.ess_status?.status === 'ACTIVE',
          ess_reason: overview.ess_status?.reason,
          open_positions_count: 0, // Not available in overview endpoint
          pending_orders_count: 0, // Not in overview endpoint
          blocked_trades_last_5m: 0, // Would need risk endpoint
          scaled_trades_last_5m: 0,
          failovers_last_5m: 0,
          daily_pnl: overview.global_pnl?.daily_pnl || 0,
          daily_pnl_pct: overview.global_pnl?.daily_pnl_pct || 0,
          total_exposure: overview.exposure_per_exchange?.reduce(
            (sum: number, ex: any) => sum + (ex.net_exposure || 0), 
            0
          ) || 0,
        });
        
        setError(null);
        setLastUpdate(Date.now());
      } catch (err) {
        console.error('[useDashboardStream] Polling error:', err);
        setError(err instanceof Error ? err.message : 'Polling failed');
      }
    };

    // Initial poll
    pollOverview();
    
    // Poll every 5 seconds
    const interval = setInterval(pollOverview, FALLBACK_POLL_INTERVAL);
    
    return () => clearInterval(interval);
  }, [connected]);

  // WebSocket real-time updates
  useEffect(() => {
    console.log('[useDashboardStream] Setting up WebSocket listener');
    
    const handleEvent = (event: DashboardEvent) => {
      // Update connection status
      if (event.type === 'connected') {
        setConnected(true);
        setError(null);
        return;
      }
      
      if (event.type === 'disconnected' || event.type === 'error') {
        setConnected(false);
        if (event.type === 'error') {
          setError('WebSocket error - using polling fallback');
        }
        return;
      }
      
      // Handle real-time data updates
      if (event.type === 'snapshot' || event.type === 'update') {
        const payload = event.payload;
        
        // Extract real-time metrics
        const streamData: DashboardStreamData = {
          timestamp: new Date().toISOString(),
          
          // GO-LIVE (from system or flags)
          go_live_active: payload.go_live_active ?? payload.system?.go_live_active ?? false,
          
          // Risk state (from risk or global)
          risk_state: payload.risk_state ?? payload.risk?.global_state ?? 'OK',
          
          // ESS status (from ESS module)
          ess_active: payload.ess_status?.active ?? payload.ess?.active ?? false,
          ess_reason: payload.ess_status?.reason ?? payload.ess?.reason,
          
          // Positions count (from positions array or count)
          open_positions_count: payload.positions?.length ?? payload.open_positions_count ?? 0,
          
          // Orders count (from orders array or count)
          pending_orders_count: payload.orders?.filter((o: any) => o.status === 'PENDING').length ?? 0,
          
          // Risk gate metrics (last 5 minutes)
          blocked_trades_last_5m: payload.blocked_trades_last_5m ?? 0,
          scaled_trades_last_5m: payload.scaled_trades_last_5m ?? 0,
          
          // Failover events
          failovers_last_5m: payload.failovers_last_5m ?? 0,
          
          // PnL (from portfolio or pnl)
          daily_pnl: payload.pnl?.daily ?? payload.portfolio?.daily_pnl ?? 0,
          daily_pnl_pct: payload.pnl?.daily_pct ?? payload.portfolio?.daily_pnl_pct ?? 0,
          
          // Total exposure
          total_exposure: payload.total_exposure ?? 
                         payload.portfolio?.total_exposure ?? 
                         payload.exposure_per_exchange?.reduce(
                           (sum: number, ex: any) => sum + (ex.net_exposure || 0), 
                           0
                         ) ?? 0,
        };
        
        setData(streamData);
        setLastUpdate(Date.now());
        console.log('[useDashboardStream] Real-time update received');
      }
    };
    
    // Subscribe to WebSocket events
    const unsubscribe = dashboardWebSocket.subscribe(handleEvent);
    
    // Connect if not already connected
    if (!dashboardWebSocket.isConnected()) {
      dashboardWebSocket.connect();
    }
    
    return () => {
      console.log('[useDashboardStream] Cleaning up WebSocket listener');
      unsubscribe();
    };
  }, []);

  return {
    data,
    connected,
    error,
    lastUpdate,
  };
}
