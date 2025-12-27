// Global state management with Zustand
import { create } from 'zustand';
import type { DashboardSnapshot, DashboardEvent, ServiceStatus } from './types';

// Connection status type (Sprint 4 Del 2)
export type ConnectionStatus = 'CONNECTED' | 'DEGRADED' | 'DISCONNECTED';

interface DashboardStore {
  // State
  snapshot: DashboardSnapshot | null;
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
  wsConnected: boolean;
  connectionStatus: ConnectionStatus; // Sprint 4 Del 2
  lastFetchTimestamp: number | null; // Sprint 4 Del 2: for 5s cache

  // Actions
  setSnapshot: (snapshot: DashboardSnapshot) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setWSConnected: (connected: boolean) => void;
  setConnectionStatus: (status: ConnectionStatus) => void; // Sprint 4 Del 2
  setLastFetchTimestamp: (timestamp: number) => void; // Sprint 4 Del 2
  handleEvent: (event: DashboardEvent) => void;
  reset: () => void;
}

export const useDashboardStore = create<DashboardStore>((set, get) => ({
  // Initial state
  snapshot: null,
  loading: false,
  error: null,
  lastUpdate: null,
  wsConnected: false,
  connectionStatus: 'DISCONNECTED',
  lastFetchTimestamp: null,

  // Set complete snapshot
  setSnapshot: (snapshot) => {
    const now = Date.now();
    
    // Determine connection status based on system health (Sprint 4 Del 2)
    let connectionStatus: ConnectionStatus = 'CONNECTED';
    if (snapshot.system.overall_status === 'DOWN') {
      connectionStatus = 'DISCONNECTED';
    } else if (snapshot.system.overall_status === 'DEGRADED') {
      connectionStatus = 'DEGRADED';
    }

    set({
      snapshot,
      lastUpdate: new Date().toISOString(),
      lastFetchTimestamp: now,
      connectionStatus,
    });
  },

  // Set WebSocket connection status
  setWSConnected: (connected) => {
    const { snapshot } = get();
    let connectionStatus: ConnectionStatus = connected ? 'CONNECTED' : 'DISCONNECTED';
    
    // If WS connected but system degraded, mark as DEGRADED
    if (connected && snapshot?.system.overall_status === 'DEGRADED') {
      connectionStatus = 'DEGRADED';
    }
    
    set({ wsConnected: connected, connectionStatus });
  },
  
  // Set connection status (Sprint 4 Del 2)
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  
  // Set last fetch timestamp (Sprint 4 Del 2)
  setLastFetchTimestamp: (timestamp) => set({ lastFetchTimestamp: timestamp }),

  // Set loading state
  setLoading: (loading) => set({ loading }),

  // Set error
  setError: (error) => set({ error, loading: false }),

  // Handle real-time event
  handleEvent: (event) => {
    const { snapshot } = get();
    if (!snapshot) return;

    console.log('[DashboardStore] Handling event:', event.type);

    switch (event.type) {
      case 'position_updated': {
        // Update specific position
        const updatedPositions = snapshot.positions.map(pos =>
          pos.symbol === event.payload.symbol ? { ...pos, ...event.payload } : pos
        );
        set({
          snapshot: { ...snapshot, positions: updatedPositions },
          lastUpdate: event.timestamp,
        });
        break;
      }

      case 'pnl_updated': {
        // Update portfolio PnL
        set({
          snapshot: {
            ...snapshot,
            portfolio: { ...snapshot.portfolio, ...event.payload },
          },
          lastUpdate: event.timestamp,
        });
        break;
      }

      case 'signal_generated': {
        // Add new signal (keep last 20)
        const updatedSignals = [event.payload, ...snapshot.signals].slice(0, 20);
        set({
          snapshot: { ...snapshot, signals: updatedSignals },
          lastUpdate: event.timestamp,
        });
        break;
      }

      case 'ess_state_changed': {
        // Update ESS state
        set({
          snapshot: {
            ...snapshot,
            risk: {
              ...snapshot.risk,
              ess_state: event.payload.ess_state,
              ess_reason: event.payload.reason,
            },
          },
          lastUpdate: event.timestamp,
        });
        break;
      }

      case 'health_alert': {
        // Update system health
        const updatedServices = snapshot.system.services.map(svc =>
          svc.name === event.payload.service
            ? { ...svc, status: event.payload.status }
            : svc
        );
        
        // Recalculate overall status
        let overallStatus: ServiceStatus = 'OK';
        for (const svc of updatedServices) {
          if (svc.status === 'DOWN') overallStatus = 'DOWN';
          else if (svc.status === 'DEGRADED' && overallStatus !== 'DOWN') {
            overallStatus = 'DEGRADED';
          }
        }
        
        set({
          snapshot: {
            ...snapshot,
            system: {
              ...snapshot.system,
              overall_status: overallStatus,
              services: updatedServices,
              alerts_count: snapshot.system.alerts_count + 1,
              last_alert: event.timestamp,
            },
          },
          lastUpdate: event.timestamp,
          connectionStatus: overallStatus === 'DOWN' ? 'DISCONNECTED' : 
                           overallStatus === 'DEGRADED' ? 'DEGRADED' : 'CONNECTED',
        });
        break;
      }
      
      case 'strategy_updated': {
        // Update strategy data (Sprint 4 Del 2)
        set({
          snapshot: {
            ...snapshot,
            strategy: event.payload,
          },
          lastUpdate: event.timestamp,
        });
        break;
      }
      
      case 'rl_sizing_updated': {
        // Update RL sizing in strategy (Sprint 4 Del 2)
        if (snapshot.strategy) {
          set({
            snapshot: {
              ...snapshot,
              strategy: {
                ...snapshot.strategy,
                rl_sizing: event.payload,
              },
            },
            lastUpdate: event.timestamp,
          });
        }
        break;
      }
      
      case 'regime_changed': {
        // Update regime in strategy (Sprint 4 Del 2)
        if (snapshot.strategy) {
          set({
            snapshot: {
              ...snapshot,
              strategy: {
                ...snapshot.strategy,
                regime: event.payload.new_regime,
              },
            },
            lastUpdate: event.timestamp,
          });
        }
        break;
      }

      case 'connected':
      case 'heartbeat':
      case 'pong':
        // Ignore connection messages
        break;

      default:
        console.log('[DashboardStore] Unknown event type:', event.type);
    }
  },

  // Reset state
  reset: () => set({
    snapshot: null,
    loading: false,
    error: null,
    lastUpdate: null,
    wsConnected: false,
    connectionStatus: 'DISCONNECTED',
    lastFetchTimestamp: null,
  }),
}));

// Helper to load cached snapshot from sessionStorage (Sprint 4 Del 2)
export function loadCachedSnapshot(): { snapshot: DashboardSnapshot; timestamp: number } | null {
  try {
    const cached = sessionStorage.getItem('dashboard_snapshot');
    if (!cached) return null;
    
    const data = JSON.parse(cached);
    const age = Date.now() - data.timestamp;
    
    // Cache valid for 5 seconds (5000ms)
    if (age < 5000) {
      return data;
    }
    
    return null;
  } catch (err) {
    console.warn('[DashboardStore] Failed to load cached snapshot:', err);
    return null;
  }
}
