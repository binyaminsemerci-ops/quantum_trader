// Main Dashboard Page - DASHBOARD-V3-001: Full Visual UI with Tabs
import { useEffect, useState } from 'react';
import { useDashboardStore, loadCachedSnapshot } from '@/lib/store';
import { fetchDashboardSnapshot } from '@/lib/api';
import { dashboardWebSocket } from '@/lib/websocket';
import Sidebar from '@/components/Sidebar';
import TopBar from '@/components/TopBar';
import PortfolioPanel from '@/components/PortfolioPanel';
import PositionsPanel from '@/components/PositionsPanel';
import SignalsPanel from '@/components/SignalsPanel';
import RiskPanel from '@/components/RiskPanel';
import SystemHealthPanel from '@/components/SystemHealthPanel';
import StrategyPanel from '@/components/dashboard/StrategyPanel';
import RLInspector from '@/components/dashboard/RLInspector';

// [DASHBOARD-V3-001] New tabbed components
import OverviewTab from '@/components/dashboard/OverviewTab';
import TradingTab from '@/components/dashboard/TradingTab';
import RiskTab from '@/components/dashboard/RiskTab';
import SystemTab from '@/components/dashboard/SystemTab';

type DashboardTab = 'classic' | 'overview' | 'trading' | 'risk' | 'system';

export default function DashboardPage() {
  const {
    snapshot,
    loading,
    error,
    lastUpdate,
    wsConnected,
    connectionStatus,
    setSnapshot,
    setLoading,
    setError,
    setWSConnected,
    setConnectionStatus,
    handleEvent,
  } = useDashboardStore();

  const [isFetching, setIsFetching] = useState(false);
  const [activeTab, setActiveTab] = useState<DashboardTab>('overview'); // [V3] Default to Overview tab

  // Load initial snapshot with 5s SWR cache (Sprint 4 Del 2)
  useEffect(() => {
    const loadSnapshot = async () => {
      // Try loading from cache first
      const cached = loadCachedSnapshot();
      if (cached) {
        console.log('[Dashboard] Using cached snapshot (age:', Date.now() - cached.timestamp, 'ms)');
        setSnapshot(cached.snapshot);
        
        // Fetch fresh data in background (stale-while-revalidate)
        setIsFetching(true);
        try {
          const data = await fetchDashboardSnapshot();
          setSnapshot(data);
          console.log('[Dashboard] Background refresh complete');
        } catch (err) {
          console.warn('[Dashboard] Background refresh failed:', err);
        } finally {
          setIsFetching(false);
        }
        return;
      }

      // No cache, do normal load with loading state
      setLoading(true);
      try {
        console.log('[Dashboard] Loading initial snapshot...');
        const data = await fetchDashboardSnapshot();
        setSnapshot(data);
        console.log('[Dashboard] Snapshot loaded successfully');
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load dashboard';
        console.error('[Dashboard] Failed to load snapshot:', err);
        setError(message);
        setConnectionStatus('DISCONNECTED');
      } finally {
        setLoading(false);
      }
    };

    loadSnapshot();
  }, [setSnapshot, setLoading, setError, setConnectionStatus]);

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    console.log('[Dashboard] Connecting to WebSocket...');
    
    // Subscribe to events
    const unsubscribe = dashboardWebSocket.subscribe((event) => {
      console.log('[Dashboard] Received event:', event.type);
      
      // Update connection status
      if (event.type === 'connected') {
        setWSConnected(true);
      }
      
      // Handle event
      handleEvent(event);
    });

    // Connect
    dashboardWebSocket.connect();

    // Update connection status
    const checkConnection = setInterval(() => {
      const connected = dashboardWebSocket.isConnected();
      setWSConnected(connected);
      
      // Update connection status based on WS and system health
      if (!connected) {
        setConnectionStatus('DISCONNECTED');
      } else if (snapshot?.system.overall_status === 'DEGRADED') {
        setConnectionStatus('DEGRADED');
      } else if (snapshot?.system.overall_status === 'OK') {
        setConnectionStatus('CONNECTED');
      }
    }, 1000);

    // Cleanup on unmount
    return () => {
      console.log('[Dashboard] Disconnecting from WebSocket...');
      clearInterval(checkConnection);
      unsubscribe();
      dashboardWebSocket.disconnect();
    };
  }, [setWSConnected, setConnectionStatus, handleEvent, snapshot]);

  // Loading state with skeleton (Sprint 4 Del 2)
  if (loading && !snapshot) {
    return (
      <div className="flex h-screen bg-gray-50 dark:bg-slate-900">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <TopBar essState="UNKNOWN" systemStatus="UNKNOWN" lastUpdate={null} wsConnected={false} />
          <div className="flex-1 overflow-auto p-6">
            <div className="max-w-[1920px] mx-auto space-y-6">
              {/* Skeleton panels */}
              <div className="dashboard-card h-32 animate-pulse bg-gray-200" />
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                <div className="xl:col-span-2 dashboard-card h-96 animate-pulse bg-gray-200" />
                <div className="dashboard-card h-96 animate-pulse bg-gray-200" />
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="dashboard-card h-80 animate-pulse bg-gray-200" />
                <div className="dashboard-card h-80 animate-pulse bg-gray-200" />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error && !snapshot) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 dark:bg-slate-900">
        <div className="text-center max-w-md">
          <div className="text-danger text-5xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
            Failed to Load Dashboard
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // No data state
  if (!snapshot) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 dark:bg-slate-900">
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400">No dashboard data available</p>
        </div>
      </div>
    );
  }

  // Degraded Mode Banner check (Sprint 4 Del 3: Extended for partial data)
  const showDegradedBanner = 
    connectionStatus === 'DEGRADED' || 
    connectionStatus === 'DISCONNECTED' ||
    (snapshot && snapshot.partial_data && snapshot.errors.length > 0);
  
  const bannerMessage = connectionStatus === 'DISCONNECTED'
    ? '⚠️ System Offline – Dashboard data may be stale'
    : (snapshot?.partial_data && snapshot.errors.length > 0)
    ? `⚠️ Partial Data – Some services unavailable: ${snapshot.errors.join(', ')}`
    : '⚠️ System Degraded – Some services experiencing issues';
  
  const bannerStyle = connectionStatus === 'DISCONNECTED'
    ? 'bg-danger text-white'
    : 'bg-warning text-white';

  // Main dashboard
  return (
    <div className="flex h-screen bg-gray-50 dark:bg-slate-900">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <TopBar
          essState={snapshot.risk.ess_state}
          systemStatus={snapshot.system.overall_status}
          lastUpdate={lastUpdate}
          wsConnected={wsConnected}
        />

        {/* Degraded Mode Banner (Sprint 4 Del 3: Extended for partial data) */}
        {showDegradedBanner && (
          <div className={`px-6 py-3 text-center text-sm font-medium ${bannerStyle}`}>
            {bannerMessage}
          </div>
        )}

        {/* Dashboard content - Tabbed Interface [V3] */}
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-[1920px] mx-auto space-y-6">
            {/* Tab Navigation - Phase 10: Enhanced styling */}
            <div className="flex flex-wrap gap-2 sm:gap-1 sm:space-x-1 border-b border-gray-300 dark:border-gray-700 mb-6">
              <button
                onClick={() => setActiveTab('overview')}
                className={`px-4 py-2.5 font-medium rounded-t-lg transition-all duration-200 ${
                  activeTab === 'overview'
                    ? 'text-primary bg-primary/10 dark:bg-primary/20 border-b-2 border-primary'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-700/50'
                }`}
              >
                📊 Overview
              </button>
              <button
                onClick={() => setActiveTab('trading')}
                className={`px-4 py-2.5 font-medium rounded-t-lg transition-all duration-200 ${
                  activeTab === 'trading'
                    ? 'text-primary bg-primary/10 dark:bg-primary/20 border-b-2 border-primary'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-700/50'
                }`}
              >
                📈 Trading
              </button>
              <button
                onClick={() => setActiveTab('risk')}
                className={`px-4 py-2.5 font-medium rounded-t-lg transition-all duration-200 ${
                  activeTab === 'risk'
                    ? 'text-primary bg-primary/10 dark:bg-primary/20 border-b-2 border-primary'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-700/50'
                }`}
              >
                🛡️ Risk & Safety
              </button>
              <button
                onClick={() => setActiveTab('system')}
                className={`px-4 py-2.5 font-medium rounded-t-lg transition-all duration-200 ${
                  activeTab === 'system'
                    ? 'text-primary bg-primary/10 dark:bg-primary/20 border-b-2 border-primary'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-700/50'
                }`}
              >
                ⚙️ System & Stress
              </button>
              <button
                onClick={() => setActiveTab('classic')}
                className={`px-4 py-2.5 font-medium rounded-t-lg transition-all duration-200 ${
                  activeTab === 'classic'
                    ? 'text-primary bg-primary/10 dark:bg-primary/20 border-b-2 border-primary'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-slate-700/50'
                }`}
              >
                🎛️ Classic
              </button>
            </div>

            {/* Tab Content - Phase 10: Improved spacing */}
            <div className="space-y-6">
              {activeTab === 'overview' && <OverviewTab />}
              {activeTab === 'trading' && <TradingTab />}
              {activeTab === 'risk' && <RiskTab />}
              {activeTab === 'system' && <SystemTab />}
              
              {/* Classic View - Original Dashboard */}
              {activeTab === 'classic' && (
                <>
                  {/* Portfolio summary */}
                  <PortfolioPanel portfolio={snapshot.portfolio} />

                  {/* Main grid: Positions + Signals */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mt-6">
                    {/* Positions (full width on mobile, 2 cols on xl) */}
                    <div className="lg:col-span-1 xl:col-span-2 h-[500px]">
                      <PositionsPanel positions={snapshot.positions} />
                    </div>

                    {/* Signals */}
                    <div className="h-[500px]">
                      <SignalsPanel signals={snapshot.signals} />
                    </div>
                  </div>

                  {/* Strategy + RL Inspector (Sprint 4 Del 2) */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <StrategyPanel strategy={snapshot.strategy} loading={isFetching} />
                    <RLInspector rlSizing={snapshot.strategy?.rl_sizing} loading={isFetching} />
                  </div>

                  {/* Bottom grid: Risk + System Health */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    {/* Risk */}
                    <RiskPanel risk={snapshot.risk} />

                    {/* System Health */}
                    <div className="h-[400px]">
                      <SystemHealthPanel system={snapshot.system} />
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
