import { useState, useEffect } from 'react';

interface DashboardData {
  systemStatus?: {
    status: string;
    uptime: string;
    timestamp: string;
  };
  portfolio?: {
    total_value: number;
    positions: number;
    pnl_percent: number;
  };
  marketData?: {
    fear_greed: number;
  };
  signals?: Array<{
    id: string;
    symbol: string;
    action: string;
    confidence: number;
    timestamp: string;
  }>;
  aiTradingStatus?: {
    status: string;
    active: boolean;
  };
}

export default function SimpleDashboard(): JSX.Element {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');

  const fetchDashboardData = async () => {
    try {
      setConnectionStatus('connecting');
      
      const [systemRes, portfolioRes, marketRes, signalsRes, aiRes] = await Promise.all([
        fetch('http://127.0.0.1:8000/api/v1/system/status').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/portfolio').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/portfolio/market-overview').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/signals/recent?limit=5').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/ai-trading/status').catch(() => null)
      ]);

      const dashboardData: DashboardData = {};

      if (systemRes?.ok) {
        dashboardData.systemStatus = await systemRes.json();
      }
      if (portfolioRes?.ok) {
        dashboardData.portfolio = await portfolioRes.json();
      }
      if (marketRes?.ok) {
        dashboardData.marketData = await marketRes.json();
      }
      if (signalsRes?.ok) {
        dashboardData.signals = await signalsRes.json();
      }
      if (aiRes?.ok) {
        dashboardData.aiTradingStatus = await aiRes.json();
      }

      setData(dashboardData);
      setConnectionStatus('connected');
      setError(null);
    } catch (err) {
      console.error('Dashboard fetch error:', err);
      setError(`Connection error: ${err}`);
      setConnectionStatus('disconnected');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Poll every 5 seconds
    const interval = setInterval(fetchDashboardData, 5000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold">Loading Quantum Trader Dashboard...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-blue-400">🚀 Quantum Trader</h1>
              <p className="text-gray-400 mt-1">Professional Trading Dashboard</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                connectionStatus === 'connected' ? 'bg-green-900 text-green-300' :
                connectionStatus === 'connecting' ? 'bg-yellow-900 text-yellow-300' :
                'bg-red-900 text-red-300'
              }`}>
                {connectionStatus === 'connected' ? '🟢 Connected' :
                 connectionStatus === 'connecting' ? '🟡 Connecting...' :
                 '🔴 Disconnected'}
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900 border border-red-600 rounded-lg">
            <h3 className="text-red-300 font-semibold">Connection Error</h3>
            <p className="text-red-200 text-sm mt-1">{error}</p>
            <button 
              onClick={fetchDashboardData}
              className="mt-2 px-4 py-2 bg-red-700 hover:bg-red-600 rounded text-sm"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {/* System Status */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-blue-300 mb-4">🖥️ System Status</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`font-semibold ${
                  data?.systemStatus?.status === 'running' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {data?.systemStatus?.status?.toUpperCase() || 'UNKNOWN'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Uptime:</span>
                <span className="text-white">{data?.systemStatus?.uptime || 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Portfolio */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-green-300 mb-4">💰 Portfolio</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Total Value:</span>
                <span className="text-white font-semibold">
                  ${data?.portfolio?.total_value?.toLocaleString() || '0'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Positions:</span>
                <span className="text-white">{data?.portfolio?.positions || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">P&L:</span>
                <span className={`font-semibold ${
                  (data?.portfolio?.pnl_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {data?.portfolio?.pnl_percent?.toFixed(2) || '0.00'}%
                </span>
              </div>
            </div>
          </div>

          {/* Market Data */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-purple-300 mb-4">📊 Market Data</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Fear/Greed Index:</span>
                <span className="text-white font-semibold">
                  {data?.marketData?.fear_greed || '50'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* AI Trading Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-cyan-300 mb-4">🤖 AI Trading</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`px-2 py-1 rounded text-sm font-medium ${
                  data?.aiTradingStatus?.active ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                }`}>
                  {data?.aiTradingStatus?.status || 'Inactive'}
                </span>
              </div>
            </div>
          </div>

          {/* Recent Signals */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-yellow-300 mb-4">📈 Recent Signals</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {data?.signals && data.signals.length > 0 ? (
                data.signals.slice(0, 3).map((signal: any, index: number) => (
                  <div key={index} className="bg-gray-700 rounded p-3">
                    <div className="flex justify-between items-center">
                      <div>
                        <span className="font-medium text-white">{signal.symbol}</span>
                        <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                          signal.action === 'BUY' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                        }`}>
                          {signal.action}
                        </span>
                      </div>
                      <span className="text-sm text-gray-400">
                        {signal.confidence}% confidence
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-400 text-center py-4">
                  No recent signals
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>Last updated: {new Date().toLocaleTimeString()}</p>
          <p>Quantum Trader • HTTP Polling Active</p>
        </div>
      </div>
    </div>
  );
}