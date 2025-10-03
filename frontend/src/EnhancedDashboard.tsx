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
    predicted_price?: number;
    current_price?: number;
    reasoning?: string;
  }>;
  aiTradingStatus?: {
    status: string;
    active: boolean;
    enabled?: boolean;
    symbols?: string[];
    learning_active?: boolean;
    symbols_monitored?: number;
    data_points?: number;
    accuracy?: number;
    total_signals_today?: number;
    successful_trades?: number;
    win_rate?: number;
  };
  learningStatus?: {
    learning_active: boolean;
    symbols_monitored: number;
    data_points: number;
    model_accuracy: number;
    status: string;
  };
  watchlist?: Array<{
    symbol: string;
    price: number;
    change24h: number;
    volume24h: number;
    sparkline?: number[];
  }>;
}

export default function EnhancedDashboard(): JSX.Element {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');

  const fetchDashboardData = async () => {
    try {
      setConnectionStatus('connecting');
      
      const [systemRes, portfolioRes, marketRes, signalsRes, aiRes, learningRes, watchlistRes] = await Promise.all([
        fetch('http://127.0.0.1:8000/api/v1/system/status').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/portfolio').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/portfolio/market-overview').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/signals/recent?limit=5').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/ai-trading/status').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/continuous-learning/status').catch(() => null),
        fetch('http://127.0.0.1:8000/api/v1/watchlist').catch(() => null)
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
      if (learningRes?.ok) {
        dashboardData.learningStatus = await learningRes.json();
      }
      if (watchlistRes?.ok) {
        dashboardData.watchlist = await watchlistRes.json();
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
    
    // Poll every 3 seconds
    const interval = setInterval(fetchDashboardData, 3000);
    
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
              <h1 className="text-3xl font-bold text-blue-400">🚀 Quantum Trader Pro</h1>
              <p className="text-gray-400 mt-1">AI-Powered Trading Dashboard • Live Data</p>
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
              <div className="text-sm text-gray-400">
                Updated: {new Date().toLocaleTimeString()}
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

        {/* Top Row - System & Portfolio */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* System Status */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-blue-300 mb-4">🖥️ System</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`font-semibold ${
                  data?.systemStatus?.status === 'online' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {data?.systemStatus?.status?.toUpperCase() || 'OFFLINE'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Uptime:</span>
                <span className="text-white text-sm">{data?.systemStatus?.uptime || 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Portfolio Summary */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-green-300 mb-4">💰 Portfolio</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Value:</span>
                <span className="text-white font-semibold text-sm">
                  ${(data?.portfolio?.total_value || 0).toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">P&L:</span>
                <span className={`font-semibold text-sm ${
                  (data?.portfolio?.pnl_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {data?.portfolio?.pnl_percent?.toFixed(2) || '0.00'}%
                </span>
              </div>
            </div>
          </div>

          {/* Market Data */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-purple-300 mb-4">📊 Market</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Fear/Greed:</span>
                <span className="text-white font-semibold">
                  {data?.marketData?.fear_greed || '50'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Positions:</span>
                <span className="text-white">{data?.portfolio?.positions || 0}</span>
              </div>
            </div>
          </div>

          {/* AI Status Quick */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-cyan-300 mb-4">🤖 AI Status</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  data?.aiTradingStatus?.active ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                }`}>
                  {data?.aiTradingStatus?.status || 'Inactive'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate:</span>
                <span className="text-white font-semibold text-sm">
                  {data?.aiTradingStatus?.win_rate?.toFixed(1) || '0.0'}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* AI Trading & Learning Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* AI Trading Details */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-cyan-300 mb-4">🤖 AI Trading Engine</h3>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-gray-400 text-sm block">Today's Signals</span>
                  <span className="text-white font-semibold text-lg">
                    {data?.aiTradingStatus?.total_signals_today || 0}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400 text-sm block">Success Rate</span>
                  <span className="text-green-400 font-semibold text-lg">
                    {data?.aiTradingStatus?.win_rate?.toFixed(1) || '0.0'}%
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-gray-400 text-sm block">Successful</span>
                  <span className="text-white font-semibold">
                    {data?.aiTradingStatus?.successful_trades || 0}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400 text-sm block">Symbols</span>
                  <span className="text-white font-semibold">
                    {data?.aiTradingStatus?.symbols_monitored || 0}
                  </span>
                </div>
              </div>
              <div className="mt-3 p-2 bg-gray-700 rounded">
                <span className="text-xs text-gray-400">Active Pairs: </span>
                <span className="text-xs text-blue-300">
                  {data?.aiTradingStatus?.symbols?.join(', ') || 'Loading...'}
                </span>
              </div>
            </div>
          </div>

          {/* Learning Status */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-orange-300 mb-4">🧠 AI Learning</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Learning Status:</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  data?.learningStatus?.learning_active ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-300'
                }`}>
                  {data?.learningStatus?.status || 'Unknown'}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Model Accuracy:</span>
                  <span className="text-white font-semibold">
                    {((data?.learningStatus?.model_accuracy || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Data Points:</span>
                  <span className="text-white">
                    {(data?.learningStatus?.data_points || 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Symbols Monitored:</span>
                  <span className="text-white">
                    {data?.learningStatus?.symbols_monitored || 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Signals & Watchlist Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Trading Signals */}
          <div className="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-yellow-300 mb-4">📈 Live Trading Signals</h3>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {data?.signals && data.signals.length > 0 ? (
                data.signals.map((signal: any, index: number) => (
                  <div key={index} className="bg-gray-700 rounded p-4 border-l-4 border-l-blue-500">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center">
                        <span className="font-bold text-white text-lg mr-3">{signal.symbol}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          signal.action === 'BUY' ? 'bg-green-900 text-green-300' : 
                          signal.action === 'SELL' ? 'bg-red-900 text-red-300' : 
                          'bg-yellow-900 text-yellow-300'
                        }`}>
                          {signal.action}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-400">Confidence</div>
                        <div className="text-lg font-bold text-white">{signal.confidence}%</div>
                      </div>
                    </div>
                    {signal.reasoning && (
                      <div className="text-sm text-gray-300 mb-2">
                        💡 {signal.reasoning}
                      </div>
                    )}
                    <div className="flex justify-between text-sm">
                      <div>
                        <span className="text-gray-400">Current: </span>
                        <span className="text-white">${signal.current_price}</span>
                      </div>
                      {signal.predicted_price && (
                        <div>
                          <span className="text-gray-400">Target: </span>
                          <span className="text-green-400">${signal.predicted_price}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-400 text-center py-8">
                  <div className="text-4xl mb-2">📊</div>
                  <div>No signals available</div>
                </div>
              )}
            </div>
          </div>

          {/* Watchlist */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-indigo-300 mb-4">👁️ Watchlist</h3>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {data?.watchlist && data.watchlist.length > 0 ? (
                data.watchlist.slice(0, 8).map((item: any, index: number) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <div>
                      <div className="font-medium text-white text-sm">{item.symbol}</div>
                      <div className="text-xs text-gray-400">${item.price?.toFixed(2)}</div>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-medium ${
                        item.change24h >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {item.change24h >= 0 ? '+' : ''}{item.change24h?.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-400 text-center py-4">
                  Loading watchlist...
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm">
          <p>🚀 Quantum Trader Pro • Live HTTP Polling • Last Updated: {new Date().toLocaleTimeString()}</p>
        </div>
      </div>
    </div>
  );
}