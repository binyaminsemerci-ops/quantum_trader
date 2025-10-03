import { useState, useEffect } from 'react';
import AISignals from './components/AISignals_NoWS';
import EnhancedDataDashboard from './components/EnhancedDataDashboard_NoWS';
import ChatPanel from './components/ChatPanel_NoWS';
import CoinTable from './components/CoinTable';

interface DashboardData {
  systemStatus: any;
  marketData: any;
  portfolio: any;
  signals: any;
  aiStatus: any;
  learningStatus: any;
}

export default function SimpleDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // HTTP Polling instead of WebSocket
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setConnectionStatus('connecting');
        
        const [systemRes, portfolioRes, marketRes, signalsRes, aiRes, learningRes] = await Promise.all([
          fetch('http://localhost:8000/api/v1/system/status'),
          fetch('http://localhost:8000/api/v1/portfolio'),
          fetch('http://localhost:8000/api/v1/portfolio/market-overview'),
          fetch('http://localhost:8000/api/v1/signals/recent?limit=5'),
          fetch('http://localhost:8000/api/v1/ai-trading/status'),
          fetch('http://localhost:8000/api/v1/continuous-learning/status')
        ]);

        if (systemRes.ok && portfolioRes.ok && marketRes.ok && signalsRes.ok && aiRes.ok && learningRes.ok) {
          const dashboardData: DashboardData = {
            systemStatus: await systemRes.json(),
            portfolio: await portfolioRes.json(),
            marketData: await marketRes.json(),
            signals: await signalsRes.json(),
            aiStatus: await aiRes.json(),
            learningStatus: await learningRes.json()
          };

          setData(dashboardData);
          setConnectionStatus('connected');
          setError(null);
          setLastUpdate(new Date());
        } else {
          throw new Error('Failed to fetch dashboard data');
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setError(`Connection error - make sure backend is running on port 8000`);
        setConnectionStatus('disconnected');
      }
    };

    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Start continuous learning
  const handleStartLearning = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/continuous-learning/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        console.log('Continuous learning started');
      }
    } catch (error) {
      console.error('Error starting continuous learning:', error);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center text-red-400">
          <p className="text-xl">⚠️ Connection Error</p>
          <p className="mt-2">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded"
          >
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
          <p className="mt-4">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-6">
        
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">🚀 Quantum Trader Dashboard</h1>
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span className={`px-2 py-1 rounded ${
              connectionStatus === 'connected' ? 'bg-green-900 text-green-300' :
              connectionStatus === 'connecting' ? 'bg-yellow-900 text-yellow-300' :
              'bg-red-900 text-red-300'
            }`}>
              HTTP: {connectionStatus}
            </span>
            {lastUpdate && (
              <span>Last update: {lastUpdate.toLocaleTimeString()}</span>
            )}
          </div>
        </div>

        {/* System Status */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-2">System Status</h3>
            <div className="text-lg font-semibold text-green-400">
              {data?.systemStatus?.status?.toUpperCase() || 'UNKNOWN'}
            </div>
            <div className="text-sm text-gray-400">
              Uptime: {data?.systemStatus?.uptime || 'N/A'}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Portfolio Value</h3>
            <div className="text-lg font-semibold text-blue-400">
              ${data?.portfolio?.total_value?.toLocaleString() || '0'}
            </div>
            <div className="text-sm text-gray-400">
              Positions: {data?.portfolio?.positions || 0}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-2">PnL</h3>
            <div className={`text-lg font-semibold ${
              (data?.portfolio?.pnl_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {data?.portfolio?.pnl_percent?.toFixed(2) || '0.00'}%
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Fear & Greed</h3>
            <div className="text-lg font-semibold text-yellow-400">
              {data?.marketData?.fear_greed || '50'}
            </div>
            <div className="text-sm text-gray-400">Neutral</div>
          </div>
        </div>

        {/* AI Learning Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-4">🤖 AI Continuous Learning</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Status:</span>
                <span className={`px-2 py-1 rounded text-sm ${
                  data?.learningStatus?.learning_active ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                }`}>
                  {data?.learningStatus?.status || 'Inactive'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Symbols Monitored:</span>
                <span className="text-white">{data?.learningStatus?.symbols_monitored || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Data Points:</span>
                <span className="text-white">{data?.learningStatus?.data_points || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Model Accuracy:</span>
                <span className="text-white">
                  {((data?.learningStatus?.model_accuracy || 0) * 100).toFixed(1)}%
                </span>
              </div>
              {!data?.learningStatus?.learning_active && (
                <button
                  onClick={handleStartLearning}
                  className="w-full mt-3 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded"
                >
                  Start Learning
                </button>
              )}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-4">📊 Recent Signals</h3>
            <div className="space-y-2">
              {data?.signals && data.signals.length > 0 ? (
                data.signals.slice(0, 3).map((signal: any, index: number) => (
                  <div key={index} className="flex justify-between items-center py-2 border-b border-gray-700">
                    <span className="text-white">{signal.symbol}</span>
                    <span className={`px-2 py-1 rounded text-sm ${
                      signal.side === 'buy' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                    }`}>
                      {signal.side?.toUpperCase()}
                    </span>
                    <span className="text-gray-400">{(signal.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))
              ) : (
                <p className="text-gray-400">No recent signals</p>
              )}
            </div>
          </div>
        </div>

        {/* Crypto Table */}
        <div className="mb-6">
          <CoinTable />
        </div>

        {/* Additional Components */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-medium mb-4">💬 AI Chat</h3>
            <ChatPanel />
          </div>
          <div>
            <h3 className="text-lg font-medium mb-4">🤖 AI Signals</h3>
            <AISignals />
          </div>
        </div>

        {/* Enhanced Data */}
        <div className="mt-6">
          <EnhancedDataDashboard />
        </div>

      </div>
    </div>
  );
}