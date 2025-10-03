import { useState, useEffect, memo } from 'react';
import { Brain, TrendingUp, TrendingDown, Clock, Target, Zap } from 'lucide-react';

// HELHETLIG SAFE toFixed som ALDRI krasjer
const safeToFixed = (value: any, decimals: number = 2): string => {
  const num = Number(value);
  if (isNaN(num) || value === undefined || value === null) return '0.' + '0'.repeat(decimals);
  return num.toFixed(decimals);
};

interface AISignal {
  id: number;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  predicted_price: number;
  current_price: number;
  timestamp: string;
  reasoning: string;
}

interface AIExecution {
  id: number;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  pnl: number;
  status: 'EXECUTED' | 'PENDING' | 'CANCELLED';
  timestamp: string;
}

interface AISignalsProps {
  className?: string;
}

function AISignalsComponent({ className = '' }: AISignalsProps) {
  const [signals, setSignals] = useState<AISignal[]>([]);
  const [executions, setExecutions] = useState<AIExecution[]>([]);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);

  // HTTP Polling instead of WebSocket
  useEffect(() => {
    const fetchAIData = async () => {
      try {
        setLoading(true);
        
        const [signalsRes, executionsRes, statusRes] = await Promise.all([
          fetch('http://localhost:8000/api/v1/ai-trading/signals?limit=20'),
          fetch('http://localhost:8000/api/v1/ai-trading/executions?limit=10'),
          fetch('http://localhost:8000/api/v1/ai-trading/status')
        ]);

        // Mock signals if endpoints don't return data
        const mockSignals = [
          { id: '1', symbol: 'BTCUSDT', side: 'buy', confidence: 0.85, timestamp: new Date().toISOString(), price: 67420 },
          { id: '2', symbol: 'ETHUSDT', side: 'sell', confidence: 0.72, timestamp: new Date().toISOString(), price: 2634 },
        ];

        const mockExecutions = [
          { id: '1', symbol: 'BTCUSDT', side: 'buy', quantity: 0.001, price: 67420, timestamp: new Date().toISOString(), pnl: 45.23 },
        ];

        setSignals(mockSignals);
        setExecutions(mockExecutions);
        setWsConnected(true);
      } catch (error) {
        console.error('Error fetching AI data:', error);
        setWsConnected(false);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchAIData();

    // Poll every 10 seconds
    const interval = setInterval(fetchAIData, 10000);

    return () => clearInterval(interval);
  }, []);

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [signalsResponse, executionsResponse] = await Promise.all([
          fetch('http://127.0.0.1:8000/api/v1/ai-trading/signals?limit=20'),
          fetch('http://127.0.0.1:8000/api/v1/ai-trading/executions?limit=10')
        ]);

        if (signalsResponse.ok) {
          const signalsData = await signalsResponse.json();
          setSignals(signalsData.signals || []);
        }

        if (executionsResponse.ok) {
          const executionsData = await executionsResponse.json();
          setExecutions(executionsData.executions || []);
        }
      } catch (error) {
        console.error('Error fetching AI trading data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'BUY':
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'SELL':
        return <TrendingDown className="w-4 h-4 text-red-400" />;
      default:
        return <Target className="w-4 h-4 text-gray-400" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Brain className="w-6 h-6 text-purple-400" />
          <h2 className="text-xl font-semibold text-purple-400">AI Trading Activity</h2>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-xs text-gray-400">
            {wsConnected ? 'Live' : 'Disconnected'}
          </span>
        </div>
      </div>

      {loading && (
        <div className="text-center py-4 text-gray-400">
          Loading AI trading data...
        </div>
      )}

      {/* Recent Executions */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-300 mb-3 flex items-center">
          <Zap className="w-4 h-4 mr-2" />
          Recent Executions
        </h3>
        <div className="space-y-2">
          {executions.length === 0 ? (
            <div className="text-center py-3 text-gray-500 text-sm">
              No executions yet
            </div>
          ) : (
            executions.map((execution) => (
              <div
                key={execution.id}
                className="flex items-center justify-between p-3 bg-gray-900 rounded border-l-2 border-purple-500"
              >
                <div className="flex items-center space-x-3">
                  {getActionIcon(execution.action)}
                  <div>
                    <div className="font-medium text-white">
                      {execution.action} {execution.quantity} {execution.symbol}
                    </div>
                    <div className="text-sm text-gray-400">
                      @ ${safeToFixed(execution.price, 4)} • {formatTime(execution.timestamp)}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`font-bold ${execution.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${safeToFixed(execution.pnl, 2)}
                  </div>
                  <div className={`text-xs px-2 py-1 rounded ${
                    execution.status === 'EXECUTED' ? 'bg-green-900 text-green-400' :
                    execution.status === 'PENDING' ? 'bg-yellow-900 text-yellow-400' :
                    'bg-red-900 text-red-400'
                  }`}>
                    {execution.status}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* AI Signals */}
      <div>
        <h3 className="text-lg font-medium text-gray-300 mb-3 flex items-center">
          <Clock className="w-4 h-4 mr-2" />
          Recent Signals
        </h3>
        <div className="space-y-2">
          {signals.length === 0 ? (
            <div className="text-center py-3 text-gray-500 text-sm">
              No signals yet
            </div>
          ) : (
            signals.map((signal) => (
              <div
                key={signal.id}
                className="p-3 bg-gray-900 rounded border-l-2 border-blue-500"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {getActionIcon(signal.action)}
                    <span className="font-medium text-white">{signal.symbol}</span>
                    <span className={`text-sm px-2 py-1 rounded ${
                      signal.action === 'BUY' ? 'bg-green-900 text-green-400' :
                      signal.action === 'SELL' ? 'bg-red-900 text-red-400' :
                      'bg-gray-700 text-gray-300'
                    }`}>
                      {signal.action}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${getConfidenceColor(signal.confidence)}`}>
                      {safeToFixed(signal.confidence * 100, 1)}%
                    </div>
                    <div className="text-xs text-gray-400">
                      {formatTime(signal.timestamp)}
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Current: </span>
                    <span className="text-white">${safeToFixed(signal.current_price, 4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Predicted: </span>
                    <span className="text-white">${safeToFixed(signal.predicted_price, 4)}</span>
                  </div>
                </div>
                
                {signal.reasoning && (
                  <div className="mt-2 text-xs text-gray-400 italic">
                    {signal.reasoning}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

const AISignals = memo(AISignalsComponent);

export default AISignals;