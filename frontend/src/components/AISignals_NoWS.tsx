import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Target } from 'lucide-react';

// HELHETLIG SAFE toFixed som ALDRI krasjer  
const safeToFixed = (value: any, decimals: number = 2): string => {
  const num = Number(value);
  if (isNaN(num) || value === undefined || value === null) return '0.' + '0'.repeat(decimals);
  return num.toFixed(decimals);
};

interface AISignal {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  confidence: number;
  timestamp: string;
  price: number;
  action: string;
  predicted_price: number;
  current_price: number;
  reasoning: string;
}

interface AIExecution {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  pnl: number;
  action: string;
  status: string;
}

interface AISignalsProps {
  className?: string;
}

function AISignalsComponent({ className = '' }: AISignalsProps) {
  const [signals, setSignals] = useState<AISignal[]>([]);
  const [executions, setExecutions] = useState<AIExecution[]>([]);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);

  // HTTP Polling instead of WebSocket
  useEffect(() => {
    const fetchAIData = async () => {
      try {
        setLoading(true);
        
        // Fetch real AI signals from backend
        const signalsResponse = await fetch('http://localhost:8000/signals/recent?limit=5');
        const signalsData = await signalsResponse.json();
        
        // Transform backend signals to frontend format
        const transformedSignals: AISignal[] = signalsData.map((signal: any) => ({
          id: signal.id,
          symbol: signal.symbol,
          side: signal.side,
          confidence: signal.confidence || 0,
          timestamp: signal.timestamp,
          price: 0, // Backend doesn't provide price in this format
          action: signal.side === 'buy' ? 'Buy Signal' : 'Sell Signal',
          predicted_price: 0, // Backend doesn't provide predicted price
          current_price: 0, // Will be updated below
          reasoning: signal.details?.note || 'AI generated signal'
        }));

        // Fetch current prices for each symbol
        for (const signal of transformedSignals) {
          try {
            const priceResponse = await fetch(`http://localhost:8000/watchlist/prices?symbols=${signal.symbol}&limit=1`);
            const priceData = await priceResponse.json();
            if (priceData.length > 0 && priceData[0].price) {
              signal.current_price = priceData[0].price;
              signal.price = priceData[0].price;
              // Add some predicted price based on signal direction
              signal.predicted_price = signal.side === 'buy' 
                ? signal.price * 1.02 
                : signal.price * 0.98;
            }
          } catch (priceError) {
            console.log(`Could not fetch price for ${signal.symbol}`);
          }
        }

        setSignals(transformedSignals);
        
        // Mock executions for now (can be replaced with real data later)
        const mockExecutions: AIExecution[] = [];
        setExecutions(mockExecutions);
        setConnected(true);
      } catch (error) {
        console.error('Error fetching AI data:', error);
        setConnected(false);
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

  if (loading && signals.length === 0) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <h3 className="text-lg font-semibold text-white mb-4">🤖 AI Trading Signals</h3>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-white">🤖 AI Trading Signals</h3>
        <div className={`px-2 py-1 rounded text-sm ${
          connected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
        }`}>
          {connected ? 'Active' : 'Disconnected'}
        </div>
      </div>

      <div className="space-y-4">
        {/* Recent Signals */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">Recent Signals</h4>
          <div className="space-y-2">
            {signals.length > 0 ? (
              signals.slice(0, 3).map((signal) => (
                <div key={signal.id} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-white font-medium">{signal.symbol}</span>
                      <div className={`flex items-center space-x-1 ${
                        signal.side === 'buy' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {signal.side === 'buy' ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                        <span className="text-sm uppercase">{signal.side}</span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-1 text-yellow-400">
                      <Target size={16} />
                      <span className="text-sm">{safeToFixed(signal.confidence * 100, 0)}%</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-400">
                    <div>Price: ${signal.current_price.toLocaleString()}</div>
                    <div>Target: ${signal.predicted_price.toLocaleString()}</div>
                    <div>{signal.reasoning}</div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-400 text-sm">No signals available</p>
            )}
          </div>
        </div>

        {/* Recent Executions */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">Recent Executions</h4>
          <div className="space-y-2">
            {executions.length > 0 ? (
              executions.slice(0, 2).map((execution) => (
                <div key={execution.id} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-white font-medium">{execution.symbol}</span>
                    <span className={`text-sm ${
                      execution.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {execution.pnl >= 0 ? '+' : ''}${safeToFixed(execution.pnl, 2)}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    <div>{execution.action} • Qty: {execution.quantity} • Price: ${execution.price}</div>
                    <div>Status: {execution.status}</div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-400 text-sm">No executions available</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AISignalsComponent;