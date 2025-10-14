import { useState, useEffect } from 'react';

interface AISignal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: string;
  reason: string;
}

interface AITrade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  status: 'executed' | 'pending' | 'cancelled';
  pnl?: number;
}

export default function AITradingMonitor(): JSX.Element {
  const [aiStatus, setAiStatus] = useState({
    isActive: true,
    mode: 'aggressive',
    performance: 15.8,
    todayTrades: 24,
    successRate: 73.5,
    activeBots: 3
  });

  const [aiSignals, setAiSignals] = useState<AISignal[]>([
    {
      id: '1',
      symbol: 'BTCUSDT',
      action: 'BUY',
      confidence: 87.5,
      price: 67450.00,
      timestamp: '2025-10-04T20:15:30Z',
      reason: 'Strong bullish momentum + RSI oversold'
    },
    {
      id: '2', 
      symbol: 'ETHUSDT',
      action: 'SELL',
      confidence: 92.3,
      price: 2680.50,
      timestamp: '2025-10-04T20:12:15Z',
      reason: 'Resistance level reached + Volume declining'
    },
    {
      id: '3',
      symbol: 'ADAUSDT', 
      action: 'HOLD',
      confidence: 65.0,
      price: 0.3845,
      timestamp: '2025-10-04T20:10:00Z',
      reason: 'Sideways trend - waiting for breakout'
    }
  ]);

  const [recentTrades] = useState<AITrade[]>([
    {
      id: '1',
      symbol: 'BTCUSDT',
      side: 'buy',
      quantity: 0.15,
      price: 67200.00,
      timestamp: '2025-10-04T19:45:30Z',
      status: 'executed',
      pnl: 250.00
    },
    {
      id: '2',
      symbol: 'ETHUSDT', 
      side: 'sell',
      quantity: 5.0,
      price: 2695.00,
      timestamp: '2025-10-04T19:30:15Z',
      status: 'executed',
      pnl: 145.50
    }
  ]);

  // Simulate real-time AI updates
  useEffect(() => {
    const interval = setInterval(() => {
      setAiStatus(prev => ({
        ...prev,
        performance: prev.performance + (Math.random() - 0.5) * 2,
        todayTrades: prev.todayTrades + (Math.random() > 0.8 ? 1 : 0),
        successRate: Math.max(50, Math.min(95, prev.successRate + (Math.random() - 0.5) * 3))
      }));

      // Occasionally add new signals
      if (Math.random() > 0.9) {
        const symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'BNBUSDT'];
        const actions: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
        const newSignal: AISignal = {
          id: Date.now().toString(),
          symbol: symbols[Math.floor(Math.random() * symbols.length)],
          action: actions[Math.floor(Math.random() * actions.length)],
          confidence: 60 + Math.random() * 35,
          price: 100 + Math.random() * 50000,
          timestamp: new Date().toISOString(),
          reason: 'AI detected pattern opportunity'
        };
        setAiSignals(prev => [newSignal, ...prev.slice(0, 4)]);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white flex items-center">
          🤖 AI Trading Monitor
        </h3>
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${aiStatus.isActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
          <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
            {aiStatus.isActive ? 'Active' : 'Inactive'} • {aiStatus.mode}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-br from-green-500 to-green-600 p-3 rounded-lg text-white">
          <div className="text-sm opacity-90">Today Performance</div>
          <div className="text-xl font-bold">+{aiStatus.performance.toFixed(1)}%</div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-3 rounded-lg text-white">
          <div className="text-sm opacity-90">Today Trades</div>
          <div className="text-xl font-bold">{aiStatus.todayTrades}</div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-3 rounded-lg text-white">
          <div className="text-sm opacity-90">Success Rate</div>
          <div className="text-xl font-bold">{aiStatus.successRate.toFixed(1)}%</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 p-3 rounded-lg text-white">
          <div className="text-sm opacity-90">Active Bots</div>
          <div className="text-xl font-bold">{aiStatus.activeBots}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* AI Signals */}
        <div>
          <h4 className="text-md font-semibold mb-3 text-gray-700 dark:text-gray-300">🎯 AI Signals</h4>
          <div className="space-y-3">
            {aiSignals.map((signal) => (
              <div key={signal.id} className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-800 dark:text-white">{signal.symbol}</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      signal.action === 'BUY' ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200' :
                      signal.action === 'SELL' ? 'bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200' :
                      'bg-yellow-100 text-yellow-700 dark:bg-yellow-800 dark:text-yellow-200'
                    }`}>
                      {signal.action}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {signal.confidence.toFixed(1)}% confidence
                  </div>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  ${signal.price.toFixed(2)} • {signal.reason}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent AI Trades */}
        <div>
          <h4 className="text-md font-semibold mb-3 text-gray-700 dark:text-gray-300">📊 Recent AI Trades</h4>
          <div className="space-y-3">
            {recentTrades.map((trade) => (
              <div key={trade.id} className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-800 dark:text-white">{trade.symbol}</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      trade.side === 'buy' ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200' :
                      'bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200'
                    }`}>
                      {trade.side.toUpperCase()}
                    </span>
                  </div>
                  {trade.pnl && (
                    <div className={`text-sm font-medium ${
                      trade.pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    }`}>
                      {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                    </div>
                  )}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {trade.quantity} @ ${trade.price.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}