import { useState, useEffect } from 'react';
import { useLivePrices } from '../services/priceService';

interface PnLData {
  totalPnL: number;
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;
  unrealizedPnL: number;
  realizedPnL: number;
  bestDay: number;
  worstDay: number;
  winningTrades: number;
  losingTrades: number;
  avgWinSize: number;
  avgLossSize: number;
}

interface TradePosition {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  duration: string;
}

export default function PnLTracker(): JSX.Element {
  const { getPrice } = useLivePrices();
  
  const [pnlData, setPnlData] = useState<PnLData>({
    totalPnL: 15750.25,
    dailyPnL: 420.15,
    weeklyPnL: 2890.40,
    monthlyPnL: 11250.75,
    unrealizedPnL: 1285.60,
    realizedPnL: 14464.65,
    bestDay: 1240.85,
    worstDay: -456.30,
    winningTrades: 244,
    losingTrades: 98,
    avgWinSize: 285.75,
    avgLossSize: -122.30
  });

  // Generate positions with live prices
  const generatePositions = (): TradePosition[] => {
    const btcPrice = getPrice('BTCUSDT') || 42741.13;
    const ethPrice = getPrice('ETHUSDT') || 2833.30;
    const solPrice = getPrice('SOLUSDT') || 95.09;
    const avaxPrice = getPrice('AVAXUSDT') || 25.67;
    
    const positions = [
      {
        symbol: 'BTCUSDT',
        side: 'long' as const,
        size: 1.25,
        entryPrice: 41800.00,
        currentPrice: btcPrice,
        duration: '3h 42m'
      },
      {
        symbol: 'ETHUSDT',
        side: 'long' as const,
        size: 8.5,
        entryPrice: 2680.00,
        currentPrice: ethPrice,
        duration: '1h 58m'
      },
      {
        symbol: 'SOLUSDT',
        side: 'short' as const,
        size: 45.0,
        entryPrice: 98.50,
        currentPrice: solPrice,
        duration: '52m'
      },
      {
        symbol: 'AVAXUSDT',
        side: 'long' as const,
        size: 25.0,
        entryPrice: 24.80,
        currentPrice: avaxPrice,
        duration: '2h 18m'
      }
    ];

    return positions.map(position => {
      const pnl = position.side === 'long' 
        ? (position.currentPrice - position.entryPrice) * position.size
        : (position.entryPrice - position.currentPrice) * position.size;
      const pnlPercent = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100 * (position.side === 'long' ? 1 : -1);

      return {
        ...position,
        pnl,
        pnlPercent
      };
    });
  };

  const [openPositions, setOpenPositions] = useState<TradePosition[]>(generatePositions());

  // Simulate real-time P&L updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update main P&L data
      setPnlData(prev => ({
        ...prev,
        dailyPnL: prev.dailyPnL + (Math.random() - 0.5) * 10,
        unrealizedPnL: prev.unrealizedPnL + (Math.random() - 0.5) * 20,
        totalPnL: prev.realizedPnL + prev.unrealizedPnL
      }));

      // Update positions with latest live prices
      setOpenPositions(generatePositions());
    }, 5000);

    return () => clearInterval(interval);
  }, [getPrice]);

  const PnLCard = ({ 
    title, 
    value, 
    period, 
    icon, 
    color = 'blue' 
  }: {
    title: string;
    value: number;
    period?: string;
    icon: string;
    color?: 'blue' | 'green' | 'red' | 'purple' | 'orange';
  }) => {
    const isPositive = value >= 0;
    const colorClasses = {
      blue: 'bg-gradient-to-br from-blue-500 to-blue-600',
      green: 'bg-gradient-to-br from-green-500 to-green-600',
      red: 'bg-gradient-to-br from-red-500 to-red-600',
      purple: 'bg-gradient-to-br from-purple-500 to-purple-600',
      orange: 'bg-gradient-to-br from-orange-500 to-orange-600'
    };

    return (
      <div className={`${colorClasses[color]} p-4 rounded-lg text-white shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-1`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-2xl">{icon}</span>
          <span className={`text-lg ${isPositive ? '📈' : '📉'}`}>
            {isPositive ? '📈' : '📉'}
          </span>
        </div>
        <div className="text-sm opacity-90 mb-1">{title}</div>
        <div className="text-xl font-bold">
          {isPositive ? '+' : ''}${Math.abs(value).toFixed(2)}
        </div>
        {period && (
          <div className="text-xs opacity-75 mt-1">{period}</div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* P&L Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
            📊 P&L Tracker
          </h3>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-500 dark:text-gray-400">Live</span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <PnLCard
            title="Total P&L"
            value={pnlData.totalPnL}
            icon="💰"
            color={pnlData.totalPnL >= 0 ? 'green' : 'red'}
          />
          <PnLCard
            title="Daily P&L"
            value={pnlData.dailyPnL}
            period="Today"
            icon="📅"
            color={pnlData.dailyPnL >= 0 ? 'blue' : 'red'}
          />
          <PnLCard
            title="Unrealized P&L"
            value={pnlData.unrealizedPnL}
            period="Open Positions"
            icon="⏳"
            color={pnlData.unrealizedPnL >= 0 ? 'purple' : 'red'}
          />
          <PnLCard
            title="Win Rate"
            value={(pnlData.winningTrades / (pnlData.winningTrades + pnlData.losingTrades)) * 100}
            period={`${pnlData.winningTrades}W/${pnlData.losingTrades}L`}
            icon="🎯"
            color="orange"
          />
        </div>

        {/* Period Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-3">Weekly P&L</h4>
            <div className={`text-2xl font-bold ${pnlData.weeklyPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {pnlData.weeklyPnL >= 0 ? '+' : ''}${pnlData.weeklyPnL.toFixed(2)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-3">Monthly P&L</h4>
            <div className={`text-2xl font-bold ${pnlData.monthlyPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {pnlData.monthlyPnL >= 0 ? '+' : ''}${pnlData.monthlyPnL.toFixed(2)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-3">Best/Worst Day</h4>
            <div className="text-sm text-green-600">Best: +${pnlData.bestDay.toFixed(2)}</div>
            <div className="text-sm text-red-600">Worst: ${pnlData.worstDay.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Open Positions */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
          🎯 Open Positions ({openPositions.length})
        </h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b dark:border-gray-700">
                <th className="text-left py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Symbol</th>
                <th className="text-center py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Side</th>
                <th className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Size</th>
                <th className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Entry</th>
                <th className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Current</th>
                <th className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">P&L</th>
                <th className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Duration</th>
              </tr>
            </thead>
            <tbody>
              {openPositions.map((position, index) => (
                <tr key={index} className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750">
                  <td className="py-3 px-2 font-semibold text-gray-800 dark:text-white">
                    {position.symbol.replace('USDT', '/USDT')}
                  </td>
                  <td className="py-3 px-2 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      position.side === 'long' 
                        ? 'bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200'
                        : 'bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200'
                    }`}>
                      {position.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3 px-2 text-right text-gray-600 dark:text-gray-300">
                    {position.size}
                  </td>
                  <td className="py-3 px-2 text-right text-gray-600 dark:text-gray-300">
                    ${position.entryPrice.toFixed(2)}
                  </td>
                  <td className="py-3 px-2 text-right font-medium text-gray-800 dark:text-white">
                    ${position.currentPrice.toFixed(2)}
                  </td>
                  <td className="py-3 px-2 text-right">
                    <div className={`font-semibold ${
                      position.pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                    }`}>
                      {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                    </div>
                    <div className={`text-xs ${
                      position.pnlPercent >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ({position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%)
                    </div>
                  </td>
                  <td className="py-3 px-2 text-right text-gray-500 dark:text-gray-400 text-sm">
                    {position.duration}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}