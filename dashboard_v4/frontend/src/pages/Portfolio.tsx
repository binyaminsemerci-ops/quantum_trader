import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = '/api';

interface SymbolData {
  symbol: string;
  reward: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_pnl: number;
  unrealized_pct: number;
  realized_pct: number;
  realized_trades: number;
  status: string;
}

interface PortfolioData {
  totalPnl: number;
  unrealizedPnl: number;
  realizedPnl: number;
  activePositions: number;
  exposure: number;
  maxDrawdown: number;
  winningPositions: number;
  losingPositions: number;
  bestPerformer: string;
  worstPerformer: string;
  avgReward: number;
}

export default function Portfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [symbols, setSymbols] = useState<SymbolData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        // Fetch from backend API instead of HTML page
        const response = await fetch(`${API_BASE_URL}/portfolio/status`);
        if (!response.ok) throw new Error('Failed to fetch');
        const portfolioData = await response.json();
        
        // portfolioData structure: { pnl, exposure, drawdown, positions }
        // Map backend data to frontend PortfolioData structure
        setData({
          totalPnl: portfolioData.pnl || 0,
          unrealizedPnl: portfolioData.pnl || 0, // Backend returns combined PnL
          realizedPnl: 0, // Not available from backend
          activePositions: portfolioData.positions || 0,
          exposure: portfolioData.exposure || 0,
          maxDrawdown: portfolioData.drawdown || 0,
          winningPositions: 0, // Not available - would need positions endpoint
          losingPositions: 0,
          bestPerformer: 'N/A',
          worstPerformer: 'N/A',
          avgReward: 0
        });
        
        // Clear symbols since backend doesn't provide per-symbol breakdown
        setSymbols([]);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load portfolio:', err);
        setLoading(false);
      }
    };

    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 5000); // 5 second refresh
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading portfolio data...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load portfolio data</div>
      </div>
    );
  }

  const winRate = data.activePositions > 0 
    ? ((data.winningPositions / data.activePositions) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-green-400">Portfolio Overview</h1>
        <div className="text-sm text-gray-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
      
      {/* Main Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="Total Portfolio P&L"
          value={`$${data.totalPnl.toFixed(2)}`}
          subtitle="Combined realized + unrealized"
          color={data.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Unrealized P&L"
          value={`$${data.unrealizedPnl.toFixed(2)}`}
          subtitle="Open positions"
          color={data.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Active Positions"
          value={data.activePositions.toString()}
          subtitle={`Exposure: ${(data.exposure * 100).toFixed(1)}%`}
          color="text-blue-400"
        />
        
        <InsightCard
          title="Max Drawdown"
          value={`${(data.maxDrawdown * 100).toFixed(2)}%`}
          subtitle="Largest loss from peak"
          color={data.maxDrawdown > 0.1 ? 'text-red-400' : 'text-yellow-400'}
        />
      </div>

      {/* Portfolio Performance Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Performance Metrics */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Performance Metrics</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Portfolio Exposure</span>
                <span className="text-blue-400 font-bold">
                  {(data.exposure * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(data.exposure * 100, 100)}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Max Drawdown</span>
                <span className={`font-bold ${data.maxDrawdown > 0.1 ? 'text-red-400' : 'text-yellow-400'}`}>
                  {(data.maxDrawdown * 100).toFixed(2)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full transition-all duration-500 ${data.maxDrawdown > 0.1 ? 'bg-red-500' : 'bg-yellow-500'}`}
                  style={{ width: `${Math.min(data.maxDrawdown * 100, 100)}%` }}
                />
              </div>
            </div>
            
            <div className="pt-4 space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate:</span>
                <span className="text-green-400 font-bold">{winRate}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Winning Positions:</span>
                <span className="text-green-400 font-bold">{data.winningPositions}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Losing Positions:</span>
                <span className="text-red-400 font-bold">{data.losingPositions}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Avg Reward:</span>
                <span className={`font-bold ${data.avgReward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {data.avgReward.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Summary */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Portfolio Summary</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total P&L:</span>
              <span className={`font-bold text-xl ${data.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.totalPnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between border-t border-gray-700 pt-2">
              <span className="text-gray-400 text-sm">Unrealized:</span>
              <span className={`font-semibold ${data.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.unrealizedPnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400 text-sm">Realized (24h):</span>
              <span className={`font-semibold ${data.realizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.realizedPnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between border-t border-gray-700 pt-3 mt-3">
              <span className="text-gray-400">Active Positions:</span>
              <span className="text-white font-bold text-xl">{data.activePositions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Portfolio Exposure:</span>
              <span className="text-blue-400 font-bold">{(data.exposure * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between border-t border-gray-700 pt-3 mt-3">
              <span className="text-gray-400">Best Performer:</span>
              <span className="text-green-400 font-bold">{data.bestPerformer}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Worst Performer:</span>
              <span className="text-red-400 font-bold">{data.worstPerformer}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Position Details Table */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Position Details</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Symbol</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Total P&L</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Unrealized</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Realized</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Reward %</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {symbols.slice(0, 15).map((symbol) => (
                <tr key={symbol.symbol} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                  <td className="py-3 px-4 text-white font-medium">{symbol.symbol}</td>
                  <td className={`py-3 px-4 text-right font-bold ${symbol.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${symbol.total_pnl.toFixed(2)}
                  </td>
                  <td className={`py-3 px-4 text-right ${symbol.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${symbol.unrealized_pnl.toFixed(2)}
                  </td>
                  <td className={`py-3 px-4 text-right ${symbol.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${symbol.realized_pnl.toFixed(2)}
                  </td>
                  <td className={`py-3 px-4 text-right font-semibold ${symbol.reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {symbol.reward.toFixed(2)}%
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      symbol.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-gray-600 text-gray-400'
                    }`}>
                      {symbol.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {symbols.length > 15 && (
            <div className="text-center py-4 text-gray-400 text-sm">
              Showing top 15 of {symbols.length} positions
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
