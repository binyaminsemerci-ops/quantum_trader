import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface PortfolioData {
  total_pnl: number;
  daily_pnl: number;
  exposure: number;
  active_positions: number;
  long_exposure: number;
  short_exposure: number;
  winning_trades: number;
  total_trades: number;
}

export default function Portfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/portfolio/status`);
        const portfolioData = await response.json();
        setData(portfolioData);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load portfolio:', err);
        setLoading(false);
      }
    };

    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 5000);
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

  const winRate = data.total_trades > 0 
    ? (data.winning_trades / data.total_trades * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-green-400">Portfolio Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="Total P&L"
          value={`$${data.total_pnl.toLocaleString()}`}
          subtitle={`Daily: $${data.daily_pnl.toLocaleString()}`}
          color={data.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Active Positions"
          value={data.active_positions.toString()}
          subtitle={`Exposure: ${(data.exposure * 100).toFixed(1)}%`}
          color="text-blue-400"
        />
        
        <InsightCard
          title="Win Rate"
          value={`${winRate}%`}
          subtitle={`${data.winning_trades}/${data.total_trades} trades`}
          color="text-purple-400"
        />
        
        <InsightCard
          title="Net Exposure"
          value={`${(data.exposure * 100).toFixed(1)}%`}
          subtitle={`Long: ${(data.long_exposure * 100).toFixed(1)}% | Short: ${(data.short_exposure * 100).toFixed(1)}%`}
          color="text-yellow-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Long/Short Breakdown</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Long Exposure</span>
                <span className="text-green-400 font-bold">
                  {(data.long_exposure * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-green-500 h-3 rounded-full"
                  style={{ width: `${data.long_exposure * 100}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Short Exposure</span>
                <span className="text-red-400 font-bold">
                  {(data.short_exposure * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-red-500 h-3 rounded-full"
                  style={{ width: `${data.short_exposure * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Performance Metrics</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total P&L:</span>
              <span className={`font-bold ${data.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.total_pnl.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Daily P&L:</span>
              <span className={`font-bold ${data.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.daily_pnl.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Win Rate:</span>
              <span className="text-white font-bold">{winRate}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Trades:</span>
              <span className="text-white font-bold">{data.total_trades}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
