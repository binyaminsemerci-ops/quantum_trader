import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface PortfolioData {
  pnl: number;
  exposure: number;
  drawdown: number;
  positions: number;
}

export default function Portfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/portfolio/status`);
        if (!response.ok) throw new Error('Failed to fetch');
        const portfolioData = await response.json();
        setData(portfolioData);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load portfolio:', err);
        setLoading(false);
      }
    };

    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 10000);
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

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-green-400">Portfolio Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <InsightCard
          title="Portfolio P&L"
          value={`$${data.pnl.toFixed(2)}`}
          subtitle="Current unrealized P&L"
          color={data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Active Positions"
          value={data.positions.toString()}
          subtitle={`Exposure: ${(data.exposure * 100).toFixed(1)}%`}
          color="text-blue-400"
        />
        
        <InsightCard
          title="Drawdown"
          value={`${(data.drawdown * 100).toFixed(2)}%`}
          subtitle="Current drawdown level"
          color={data.drawdown > 0.1 ? 'text-red-400' : 'text-yellow-400'}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Portfolio Metrics</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Exposure</span>
                <span className="text-blue-400 font-bold">
                  {(data.exposure * 100).toFixed(2)}%
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
                <span className="text-gray-400">Drawdown</span>
                <span className={`font-bold ${data.drawdown > 0.1 ? 'text-red-400' : 'text-yellow-400'}`}>
                  {(data.drawdown * 100).toFixed(2)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full transition-all duration-500 ${data.drawdown > 0.1 ? 'bg-red-500' : 'bg-yellow-500'}`}
                  style={{ width: `${Math.min(data.drawdown * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Performance Summary</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Current P&L:</span>
              <span className={`font-bold text-xl ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.pnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Active Positions:</span>
              <span className="text-white font-bold text-xl">{data.positions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Portfolio Exposure:</span>
              <span className="text-blue-400 font-bold">{(data.exposure * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Drawdown:</span>
              <span className={`font-bold ${data.drawdown > 0.1 ? 'text-red-400' : 'text-yellow-400'}`}>
                {(data.drawdown * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
