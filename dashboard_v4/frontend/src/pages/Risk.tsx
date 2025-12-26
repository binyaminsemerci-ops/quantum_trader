import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface RiskData {
  var_95: number;
  cvar_95: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  market_regime: string;
  risk_score: number;
}

export default function Risk() {
  const [data, setData] = useState<RiskData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRisk = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/risk/metrics`);
        const riskData = await response.json();
        setData(riskData);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load risk data:', err);
        setLoading(false);
      }
    };

    fetchRisk();
    const interval = setInterval(fetchRisk, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading risk metrics...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load risk data</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-yellow-400">Risk Management</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="VaR (95%)"
          value={`${((data?.var_95 ?? 0) * 100).toFixed(2)}%`}
          subtitle="Value at Risk"
          color="text-yellow-400"
        />
        
        <InsightCard
          title="CVaR (95%)"
          value={`${((data?.cvar_95 ?? 0) * 100).toFixed(2)}%`}
          subtitle="Conditional VaR"
          color="text-yellow-400"
        />
        
        <InsightCard
          title="Volatility"
          value={`${((data?.volatility ?? 0) * 100).toFixed(2)}%`}
          subtitle="Portfolio volatility"
          color="text-purple-400"
        />
        
        <InsightCard
          title="Max Drawdown"
          value={`${((data?.max_drawdown ?? 0) * 100).toFixed(2)}%`}
          subtitle="Largest peak-to-trough"
          color="text-red-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Risk Score</h2>
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className="relative w-48 h-48">
                <svg className="transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke="#374151"
                    strokeWidth="10"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke={
                      data.risk_score < 30 ? '#10b981' :
                      data.risk_score < 70 ? '#f59e0b' : '#ef4444'
                    }
                    strokeWidth="10"
                    strokeDasharray={`${data.risk_score * 2.51} 251`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-4xl font-bold text-white">{data.risk_score}</span>
                  <span className="text-sm text-gray-400">Risk Score</span>
                </div>
              </div>
            </div>
            <div className="text-center text-gray-400 text-sm">
              {data.risk_score < 30 ? 'Low Risk' : 
               data.risk_score < 70 ? 'Moderate Risk' : 'High Risk'}
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Market Regime</h2>
          <div className="space-y-4">
            <div className="bg-gray-700 rounded-lg p-6 text-center">
              <div className="text-3xl font-bold text-blue-400 mb-2">
                {data?.market_regime ?? 'UNKNOWN'}
              </div>
              <div className="text-sm text-gray-400">Current Market State</div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Sharpe Ratio:</span>
                <span className="text-white font-bold">{(data?.sharpe_ratio ?? 0).toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volatility:</span>
                <span className="text-white font-bold">{((data?.volatility ?? 0) * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Risk Metrics Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">VaR (95%)</span>
                <span className="text-white">{((data?.var_95 ?? 0) * 100).toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-yellow-500 h-2 rounded-full"
                  style={{ width: `${Math.min(((data?.var_95 ?? 0) * 100 * 10), 100)}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">CVaR (95%)</span>
                <span className="text-white">{(data.cvar_95 * 100).toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-orange-500 h-2 rounded-full"
                  style={{ width: `${Math.min(data.cvar_95 * 100 * 10, 100)}%` }}
                />
              </div>
            </div>
          </div>
          
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Volatility</span>
                <span className="text-white">{(data.volatility * 100).toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full"
                  style={{ width: `${Math.min(data.volatility * 100 * 5, 100)}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Max Drawdown</span>
                <span className="text-white">{(data.max_drawdown * 100).toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-red-500 h-2 rounded-full"
                  style={{ width: `${Math.min(data.max_drawdown * 100 * 5, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
