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

// RLDashboardData interface - RL Intelligence dashboard data structure
// @ts-ignore - Used for type safety
interface RLDashboardData {
  status: string;
  symbols_tracked: number;
  symbols: SymbolData[];
  best_performer: string;
  best_reward: number;
  avg_reward: number;
  message: string;
}

interface RiskData {
  var95: number;
  cvar95: number;
  volatility: number;
  regime: string;
  riskScore: number;
  maxDrawdown: number;
  sharpeRatio: number;
  concentrationRisk: number;
  totalExposure: number;
  positionsAtRisk: number;
}

// Calculate VaR (Value at Risk) at 95% confidence level
// @ts-ignore - Used dynamically
function calculateVaR(returns: number[], confidenceLevel: number = 0.95): number {
  const sorted = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidenceLevel) * sorted.length);
  return sorted[index] || 0;
}

// Calculate CVaR (Conditional Value at Risk) at 95% confidence level
// @ts-ignore - Used dynamically
function calculateCVaR(returns: number[], confidenceLevel: number = 0.95): number {
  const sorted = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidenceLevel) * sorted.length);
  const tail = sorted.slice(0, index + 1);
  return tail.length > 0 ? tail.reduce((sum, val) => sum + val, 0) / tail.length : 0;
}

// @ts-ignore - Used dynamically
// Calculate portfolio volatility (standard deviation of returns)
function calculateVolatility(returns: number[]): number {
  if (returns.length < 2) return 0;
  const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length;
  const variance = returns.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / returns.length;
  return Math.sqrt(variance);
}
// @ts-ignore - Used dynamically

// Determine market regime based on average returns and volatility
function determineMarketRegime(avgReturn: number, volatility: number): string {
  if (avgReturn > 2 && volatility < 15) return 'Bullish';
  if (avgReturn < -2 && volatility > 20) return 'Bearish';
  if (volatility > 25) return 'Volatile';
  return 'Neutral';
}
// @ts-ignore - Used dynamically

// Calculate Sharpe Ratio (simplified, assuming 0% risk-free rate)
function calculateSharpeRatio(returns: number[], volatility: number): number {
  if (volatility === 0) return 0;
  const avgReturn = returns.reduce((sum, val) => sum + val, 0) / returns.length;
  return avgReturn / volatility;
}

export default function Risk() {
  const [data, setData] = useState<RiskData | null>(null);
  const [topRisks, setTopRisks] = useState<SymbolData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRisk = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/risk/metrics`);
        if (!response.ok) throw new Error('Failed to fetch');
        const riskMetrics = await response.json();
        
        // Backend returns: { var, cvar, volatility, regime }
        const var95 = riskMetrics.var ? Math.abs(riskMetrics.var) * 100 : 0;
        const cvar95 = riskMetrics.cvar ? Math.abs(riskMetrics.cvar) * 100 : 0;
        const volatility = riskMetrics.volatility ? riskMetrics.volatility * 100 : 0;
        const regime = riskMetrics.regime || 'Unknown';
        
        // Calculate risk score (0-100 based on volatility)
        const riskScore = Math.min(Math.round(volatility), 100);
        
        // Simplified metrics using backend data
        const maxDrawdown = var95;
        const sharpeRatio = 0; // Would need additional endpoint
        const concentrationRisk = 0; // Would need position breakdown
        const totalExposure = var95;
        const positionsAtRisk = 0; // Would need position breakdown
        
        setData({
          var95,
          cvar95,
          volatility,
          regime,
          riskScore,
          maxDrawdown,
          sharpeRatio,
          concentrationRisk,
          totalExposure,
          positionsAtRisk
        });
        
        // Clear top risks (would need separate endpoint)
        setTopRisks([]);
        
        setLoading(false);
      } catch (err) {
        console.error('Failed to load risk data:', err);
        setLoading(false);
      }
    };

    fetchRisk();
    const interval = setInterval(fetchRisk, 5000); // 5 second refresh
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

  const getRiskLevel = (score: number): { text: string; color: string } => {
    if (score < 30) return { text: 'Low Risk', color: 'text-green-400' };
    if (score < 60) return { text: 'Moderate Risk', color: 'text-yellow-400' };
    return { text: 'High Risk', color: 'text-red-400' };
  };

  const riskLevel = getRiskLevel(data.riskScore);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-yellow-400">Risk Management</h1>
        <div className="text-sm text-gray-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
      
      {/* Main Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="VaR (95%)"
          value={`${data.var95.toFixed(2)}%`}
          subtitle="Value at Risk"
          color="text-yellow-400"
        />
        
        <InsightCard
          title="CVaR (95%)"
          value={`${data.cvar95.toFixed(2)}%`}
          subtitle="Conditional VaR"
          color="text-orange-400"
        />
        
        <InsightCard
          title="Volatility"
          value={`${data.volatility.toFixed(2)}%`}
          subtitle="Portfolio volatility"
          color="text-purple-400"
        />
        
        <InsightCard
          title="Market Regime"
          value={data.regime}
          subtitle="Current market state"
          color={
            data.regime === "Bullish" ? "text-green-400" :
            data.regime === "Bearish" ? "text-red-400" :
            data.regime === "Volatile" ? "text-orange-400" : "text-blue-400"
          }
        />
      </div>

      {/* Risk Score and Market Regime */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Risk Score Gauge */}
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
                      data.riskScore < 30 ? '#10b981' :
                      data.riskScore < 60 ? '#f59e0b' : '#ef4444'
                    }
                    strokeWidth="10"
                    strokeDasharray={`${data.riskScore * 2.51} 251`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className={`text-4xl font-bold ${riskLevel.color}`}>
                    {data.riskScore}
                  </span>
                  <span className="text-sm text-gray-400">out of 100</span>
                </div>
              </div>
            </div>
            <div className={`text-center font-bold text-lg ${riskLevel.color}`}>
              {riskLevel.text}
            </div>
            <div className="space-y-2 pt-4">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Max Drawdown:</span>
                <span className="text-red-400 font-bold">{data.maxDrawdown.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Sharpe Ratio:</span>
                <span className={`font-bold ${data.sharpeRatio > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {data.sharpeRatio.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Positions at Risk:</span>
                <span className="text-orange-400 font-bold">{data.positionsAtRisk}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Market Regime Details */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Market Regime</h2>
          <div className="space-y-4">
            <div className="bg-gray-700 rounded-lg p-6 text-center">
              <div className={`text-3xl font-bold mb-2 ${
                data.regime === "Bullish" ? "text-green-400" :
                data.regime === "Bearish" ? "text-red-400" :
                data.regime === "Volatile" ? "text-orange-400" : "text-blue-400"
              }`}>
                {data.regime.toUpperCase()}
              </div>
              <div className="text-sm text-gray-400">Current Market State</div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">VaR (95%):</span>
                <span className="text-yellow-400 font-bold">{data.var95.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">CVaR (95%):</span>
                <span className="text-orange-400 font-bold">{data.cvar95.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volatility:</span>
                <span className="text-purple-400 font-bold">{data.volatility.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between border-t border-gray-600 pt-2 mt-2">
                <span className="text-gray-400">Concentration Risk:</span>
                <span className={`font-bold ${data.concentrationRisk > 50 ? 'text-red-400' : 'text-green-400'}`}>
                  {data.concentrationRisk.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Total Exposure:</span>
                <span className="text-blue-400 font-bold">${data.totalExposure.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Metrics Summary with Progress Bars */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Risk Metrics Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">VaR (95%)</span>
                <span className="text-white">{data.var95.toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-yellow-500 h-2 rounded-full"
                  style={{ width: `${Math.min(Math.abs(data.var95) * 2, 100)}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">CVaR (95%)</span>
                <span className="text-white">{data.cvar95.toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-orange-500 h-2 rounded-full"
                  style={{ width: `${Math.min(Math.abs(data.cvar95) * 2, 100)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Volatility</span>
                <span className="text-white">{data.volatility.toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full"
                  style={{ width: `${Math.min(data.volatility * 2, 100)}%` }}
                />
              </div>
            </div>
          </div>
          
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Risk Score</span>
                <span className="text-white">{data.riskScore}/100</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    data.riskScore < 30 ? 'bg-green-500' :
                    data.riskScore < 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${data.riskScore}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Concentration Risk</span>
                <span className="text-white">{data.concentrationRisk.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    data.concentrationRisk < 40 ? 'bg-green-500' :
                    data.concentrationRisk < 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${Math.min(data.concentrationRisk, 100)}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Market Regime</span>
                <span className={`font-bold ${
                  data.regime === "Bullish" ? "text-green-400" :
                  data.regime === "Bearish" ? "text-red-400" :
                  data.regime === "Volatile" ? "text-orange-400" : "text-blue-400"
                }`}>
                  {data.regime}
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={
                    data.regime === "Bullish" ? "bg-green-500 h-2 rounded-full" :
                    data.regime === "Bearish" ? "bg-red-500 h-2 rounded-full" :
                    data.regime === "Volatile" ? "bg-orange-500 h-2 rounded-full" : "bg-blue-500 h-2 rounded-full"
                  }
                  style={{ width: `${
                    data.regime === "Bullish" ? 75 :
                    data.regime === "Bearish" ? 25 :
                    data.regime === "Volatile" ? 90 : 50
                  }%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top 5 Risky Positions */}
      {topRisks.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Positions at Highest Risk</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Symbol</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-semibold">Reward %</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-semibold">Total P&L</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-semibold">Unrealized</th>
                  <th className="text-center py-3 px-4 text-gray-400 font-semibold">Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {topRisks.map((symbol) => {
                  const riskSeverity = Math.abs(symbol.reward);
                  const level = riskSeverity > 50 ? 'Critical' : riskSeverity > 20 ? 'High' : 'Moderate';
                  const levelColor = riskSeverity > 50 ? 'bg-red-500/20 text-red-400' : 
                                    riskSeverity > 20 ? 'bg-orange-500/20 text-orange-400' : 
                                    'bg-yellow-500/20 text-yellow-400';
                  
                  return (
                    <tr key={symbol.symbol} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                      <td className="py-3 px-4 text-white font-medium">{symbol.symbol}</td>
                      <td className="py-3 px-4 text-right font-bold text-red-400">
                        {symbol.reward.toFixed(2)}%
                      </td>
                      <td className="py-3 px-4 text-right font-bold text-red-400">
                        ${symbol.total_pnl.toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-right text-red-400">
                        ${symbol.unrealized_pnl.toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${levelColor}`}>
                          {level}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
