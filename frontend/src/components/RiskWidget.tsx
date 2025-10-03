import { useState, useEffect } from 'react';
import { Shield, AlertTriangle, Activity, Zap } from 'lucide-react';

interface RiskMetrics {
  riskScore: number; // 0-100
  volatility: number;
  var95: number; // Value at Risk 95%
  exposure: number;
  leverage: number;
  marginUsed: number;
  liquidationPrice: number | null;
  drawdownRisk: 'LOW' | 'MEDIUM' | 'HIGH';
  correlationRisk: 'LOW' | 'MEDIUM' | 'HIGH';
}

interface RiskWidgetProps {
  symbol?: string;
}

const getRiskColor = (score: number) => {
  if (score <= 30) return 'text-green-600';
  if (score <= 60) return 'text-yellow-600';
  return 'text-red-600';
};

const getRiskBgColor = (score: number) => {
  if (score <= 30) return 'bg-green-100 dark:bg-green-900/20';
  if (score <= 60) return 'bg-yellow-100 dark:bg-yellow-900/20';
  return 'bg-red-100 dark:bg-red-900/20';
};

export default function RiskWidget({ symbol }: RiskWidgetProps) {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate risk data
    const mockData: RiskMetrics = {
      riskScore: 42,
      volatility: 0.35,
      var95: -1250.50,
      exposure: 75.2,
      leverage: 2.5,
      marginUsed: 45.8,
      liquidationPrice: symbol?.includes('BTC') ? 45200 : null,
      drawdownRisk: 'MEDIUM',
      correlationRisk: 'LOW',
    };

    setTimeout(() => {
      setRiskMetrics(mockData);
      setLoading(false);
    }, 400);
  }, [symbol]);

  if (loading || !riskMetrics) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  const riskLevel = riskMetrics.riskScore <= 30 ? 'LOW' : 
                   riskMetrics.riskScore <= 60 ? 'MEDIUM' : 'HIGH';

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Risk Score Header */}
      <div className={`rounded-lg p-4 ${getRiskBgColor(riskMetrics.riskScore)}`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Shield className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span className="font-semibold text-gray-900 dark:text-white">Risk Monitor</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <AlertTriangle className={`w-4 h-4 ${getRiskColor(riskMetrics.riskScore)}`} />
            <span className={`text-sm font-semibold ${getRiskColor(riskMetrics.riskScore)}`}>
              {riskLevel}
            </span>
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">Risk Score</span>
          <span className={`text-2xl font-bold ${getRiskColor(riskMetrics.riskScore)}`}>
            {riskMetrics.riskScore}/100
          </span>
        </div>
        
        {/* Risk Progress Bar */}
        <div className="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 relative overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              riskMetrics.riskScore <= 30 ? 'bg-green-500 risk-progress-30' :
              riskMetrics.riskScore <= 60 ? 'bg-yellow-500 risk-progress-60' : 'bg-red-500 risk-progress-100'
            }`}
          >
            <div className="sr-only">{riskMetrics.riskScore}% risk level</div>
          </div>
        </div>
      </div>

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 flex-1">
        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="flex items-center space-x-2 mb-1">
            <Activity className="w-4 h-4 text-purple-500" />
            <div className="text-xs text-gray-500 dark:text-gray-400">Volatility</div>
          </div>
          <div className="text-lg font-semibold text-gray-900 dark:text-white">
            {(riskMetrics.volatility * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="flex items-center space-x-2 mb-1">
            <Zap className="w-4 h-4 text-blue-500" />
            <div className="text-xs text-gray-500 dark:text-gray-400">Leverage</div>
          </div>
          <div className="text-lg font-semibold text-gray-900 dark:text-white">
            {riskMetrics.leverage}x
          </div>
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">VaR (95%)</div>
          <div className="text-lg font-semibold text-red-600">
            ${Math.abs(riskMetrics.var95).toLocaleString()}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Margin Used</div>
          <div className={`text-lg font-semibold ${
            riskMetrics.marginUsed > 80 ? 'text-red-600' :
            riskMetrics.marginUsed > 60 ? 'text-yellow-600' : 'text-green-600'
          }`}>
            {riskMetrics.marginUsed}%
          </div>
        </div>
      </div>

      {/* Risk Factors */}
      <div className="space-y-2 flex-1 overflow-hidden">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white">Risk Factors</h4>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-800 rounded">
            <span className="text-sm text-gray-700 dark:text-gray-300">Drawdown Risk</span>
            <span className={`text-xs font-semibold px-2 py-1 rounded ${
              riskMetrics.drawdownRisk === 'LOW' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
              riskMetrics.drawdownRisk === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
              'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
            }`}>
              {riskMetrics.drawdownRisk}
            </span>
          </div>

          <div className="flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-800 rounded">
            <span className="text-sm text-gray-700 dark:text-gray-300">Correlation Risk</span>
            <span className={`text-xs font-semibold px-2 py-1 rounded ${
              riskMetrics.correlationRisk === 'LOW' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
              riskMetrics.correlationRisk === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
              'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
            }`}>
              {riskMetrics.correlationRisk}
            </span>
          </div>

          {riskMetrics.liquidationPrice && (
            <div className="flex items-center justify-between py-2 px-3 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
              <span className="text-sm text-red-700 dark:text-red-300">Liquidation Price</span>
              <span className="text-sm font-semibold text-red-600">
                ${riskMetrics.liquidationPrice.toLocaleString()}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}