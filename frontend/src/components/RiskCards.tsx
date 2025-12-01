import { useMemo } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';

export default function RiskCards(): JSX.Element {
  const { data } = useDashboardData();
  const riskMetrics = useMemo(() => {
    const r: any = data?.stats?.risk || {};
    return {
      maxDrawdown: r.max_drawdown,
      currentDrawdown: r.current_drawdown,
      portfolioVaR: r.max_trade_exposure,
      leverage: r.leverage,
      marginUsed: r.margin_used,
      riskScore: r.risk_score,
      volatility: r.volatility
    };
  }, [data]);

  const RiskCard = ({ 
    icon, 
    title, 
    value, 
    unit = '', 
    status, 
    description 
  }: {
    icon: string;
    title: string;
    value: number | string;
    unit?: string;
    status: 'safe' | 'warning' | 'danger';
    description: string;
  }) => {
    const statusColors = {
      safe: 'border-green-200 dark:border-green-700 bg-green-50 dark:bg-green-900/20',
      warning: 'border-yellow-200 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20',
      danger: 'border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/20'
    };

    const statusDots = {
      safe: 'bg-green-500',
      warning: 'bg-yellow-500', 
      danger: 'bg-red-500'
    };

    return (
      <div className={`p-4 rounded-lg border-2 ${statusColors[status]} hover:shadow-md transition-all`}>
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-lg">{icon}</span>
            <div className={`w-2 h-2 rounded-full ${statusDots[status]} animate-pulse`}></div>
          </div>
          <div className="text-right">
            <div className="text-lg font-bold text-gray-800 dark:text-white">
              {typeof value === 'number' ? value.toFixed(1) : value}{unit}
            </div>
          </div>
        </div>
        <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">{title}</div>
        <div className="text-xs text-gray-500 dark:text-gray-400">{description}</div>
      </div>
    );
  };

  const getRiskStatus = (value: number, thresholds: [number, number]): 'safe' | 'warning' | 'danger' => {
    if (value <= thresholds[0]) return 'safe';
    if (value <= thresholds[1]) return 'warning';
    return 'danger';
  };

  return (
    <div className="space-y-4">
      {/* Risk Overview */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 rounded-lg text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium opacity-90">Overall Risk Score</h3>
            <div className="text-2xl font-bold">{riskMetrics.riskScore.toFixed(1)}/10</div>
          </div>
          <div className="text-3xl">🛡️</div>
        </div>
        <div className="mt-2 bg-white/20 rounded-full h-2">
          <div 
            className={`bg-white rounded-full h-2 transition-all duration-500 ${
              riskMetrics.riskScore <= 3 ? 'w-3/10' :
              riskMetrics.riskScore <= 5 ? 'w-5/10' :
              riskMetrics.riskScore <= 7 ? 'w-7/10' :
              'w-9/10'
            }`}
          ></div>
        </div>
      </div>

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-1 gap-3">
        <RiskCard
          icon="📉"
          title="Max Drawdown"
          value={riskMetrics.maxDrawdown}
          unit="%"
          status="warning"
          description="Maximum historical portfolio decline"
        />
        
        <RiskCard
          icon="⚡"
          title="Current Drawdown"
          value={riskMetrics.currentDrawdown}
          unit="%"
          status={getRiskStatus(Math.abs(riskMetrics.currentDrawdown), [3, 5])}
          description="Current portfolio decline from peak"
        />
        
        <RiskCard
          icon="💥"
          title="Value at Risk (1d)"
          value={riskMetrics.portfolioVaR}
          unit="$"
          status={getRiskStatus(riskMetrics.portfolioVaR, [500, 1000])}
          description="Potential 1-day loss at 95% confidence"
        />
        
        <RiskCard
          icon="📊"
          title="Volatility"
          value={riskMetrics.volatility}
          unit="%"
          status={getRiskStatus(riskMetrics.volatility, [15, 25])}
          description="Portfolio volatility (30-day)"
        />
      </div>
    </div>
  );
}
