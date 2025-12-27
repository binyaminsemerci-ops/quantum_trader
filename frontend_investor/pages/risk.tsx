// pages/risk.tsx
import { useEffect, useState } from 'react';
import InvestorNavbar from '@/components/InvestorNavbar';
import LoadingSpinner from '@/components/LoadingSpinner';
import { useAuth } from '@/hooks/useAuth';

interface RiskData {
  portfolio_exposure?: number;
  var95?: number;
  es?: number;
  ess_state?: string;
  current_drawdown?: number;
  risk_level?: string;
}

export default function Risk() {
  const [risk, setRisk] = useState<RiskData>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { getToken } = useAuth();

  useEffect(() => {
    fetchRisk();
  }, []);

  const fetchRisk = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com';
      const token = getToken();
      
      const response = await fetch(`${apiUrl}/risk/summary`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Failed to fetch risk data');
      
      const data = await response.json();
      setRisk(data);
    } catch (err) {
      console.error('Risk fetch error:', err);
      setError('Failed to load risk data');
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevelColor = (level?: string) => {
    switch (level?.toUpperCase()) {
      case 'LOW':
        return 'text-green-400';
      case 'MODERATE':
        return 'text-yellow-400';
      case 'HIGH':
        return 'text-red-400';
      default:
        return 'text-quantum-muted';
    }
  };

  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-quantum-text mb-2">Risk Overview</h1>
          <p className="text-quantum-muted">Portfolio risk metrics and monitoring</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {loading ? (
          <LoadingSpinner />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Risk Metrics Card */}
            <div className="bg-quantum-card border border-quantum-border rounded-lg p-6">
              <h3 className="text-lg font-semibold text-quantum-text mb-4 flex items-center">
                <span className="mr-2">📊</span>
                Risk Metrics
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-3 border-b border-quantum-border">
                  <span className="text-quantum-muted">Portfolio Exposure</span>
                  <span className="text-quantum-text font-medium">
                    {risk.portfolio_exposure !== undefined 
                      ? `${(risk.portfolio_exposure * 100).toFixed(1)}%` 
                      : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-3 border-b border-quantum-border">
                  <span className="text-quantum-muted">VaR (95%)</span>
                  <span className="text-quantum-text font-medium">
                    {risk.var95 !== undefined ? `$${risk.var95.toFixed(2)}` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-3 border-b border-quantum-border">
                  <span className="text-quantum-muted">Expected Shortfall</span>
                  <span className="text-quantum-text font-medium">
                    {risk.es !== undefined ? `$${risk.es.toFixed(2)}` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-3">
                  <span className="text-quantum-muted">Current Drawdown</span>
                  <span className="text-red-400 font-medium">
                    {risk.current_drawdown !== undefined 
                      ? `${(risk.current_drawdown * 100).toFixed(2)}%` 
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* System Status Card */}
            <div className="bg-quantum-card border border-quantum-border rounded-lg p-6">
              <h3 className="text-lg font-semibold text-quantum-text mb-4 flex items-center">
                <span className="mr-2">⚙️</span>
                System Status
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-3 border-b border-quantum-border">
                  <span className="text-quantum-muted">Governor State</span>
                  <span className="text-green-400 font-medium">
                    {risk.ess_state || 'NORMAL'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-3 border-b border-quantum-border">
                  <span className="text-quantum-muted">Risk Level</span>
                  <span className={`font-medium ${getRiskLevelColor(risk.risk_level)}`}>
                    {risk.risk_level || 'MODERATE'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-3">
                  <span className="text-quantum-muted">Protection</span>
                  <span className="text-green-400 font-medium">ACTIVE</span>
                </div>
              </div>
            </div>

            {/* Risk Explanation Card */}
            <div className="md:col-span-2 bg-quantum-card border border-quantum-border rounded-lg p-6">
              <h3 className="text-lg font-semibold text-quantum-text mb-4">
                📚 Risk Metrics Explained
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">VaR (Value at Risk)</h4>
                  <p className="text-quantum-muted">
                    Maximum expected loss over a given time period with 95% confidence.
                  </p>
                </div>
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Expected Shortfall</h4>
                  <p className="text-quantum-muted">
                    Average loss in worst-case scenarios beyond the VaR threshold.
                  </p>
                </div>
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Portfolio Exposure</h4>
                  <p className="text-quantum-muted">
                    Percentage of total capital currently deployed in active positions.
                  </p>
                </div>
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Governor State</h4>
                  <p className="text-quantum-muted">
                    Emergency Safety System (ESS) state controlling risk limits and circuit breakers.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
