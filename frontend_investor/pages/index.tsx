// pages/index.tsx
import { useEffect, useState } from 'react';
import InvestorNavbar from '@/components/InvestorNavbar';
import MetricCard from '@/components/MetricCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import { useAuth } from '@/hooks/useAuth';

interface Metrics {
  total_return?: number;
  winrate?: number;
  profit_factor?: number;
  sharpe?: number;
  sortino?: number;
  max_drawdown?: number;
}

export default function Home() {
  const [metrics, setMetrics] = useState<Metrics>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { getToken } = useAuth();

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com';
      const token = getToken();
      
      const response = await fetch(`${apiUrl}/performance/metrics`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Failed to fetch metrics');
      
      const data = await response.json();
      setMetrics(data.metrics || {});
    } catch (err) {
      console.error('Metrics fetch error:', err);
      setError('Failed to load performance metrics');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-quantum-text mb-2">
            QuantumFond Investor Dashboard
          </h1>
          <p className="text-quantum-muted">
            Real-time fund performance and AI-powered analytics
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading ? (
          <LoadingSpinner />
        ) : (
          <>
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              <MetricCard
                label="Total Return"
                value={metrics.total_return}
                format="currency"
                icon="💰"
                trend={metrics.total_return && metrics.total_return > 0 ? 'up' : 'down'}
              />
              <MetricCard
                label="Win Rate"
                value={metrics.winrate}
                format="percentage"
                icon="🎯"
              />
              <MetricCard
                label="Profit Factor"
                value={metrics.profit_factor}
                format="number"
                icon="📊"
              />
              <MetricCard
                label="Sharpe Ratio"
                value={metrics.sharpe}
                format="number"
                icon="📈"
              />
              <MetricCard
                label="Sortino Ratio"
                value={metrics.sortino}
                format="number"
                icon="🎲"
              />
              <MetricCard
                label="Max Drawdown"
                value={metrics.max_drawdown}
                format="percentage"
                icon="⚠️"
                trend="down"
              />
            </div>

            {/* Quick Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-quantum-card border border-quantum-border rounded-lg p-6">
                <h3 className="text-lg font-semibold text-quantum-text mb-4">
                  🤖 AI Engine Status
                </h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-quantum-muted">Active Models:</span>
                    <span className="text-quantum-accent font-medium">4 Ensemble</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-quantum-muted">Trading Mode:</span>
                    <span className="text-green-400 font-medium">AUTONOMOUS</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-quantum-muted">System Health:</span>
                    <span className="text-green-400 font-medium">OPERATIONAL</span>
                  </div>
                </div>
              </div>

              <div className="bg-quantum-card border border-quantum-border rounded-lg p-6">
                <h3 className="text-lg font-semibold text-quantum-text mb-4">
                  ⚠️ Risk Overview
                </h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-quantum-muted">Risk Level:</span>
                    <span className="text-yellow-400 font-medium">MODERATE</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-quantum-muted">Governor State:</span>
                    <span className="text-green-400 font-medium">NORMAL</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-quantum-muted">Exposure:</span>
                    <span className="text-quantum-text font-medium">65%</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
