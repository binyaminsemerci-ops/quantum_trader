// pages/performance.tsx
import { useEffect, useState } from 'react';
import InvestorNavbar from '@/components/InvestorNavbar';
import EquityChart from '@/components/EquityChart';
import LoadingSpinner from '@/components/LoadingSpinner';
import { useAuth } from '@/hooks/useAuth';

export default function Performance() {
  const [curve, setCurve] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { getToken } = useAuth();

  useEffect(() => {
    fetchPerformance();
  }, []);

  const fetchPerformance = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com';
      const token = getToken();
      
      const response = await fetch(`${apiUrl}/performance/metrics`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Failed to fetch performance');
      
      const data = await response.json();
      setCurve(data.curve || []);
    } catch (err) {
      console.error('Performance fetch error:', err);
      setError('Failed to load performance data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-quantum-text mb-2">Performance Analytics</h1>
          <p className="text-quantum-muted">Equity curve and historical performance</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {loading ? (
          <LoadingSpinner />
        ) : (
          <div className="space-y-6">
            <EquityChart data={curve} height={500} />
            
            {curve.length === 0 && (
              <div className="bg-quantum-card border border-quantum-border rounded-lg p-8 text-center text-quantum-muted">
                No performance data available yet
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
